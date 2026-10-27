"""Capacity-matched masked group Transformer pilot on GaussianPrediction.

This is a structured-keypoint experiment, not an exact MoGaF reproduction.
It uses only the 80-frame observed Hook prefix to form groups, label their
rigidity, and train one small masked Transformer per group.  A shuffled-route
control preserves group counts, expert capacity, loss types, seed, and updates.
"""

from __future__ import print_function

import json
import math
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get(
    "GAUSSIAN_PREDICTION_ROOT", ROOT / "external" / "gaussian_prediction"
))
sys.path.insert(0, str(OFFICIAL))

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args  # noqa: E402
from eval import render_kpts  # noqa: E402
from gaussian_renderer import GaussianModel  # noqa: E402
from metrics import evaluate  # noqa: E402
from motion_model.dataset import GCN3DDataset  # noqa: E402
from options.gaussian_option import Gaussian_Options  # noqa: E402
from scene import Scene  # noqa: E402


class MaskedMotionTransformer(nn.Module):
    def __init__(self, sequence_length=10, hidden=32, heads=8, feedforward=64,
                 dropout=0.2):
        super(MaskedMotionTransformer, self).__init__()
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(7, hidden)
        self.position = nn.Parameter(torch.zeros(sequence_length, hidden))
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(sequence_length * hidden, 7)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, sequence, temporal_mask=None):
        mean = sequence.mean(dim=1, keepdim=True)
        std = sequence.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-4)
        normalized = (sequence - mean) / std
        hidden = self.input_projection(normalized) + self.position.unsqueeze(0)
        if temporal_mask is not None and bool(temporal_mask.any()):
            replacement = self.mask_token.view(1, 1, -1) + self.position.unsqueeze(0)
            hidden = torch.where(temporal_mask.view(1, -1, 1), replacement, hidden)
        encoded = self.encoder(hidden)
        normalized_prediction = self.head(encoded.reshape(encoded.shape[0], -1))
        return normalized_prediction * std[:, 0] + mean[:, 0]


def _finite_state(state):
    for value in state.values():
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
            return False
    return True


def _align_quaternion_sequence(rotation):
    output = rotation.clone()
    for index in range(1, output.shape[0]):
        sign = torch.where(
            torch.sum(output[index] * output[index - 1], dim=-1, keepdim=True) < 0,
            -torch.ones_like(output[index, :, :1]),
            torch.ones_like(output[index, :, :1]),
        )
        output[index] = output[index] * sign
    return F.normalize(output, dim=-1)


def _kmeans(features, clusters, iterations=40):
    centers = [features[torch.argmax(torch.sum(features * features, dim=1))]]
    for _ in range(1, clusters):
        distance = torch.cdist(features, torch.stack(centers)).pow(2).min(dim=1)[0]
        centers.append(features[torch.argmax(distance)])
    centers = torch.stack(centers)
    labels = torch.full(
        (features.shape[0],), -1, dtype=torch.long, device=features.device
    )
    for _ in range(iterations):
        new_labels = torch.cdist(features, centers).argmin(dim=1)
        new_centers = []
        for group in range(clusters):
            mask = new_labels == group
            if bool(mask.any()):
                new_centers.append(features[mask].mean(dim=0))
            else:
                distance = torch.cdist(features, centers).pow(2).min(dim=1)[0]
                new_centers.append(features[torch.argmax(distance)])
        new_centers = torch.stack(new_centers)
        converged = torch.equal(new_labels, labels) and torch.allclose(
            new_centers, centers
        )
        labels = new_labels
        centers = new_centers
        if converged:
            break
    return labels, centers


def _motion_features(canonical, observed_xyz):
    trajectory = observed_xyz.permute(1, 0, 2).reshape(observed_xyz.shape[1], -1)
    trajectory = trajectory - trajectory.mean(dim=0, keepdim=True)
    _, _, right = torch.pca_lowrank(trajectory, q=6, center=False)
    coefficients = trajectory @ right[:, :6]
    canonical = (canonical - canonical.mean(dim=0, keepdim=True)) / canonical.std(
        dim=0, keepdim=True
    ).clamp_min(1.0e-6)
    coefficients = (
        coefficients - coefficients.mean(dim=0, keepdim=True)
    ) / coefficients.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    return torch.cat([canonical, coefficients], dim=1)


def _kabsch_align(source, target):
    source_mean = source.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.transpose(0, 1) @ target_centered
    left, _, right_t = torch.linalg.svd(covariance)
    correction = torch.ones(3, device=source.device, dtype=source.dtype)
    correction[-1] = torch.sign(torch.det(left @ right_t)).detach()
    rotation = left @ torch.diag(correction) @ right_t
    return source_centered @ rotation + target_mean


def _rigidity_labels(observed_xyz, labels, groups, threshold=0.10):
    rigid = []
    scores = []
    radii = []
    for group in range(groups):
        mask = labels == group
        reference = observed_xyz[0, mask]
        center = reference.mean(dim=0, keepdim=True)
        radius = torch.sqrt(torch.mean(torch.sum((reference - center) ** 2, dim=-1)))
        radius = radius.clamp_min(1.0e-5)
        residuals = []
        for time_index in range(0, observed_xyz.shape[0], 4):
            target = observed_xyz[time_index, mask]
            aligned = _kabsch_align(reference, target)
            residual = torch.sqrt(torch.mean(torch.sum((aligned - target) ** 2, dim=-1)))
            residuals.append(residual / radius)
        score = torch.median(torch.stack(residuals)).item()
        scores.append(score)
        radii.append(radius.item())
        rigid.append(bool(score <= threshold))
    return rigid, scores, radii


def _shuffled_labels(structured_labels, groups, seed):
    counts = [int((structured_labels == group).sum().item()) for group in range(groups)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(structured_labels.shape[0], generator=generator)
    labels = torch.empty_like(structured_labels.cpu())
    offset = 0
    for group, count in enumerate(counts):
        labels[permutation[offset:offset + count]] = group
        offset += count
    return labels.to(structured_labels.device)


def _group_edges(canonical, labels, groups, neighbors=4):
    edges = []
    for group in range(groups):
        indices = torch.nonzero(labels == group, as_tuple=False).squeeze(1)
        points = canonical[indices]
        if points.shape[0] <= 1:
            edges.append((torch.empty(0, dtype=torch.long, device=points.device),
                          torch.empty(0, dtype=torch.long, device=points.device)))
            continue
        k = min(neighbors + 1, points.shape[0])
        nearest = torch.cdist(points, points).topk(k, largest=False).indices[:, 1:]
        source = torch.arange(points.shape[0], device=points.device).view(-1, 1)
        source = source.expand_as(nearest).reshape(-1)
        target = nearest.reshape(-1)
        edges.append((source, target))
    return edges


def _temporal_mask(epoch, epochs, length, group):
    if epochs <= 1:
        ratio = 0.0
    else:
        ratio = 0.4 * (1.0 - float(epoch) / float(epochs - 1))
    span = int(round(ratio * length))
    mask = torch.zeros(length, dtype=torch.bool, device="cuda")
    if span > 0:
        start = (epoch * 3 + group * 5) % (length - span + 1)
        mask[start:start + span] = True
    return mask, ratio


def _physical_loss(prediction, last_state, canonical, rigid, radius, edges):
    predicted_xyz = prediction[:, :3]
    last_xyz = last_state[:, :3]
    radius_squared = max(radius * radius, 1.0e-8)
    if rigid and predicted_xyz.shape[0] >= 3:
        aligned = _kabsch_align(last_xyz, predicted_xyz)
        return torch.mean((predicted_xyz - aligned) ** 2) / radius_squared
    source, target = edges
    if source.numel() == 0:
        return torch.zeros((), device=prediction.device)
    displacement = predicted_xyz - last_xyz
    return torch.mean((displacement[source] - displacement[target]) ** 2) / radius_squared


def _build_state(observed_xyz, observed_r):
    observed_r = _align_quaternion_sequence(observed_r)
    return torch.cat([observed_xyz, observed_r], dim=-1)


def _load_runtime(args, model_parser, optimization_parser, pipeline_parser):
    dataset = model_parser.extract(args)
    optimization = optimization_parser.extract(args)
    pipeline = pipeline_parser.extract(args)
    checkpoint_path = os.path.join(
        dataset.model_path, "chkpnt%d.pth" % args.ckpt_iteration
    )
    model_state, optimizer_state, checkpoint_iteration = torch.load(
        checkpoint_path, map_location="cuda"
    )
    gaussians = GaussianModel(dataset.sh_degree, args)
    gaussians.set_inputDim(2 * optimization.time_freq, 6 * optimization.xyz_freq)
    gaussians.N_pcd_init = model_state["_xyz"].shape[0]
    gaussians.active_sh_degree = gaussians.max_sh_degree
    gaussians.final_kpts_num = model_state["super_gaussians"].shape[0]
    scene = Scene(dataset, gaussians, shuffle=False)
    gaussians.restore(optimizer_state, optimization, checkpoint_iteration)
    gaussians.load_state_dict(model_state, strict=False)
    gaussians.compile_sparse_bindings()
    gaussians.eval()
    for parameter in gaussians.parameters():
        parameter.requires_grad_(False)
    # The 4D representation is frozen for this forecasting experiment.  The
    # released dataset object otherwise retains its deformation autograd graph,
    # which is both wasteful and invalid across multiple forecaster epochs.
    with torch.no_grad():
        observed = GCN3DDataset(
            gaussians,
            optimization.time_freq,
            args.ckpt_iteration,
            args.model_path,
            args.source_path,
            max_time=args.max_time,
            input_size=args.sequence_length,
            output_size=1,
            split="train",
        )
    # Sparse bindings are only an exact trajectory-extraction cache here.
    # Restore the released renderer path for matched latency/evaluation.
    gaussians.compact_nearest = None
    gaussians.compact_weights_xyz = None
    gaussians.compact_weights_r = None
    future_views = scene.getTestCameras()
    strict_fit = bool(getattr(args, "strict_fit_no_future_views", False))
    if strict_fit:
        if future_views:
            raise RuntimeError("Strict structured fit must expose zero future views")
        future_count = 0
    elif getattr(args, "allow_repeated_future_times", False):
        future_times = sorted(set(
            round(float(np.asarray(view.time).reshape(-1)[0]), 7)
            for view in future_views
        ))
        future_count = len(future_times)
    else:
        future_count = len(future_views)
    expected_runtime_future = 0 if strict_fit else args.expected_future_frames
    if (len(observed.train_times) != args.expected_observed_frames or
            future_count != expected_runtime_future):
        raise RuntimeError(
            "Expected %d/%d observed/future times, got %d/%d (%d eval views)"
            % (
                args.expected_observed_frames,
                args.expected_future_frames,
                len(observed.train_times),
                future_count,
                len(future_views),
            )
        )
    return dataset, optimization, pipeline, gaussians, scene, observed, checkpoint_path


def _prepare_groups(args, gaussians, observed):
    features = _motion_features(
        gaussians.get_superGaussians.detach(), observed.kpts_xyz_train
    )
    structured_labels, centers = _kmeans(features, args.groups)
    rigid, rigidity_scores, radii = _rigidity_labels(
        observed.kpts_xyz_train, structured_labels, args.groups
    )
    if args.routing in ("structured", "structured_no_physics", "global"):
        labels = structured_labels
    elif args.routing == "shuffled":
        labels = _shuffled_labels(structured_labels, args.groups, 2024 * args.seed)
    else:
        raise ValueError("Unknown routing")
    counts = [int((labels == group).sum().item()) for group in range(args.groups)]
    edges = _group_edges(gaussians.get_superGaussians.detach(), labels, args.groups)
    return labels, structured_labels, centers, rigid, rigidity_scores, radii, counts, edges


def _new_models(args):
    model_count = 1 if args.routing == "global" else args.groups
    return nn.ModuleList([
        MaskedMotionTransformer(
            sequence_length=args.sequence_length,
            hidden=32,
            heads=8,
            feedforward=64,
            dropout=0.2,
        ) for _ in range(model_count)
    ]).cuda()


def _model_for_group(models, routing, group):
    return models[0] if routing == "global" else models[group]


def _audit_trainable(models, gaussians):
    representation_trainable = [
        name for name, parameter in gaussians.named_parameters()
        if parameter.requires_grad
    ]
    if representation_trainable:
        raise RuntimeError(
            "Unintended trainable Gaussian tensors: %s"
            % ", ".join(representation_trainable)
        )
    groups = []
    for name, parameter in models.named_parameters():
        if parameter.requires_grad:
            groups.append({
                "name": name,
                "shape": list(parameter.shape),
                "parameters": int(parameter.numel()),
            })
    print(json.dumps({
        "trainable_parameter_groups": groups,
        "trainable_parameter_count": int(sum(item["parameters"] for item in groups)),
        "trainable_gaussian_or_representation_tensors": representation_trainable,
    }))


def train(args, model_parser, optimization_parser, pipeline_parser):
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    dataset, _, _, gaussians, _, observed, representation_path = _load_runtime(
        args, model_parser, optimization_parser, pipeline_parser
    )
    groups = _prepare_groups(args, gaussians, observed)
    labels, structured_labels, centers, rigid, rigidity_scores, radii, counts, edges = groups
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    models = _new_models(args)
    _audit_trainable(models, gaussians)
    optimizer = torch.optim.Adam(models.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1.0e-5
    )
    windows = state.shape[0] - args.sequence_length
    history = []
    models.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        prediction_losses = []
        acceleration_losses = []
        physical_losses = []
        mask_ratio = 0.0
        for group in range(args.groups):
            node_mask = labels == group
            window = (epoch * 17 + group * 11) % windows
            sequence = state[window:window + args.sequence_length, node_mask].permute(1, 0, 2)
            target = state[window + args.sequence_length, node_mask]
            temporal_mask, mask_ratio = _temporal_mask(
                epoch, args.epochs, args.sequence_length, group
            )
            prediction = _model_for_group(models, args.routing, group)(
                sequence, temporal_mask
            )
            prediction_r = F.normalize(prediction[:, 3:], dim=-1)
            target_r = target[:, 3:]
            sign = torch.where(
                torch.sum(prediction_r * target_r, dim=-1, keepdim=True) < 0,
                -torch.ones_like(target_r[:, :1]),
                torch.ones_like(target_r[:, :1]),
            )
            target_r = target_r * sign
            prediction_loss = F.mse_loss(prediction[:, :3], target[:, :3])
            prediction_loss = prediction_loss + F.mse_loss(prediction_r, target_r)
            acceleration = prediction[:, :3] - 2.0 * sequence[:, -1, :3] + sequence[:, -2, :3]
            acceleration_loss = torch.mean(acceleration ** 2)
            physical_loss = _physical_loss(
                prediction,
                sequence[:, -1],
                gaussians.get_superGaussians.detach()[node_mask],
                rigid[group],
                radii[group],
                edges[group],
            )
            prediction_losses.append(prediction_loss)
            acceleration_losses.append(acceleration_loss)
            physical_losses.append(physical_loss)
        prediction_loss = torch.stack(prediction_losses).mean()
        acceleration_loss = torch.stack(acceleration_losses).mean()
        physical_weighted = []
        for group, value in enumerate(physical_losses):
            physical_weighted.append((0.1 if rigid[group] else 0.02) * value)
        physical_loss = torch.stack(physical_weighted).mean()
        if args.routing == "structured_no_physics":
            physical_loss = physical_loss * 0.0
        loss = prediction_loss + acceleration_loss + physical_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite loss at epoch %d" % epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            item = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "prediction": float(prediction_loss.item()),
                "acceleration": float(acceleration_loss.item()),
                "physical": float(physical_loss.item()),
                "mask_ratio": float(mask_ratio),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(item)
            print(json.dumps(item))

    checkpoint = {
        "method": args.experiment_name,
        "routing": args.routing,
        "model": models.state_dict(),
        "labels": labels.detach().cpu(),
        "structured_labels": structured_labels.detach().cpu(),
        "centers": centers.detach().cpu(),
        "rigid": rigid,
        "rigidity_scores": rigidity_scores,
        "radii": radii,
        "counts": counts,
        "history": history,
        "config": {
            "groups": args.groups,
            "sequence_length": args.sequence_length,
            "hidden": 32,
            "heads": 8,
            "feedforward": 64,
            "dropout": 0.2,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "mask_ratio_start": 0.4,
            "mask_ratio_end": 0.0,
            "acceleration_weight": 1.0,
            "rigid_weight": 0.1,
            "nonrigid_weight": 0.02,
            "physical_losses_enabled": args.routing != "structured_no_physics",
            "rigidity_threshold": 0.10,
            "seed": args.seed,
            "observed_frames": args.expected_observed_frames,
            "future_frames_used": 0,
        },
        "representation_checkpoint": representation_path,
        "train_seconds": time.time() - started,
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "parameter_count": sum(parameter.numel() for parameter in models.parameters()),
    }
    output_path = Path(args.checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(output_path))

    reloaded = torch.load(str(output_path), map_location="cuda")
    reload_models = _new_models(args)
    reload_models.load_state_dict(reloaded["model"])
    if not _finite_state(reloaded["model"]):
        raise RuntimeError("Saved model contains non-finite tensors")
    reload_models.eval()
    with torch.no_grad():
        group = 0
        node_mask = labels == group
        smoke_sequence = state[-args.sequence_length:, node_mask].permute(1, 0, 2)
        smoke_prediction = _model_for_group(
            reload_models, args.routing, group
        )(smoke_sequence)
        if not bool(torch.isfinite(smoke_prediction).all()):
            raise RuntimeError("Cold-reloaded inference is non-finite")
    print(json.dumps({
        "checkpoint": str(output_path),
        "bytes": os.path.getsize(str(output_path)),
        "parameter_count": checkpoint["parameter_count"],
        "counts": counts,
        "rigid": rigid,
        "rigidity_scores": rigidity_scores,
        "cold_reload": True,
    }, indent=2))


def _rollout(models, state, labels, groups, sequence_length, frames, routing):
    sequence = state[-sequence_length:].clone()
    xyz_predictions = []
    rotation_predictions = []
    models.eval()
    with torch.no_grad():
        for _ in range(frames):
            next_state = torch.empty_like(sequence[-1])
            for group in range(groups):
                mask = labels == group
                group_sequence = sequence[:, mask].permute(1, 0, 2)
                prediction = _model_for_group(models, routing, group)(group_sequence)
                prediction[:, 3:] = F.normalize(prediction[:, 3:], dim=-1)
                sign = torch.where(
                    torch.sum(prediction[:, 3:] * group_sequence[:, -1, 3:], dim=-1, keepdim=True) < 0,
                    -torch.ones_like(prediction[:, 3:4]),
                    torch.ones_like(prediction[:, 3:4]),
                )
                prediction[:, 3:] = prediction[:, 3:] * sign
                next_state[mask] = prediction
            xyz_predictions.append(next_state[:, :3])
            rotation_predictions.append(next_state[:, 3:])
            sequence = torch.cat([sequence[1:], next_state.unsqueeze(0)], dim=0)
    return torch.stack(xyz_predictions), torch.stack(rotation_predictions)


def evaluate_checkpoint(args, model_parser, optimization_parser, pipeline_parser):
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    dataset, _, pipeline, gaussians, scene, observed, representation_path = _load_runtime(
        args, model_parser, optimization_parser, pipeline_parser
    )
    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    if checkpoint["routing"] != args.routing:
        raise RuntimeError("Routing/checkpoint mismatch")
    if (checkpoint["config"]["observed_frames"] != args.expected_observed_frames or
            checkpoint["config"]["future_frames_used"] != 0):
        raise RuntimeError("Checkpoint protocol mismatch")
    models = _new_models(args)
    models.load_state_dict(checkpoint["model"])
    if not _finite_state(checkpoint["model"]):
        raise RuntimeError("Non-finite checkpoint")
    labels = checkpoint["labels"].to("cuda")
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    torch.cuda.synchronize()
    forecast_started = time.time()
    predicted_xyz, predicted_r = _rollout(
        models,
        state,
        labels,
        args.groups,
        args.sequence_length,
        args.expected_future_frames,
        args.routing,
    )
    torch.cuda.synchronize()
    forecast_seconds = time.time() - forecast_started
    if not bool(torch.isfinite(predicted_xyz).all()) or not bool(torch.isfinite(predicted_r).all()):
        raise RuntimeError("Non-finite rollout")

    views = scene.getTestCameras()
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    render_started = time.time()
    eval_root, _ = render_kpts(
        args.model_path,
        args.experiment_name,
        "[metrics]%s" % args.routing,
        args.ckpt_iteration,
        views,
        gaussians,
        pipeline,
        background,
        predicted_xyz,
        predicted_r,
        metrics=True,
    )
    render_seconds = time.time() - render_started
    evaluate(eval_root)
    with open(os.path.join(eval_root, "results.json"), "r") as handle:
        metrics = json.load(handle)
    summary = {
        "method": args.experiment_name,
        "routing": args.routing,
        "cold_reload": True,
        "scene": args.scene_name,
        "protocol": {
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "future_images_used_for_training_grouping_or_routing": False,
            "background": "white" if dataset.white_background else "black",
            "seed": args.seed,
        },
        "metrics": metrics,
        "resources": {
            "train_seconds": checkpoint["train_seconds"],
            "train_peak_gpu_allocated_mb": checkpoint["peak_gpu_allocated_mb"],
            "eval_seconds": time.time() - started,
            "forecast_head_seconds": forecast_seconds,
            "forecast_head_fps": args.expected_future_frames / forecast_seconds,
            "eval_peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
            "render_and_png_fps": args.expected_future_frames / render_seconds,
            "parameters": checkpoint["parameter_count"],
            "checkpoint_bytes": os.path.getsize(args.checkpoint),
        },
        "structure": {
            "groups": args.groups,
            "counts": checkpoint["counts"],
            "rigid": checkpoint["rigid"],
            "rigidity_scores": checkpoint["rigidity_scores"],
        },
        "artifacts": {
            "checkpoint": args.checkpoint,
            "representation_checkpoint": representation_path,
            "render_root": eval_root,
        },
    }
    output_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = ArgumentParser(description="Phase 77 masked group Transformer")
    optimization_parser = OptimizationParams(parser)
    model_parser = ModelParams(parser, sentinel=True)
    pipeline_parser = PipelineParams(parser)
    options = Gaussian_Options(parser)
    options.initial()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument(
        "--routing",
        choices=["structured", "shuffled", "structured_no_physics", "global"],
        required=True,
    )
    parser.add_argument("--ckpt_iteration", default=60000, type=int)
    parser.add_argument("--groups", default=8, type=int)
    parser.add_argument("--sequence_length", default=10, type=int)
    parser.add_argument("--epochs", default=2001, type=int)
    parser.add_argument("--learning_rate", default=1.0e-3, type=float)
    parser.add_argument("--scene_name", default="D-NeRF Hook", type=str)
    parser.add_argument("--expected_observed_frames", default=80, type=int)
    parser.add_argument("--expected_future_frames", default=20, type=int)
    parser.add_argument("--allow_repeated_future_times", action="store_true")
    parser.add_argument("--strict_fit_no_future_views", action="store_true")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--summary", default="", type=str)
    parser.add_argument(
        "--experiment_name", default="moregs_structured", type=str
    )
    args = get_combined_args(parser)

    seed = 2024 * args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if args.mode == "train":
        train(args, model_parser, optimization_parser, pipeline_parser)
    else:
        if not args.summary:
            raise ValueError("--summary is required for evaluation")
        evaluate_checkpoint(args, model_parser, optimization_parser, pipeline_parser)


if __name__ == "__main__":
    main()
