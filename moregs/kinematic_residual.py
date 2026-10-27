"""Causal direct forecasting with a shared group-SE(3) residual.

The base is the Phase-80 non-autoregressive direct Transformer.  One shared
group encoder predicts a continuously gated rigid transform for each
support-only motion group and horizon.  The transform is added as a residual
to the base prediction; zero transform is exactly the Phase-80 architecture.
No Gaussian, appearance, opacity, or representation tensor is trainable.
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
from metrics import evaluate  # noqa: E402
from options.gaussian_option import Gaussian_Options  # noqa: E402
from moregs.forecasting import (  # noqa: E402
    DirectHorizonTransformer,
    _build_state,
    _forecast_losses,
    _load_runtime,
    _temporal_mask,
)


def _axis_angle_matrix(vector):
    """Stable Rodrigues map for [..., 3] axis-angle vectors."""
    angle_squared = torch.sum(vector * vector, dim=-1, keepdim=True)
    angle = torch.sqrt(angle_squared.clamp_min(1.0e-16))
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ], dim=-1).reshape(vector.shape[:-1] + (3, 3))
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    identity = identity.expand(vector.shape[:-1] + (3, 3))
    small = angle_squared < 1.0e-8
    coefficient_a = torch.where(
        small,
        1.0 - angle_squared / 6.0,
        torch.sin(angle) / angle.clamp_min(1.0e-8),
    ).unsqueeze(-1)
    coefficient_b = torch.where(
        small,
        0.5 - angle_squared / 24.0,
        (1.0 - torch.cos(angle)) / angle_squared.clamp_min(1.0e-8),
    ).unsqueeze(-1)
    return identity + coefficient_a * skew + coefficient_b * torch.matmul(skew, skew)


def _axis_angle_quaternion(vector):
    angle_squared = torch.sum(vector * vector, dim=-1, keepdim=True)
    angle = torch.sqrt(angle_squared.clamp_min(1.0e-16))
    half = 0.5 * angle
    scale = torch.where(
        angle_squared < 1.0e-8,
        0.5 - angle_squared / 48.0,
        torch.sin(half) / angle.clamp_min(1.0e-8),
    )
    xyz = vector * scale
    return torch.cat([torch.cos(half), xyz], dim=-1)


def _quaternion_multiply(left, right):
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], dim=-1)


class SharedKinematicResidual(nn.Module):
    def __init__(self, labels, rigidity_scores, sequence_length=10, hidden=32,
                 heads=8, feedforward=64, dropout=0.2):
        super(SharedKinematicResidual, self).__init__()
        self.sequence_length = sequence_length
        self.groups = int(labels.max().item()) + 1
        self.register_buffer("labels", labels.long())
        self.register_buffer(
            "rigidity_scores", torch.as_tensor(rigidity_scores, dtype=torch.float32)
        )
        self.base = DirectHorizonTransformer(
            sequence_length=sequence_length,
            hidden=hidden,
            heads=heads,
            feedforward=feedforward,
            dropout=dropout,
        )
        self.group_input = nn.Linear(7, hidden)
        self.group_position = nn.Parameter(torch.zeros(sequence_length, hidden))
        self.group_token = nn.Embedding(self.groups, hidden)
        self.rigidity_projection = nn.Linear(1, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.group_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.horizon_projection = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.transform_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 6)
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )
        nn.init.normal_(self.group_position, std=0.02)
        nn.init.normal_(self.group_token.weight, std=0.02)
        nn.init.zeros_(self.transform_head[-1].weight)
        nn.init.zeros_(self.transform_head[-1].bias)
        nn.init.zeros_(self.residual_gate[-1].weight)
        nn.init.constant_(self.residual_gate[-1].bias, -3.0)

    def _group_features(self, sequence):
        features = []
        for group in range(self.groups):
            mask = self.labels == group
            xyz = sequence[mask, :, :3]
            centroid = xyz.mean(dim=0)
            velocity = torch.cat(
                [torch.zeros_like(centroid[:1]), centroid[1:] - centroid[:-1]], dim=0
            )
            radius = torch.sqrt(
                torch.mean(torch.sum((xyz - centroid.unsqueeze(0)) ** 2, dim=-1), dim=0)
            ).unsqueeze(-1)
            features.append(torch.cat([centroid, velocity, radius], dim=-1))
        features = torch.stack(features)
        mean = features.mean(dim=1, keepdim=True)
        std = features.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-4)
        return (features - mean) / std

    def forward(self, sequence, canonical, horizons, horizon_scale,
                temporal_mask=None):
        base_prediction, base_gates = self.base(
            sequence, canonical, horizons, horizon_scale, temporal_mask
        )
        group_features = self._group_features(sequence)
        hidden = self.group_input(group_features) + self.group_position.unsqueeze(0)
        indices = torch.arange(self.groups, device=sequence.device)
        hidden = hidden + self.group_token(indices).unsqueeze(1)
        hidden = hidden + self.rigidity_projection(
            self.rigidity_scores.view(-1, 1)
        ).unsqueeze(1)
        if temporal_mask is not None and bool(temporal_mask.any()):
            hidden = torch.where(
                temporal_mask.view(1, -1, 1),
                (self.group_token(indices) + self.rigidity_projection(
                    self.rigidity_scores.view(-1, 1)
                )).unsqueeze(1) + self.group_position.unsqueeze(0),
                hidden,
            )
        context = self.group_encoder(hidden).mean(dim=1)
        horizon_features = DirectHorizonTransformer._horizon_features(
            horizons, horizon_scale
        )
        horizon_hidden = self.horizon_projection(horizon_features)
        joint = torch.cat([
            context[:, None, :].expand(-1, horizons.shape[0], -1),
            horizon_hidden[None, :, :].expand(self.groups, -1, -1),
        ], dim=-1)
        transform = self.transform_head(joint)
        gate = torch.sigmoid(self.residual_gate(joint))

        scene_scale = canonical.std(dim=0, unbiased=False).mean().clamp_min(1.0e-4)
        translation = torch.tanh(transform[:, :, :3]) * (0.25 * scene_scale)
        axis_angle = torch.tanh(transform[:, :, 3:]) * 0.5
        translation = gate * translation
        axis_angle = gate * axis_angle

        output = base_prediction.clone()
        for group in range(self.groups):
            mask = self.labels == group
            last_xyz = sequence[mask, -1, :3]
            center = last_xyz.mean(dim=0)
            relative = last_xyz - center
            rotation_matrix = _axis_angle_matrix(axis_angle[group])
            rotated = torch.einsum("hij,nj->nhi", rotation_matrix, relative)
            rigid_xyz = center.view(1, 1, 3) + rotated
            rigid_xyz = rigid_xyz + translation[group].unsqueeze(0)
            coherent_delta = rigid_xyz - last_xyz.unsqueeze(1)
            output[mask, :, :3] = base_prediction[mask, :, :3] + coherent_delta

            delta_quaternion = _axis_angle_quaternion(axis_angle[group])
            delta_quaternion = delta_quaternion.unsqueeze(0).expand(
                int(mask.sum().item()), -1, -1
            )
            rotation = _quaternion_multiply(
                delta_quaternion, base_prediction[mask, :, 3:]
            )
            rotation = F.normalize(rotation, dim=-1)
            reference = base_prediction[mask, :, 3:]
            sign = torch.where(
                torch.sum(rotation * reference, dim=-1, keepdim=True) < 0,
                -torch.ones_like(rotation[..., :1]),
                torch.ones_like(rotation[..., :1]),
            )
            output[mask, :, 3:] = rotation * sign

        node_gate = gate[self.labels]
        all_gates = torch.cat([base_gates, node_gate], dim=-1)
        return output, all_gates, gate, transform


def _load_group_metadata(args, node_count):
    checkpoint = torch.load(args.group_checkpoint, map_location="cpu")
    if checkpoint.get("routing") != "structured":
        raise RuntimeError("Group checkpoint is not structured")
    if checkpoint["config"]["observed_frames"] != args.expected_observed_frames:
        raise RuntimeError("Group checkpoint observed-frame mismatch")
    if checkpoint["config"]["future_frames_used"] != 0:
        raise RuntimeError("Group checkpoint used future frames")
    labels = checkpoint["labels"].long()
    if labels.numel() != node_count:
        raise RuntimeError("Group/node count mismatch")
    if sorted(torch.unique(labels).tolist()) != list(range(args.groups)):
        raise RuntimeError("Invalid group labels")
    return labels.cuda(), checkpoint["rigidity_scores"], checkpoint["counts"]


def _new_model(args, labels, rigidity_scores):
    return SharedKinematicResidual(
        labels=labels,
        rigidity_scores=rigidity_scores,
        sequence_length=args.sequence_length,
        hidden=args.hidden,
        heads=args.heads,
        feedforward=args.feedforward,
        dropout=args.dropout,
    ).cuda()


def _audit_trainable(model, gaussians):
    representation = [
        name for name, parameter in gaussians.named_parameters()
        if parameter.requires_grad
    ]
    if representation:
        raise RuntimeError("Unintended trainable Gaussian tensors: %s" % representation)
    groups = [
        {"name": name, "shape": list(parameter.shape), "parameters": parameter.numel()}
        for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    print(json.dumps({
        "trainable_parameter_groups": groups,
        "trainable_parameter_count": int(sum(item["parameters"] for item in groups)),
        "trainable_gaussian_or_representation_tensors": representation,
    }))


def _finite_state(state):
    return all(
        not torch.is_tensor(value) or bool(torch.isfinite(value).all())
        for value in state.values()
    )


def train(args, model_parser, optimization_parser, pipeline_parser):
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    runtime = _load_runtime(args, model_parser, optimization_parser, pipeline_parser)
    _, _, gaussians, _, observed, _, _, representation_path = runtime
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    labels, rigidity_scores, counts = _load_group_metadata(args, state.shape[1])
    model = _new_model(args, labels, rigidity_scores)
    _audit_trainable(model, gaussians)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1.0e-5
    )
    max_horizon = min(
        args.max_train_horizon,
        args.expected_future_frames,
        state.shape[0] - args.sequence_length,
    )
    valid_cutoffs = list(range(args.sequence_length, state.shape[0]))
    history = []
    model.train()
    for epoch in range(args.epochs):
        cutoff = valid_cutoffs[(epoch * 17 + 11) % len(valid_cutoffs)]
        available = min(max_horizon, state.shape[0] - cutoff)
        sequence = state[cutoff - args.sequence_length:cutoff].permute(1, 0, 2)
        target = state[cutoff:cutoff + available].permute(1, 0, 2)
        horizons = torch.arange(1, available + 1, device="cuda", dtype=state.dtype)
        temporal_mask, mask_ratio = _temporal_mask(
            epoch, args.epochs, args.sequence_length, state.device
        )
        prediction, gates, group_gate, transform = model(
            sequence, canonical, horizons, args.expected_future_frames, temporal_mask
        )
        losses = _forecast_losses(prediction, target, sequence, gates)
        loss, xyz, rotation, velocity, acceleration, gate_regularizer = losses
        transform_regularizer = torch.mean(transform ** 2)
        loss = loss + args.transform_regularizer * transform_regularizer
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite loss at epoch %d" % epoch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            item = {
                "epoch": epoch,
                "cutoff": cutoff,
                "horizons": available,
                "loss": float(loss.item()),
                "xyz": float(xyz.item()),
                "rotation": float(rotation.item()),
                "velocity": float(velocity.item()),
                "acceleration": float(acceleration.item()),
                "mean_base_and_group_gate": float(gate_regularizer.item()),
                "mean_group_gate": float(group_gate.mean().item()),
                "transform_regularizer": float(transform_regularizer.item()),
                "mask_ratio": float(mask_ratio),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(item)
            print(json.dumps(item))

    checkpoint = {
        "method": "phase82_shared_kinematic_residual",
        "model": model.state_dict(),
        "history": history,
        "config": {
            "sequence_length": args.sequence_length,
            "hidden": args.hidden,
            "heads": args.heads,
            "feedforward": args.feedforward,
            "dropout": args.dropout,
            "groups": args.groups,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_train_horizon": max_horizon,
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "future_frames_used": 0,
            "autoregressive": False,
            "persistence_anchored": True,
            "grouping_observed_prefix_only": True,
            "shared_group_encoder": True,
            "se3_residual": True,
            "opacity_trainable": False,
            "seed": args.seed,
        },
        "group_checkpoint": args.group_checkpoint,
        "group_counts": counts,
        "representation_checkpoint": representation_path,
        "train_seconds": time.time() - started,
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    path = Path(args.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))
    reloaded = torch.load(str(path), map_location="cuda")
    reload_model = _new_model(args, labels, rigidity_scores)
    reload_model.load_state_dict(reloaded["model"])
    reload_model.eval()
    with torch.no_grad():
        horizons = torch.arange(
            1, args.expected_future_frames + 1, device="cuda", dtype=state.dtype
        )
        prediction, _, _, _ = reload_model(
            state[-args.sequence_length:].permute(1, 0, 2),
            canonical,
            horizons,
            args.expected_future_frames,
        )
    if not _finite_state(reloaded["model"]) or not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Cold reload failed")
    print(json.dumps({
        "checkpoint": str(path),
        "checkpoint_bytes": os.path.getsize(str(path)),
        "parameter_count": checkpoint["parameter_count"],
        "cold_reload": True,
    }, indent=2))


def evaluate_checkpoint(args, model_parser, optimization_parser, pipeline_parser):
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    runtime = _load_runtime(args, model_parser, optimization_parser, pipeline_parser)
    dataset, pipeline, gaussians, _, observed, _, test_views, representation_path = runtime
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    labels, rigidity_scores, counts = _load_group_metadata(args, state.shape[1])
    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    config = checkpoint["config"]
    if (config["future_frames_used"] != 0 or config["autoregressive"] or
            config["observed_frames"] != args.expected_observed_frames or
            config["future_frames"] != args.expected_future_frames):
        raise RuntimeError("Checkpoint protocol mismatch")
    model = _new_model(args, labels, rigidity_scores)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    horizons = torch.arange(
        1, args.expected_future_frames + 1, device="cuda", dtype=state.dtype
    )
    torch.cuda.synchronize()
    forecast_started = time.time()
    with torch.no_grad():
        prediction, _, group_gate, transform = model(
            state[-args.sequence_length:].permute(1, 0, 2),
            canonical,
            horizons,
            args.expected_future_frames,
        )
    torch.cuda.synchronize()
    forecast_seconds = time.time() - forecast_started
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Non-finite forecast")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda"
    )
    render_started = time.time()
    eval_root, _ = render_kpts(
        args.model_path,
        args.experiment_name,
        "[metrics]shared_kinematic",
        args.ckpt_iteration,
        test_views,
        gaussians,
        pipeline,
        background,
        prediction[:, :, :3].permute(1, 0, 2).contiguous(),
        prediction[:, :, 3:].permute(1, 0, 2).contiguous(),
        metrics=True,
    )
    render_seconds = time.time() - render_started
    evaluate(eval_root)
    with open(os.path.join(eval_root, "results.json"), "r") as handle:
        metrics = json.load(handle)
    with open(os.path.join(eval_root, "per_view.json"), "r") as handle:
        per_view = json.load(handle)
    summary = {
        "method": "phase82_shared_kinematic_residual",
        "scene": args.scene_name,
        "cold_reload": True,
        "protocol": config,
        "metrics": metrics,
        "per_view": per_view,
        "kinematic": {
            "group_counts": counts,
            "rigidity_scores": [float(value) for value in rigidity_scores],
            "mean_gate": float(group_gate.mean().item()),
            "gate_by_group": group_gate.mean(dim=1).squeeze(-1).detach().cpu().tolist(),
            "mean_absolute_transform": float(transform.abs().mean().item()),
        },
        "resources": {
            "train_seconds": checkpoint["train_seconds"],
            "train_peak_gpu_allocated_mb": checkpoint["peak_gpu_allocated_mb"],
            "forecast_head_seconds": forecast_seconds,
            "forecast_head_fps": args.expected_future_frames / forecast_seconds,
            "render_and_png_fps": args.expected_future_frames / render_seconds,
            "eval_peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            "parameters": checkpoint["parameter_count"],
            "checkpoint_bytes": os.path.getsize(args.checkpoint),
        },
        "artifacts": {
            "checkpoint": args.checkpoint,
            "group_checkpoint": args.group_checkpoint,
            "representation_checkpoint": representation_path,
            "render_root": eval_root,
        },
        "evaluation_seconds": time.time() - started,
    }
    path = Path(args.summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = ArgumentParser(description="Phase 82 shared group-SE3 residual")
    optimization_parser = OptimizationParams(parser)
    model_parser = ModelParams(parser, sentinel=True)
    pipeline_parser = PipelineParams(parser)
    options = Gaussian_Options(parser)
    options.initial()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--ckpt_iteration", default=60000, type=int)
    parser.add_argument("--sequence_length", default=10, type=int)
    parser.add_argument("--max_train_horizon", default=40, type=int)
    parser.add_argument("--epochs", default=2001, type=int)
    parser.add_argument("--learning_rate", default=1.0e-3, type=float)
    parser.add_argument("--weight_decay", default=1.0e-4, type=float)
    parser.add_argument("--transform_regularizer", default=1.0e-5, type=float)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--feedforward", default=64, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--groups", default=8, type=int)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--expected_observed_frames", required=True, type=int)
    parser.add_argument("--expected_future_frames", required=True, type=int)
    parser.add_argument("--group_checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--experiment_name", default="moregs_kinematic_residual")
    args = get_combined_args(parser)

    seed = 2024 * args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if args.mode == "train":
        train(args, model_parser, optimization_parser, pipeline_parser)
    else:
        if not args.summary:
            raise ValueError("--summary required in eval mode")
        evaluate_checkpoint(args, model_parser, optimization_parser, pipeline_parser)


if __name__ == "__main__":
    main()
