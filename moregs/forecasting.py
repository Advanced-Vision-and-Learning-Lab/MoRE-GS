"""Causal direct multi-horizon forecasting on frozen GaussianPrediction states.

The model sees only the observed prefix.  During training, cut points and
future horizons are sampled entirely inside that prefix.  At evaluation it
predicts the complete held-out suffix in one forward pass; no predicted state
is ever fed back as an input.  Motion is a gated residual around exact
persistence, and the Gaussian representation remains frozen.
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


def _finite_state(state):
    return all(
        not torch.is_tensor(value) or bool(torch.isfinite(value).all())
        for value in state.values()
    )


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


def _build_state(observed_xyz, observed_rotation):
    return torch.cat(
        [observed_xyz, _align_quaternion_sequence(observed_rotation)], dim=-1
    )


class DirectHorizonTransformer(nn.Module):
    """Shared target-scene model that predicts every horizon in parallel."""

    def __init__(self, sequence_length=10, hidden=32, heads=8, feedforward=64,
                 dropout=0.2):
        super(DirectHorizonTransformer, self).__init__()
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(7, hidden)
        self.canonical_projection = nn.Linear(3, hidden)
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
        self.horizon_projection = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, 7),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),
        )
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -3.0)

    @staticmethod
    def _horizon_features(horizon, scale):
        normalized = horizon / float(max(scale, 1))
        return torch.stack(
            [
                normalized,
                normalized.square(),
                torch.sin(math.pi * normalized),
                torch.cos(math.pi * normalized),
                torch.log1p(horizon) / math.log1p(float(max(scale, 1))),
            ],
            dim=-1,
        )

    def forward(self, sequence, canonical, horizons, horizon_scale,
                temporal_mask=None):
        # Per-keypoint temporal normalization prevents large static scene
        # coordinates from overwhelming the small motion residual.
        mean = sequence.mean(dim=1, keepdim=True)
        std = sequence.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-4)
        normalized = (sequence - mean) / std
        canonical_mean = canonical.mean(dim=0, keepdim=True)
        canonical_std = canonical.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-4)
        canonical_normalized = (canonical - canonical_mean) / canonical_std

        hidden = self.input_projection(normalized) + self.position.unsqueeze(0)
        hidden = hidden + self.canonical_projection(canonical_normalized).unsqueeze(1)
        if temporal_mask is not None and bool(temporal_mask.any()):
            replacement = self.mask_token.view(1, 1, -1) + self.position.unsqueeze(0)
            replacement = replacement + self.canonical_projection(
                canonical_normalized
            ).unsqueeze(1)
            hidden = torch.where(temporal_mask.view(1, -1, 1), replacement, hidden)
        context = self.encoder(hidden).mean(dim=1)

        horizon_features = self._horizon_features(horizons, horizon_scale)
        horizon_hidden = self.horizon_projection(horizon_features)
        context = context[:, None, :].expand(-1, horizons.shape[0], -1)
        horizon_hidden = horizon_hidden[None, :, :].expand(context.shape[0], -1, -1)
        joint = torch.cat([context, horizon_hidden], dim=-1)
        raw_residual = self.decoder(joint) * std[:, 0, :].unsqueeze(1)
        gates = torch.sigmoid(self.gate(joint))

        last = sequence[:, -1, :].unsqueeze(1)
        xyz = last[:, :, :3] + gates[:, :, :1] * raw_residual[:, :, :3]
        rotation = last[:, :, 3:] + gates[:, :, 1:] * raw_residual[:, :, 3:]
        rotation = F.normalize(rotation, dim=-1)
        sign = torch.where(
            torch.sum(rotation * last[:, :, 3:], dim=-1, keepdim=True) < 0,
            -torch.ones_like(rotation[:, :, :1]),
            torch.ones_like(rotation[:, :, :1]),
        )
        rotation = rotation * sign
        return torch.cat([xyz, rotation], dim=-1), gates


def _load_runtime(args, model_parser, optimization_parser, pipeline_parser):
    dataset = model_parser.extract(args)
    optimization = optimization_parser.extract(args)
    pipeline = pipeline_parser.extract(args)
    compact_path = getattr(args, "compact_checkpoint", None)
    if compact_path:
        payload = torch.load(compact_path, map_location="cpu")
        if payload.get("format") != "gp_compiled_sparse_v1":
            raise RuntimeError("Unsupported compact checkpoint format")
        model_state = payload["model_state"]
        checkpoint_iteration = int(payload["iteration"])
        args.compact_inference = True
        checkpoint_path = compact_path
    else:
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
    if compact_path:
        gaussians.inference_setup(checkpoint_iteration)
        gaussians.load_state_dict(model_state, strict=False)
        gaussians.load_sparse_bindings(payload["bindings"])
    else:
        gaussians.restore(optimizer_state, optimization, checkpoint_iteration)
        gaussians.load_state_dict(model_state, strict=False)
        gaussians.compile_sparse_bindings()
    gaussians.eval()
    for parameter in gaussians.parameters():
        parameter.requires_grad_(False)
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
    # Use compiled bindings only as an exact preprocessing cache; subsequent
    # rendering remains on the released dense-binding path for matched timing.
    gaussians.compact_nearest = None
    gaussians.compact_weights_xyz = None
    gaussians.compact_weights_r = None
    train_views = sorted(
        scene.getTrainCameras(),
        key=lambda view: float(np.asarray(view.time).reshape(-1)[0]),
    )
    test_views = sorted(
        scene.getTestCameras(),
        key=lambda view: float(np.asarray(view.time).reshape(-1)[0]),
    )
    if len(observed.train_times) != args.expected_observed_frames:
        raise RuntimeError("Observed-frame count mismatch")
    observed_times = [float(value) for value in observed.train_times]
    future_times = [float(np.asarray(view.time).reshape(-1)[0]) for view in test_views]
    strict_fit = bool(getattr(args, "strict_fit_no_future_views", False))
    if strict_fit:
        if test_views:
            raise RuntimeError("Strict fit runtime must expose zero future views")
        args.future_view_horizon_indices = []
        args.future_evaluation_view_count = 0
        args.future_unique_times = []
    elif getattr(args, "allow_repeated_future_times", False):
        unique_future_times = []
        for value in future_times:
            if not unique_future_times or not np.isclose(
                    value, unique_future_times[-1], rtol=0.0, atol=1.0e-7):
                unique_future_times.append(value)
        if len(unique_future_times) != args.expected_future_frames:
            raise RuntimeError(
                "Unique future-time count mismatch: expected %d, got %d views/%d times"
                % (args.expected_future_frames, len(test_views), len(unique_future_times))
            )
        horizon_indices = []
        for value in future_times:
            distances = np.abs(np.asarray(unique_future_times) - value)
            index = int(np.argmin(distances))
            if distances[index] > 1.0e-7:
                raise RuntimeError("Future view does not map to a forecast horizon")
            horizon_indices.append(index)
        args.future_view_horizon_indices = horizon_indices
        args.future_evaluation_view_count = len(test_views)
        args.future_unique_times = unique_future_times
    else:
        if len(test_views) != args.expected_future_frames:
            raise RuntimeError("Future-frame count mismatch")
        args.future_view_horizon_indices = list(range(len(test_views)))
        args.future_evaluation_view_count = len(test_views)
        args.future_unique_times = future_times
    if future_times and (
        set(observed_times) & set(future_times)
        or max(observed_times) >= min(future_times)
    ):
        raise RuntimeError("Training/evaluation time leakage")
    if len(train_views) != len(observed_times):
        raise RuntimeError("Observed state/view count mismatch")
    return (
        dataset, pipeline, gaussians, scene, observed, train_views, test_views,
        checkpoint_path,
    )


def _new_model(args):
    return DirectHorizonTransformer(
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


def _temporal_mask(epoch, epochs, length, device):
    ratio = 0.4 * (1.0 - float(epoch) / float(max(epochs - 1, 1)))
    span = int(round(ratio * length))
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    if span:
        start = (epoch * 3) % (length - span + 1)
        mask[start:start + span] = True
    return mask, ratio


def _forecast_losses(prediction, target, sequence, gates):
    target_rotation = target[:, :, 3:]
    sign = torch.where(
        torch.sum(prediction[:, :, 3:] * target_rotation, dim=-1, keepdim=True) < 0,
        -torch.ones_like(target_rotation[:, :, :1]),
        torch.ones_like(target_rotation[:, :, :1]),
    )
    target_rotation = target_rotation * sign
    xyz = F.mse_loss(prediction[:, :, :3], target[:, :, :3])
    rotation = F.mse_loss(prediction[:, :, 3:], target_rotation)

    last_xyz = sequence[:, -1, :3].unsqueeze(1)
    prediction_velocity = torch.cat(
        [prediction[:, :1, :3] - last_xyz,
         prediction[:, 1:, :3] - prediction[:, :-1, :3]], dim=1
    )
    target_velocity = torch.cat(
        [target[:, :1, :3] - last_xyz,
         target[:, 1:, :3] - target[:, :-1, :3]], dim=1
    )
    velocity = F.mse_loss(prediction_velocity, target_velocity)
    if prediction.shape[1] > 1:
        previous_velocity = sequence[:, -1, :3] - sequence[:, -2, :3]
        prediction_acceleration = torch.cat(
            [prediction_velocity[:, :1] - previous_velocity.unsqueeze(1),
             prediction_velocity[:, 1:] - prediction_velocity[:, :-1]], dim=1
        )
        target_acceleration = torch.cat(
            [target_velocity[:, :1] - previous_velocity.unsqueeze(1),
             target_velocity[:, 1:] - target_velocity[:, :-1]], dim=1
        )
        acceleration = F.mse_loss(prediction_acceleration, target_acceleration)
    else:
        acceleration = torch.zeros((), device=prediction.device)
    gate_regularizer = gates.mean()
    total = xyz + rotation + 0.5 * velocity + 0.1 * acceleration + 1.0e-5 * gate_regularizer
    return total, xyz, rotation, velocity, acceleration, gate_regularizer


def train(args, model_parser, optimization_parser, pipeline_parser):
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    runtime = _load_runtime(args, model_parser, optimization_parser, pipeline_parser)
    _, _, gaussians, _, observed, _, _, representation_path = runtime
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    model = _new_model(args)
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
        # Deterministic coverage of all legal prefix cut points.  The target
        # length is truncated near frame 79; future frames are never loaded.
        cutoff = valid_cutoffs[(epoch * 17 + 11) % len(valid_cutoffs)]
        available = min(max_horizon, state.shape[0] - cutoff)
        sequence = state[cutoff - args.sequence_length:cutoff].permute(1, 0, 2)
        target = state[cutoff:cutoff + available].permute(1, 0, 2)
        horizons = torch.arange(1, available + 1, device="cuda", dtype=state.dtype)
        temporal_mask, mask_ratio = _temporal_mask(
            epoch, args.epochs, args.sequence_length, state.device
        )
        prediction, gates = model(
            sequence, canonical, horizons, args.expected_future_frames, temporal_mask
        )
        losses = _forecast_losses(prediction, target, sequence, gates)
        loss, xyz, rotation, velocity, acceleration, gate_regularizer = losses
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
                "mean_gate": float(gate_regularizer.item()),
                "mask_ratio": float(mask_ratio),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(item)
            print(json.dumps(item))

    checkpoint = {
        "method": "phase80_direct_multihorizon",
        "model": model.state_dict(),
        "history": history,
        "config": {
            "sequence_length": args.sequence_length,
            "hidden": args.hidden,
            "heads": args.heads,
            "feedforward": args.feedforward,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_train_horizon": max_horizon,
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "future_frames_used": 0,
            "autoregressive": False,
            "persistence_anchored": True,
            "seed": args.seed,
        },
        "representation_checkpoint": representation_path,
        "train_seconds": time.time() - started,
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    output_path = Path(args.checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(output_path))
    reloaded = torch.load(str(output_path), map_location="cuda")
    reload_model = _new_model(args)
    reload_model.load_state_dict(reloaded["model"])
    reload_model.eval()
    with torch.no_grad():
        horizons = torch.arange(
            1, args.expected_future_frames + 1, device="cuda", dtype=state.dtype
        )
        prediction, _ = reload_model(
            state[-args.sequence_length:].permute(1, 0, 2),
            canonical, horizons, args.expected_future_frames,
        )
    if not _finite_state(reloaded["model"]) or not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Cold-reloaded checkpoint is non-finite")
    print(json.dumps({
        "checkpoint": str(output_path),
        "bytes": os.path.getsize(str(output_path)),
        "parameter_count": checkpoint["parameter_count"],
        "cold_reload": True,
    }, indent=2))


def evaluate_checkpoint(args, model_parser, optimization_parser, pipeline_parser):
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    runtime = _load_runtime(args, model_parser, optimization_parser, pipeline_parser)
    dataset, pipeline, gaussians, _, observed, _, test_views, representation_path = runtime
    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    config = checkpoint["config"]
    if (config["future_frames_used"] != 0 or config["autoregressive"] or
            config["observed_frames"] != args.expected_observed_frames or
            config["future_frames"] != args.expected_future_frames):
        raise RuntimeError("Checkpoint protocol mismatch")
    model = _new_model(args)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    horizons = torch.arange(
        1, args.expected_future_frames + 1, device="cuda", dtype=state.dtype
    )
    torch.cuda.synchronize()
    forecast_started = time.time()
    with torch.no_grad():
        prediction, gates = model(
            state[-args.sequence_length:].permute(1, 0, 2),
            canonical,
            horizons,
            args.expected_future_frames,
        )
    torch.cuda.synchronize()
    forecast_seconds = time.time() - forecast_started
    predicted_xyz = prediction[:, :, :3].permute(1, 0, 2).contiguous()
    predicted_rotation = prediction[:, :, 3:].permute(1, 0, 2).contiguous()
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Non-finite direct forecast")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    render_started = time.time()
    eval_root, _ = render_kpts(
        args.model_path,
        args.experiment_name,
        "[metrics]direct",
        args.ckpt_iteration,
        test_views,
        gaussians,
        pipeline,
        background,
        predicted_xyz,
        predicted_rotation,
        metrics=True,
    )
    render_seconds = time.time() - render_started
    evaluate(eval_root)
    with open(os.path.join(eval_root, "results.json"), "r") as handle:
        metrics = json.load(handle)
    with open(os.path.join(eval_root, "per_view.json"), "r") as handle:
        per_view = json.load(handle)
    psnr_values = list(per_view["PSNR"].values())
    quarter = max(1, len(psnr_values) // 4)
    horizon_psnr = [
        float(np.mean(psnr_values[start:min(start + quarter, len(psnr_values))]))
        for start in range(0, len(psnr_values), quarter)
    ]
    summary = {
        "method": "phase80_direct_multihorizon",
        "scene": args.scene_name,
        "cold_reload": True,
        "protocol": {
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "future_images_used_for_training_selection_or_early_stopping": False,
            "internal_pseudo_future_training": True,
            "autoregressive": False,
            "persistence_anchored": True,
            "seed": args.seed,
        },
        "metrics": metrics,
        "horizon_psnr": horizon_psnr,
        "gates": {
            "mean_xyz": float(gates[:, :, 0].mean().item()),
            "mean_rotation": float(gates[:, :, 1].mean().item()),
            "last_horizon_xyz": float(gates[:, -1, 0].mean().item()),
            "last_horizon_rotation": float(gates[:, -1, 1].mean().item()),
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
    parser = ArgumentParser(description="Phase 80 direct multi-horizon forecaster")
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
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--feedforward", default=64, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--scene_name", required=True, type=str)
    parser.add_argument("--expected_observed_frames", required=True, type=int)
    parser.add_argument("--expected_future_frames", required=True, type=int)
    parser.add_argument("--allow_repeated_future_times", action="store_true")
    parser.add_argument("--strict_fit_no_future_views", action="store_true")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--summary", default="", type=str)
    parser.add_argument("--experiment_name", default="moregs_forecasting", type=str)
    args = get_combined_args(parser)

    seed = 2024 * args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if args.mode == "train":
        train(args, model_parser, optimization_parser, pipeline_parser)
    else:
        if not args.summary:
            raise ValueError("--summary is required in eval mode")
        evaluate_checkpoint(args, model_parser, optimization_parser, pipeline_parser)


if __name__ == "__main__":
    main()
