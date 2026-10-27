"""Cross-fitted rendered-risk mixture of motion experts.

This experiment isolates the routing failure found in Phases 81 and 86.  A
calibration expert bank is trained on the first 75% of the *observed* prefix.
Its persistence, direct, rigid-SE(3), and local-nonrigid candidates are scored
by rendering the disjoint final 25% of that observed prefix.  The resulting
group/horizon routes are frozen.  A fresh, identically shaped expert bank is
then trained on the complete observed prefix and evaluated on the untouched
future suffix.  A matched latent-trajectory router is retained as the causal
control.

No future image, keypoint, or metric is read before both router checkpoints
and the final expert checkpoint have been written.
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
from gaussian_renderer import render_motion  # noqa: E402
from metrics import evaluate  # noqa: E402
from options.gaussian_option import Gaussian_Options  # noqa: E402
from moregs.forecasting import (  # noqa: E402
    DirectHorizonTransformer,
    _build_state,
    _finite_state,
    _forecast_losses,
    _load_runtime,
    _temporal_mask,
)
from moregs.grouping import (  # noqa: E402
    RegimeAwareForecaster,
    _candidate_point_error,
    _weighted_group_mean,
    build_soft_motion_groups,
)


EXPERTS = ("persistence", "direct", "rigid_se3", "local_nonrigid")


class RenderedRiskExpertBank(nn.Module):
    """Independent global-direct and typed group/local motion experts."""

    def __init__(self, assignment, rigidity, args):
        super(RenderedRiskExpertBank, self).__init__()
        self.direct = DirectHorizonTransformer(
            sequence_length=args.sequence_length,
            hidden=args.hidden,
            heads=args.heads,
            feedforward=args.feedforward,
            dropout=args.dropout,
        )
        self.typed = RegimeAwareForecaster(
            assignment=assignment,
            rigidity=rigidity,
            sequence_length=args.sequence_length,
            hidden=args.hidden,
            heads=args.heads,
            feedforward=args.feedforward,
            dropout=args.dropout,
            local_rank=args.local_rank,
        )

    def forward(self, sequence, canonical, horizons, horizon_scale,
                temporal_mask=None):
        direct, _ = self.direct(
            sequence, canonical, horizons, horizon_scale, temporal_mask
        )
        typed, transform = self.typed(
            sequence, canonical, horizons, horizon_scale, temporal_mask
        )
        return torch.stack([
            typed[:, :, 0], direct, typed[:, :, 1], typed[:, :, 2]
        ], dim=2), transform


def _new_bank(args, assignment, rigidity):
    return RenderedRiskExpertBank(assignment, rigidity, args).cuda()


def _audit_trainable(model, gaussians):
    representation = [
        name for name, parameter in gaussians.named_parameters()
        if parameter.requires_grad
    ]
    if representation:
        raise RuntimeError("Unintended trainable Gaussian tensors: %s" % representation)
    groups = [
        {"name": name, "shape": list(parameter.shape),
         "parameters": int(parameter.numel())}
        for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    print(json.dumps({
        "trainable_parameter_groups": groups,
        "trainable_parameter_count": int(sum(item["parameters"] for item in groups)),
        "trainable_gaussian_or_representation_tensors": representation,
    }))


def _fit_bank(args, model, state, canonical, end_frame, epochs, tag):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1.0e-5
    )
    valid_cutoffs = list(range(args.sequence_length, end_frame))
    if not valid_cutoffs:
        raise RuntimeError("Insufficient frames for expert training")
    max_horizon = min(
        args.max_train_horizon,
        args.expected_future_frames,
        end_frame - args.sequence_length,
    )
    history = []
    model.train()
    for epoch in range(epochs):
        cutoff = valid_cutoffs[(epoch * 17 + 11) % len(valid_cutoffs)]
        available = min(max_horizon, end_frame - cutoff)
        sequence = state[cutoff - args.sequence_length:cutoff].permute(1, 0, 2)
        target = state[cutoff:cutoff + available].permute(1, 0, 2)
        horizons = torch.arange(
            1, available + 1, device=state.device, dtype=state.dtype
        )
        temporal_mask, mask_ratio = _temporal_mask(
            epoch, epochs, args.sequence_length, state.device
        )
        candidates, transform = model(
            sequence, canonical, horizons, args.expected_future_frames,
            temporal_mask,
        )
        expert_losses = []
        for expert_index in range(1, len(EXPERTS)):
            expert_losses.append(_forecast_losses(
                candidates[:, :, expert_index], target, sequence,
                torch.ones(candidates.shape[0], available, 2, device=state.device),
            )[0])
        local_delta = candidates[:, :, 3, :3] - sequence[:, -1:, :3]
        assignment = model.typed.assignment
        group_delta = _weighted_group_mean(local_delta, assignment)
        smooth_delta = torch.einsum("ng,ghd->nhd", assignment, group_delta)
        local_smoothness = F.mse_loss(local_delta, smooth_delta)
        loss = (
            sum(expert_losses)
            + args.local_smoothness_weight * local_smoothness
            + args.transform_regularizer * torch.mean(transform.square())
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite %s loss at epoch %d" % (tag, epoch))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch in (0, epochs - 1):
            item = {
                "bank": tag, "epoch": epoch, "cutoff": cutoff,
                "horizons": available, "loss": float(loss.item()),
                "expert_losses": [float(value.item()) for value in expert_losses],
                "mask_ratio": float(mask_ratio),
            }
            history.append(item)
            print(json.dumps(item))
    model.eval()
    return history


def _forecast(model, state, canonical, args, frames=None):
    if frames is None:
        frames = args.expected_future_frames
    horizons = torch.arange(1, frames + 1, device=state.device, dtype=state.dtype)
    with torch.no_grad():
        candidates, _ = model(
            state[-args.sequence_length:].permute(1, 0, 2), canonical,
            horizons, args.expected_future_frames,
        )
    return candidates


def _render_prediction(view, prediction, gaussians, pipeline, background,
                       weights_xyz, weights_rotation):
    xyz_final = gaussians.get_xyz + weights_xyz @ (
        prediction[:, :3] - gaussians.super_gaussians
    )
    delta_rotation = gaussians.get_rotation_(gaussians.rotation_activation(
        weights_rotation @ prediction[:, 3:]
    ))
    return render_motion(
        view, gaussians, pipeline, background, xyz_t=xyz_final,
        r_t=delta_rotation, opacity=gaussians.lifecycle_opacity,
    )["render"]


def _foreground_mask(gt, background):
    distance = torch.mean(torch.abs(gt - background.view(3, 1, 1)), dim=0)
    return distance > 0.04


def _image_risk(rendering, gt, persistence_render, background, args):
    squared = torch.mean((rendering - gt) ** 2, dim=0)
    full = squared.mean()
    foreground = _foreground_mask(gt, background)
    dynamic = torch.mean(torch.abs(gt - persistence_render), dim=0) > 0.025
    important = foreground | dynamic
    if bool(important.any()):
        important_error = squared[important].mean()
    else:
        important_error = full
    return full + args.rendered_region_weight * important_error


def _bin_routes(risks, args):
    """Convert [G,H,E] observed-prefix risks to conservative one-hot routes."""
    groups, horizons, experts = risks.shape
    weights = torch.zeros(
        groups, args.calibration_bins, experts, dtype=torch.float32
    )
    records = []
    for group in range(groups):
        for bin_index, indices in enumerate(np.array_split(np.arange(horizons), args.calibration_bins)):
            mean = risks[group, indices].mean(dim=0)
            best_value, best_index = mean.min(dim=0)
            persistence = mean[0].clamp_min(1.0e-10)
            relative_gain = float(((persistence - best_value) / persistence).item())
            selected = int(best_index.item())
            if selected != 0 and relative_gain < args.minimum_relative_gain:
                selected = 0
            weights[group, bin_index, selected] = 1.0
            records.append({
                "group": group,
                "bin": bin_index,
                "samples": int(len(indices)),
                "mean_risk": mean.tolist(),
                "selected": EXPERTS[selected],
                "relative_gain_over_persistence": relative_gain,
            })
    return weights, records


def _latent_calibration(args, candidates, target, assignment):
    point_error = _candidate_point_error(candidates, target)
    group_error = torch.einsum(
        "ng,nhe->ghe", assignment, point_error
    ) / assignment.sum(dim=0).clamp_min(1.0e-6)[:, None, None]
    return _bin_routes(group_error.detach().cpu(), args)


def _rendered_calibration(args, candidates, target, assignment, views,
                          gaussians, pipeline, background):
    groups = assignment.shape[1]
    horizons = candidates.shape[1]
    risks = torch.empty(groups, horizons, len(EXPERTS), dtype=torch.float64)
    time_value = torch.from_numpy(views[0].time).to(torch.float32).cuda()
    with torch.no_grad():
        _, _, _, _, weights_xyz, weights_rotation = gaussians(
            time_value, args.ckpt_iteration, return_weights=True
        )
        for horizon in range(horizons):
            gt = views[horizon].original_image[:3].cuda()
            persistence = candidates[:, horizon, 0]
            persistence_render = _render_prediction(
                views[horizon], persistence, gaussians, pipeline, background,
                weights_xyz, weights_rotation,
            )
            base_risk = _image_risk(
                persistence_render, gt, persistence_render, background, args
            )
            for group in range(groups):
                risks[group, horizon, 0] = float(base_risk.item())
                membership = assignment[:, group:group + 1]
                for expert in range(1, len(EXPERTS)):
                    prediction = persistence + membership * (
                        candidates[:, horizon, expert] - persistence
                    )
                    prediction[:, 3:] = F.normalize(prediction[:, 3:], dim=-1)
                    rendering = _render_prediction(
                        views[horizon], prediction, gaussians, pipeline,
                        background, weights_xyz, weights_rotation,
                    )
                    risks[group, horizon, expert] = float(_image_risk(
                        rendering, gt, persistence_render, background, args
                    ).item())
    weights, records = _bin_routes(risks, args)
    return weights, records, risks


def _mix(candidates, assignment, group_weights, calibration_bins):
    nodes, horizons = candidates.shape[:2]
    point_weights = []
    for horizon_index in range(horizons):
        bin_index = min(
            calibration_bins - 1,
            int(horizon_index * calibration_bins / float(horizons)),
        )
        value = torch.matmul(assignment, group_weights[:, bin_index])
        point_weights.append(value)
    point_weights = torch.stack(point_weights, dim=1)
    point_weights = point_weights / point_weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    xyz = torch.sum(point_weights.unsqueeze(-1) * candidates[..., :3], dim=2)
    rotation_candidates = candidates[..., 3:]
    reference = rotation_candidates[:, :, :1]
    sign = torch.where(
        torch.sum(rotation_candidates * reference, dim=-1, keepdim=True) < 0,
        -torch.ones_like(rotation_candidates[..., :1]),
        torch.ones_like(rotation_candidates[..., :1]),
    )
    rotation = torch.sum(
        point_weights.unsqueeze(-1) * rotation_candidates * sign, dim=2
    )
    rotation = F.normalize(rotation, dim=-1)
    return torch.cat([xyz, rotation], dim=-1), point_weights


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _render_and_measure(args, name, prediction, runtime, background):
    dataset, pipeline, gaussians, _, _, _, test_views, _ = runtime
    horizon_indices = getattr(
        args, "future_view_horizon_indices", list(range(prediction.shape[1]))
    )
    if len(horizon_indices) != len(test_views):
        raise RuntimeError("Forecast-horizon/evaluation-view mapping mismatch")
    prediction = prediction[:, horizon_indices]
    eval_root, _ = render_kpts(
        args.model_path, args.experiment_name, "[metrics]%s" % name,
        args.ckpt_iteration, test_views, gaussians, pipeline, background,
        prediction[:, :, :3].permute(1, 0, 2).contiguous(),
        prediction[:, :, 3:].permute(1, 0, 2).contiguous(), metrics=True,
    )
    evaluate(eval_root)
    return {
        "metrics": _read_json(os.path.join(eval_root, "results.json")),
        "per_view": _read_json(os.path.join(eval_root, "per_view.json")),
        "render_root": eval_root,
    }


def main():
    parser = ArgumentParser(description="Phase 90 rendered-risk MoE")
    optimization_parser = OptimizationParams(parser)
    model_parser = ModelParams(parser, sentinel=True)
    pipeline_parser = PipelineParams(parser)
    options = Gaussian_Options(parser)
    options.initial()
    parser.add_argument("--mode", choices=["smoke", "run"], required=True)
    parser.add_argument("--ckpt_iteration", default=60000, type=int)
    parser.add_argument("--sequence_length", default=10, type=int)
    parser.add_argument("--max_train_horizon", default=40, type=int)
    parser.add_argument("--epochs", default=2001, type=int)
    parser.add_argument("--learning_rate", default=1.0e-3, type=float)
    parser.add_argument("--weight_decay", default=1.0e-4, type=float)
    parser.add_argument("--transform_regularizer", default=1.0e-5, type=float)
    parser.add_argument("--local_smoothness_weight", default=0.02, type=float)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--feedforward", default=64, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--local_rank", default=8, type=int)
    parser.add_argument("--groups", default=8, type=int)
    parser.add_argument("--group_temperature", default=0.18, type=float)
    parser.add_argument("--fit_fraction", default=0.75, type=float)
    parser.add_argument("--calibration_bins", default=4, type=int)
    parser.add_argument("--minimum_relative_gain", default=0.005, type=float)
    parser.add_argument("--rendered_region_weight", default=1.0, type=float)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--expected_observed_frames", required=True, type=int)
    parser.add_argument("--expected_future_frames", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--experiment_name", required=True)
    args = get_combined_args(parser)

    seed = 2024 * args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    runtime = _load_runtime(
        args, model_parser, optimization_parser, pipeline_parser
    )
    dataset, pipeline, gaussians, _, observed, train_views, _, representation = runtime
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    fit_frames = int(math.floor(state.shape[0] * args.fit_fraction))
    fit_frames = max(
        args.sequence_length + 2, min(fit_frames, state.shape[0] - 2)
    )
    assignment, rigidity, diagnostics = build_soft_motion_groups(
        state[:fit_frames], canonical, args.groups, args.group_temperature
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )

    calibration_bank = _new_bank(args, assignment, rigidity)
    _audit_trainable(calibration_bank, gaussians)
    calibration_epochs = 2 if args.mode == "smoke" else args.epochs
    calibration_history = _fit_bank(
        args, calibration_bank, state, canonical, fit_frames,
        calibration_epochs, "calibration",
    )
    calibration_candidates = _forecast(
        calibration_bank, state[:fit_frames], canonical, args,
        state.shape[0] - fit_frames,
    )
    calibration_target = state[fit_frames:].permute(1, 0, 2)
    latent_weights, latent_records = _latent_calibration(
        args, calibration_candidates, calibration_target, assignment
    )
    rendered_weights, rendered_records, rendered_risks = _rendered_calibration(
        args, calibration_candidates, calibration_target, assignment,
        train_views[fit_frames:], gaussians, pipeline, background,
    )
    del calibration_bank, calibration_candidates
    torch.cuda.empty_cache()

    # A fresh final bank sees all observed frames.  Router choices remain frozen
    # from the disjoint calibration bank above.
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 1)
    final_bank = _new_bank(args, assignment, rigidity)
    final_epochs = 2 if args.mode == "smoke" else args.epochs
    final_history = _fit_bank(
        args, final_bank, state, canonical, state.shape[0], final_epochs, "final"
    )
    checkpoint = {
        "method": "phase90_crossfit_rendered_risk_moe",
        "model": final_bank.state_dict(),
        "assignment": assignment.cpu(),
        "rigidity": rigidity.cpu(),
        "latent_weights": latent_weights,
        "rendered_weights": rendered_weights,
        "latent_calibration": latent_records,
        "rendered_calibration": rendered_records,
        "rendered_risks": rendered_risks,
        "regime_diagnostics": diagnostics,
        "history": calibration_history + final_history,
        "config": {
            "experts": list(EXPERTS),
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "fit_frames": fit_frames,
            "calibration_frames": int(state.shape[0] - fit_frames),
            "fit_fraction": args.fit_fraction,
            "calibration_bins": args.calibration_bins,
            "minimum_relative_gain": args.minimum_relative_gain,
            "rendered_region_weight": args.rendered_region_weight,
            "future_frames_used_for_training_grouping_or_routing": 0,
            "calibration_expert_trained_on_fit_only": True,
            "router_calibration_disjoint_from_calibration_expert_training": True,
            "final_expert_trained_on_complete_observed_prefix": True,
            "exact_persistence_expert": True,
            "seed": args.seed,
        },
        "representation_checkpoint": representation,
        "parameter_count": int(sum(p.numel() for p in final_bank.parameters())),
        "train_and_calibration_seconds": time.time() - started,
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(checkpoint_path))
    reloaded = torch.load(str(checkpoint_path), map_location="cuda")
    reload_bank = _new_bank(
        args, reloaded["assignment"].cuda(), reloaded["rigidity"].cuda()
    )
    reload_bank.load_state_dict(reloaded["model"])
    reload_bank.eval()
    candidates = _forecast(reload_bank, state, canonical, args)
    if not _finite_state(reloaded["model"]) or not bool(torch.isfinite(candidates).all()):
        raise RuntimeError("Cold-reloaded expert bank is non-finite")
    latent_prediction, latent_point_weights = _mix(
        candidates, reloaded["assignment"].cuda(),
        reloaded["latent_weights"].cuda(), args.calibration_bins,
    )
    rendered_prediction, rendered_point_weights = _mix(
        candidates, reloaded["assignment"].cuda(),
        reloaded["rendered_weights"].cuda(), args.calibration_bins,
    )
    if args.mode == "smoke":
        print(json.dumps({
            "smoke": "passed", "cold_reload": True,
            "fit_frames": fit_frames,
            "calibration_frames": int(state.shape[0] - fit_frames),
            "parameters": checkpoint["parameter_count"],
            "rendered_route_counts": {
                EXPERTS[index]: int((rendered_weights.argmax(-1) == index).sum())
                for index in range(len(EXPERTS))
            },
        }, indent=2))
        return

    # The untouched suffix is accessed for the first time inside these renders.
    methods = {
        "persistence": candidates[:, :, 0],
        "direct": candidates[:, :, 1],
        "rigid_se3": candidates[:, :, 2],
        "local_nonrigid": candidates[:, :, 3],
        "latent_router": latent_prediction,
        "rendered_router": rendered_prediction,
    }
    results = {}
    render_started = time.time()
    for name, prediction in methods.items():
        results[name] = _render_and_measure(
            args, name, prediction, runtime, background
        )
    render_seconds = time.time() - render_started
    summary = {
        "method": checkpoint["method"],
        "scene": args.scene_name,
        "cold_reload": True,
        "protocol": checkpoint["config"],
        "results": results,
        "routing": {
            "expert_order": list(EXPERTS),
            "latent_route_counts": {
                EXPERTS[index]: int((latent_weights.argmax(-1) == index).sum())
                for index in range(len(EXPERTS))
            },
            "rendered_route_counts": {
                EXPERTS[index]: int((rendered_weights.argmax(-1) == index).sum())
                for index in range(len(EXPERTS))
            },
            "latent_mean_point_weights": latent_point_weights.mean((0, 1)).cpu().tolist(),
            "rendered_mean_point_weights": rendered_point_weights.mean((0, 1)).cpu().tolist(),
        },
        "regime_diagnostics": diagnostics,
        "resources": {
            "train_and_calibration_seconds": checkpoint["train_and_calibration_seconds"],
            "render_all_methods_seconds": render_seconds,
            "peak_gpu_allocated_mb": max(
                checkpoint["peak_gpu_allocated_mb"],
                torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            ),
            "parameters": checkpoint["parameter_count"],
            "checkpoint_bytes": os.path.getsize(str(checkpoint_path)),
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "representation_checkpoint": representation,
        },
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(summary_path), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
