"""Cross-fitted winner-expert bank with joint rendered routing.

The bank contains the empirically complementary experts identified in the
Phase-90--92 audit: exact persistence, compact random-multiband, the released
GaussianPrediction GCN, and the structured articulated Transformer.  Fit-only
versions of every moving expert predict a disjoint observed-prefix calibration
tail.  A 128-parameter router is optimized jointly through the frozen renderer.
The router is then frozen and applied to independently trained full-prefix
experts.  True future frames are accessed only after cold reload.

This is integration/development evidence on previously inspected suffixes, not
an untouched promotion experiment.
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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get(
    "GAUSSIAN_PREDICTION_ROOT", ROOT / "external" / "gaussian_prediction"
))
sys.path.insert(0, str(OFFICIAL))

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args  # noqa: E402
from motion_model.gcn import GCN_xyzr  # noqa: E402
from options.gaussian_option import Gaussian_Options  # noqa: E402
from moregs.structured_expert import (  # noqa: E402
    _group_edges,
    _kmeans,
    _model_for_group,
    _motion_features,
    _new_models as new_structured_models,
    _physical_loss,
    _rigidity_labels,
    _rollout as structured_rollout,
    _temporal_mask as structured_temporal_mask,
)
from moregs.forecasting import (  # noqa: E402
    _build_state,
    _forecast_losses,
    _load_runtime,
    _temporal_mask,
)
from moregs.graph_expert import _gcn_rollout  # noqa: E402
from moregs.grouping import build_soft_motion_groups  # noqa: E402
from moregs.rendered_risk import (  # noqa: E402
    _bin_routes,
    _candidate_point_error,
    _mix,
    _render_and_measure,
)
from moregs.router import (  # noqa: E402
    JointRenderedRouter,
    _audit_router,
    _optimize_router,
)
from moregs.coefficient_expert import FrequencyBasisTransformer  # noqa: E402


EXPERTS = ("persistence", "random_multiband", "gcn", "structured")


def _new_random(args):
    return FrequencyBasisTransformer(
        sequence_length=args.sequence_length,
        hidden=args.hidden,
        heads=args.heads,
        feedforward=args.feedforward,
        dropout=args.dropout,
        basis_variant="random",
        seed=args.seed,
    ).cuda()


def _new_gcn(args, nodes):
    return GCN_xyzr(
        input_feature=args.sequence_length,
        hidden_feature=args.gcn_linear_size,
        output_feature=1,
        p_dropout=0.0,
        num_stage=args.gcn_num_stage,
        node_n=nodes,
        no_mapping=False,
    ).cuda()


def _parameter_count(module):
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _audit_modules(gaussians, modules):
    representation = [
        name for name, parameter in gaussians.named_parameters()
        if parameter.requires_grad
    ]
    if representation:
        raise RuntimeError("Unintended trainable Gaussian tensors: %s" % representation)
    report = {}
    for name, module in modules.items():
        report[name] = {
            "parameters": _parameter_count(module),
            "trainable": int(sum(
                parameter.numel() for parameter in module.parameters()
                if parameter.requires_grad
            )),
        }
    print(json.dumps({
        "expert_parameter_groups": report,
        "trainable_gaussian_or_representation_tensors": representation,
    }))


def _fit_random(args, model, state, canonical, end_frame, epochs):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1.0e-5
    )
    valid = list(range(args.sequence_length, end_frame))
    max_horizon = min(
        args.expected_future_frames,
        args.max_train_horizon,
        end_frame - args.sequence_length,
    )
    model.train()
    history = []
    for epoch in range(epochs):
        cutoff = valid[(epoch * 17 + 11) % len(valid)]
        available = min(max_horizon, end_frame - cutoff)
        sequence = state[
            cutoff - args.sequence_length:cutoff
        ].permute(1, 0, 2)
        target = state[cutoff:cutoff + available].permute(1, 0, 2)
        horizons = torch.arange(
            1, available + 1, device=state.device, dtype=state.dtype
        )
        mask, ratio = _temporal_mask(
            epoch, epochs, args.sequence_length, state.device
        )
        prediction, gates = model(
            sequence, canonical, horizons, args.expected_future_frames, mask
        )
        loss = _forecast_losses(prediction, target, sequence, gates)[0]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch in (0, epochs - 1):
            history.append({
                "expert": "random_multiband", "epoch": epoch,
                "loss": float(loss.item()), "cutoff": cutoff,
                "horizons": available, "mask_ratio": float(ratio),
            })
            print(json.dumps(history[-1]))
    model.eval()
    return history


def _prepare_structured_groups(args, canonical, state_fit):
    features = _motion_features(canonical, state_fit[:, :, :3])
    labels, centers = _kmeans(features, args.groups)
    rigid, scores, radii = _rigidity_labels(
        state_fit[:, :, :3], labels, args.groups
    )
    edges = _group_edges(canonical, labels, args.groups)
    return labels, centers, rigid, scores, radii, edges


def _fit_structured(args, models, state, canonical, end_frame, epochs,
                    group_data):
    labels, _, rigid, _, radii, edges = group_data
    optimizer = torch.optim.Adam(
        models.parameters(), lr=args.structured_learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1.0e-5
    )
    windows = end_frame - args.sequence_length
    history = []
    models.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction_losses = []
        acceleration_losses = []
        physical_losses = []
        for group in range(args.groups):
            node_mask = labels == group
            window = (epoch * 17 + group * 11) % windows
            sequence = state[
                window:window + args.sequence_length, node_mask
            ].permute(1, 0, 2)
            target = state[window + args.sequence_length, node_mask]
            mask, _ = structured_temporal_mask(
                epoch, epochs, args.sequence_length, group
            )
            prediction = _model_for_group(models, "structured", group)(
                sequence, mask
            )
            prediction_rotation = F.normalize(prediction[:, 3:], dim=-1)
            target_rotation = target[:, 3:]
            sign = torch.where(
                torch.sum(
                    prediction_rotation * target_rotation,
                    dim=-1, keepdim=True,
                ) < 0,
                -torch.ones_like(target_rotation[:, :1]),
                torch.ones_like(target_rotation[:, :1]),
            )
            prediction_loss = F.mse_loss(
                prediction[:, :3], target[:, :3]
            ) + F.mse_loss(prediction_rotation, target_rotation * sign)
            acceleration = (
                prediction[:, :3] - 2.0 * sequence[:, -1, :3]
                + sequence[:, -2, :3]
            )
            physical = _physical_loss(
                prediction, sequence[:, -1], canonical[node_mask],
                rigid[group], radii[group], edges[group],
            )
            prediction_losses.append(prediction_loss)
            acceleration_losses.append(torch.mean(acceleration.square()))
            physical_losses.append((0.1 if rigid[group] else 0.02) * physical)
        prediction_loss = torch.stack(prediction_losses).mean()
        acceleration_loss = torch.stack(acceleration_losses).mean()
        physical_loss = torch.stack(physical_losses).mean()
        loss = prediction_loss + acceleration_loss + physical_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch in (0, epochs - 1):
            history.append({
                "expert": "structured", "epoch": epoch,
                "loss": float(loss.item()),
                "prediction": float(prediction_loss.item()),
                "physical": float(physical_loss.item()),
            })
            print(json.dumps(history[-1]))
    models.eval()
    return history


def _random_forecast(model, state, canonical, args, frames):
    horizons = torch.arange(1, frames + 1, device=state.device, dtype=state.dtype)
    with torch.no_grad():
        prediction, _ = model(
            state[-args.sequence_length:].permute(1, 0, 2), canonical,
            horizons, args.expected_future_frames,
        )
    return prediction


def _gcn_forecast(model, state, args, frames):
    return _gcn_rollout(
        model, state, frames, args.sequence_length
    ).permute(1, 0, 2).contiguous()


def _structured_forecast(models, state, labels, args, frames):
    xyz, rotation = structured_rollout(
        models, state, labels, args.groups, args.sequence_length,
        frames, "structured",
    )
    return torch.cat([xyz, rotation], dim=-1).permute(1, 0, 2).contiguous()


def _candidate_bank(random_model, gcn_model, structured_models,
                    structured_labels, state, canonical, args, frames,
                    context):
    persistence = state[-1].unsqueeze(1).expand(-1, frames, -1).clone()
    if not bool(torch.isfinite(persistence).all()):
        raise RuntimeError("Non-finite persistence state")
    random_prediction = _random_forecast(
        random_model, state, canonical, args, frames
    )
    gcn_prediction = _gcn_forecast(gcn_model, state, args, frames)
    structured_prediction = _structured_forecast(
        structured_models, state, structured_labels, args, frames
    )
    predictions = [
        persistence, random_prediction, gcn_prediction, structured_prediction
    ]
    fallback_events = getattr(args, "nonfinite_fallback_events", [])
    for index, (name, prediction) in enumerate(zip(EXPERTS, predictions)):
        invalid = int((~torch.isfinite(prediction)).sum().item())
        if invalid == 0:
            continue
        event = {
            "context": context,
            "expert": name,
            "frames": int(frames),
            "invalid_scalars": invalid,
            "total_scalars": int(prediction.numel()),
            "policy": args.nonfinite_policy,
        }
        print(json.dumps({"nonfinite_expert_audit": event}))
        if args.nonfinite_policy == "error":
            raise RuntimeError(
                "Non-finite winner-bank forecast: " + json.dumps(event)
            )
        # A numerically unstable expert abstains as a whole.  Partial
        # coordinate replacement could create invalid rotations or geometry.
        predictions[index] = persistence.clone()
        fallback_events.append(event)
    args.nonfinite_fallback_events = fallback_events
    candidates = torch.stack(predictions, dim=2)
    return candidates


def _latent_initialization(args, candidates, target, assignment):
    point_error = _candidate_point_error(candidates, target)
    group_error = torch.einsum(
        "ng,nhe->ghe", assignment, point_error
    ) / assignment.sum(dim=0).clamp_min(1.0e-6)[:, None, None]
    weights, _ = _bin_routes(group_error.detach().cpu(), args)
    return weights


def _load_full_experts(args, state, canonical):
    random_checkpoint = torch.load(args.random_checkpoint, map_location="cuda")
    random_model = _new_random(args)
    random_model.load_state_dict(random_checkpoint["model"])
    random_model.eval()

    gcn_model = _new_gcn(args, state.shape[1])
    gcn_model.load_state_dict(torch.load(args.gcn_checkpoint, map_location="cuda"))
    gcn_model.eval()

    structured_checkpoint = torch.load(
        args.structured_checkpoint, map_location="cuda"
    )
    if structured_checkpoint["routing"] != "structured":
        raise RuntimeError("Structured checkpoint routing mismatch")
    args.routing = "structured"
    structured_models = new_structured_models(args)
    structured_models.load_state_dict(structured_checkpoint["model"])
    structured_models.eval()
    structured_labels = structured_checkpoint["labels"].cuda()
    for module in (random_model, gcn_model, structured_models):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return random_model, gcn_model, structured_models, structured_labels


def main():
    parser = ArgumentParser(description="Phase 93 crossfit winner bank")
    optimization_parser = OptimizationParams(parser)
    model_parser = ModelParams(parser, sentinel=True)
    pipeline_parser = PipelineParams(parser)
    options = Gaussian_Options(parser)
    options.initial()
    parser.add_argument("--mode", choices=["smoke", "fit", "run"], required=True)
    parser.add_argument("--ckpt_iteration", default=60000, type=int)
    parser.add_argument("--sequence_length", default=10, type=int)
    parser.add_argument("--max_train_horizon", default=40, type=int)
    parser.add_argument("--epochs", default=2001, type=int)
    parser.add_argument("--learning_rate", default=1.0e-3, type=float)
    parser.add_argument("--structured_learning_rate", default=1.0e-3, type=float)
    parser.add_argument("--weight_decay", default=1.0e-4, type=float)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--feedforward", default=64, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--groups", default=8, type=int)
    parser.add_argument("--group_temperature", default=0.18, type=float)
    parser.add_argument("--fit_fraction", default=0.75, type=float)
    parser.add_argument("--calibration_bins", default=4, type=int)
    parser.add_argument("--minimum_relative_gain", default=0.005, type=float)
    parser.add_argument("--rendered_region_weight", default=1.0, type=float)
    parser.add_argument("--router_steps", default=200, type=int)
    parser.add_argument("--router_learning_rate", default=0.05, type=float)
    parser.add_argument("--router_weight_decay", default=0.0, type=float)
    parser.add_argument("--router_smoothness_weight", default=0.002, type=float)
    parser.add_argument("--router_residual_weight", default=0.0001, type=float)
    parser.add_argument("--router_entropy_weight", default=0.0001, type=float)
    parser.add_argument("--gcn_linear_size", default=128, type=int)
    parser.add_argument("--gcn_num_stage", default=6, type=int)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--expected_observed_frames", required=True, type=int)
    parser.add_argument("--expected_future_frames", required=True, type=int)
    parser.add_argument("--allow_repeated_future_times", action="store_true")
    parser.add_argument("--strict_fit_no_future_views", action="store_true")
    parser.add_argument(
        "--nonfinite_policy", choices=["error", "persistence"], default="error"
    )
    parser.add_argument("--calibration_gcn_checkpoint", required=True)
    parser.add_argument("--gcn_checkpoint", required=True)
    parser.add_argument("--random_checkpoint", required=True)
    parser.add_argument("--structured_checkpoint", required=True)
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
    dataset, _, gaussians, _, observed, train_views, _, representation = runtime
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    fit_frames = int(math.floor(state.shape[0] * args.fit_fraction))
    assignment, _, assignment_diagnostics = build_soft_motion_groups(
        state[:fit_frames], canonical, args.groups, args.group_temperature
    )
    structured_group_data = _prepare_structured_groups(
        args, canonical, state[:fit_frames]
    )
    structured_labels = structured_group_data[0]

    calibration_random = _new_random(args)
    args.routing = "structured"
    calibration_structured = new_structured_models(args)
    calibration_gcn = _new_gcn(args, state.shape[1])
    calibration_gcn.load_state_dict(torch.load(
        args.calibration_gcn_checkpoint, map_location="cuda"
    ))
    calibration_gcn.eval()
    for parameter in calibration_gcn.parameters():
        parameter.requires_grad_(False)
    _audit_modules(gaussians, {
        "calibration_random_multiband": calibration_random,
        "calibration_gcn_frozen": calibration_gcn,
        "calibration_structured": calibration_structured,
    })
    calibration_epochs = 2 if args.mode == "smoke" else args.epochs
    history = _fit_random(
        args, calibration_random, state, canonical, fit_frames,
        calibration_epochs,
    )
    history.extend(_fit_structured(
        args, calibration_structured, state, canonical, fit_frames,
        calibration_epochs, structured_group_data,
    ))
    calibration_frames = state.shape[0] - fit_frames
    calibration_candidates = _candidate_bank(
        calibration_random, calibration_gcn, calibration_structured,
        structured_labels, state[:fit_frames], canonical, args,
        calibration_frames, "observed_tail_calibration",
    )
    target = state[fit_frames:].permute(1, 0, 2)
    latent_weights = _latent_initialization(
        args, calibration_candidates, target, assignment
    )
    router = JointRenderedRouter(latent_weights.float()).cuda()
    _audit_router(router)
    if args.mode == "smoke":
        args.router_steps = 2
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )
    joint_weights, router_history = _optimize_router(
        args, router, calibration_candidates, assignment,
        train_views[fit_frames:], runtime, background,
    )

    del calibration_random, calibration_gcn, calibration_structured
    del calibration_candidates, router
    torch.cuda.empty_cache()
    full_experts = _load_full_experts(args, state, canonical)
    random_model, gcn_model, structured_models, full_structured_labels = full_experts
    _audit_modules(gaussians, {
        "full_random_multiband_frozen": random_model,
        "full_gcn_frozen": gcn_model,
        "full_structured_frozen": structured_models,
    })
    final_candidates = _candidate_bank(
        random_model, gcn_model, structured_models, full_structured_labels,
        state, canonical, args, args.expected_future_frames, "future_forecast",
    )
    prediction, point_weights = _mix(
        final_candidates, assignment, joint_weights.cuda(),
        args.calibration_bins,
    )
    checkpoint = {
        "method": "phase93_crossfit_winner_bank",
        # Keep the cold-reloadable routed trajectory with the router state so
        # downstream observed-only lifecycle/appearance experts can consume
        # the MoRE-GS forecast without reopening future images or refitting the
        # expert bank.  The top-level protocol mirrors the compact forecast
        # checkpoint contract used by the composition runners.
        "future_keypoint_prediction": prediction.detach().cpu(),
        "protocol": {
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "future_rgb_used_before_checkpoint": 0,
            "future_states_used_before_checkpoint": 0,
            "seed": args.seed,
        },
        "joint_weights": joint_weights.cpu(),
        "assignment": assignment.cpu(),
        "router_history": router_history,
        "expert_fit_history": history,
        "assignment_diagnostics": assignment_diagnostics,
        "config": {
            "experts": list(EXPERTS),
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "future_evaluation_views": getattr(
                args, "future_evaluation_view_count", args.expected_future_frames
            ),
            "repeated_future_view_times": bool(
                getattr(args, "allow_repeated_future_times", False)
            ),
            "fit_frames": fit_frames,
            "calibration_frames": calibration_frames,
            "calibration_bins": args.calibration_bins,
            "router_steps": args.router_steps,
            "future_frames_used_for_training_grouping_or_routing": 0,
            "all_calibration_experts_fit_only": True,
            "full_experts_independent_full_prefix_checkpoints": True,
            "joint_router_optimized_through_renderer": True,
            "nonfinite_policy": args.nonfinite_policy,
            "nonfinite_fallback_events": list(
                getattr(args, "nonfinite_fallback_events", [])
            ),
            "development_suffix_previously_inspected": True,
            "seed": args.seed,
        },
        "expert_checkpoints": {
            "calibration_gcn": args.calibration_gcn_checkpoint,
            "gcn": args.gcn_checkpoint,
            "random_multiband": args.random_checkpoint,
            "structured": args.structured_checkpoint,
        },
        "expert_parameters": {
            "random_multiband": _parameter_count(random_model),
            "gcn": _parameter_count(gcn_model),
            "structured": _parameter_count(structured_models),
            "router": int(joint_weights.numel()),
        },
        "train_and_calibration_seconds": time.time() - started,
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
        "representation_checkpoint": representation,
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(checkpoint_path))
    reloaded = torch.load(str(checkpoint_path), map_location="cuda")
    reloaded_prediction, reloaded_point_weights = _mix(
        final_candidates, reloaded["assignment"],
        reloaded["joint_weights"], args.calibration_bins,
    )
    if not torch.allclose(prediction, reloaded_prediction, atol=0.0, rtol=0.0):
        raise RuntimeError("Cold-reloaded routed forecast mismatch")
    if args.mode == "smoke":
        print(json.dumps({
            "smoke": "passed", "cold_reload": True,
            "fit_frames": fit_frames,
            "calibration_frames": calibration_frames,
            "expert_parameters": checkpoint["expert_parameters"],
            "mean_point_weights": reloaded_point_weights.mean((0, 1)).cpu().tolist(),
        }, indent=2))
        return

    if args.mode == "fit":
        if not args.strict_fit_no_future_views:
            raise RuntimeError("Fit-only mode requires --strict_fit_no_future_views")
        fit_summary = {
            "method": checkpoint["method"],
            "scene": args.scene_name,
            "fit_only": True,
            "future_sensory_views_loaded": 0,
            "cold_reload": True,
            "protocol": checkpoint["config"],
            "routing": {
                "expert_order": list(EXPERTS),
                "group_bin_weights": reloaded["joint_weights"].cpu().tolist(),
                "mean_point_weights": reloaded_point_weights.mean((0, 1)).cpu().tolist(),
            },
            "resources": {
                "train_and_calibration_seconds": checkpoint["train_and_calibration_seconds"],
                "peak_gpu_allocated_mb": checkpoint["peak_gpu_allocated_mb"],
                "router_checkpoint_bytes": os.path.getsize(str(checkpoint_path)),
                "expert_parameters": checkpoint["expert_parameters"],
            },
            "artifacts": {
                "checkpoint": str(checkpoint_path),
                "representation_checkpoint": representation,
            },
        }
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w") as handle:
            json.dump(fit_summary, handle, indent=2)
        print(json.dumps(fit_summary, indent=2))
        return

    results = {}
    for index, name in enumerate(EXPERTS):
        results[name] = _render_and_measure(
            args, name, final_candidates[:, :, index], runtime, background
        )
    results["joint_winner_bank"] = _render_and_measure(
        args, "joint_winner_bank", reloaded_prediction, runtime, background
    )
    summary = {
        "method": checkpoint["method"],
        "scene": args.scene_name,
        "development_evidence_not_untouched_promotion": True,
        "cold_reload": True,
        "protocol": checkpoint["config"],
        "results": results,
        "routing": {
            "expert_order": list(EXPERTS),
            "group_bin_weights": reloaded["joint_weights"].cpu().tolist(),
            "mean_point_weights": reloaded_point_weights.mean((0, 1)).cpu().tolist(),
        },
        "resources": {
            "train_and_calibration_seconds": checkpoint["train_and_calibration_seconds"],
            "peak_gpu_allocated_mb": checkpoint["peak_gpu_allocated_mb"],
            "router_checkpoint_bytes": os.path.getsize(str(checkpoint_path)),
            "expert_parameters": checkpoint["expert_parameters"],
            "expert_checkpoint_bytes": {
                name: os.path.getsize(path)
                for name, path in checkpoint["expert_checkpoints"].items()
            },
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
