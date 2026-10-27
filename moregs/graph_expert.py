"""Leakage-free support-only motion-expert router.

The selector is fitted exclusively on pseudo-future windows contained in the
observed prefix.  It routes motion groups and horizons among exact
persistence, the released GCN, the structured Transformer, and the direct
multi-horizon Transformer.  True suffix frames are loaded only after the
router checkpoint has been written and cold reloaded.
"""

from __future__ import print_function

import json
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
from eval import render_kpts  # noqa: E402
from metrics import evaluate  # noqa: E402
from motion_model.gcn import GCN_xyzr  # noqa: E402
from options.gaussian_option import Gaussian_Options  # noqa: E402
from moregs.structured_expert import (  # noqa: E402
    _new_models as new_structured_models,
    _rollout as structured_rollout,
)
from moregs.forecasting import (  # noqa: E402
    _build_state,
    _load_runtime,
    _new_model as new_direct_model,
)


EXPERTS = ("persistence", "gcn", "structured", "direct")


def _finite(tensor):
    return bool(torch.isfinite(tensor).all())


def _align_rotation(rotation, reference):
    sign = torch.where(
        torch.sum(rotation * reference, dim=-1, keepdim=True) < 0,
        -torch.ones_like(rotation[..., :1]),
        torch.ones_like(rotation[..., :1]),
    )
    return F.normalize(rotation * sign, dim=-1)


def _load_models(args, node_count):
    gcn = GCN_xyzr(
        input_feature=args.input_size,
        hidden_feature=args.linear_size,
        output_feature=args.output_size,
        p_dropout=args.dropout,
        num_stage=args.num_stage,
        node_n=node_count,
        no_mapping=args.no_mapping,
    ).cuda()
    gcn.load_state_dict(torch.load(args.gcn_checkpoint, map_location="cuda"))
    gcn.eval()

    structured_checkpoint = torch.load(
        args.structured_checkpoint, map_location="cuda"
    )
    if structured_checkpoint["routing"] != "structured":
        raise RuntimeError("Structured checkpoint routing mismatch")
    args.routing = "structured"
    structured = new_structured_models(args)
    structured.load_state_dict(structured_checkpoint["model"])
    structured.eval()
    labels = structured_checkpoint["labels"].to("cuda")

    direct_checkpoint = torch.load(args.direct_checkpoint, map_location="cuda")
    direct_config = direct_checkpoint["config"]
    if (direct_config["future_frames_used"] != 0 or
            direct_config["observed_frames"] != args.expected_observed_frames):
        raise RuntimeError("Direct checkpoint protocol mismatch")
    direct = new_direct_model(args)
    direct.load_state_dict(direct_checkpoint["model"])
    direct.eval()

    for module in [gcn, structured, direct]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    trainable = []
    for prefix, module in [("gcn", gcn), ("structured", structured), ("direct", direct)]:
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                trainable.append(prefix + "." + name)
    if trainable:
        raise RuntimeError("Unexpected trainable tensors: %s" % trainable)

    parameter_counts = {
        "persistence": 0,
        "gcn": int(sum(p.numel() for p in gcn.parameters())),
        "structured": int(sum(p.numel() for p in structured.parameters())),
        "direct": int(sum(p.numel() for p in direct.parameters())),
        "router": 0,
    }
    print(json.dumps({
        "trainable_parameter_groups": [],
        "trainable_parameter_count": 0,
        "frozen_expert_parameter_counts": parameter_counts,
        "trainable_gaussian_or_representation_tensors": [],
    }))
    return gcn, structured, direct, labels, parameter_counts


def _gcn_rollout(model, state, frames, sequence_length):
    xyz = state[-sequence_length:, :, :3].unsqueeze(0)
    rotation = state[-sequence_length:, :, 3:].unsqueeze(0)
    xyz_output = []
    rotation_output = []
    with torch.no_grad():
        for _ in range(frames):
            xyz_prediction, rotation_prediction = model(
                xyz.permute(0, 3, 2, 1), rotation.permute(0, 3, 2, 1)
            )
            xyz_prediction = xyz_prediction.permute(0, 3, 2, 1)[:, -1]
            rotation_prediction = rotation_prediction.permute(0, 3, 2, 1)[:, -1]
            rotation_prediction = _align_rotation(
                rotation_prediction, rotation[:, -1]
            )
            xyz_output.append(xyz_prediction[0])
            rotation_output.append(rotation_prediction[0])
            xyz = torch.cat([xyz[:, 1:], xyz_prediction.unsqueeze(1)], dim=1)
            rotation = torch.cat(
                [rotation[:, 1:], rotation_prediction.unsqueeze(1)], dim=1
            )
    return torch.cat(
        [torch.stack(xyz_output), torch.stack(rotation_output)], dim=-1
    )


def _direct_forecast(model, state, canonical, frames, sequence_length, scale):
    horizons = torch.arange(1, frames + 1, device="cuda", dtype=state.dtype)
    with torch.no_grad():
        prediction, _ = model(
            state[-sequence_length:].permute(1, 0, 2),
            canonical,
            horizons,
            scale,
        )
    return prediction.permute(1, 0, 2).contiguous()


def _structured_forecast(models, state, labels, args, frames):
    xyz, rotation = structured_rollout(
        models,
        state,
        labels,
        args.groups,
        args.sequence_length,
        frames,
        "structured",
    )
    return torch.cat([xyz, rotation], dim=-1)


def _forecasts(models, state, canonical, labels, args, frames):
    gcn, structured, direct = models
    persistence = state[-1:].expand(frames, -1, -1).clone()
    predictions = {
        "persistence": persistence,
        "gcn": _gcn_rollout(gcn, state, frames, args.sequence_length),
        "structured": _structured_forecast(
            structured, state, labels, args, frames
        ),
        "direct": _direct_forecast(
            direct,
            state,
            canonical,
            frames,
            args.sequence_length,
            args.expected_future_frames,
        ),
    }
    for name, value in predictions.items():
        if value.shape != persistence.shape or not _finite(value):
            raise RuntimeError("Invalid %s forecast" % name)
    return predictions


def _parse_cutoffs(value, observed_frames, future_frames, sequence_length):
    if value:
        cutoffs = [int(item) for item in value.split(",")]
    else:
        first = max(sequence_length + future_frames, observed_frames // 2)
        last = observed_frames - future_frames
        step = max(1, future_frames // 2)
        cutoffs = list(range(first, last + 1, step))
    if not cutoffs:
        raise RuntimeError("No legal pseudo-future cutoff")
    for cutoff in cutoffs:
        if cutoff < sequence_length or cutoff + future_frames > observed_frames:
            raise RuntimeError("Illegal support cutoff %d" % cutoff)
    return cutoffs


def _fit_router(models, state, canonical, labels, args, cutoffs):
    errors = torch.zeros(
        len(EXPERTS), args.expected_future_frames, args.groups,
        dtype=torch.float64, device="cuda"
    )
    counts = torch.zeros_like(errors)
    scene_scale = canonical.std(dim=0, unbiased=False).mean().clamp_min(1.0e-4)
    for cutoff in cutoffs:
        prediction = _forecasts(
            models, state[:cutoff], canonical, labels, args,
            args.expected_future_frames
        )
        target = state[cutoff:cutoff + args.expected_future_frames]
        for expert_index, expert in enumerate(EXPERTS):
            xyz_error = torch.sum(
                (prediction[expert][:, :, :3] - target[:, :, :3]) ** 2,
                dim=-1,
            ) / (scene_scale ** 2)
            predicted_rotation = _align_rotation(
                prediction[expert][:, :, 3:], target[:, :, 3:]
            )
            rotation_error = 1.0 - torch.abs(torch.sum(
                predicted_rotation * target[:, :, 3:], dim=-1
            )).clamp(0.0, 1.0)
            combined = xyz_error + 0.25 * rotation_error
            for group in range(args.groups):
                mask = labels == group
                errors[expert_index, :, group] += combined[:, mask].sum(dim=1).double()
                counts[expert_index, :, group] += float(mask.sum().item())
    mean_errors = errors / counts.clamp_min(1.0)
    best_errors, choices = mean_errors.min(dim=0)
    persistence_errors = mean_errors[0]
    reliable = best_errors <= (1.0 - args.minimum_relative_gain) * persistence_errors
    choices = torch.where(reliable, choices, torch.zeros_like(choices))
    return choices, mean_errors, reliable


def _combine(predictions, choices, labels, groups):
    frames = choices.shape[0]
    output = predictions["persistence"].clone()
    for horizon in range(frames):
        for group in range(groups):
            mask = labels == group
            expert = EXPERTS[int(choices[horizon, group].item())]
            output[horizon, mask] = predictions[expert][horizon, mask]
    output[:, :, 3:] = F.normalize(output[:, :, 3:], dim=-1)
    return output


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def main():
    parser = ArgumentParser(description="Phase 81 support-only motion router")
    optimization_parser = OptimizationParams(parser)
    model_parser = ModelParams(parser, sentinel=True)
    pipeline_parser = PipelineParams(parser)
    options = Gaussian_Options(parser)
    options.initial()
    options.gcn_training()
    parser.add_argument("--mode", choices=["smoke", "run"], required=True)
    parser.add_argument("--ckpt_iteration", default=60000, type=int)
    parser.add_argument("--sequence_length", default=10, type=int)
    parser.add_argument("--groups", default=8, type=int)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--feedforward", default=64, type=int)
    parser.add_argument("--weight_decay", default=1.0e-4, type=float)
    parser.add_argument("--learning_rate", default=1.0e-3, type=float)
    parser.add_argument("--epochs", default=2001, type=int)
    parser.add_argument("--max_train_horizon", default=10, type=int)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--expected_observed_frames", required=True, type=int)
    parser.add_argument("--expected_future_frames", required=True, type=int)
    parser.add_argument("--gcn_checkpoint", required=True)
    parser.add_argument("--structured_checkpoint", required=True)
    parser.add_argument("--direct_checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--cutoffs", default="")
    parser.add_argument("--minimum_relative_gain", default=0.05, type=float)
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
    dataset, pipeline, gaussians, _, observed, _, test_views, representation = runtime
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    models_and_meta = _load_models(args, state.shape[1])
    models = models_and_meta[:3]
    labels = models_and_meta[3]
    parameter_counts = models_and_meta[4]
    if labels.shape[0] != state.shape[1] or sorted(torch.unique(labels).tolist()) != list(range(args.groups)):
        raise RuntimeError("Invalid support-only group labels")

    cutoffs = _parse_cutoffs(
        args.cutoffs, args.expected_observed_frames,
        args.expected_future_frames, args.sequence_length
    )
    if args.mode == "smoke":
        cutoffs = cutoffs[:2]
    choices, errors, reliable = _fit_router(
        models, state, canonical, labels, args, cutoffs
    )
    choice_counts = {
        name: int((choices == index).sum().item())
        for index, name in enumerate(EXPERTS)
    }
    router_checkpoint = {
        "method": "phase81_support_only_motion_router",
        "choices": choices.cpu(),
        "labels": labels.cpu(),
        "calibration_errors": errors.cpu(),
        "reliable": reliable.cpu(),
        "experts": list(EXPERTS),
        "config": {
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "cutoffs": cutoffs,
            "groups": args.groups,
            "minimum_relative_gain": args.minimum_relative_gain,
            "true_future_frames_used_for_routing": 0,
            "grouping_uses_observed_prefix_only": True,
            "exact_persistence_fallback": True,
            "seed": args.seed,
        },
        "expert_checkpoints": {
            "gcn": args.gcn_checkpoint,
            "structured": args.structured_checkpoint,
            "direct": args.direct_checkpoint,
        },
    }
    output = Path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(router_checkpoint, str(output))
    reloaded = torch.load(str(output), map_location="cuda")
    if reloaded["config"]["true_future_frames_used_for_routing"] != 0:
        raise RuntimeError("Router checkpoint leakage")

    if args.mode == "smoke":
        print(json.dumps({
            "smoke": "passed",
            "iterations": len(cutoffs),
            "cutoffs": cutoffs,
            "choice_counts": choice_counts,
            "cold_reload": True,
        }, indent=2))
        return

    # The true suffix remains untouched until after the router is finalized and
    # cold reloaded above.
    torch.cuda.synchronize()
    forecast_started = time.time()
    predictions = _forecasts(
        models, state, canonical, labels, args, args.expected_future_frames
    )
    combined = _combine(
        predictions,
        reloaded["choices"].to("cuda"),
        reloaded["labels"].to("cuda"),
        args.groups,
    )
    combined_again = _combine(
        predictions,
        choices,
        labels,
        args.groups,
    )
    if not torch.equal(combined, combined_again):
        raise RuntimeError("Cold-reloaded router does not reproduce forecast")
    torch.cuda.synchronize()
    forecast_seconds = time.time() - forecast_started

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda"
    )
    render_started = time.time()
    eval_root, _ = render_kpts(
        args.model_path,
        "phase81_support_motion_router",
        "[metrics]support_only_router",
        args.ckpt_iteration,
        test_views,
        gaussians,
        pipeline,
        background,
        combined[:, :, :3],
        combined[:, :, 3:],
        metrics=True,
    )
    render_seconds = time.time() - render_started
    evaluate(eval_root)
    metrics = _read_json(os.path.join(eval_root, "results.json"))
    per_view = _read_json(os.path.join(eval_root, "per_view.json"))

    summary = {
        "method": "phase81_support_only_motion_router",
        "scene": args.scene_name,
        "cold_reload": True,
        "protocol": router_checkpoint["config"],
        "metrics": metrics,
        "per_view": per_view,
        "router": {
            "experts": list(EXPERTS),
            "choice_counts": choice_counts,
            "choices_by_horizon_group": choices.cpu().tolist(),
            "support_error_by_expert_horizon_group": errors.cpu().tolist(),
            "reliable_cells": int(reliable.sum().item()),
            "total_cells": int(reliable.numel()),
        },
        "resources": {
            "fit_and_evaluation_seconds": time.time() - started,
            "forecast_head_seconds": forecast_seconds,
            "forecast_head_fps": args.expected_future_frames / forecast_seconds,
            "render_and_png_fps": args.expected_future_frames / render_seconds,
            "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            "router_trainable_parameters": 0,
            "frozen_expert_parameters": parameter_counts,
            "router_checkpoint_bytes": os.path.getsize(str(output)),
            "expert_checkpoint_bytes": {
                "gcn": os.path.getsize(args.gcn_checkpoint),
                "structured": os.path.getsize(args.structured_checkpoint),
                "direct": os.path.getsize(args.direct_checkpoint),
            },
        },
        "artifacts": {
            "checkpoint": str(output),
            "representation_checkpoint": representation,
            "render_root": eval_root,
        },
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(summary_path), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
