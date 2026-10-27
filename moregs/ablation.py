"""Matched causal routing ablations for the frozen Phase-93 expert bank.

All route fitting uses only the observed-prefix calibration tail.  Full-prefix
experts and the Gaussian representation are frozen.  The future suffix is
opened only after every causal route has been saved and cold reloaded.
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


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get(
    "GAUSSIAN_PREDICTION_ROOT", ROOT / "external" / "gaussian_prediction"
))
sys.path.insert(0, str(OFFICIAL))

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args  # noqa: E402
from options.gaussian_option import Gaussian_Options  # noqa: E402
from moregs.forecasting import _build_state, _load_runtime  # noqa: E402
from moregs.grouping import build_soft_motion_groups  # noqa: E402
from moregs.rendered_risk import (  # noqa: E402
    _image_risk,
    _mix,
    _render_and_measure,
    _render_prediction,
)
from moregs.router import JointRenderedRouter, _optimize_router  # noqa: E402
from moregs.train import (  # noqa: E402
    EXPERTS,
    _audit_modules,
    _candidate_bank,
    _fit_random,
    _fit_structured,
    _latent_initialization,
    _load_full_experts,
    _new_gcn,
    _new_random,
    _prepare_structured_groups,
)
from moregs.structured_expert import _new_models as new_structured_models  # noqa: E402


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _fit_route(args, candidates, target, assignment, views, runtime,
               background, bins, steps):
    previous_bins = args.calibration_bins
    previous_steps = args.router_steps
    args.calibration_bins = bins
    args.router_steps = steps
    try:
        initial = _latent_initialization(args, candidates, target, assignment)
        router = JointRenderedRouter(initial.float()).cuda()
        weights, history = _optimize_router(
            args, router, candidates, assignment, views, runtime, background
        )
    finally:
        args.calibration_bins = previous_bins
        args.router_steps = previous_steps
    return weights.cpu(), history


def _global_calibration_risk(args, candidates, views, runtime, background):
    _, pipeline, gaussians, _, _, _, _, _ = runtime
    time_value = torch.from_numpy(views[0].time).to(torch.float32).cuda()
    risk = torch.zeros(len(EXPERTS), dtype=torch.float64)
    with torch.no_grad():
        _, _, _, _, weights_xyz, weights_rotation = gaussians(
            time_value, args.ckpt_iteration, return_weights=True
        )
        for horizon, view in enumerate(views):
            gt = view.original_image[:3].cuda()
            persistence = _render_prediction(
                view, candidates[:, horizon, 0], gaussians, pipeline,
                background, weights_xyz, weights_rotation,
            )
            for expert in range(len(EXPERTS)):
                rendering = persistence if expert == 0 else _render_prediction(
                    view, candidates[:, horizon, expert], gaussians, pipeline,
                    background, weights_xyz, weights_rotation,
                )
                risk[expert] += float(_image_risk(
                    rendering, gt, persistence, background, args
                ).item())
    return risk / float(len(views))


def _oracle_from_expert_results(results):
    psnr_maps = {name: value["per_view"]["PSNR"] for name, value in results.items()}
    keys = sorted(next(iter(psnr_maps.values())).keys())
    selections = []
    values = {"PSNR": [], "SSIM": [], "LPIPS-vgg": []}
    wins = {name: 0 for name in results}
    for key in keys:
        winner = max(results, key=lambda name: psnr_maps[name][key])
        wins[winner] += 1
        selections.append({"view": key, "expert": winner})
        for metric in values:
            values[metric].append(float(results[winner]["per_view"][metric][key]))
    return {
        "diagnostic_only_non_deployable": True,
        "future_ground_truth_used_for_selection": True,
        "selection": "best expert independently per future evaluation view by PSNR",
        "metrics": {metric: float(np.mean(items)) for metric, items in values.items()},
        "expert_wins": wins,
        "per_view_selection": selections,
    }


def _prediction(candidates, assignment, weights, bins):
    return _mix(candidates, assignment.cuda(), weights.cuda(), bins)[0]


def main():
    parser = ArgumentParser(description="Phase-167 matched Phase-93 ablations")
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
    parser.add_argument(
        "--nonfinite_policy", choices=["error", "persistence"], default="error"
    )
    parser.add_argument("--calibration_gcn_checkpoint", required=True)
    parser.add_argument("--gcn_checkpoint", required=True)
    parser.add_argument("--random_checkpoint", required=True)
    parser.add_argument("--structured_checkpoint", required=True)
    parser.add_argument(
        "--base_checkpoint", "--phase93_checkpoint", dest="base_checkpoint", required=True
    )
    parser.add_argument(
        "--base_summary", "--phase93_summary", dest="base_summary", required=True
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument(
        "--evaluation_variants",
        nargs="+",
        choices=[
            "uniform_mixture",
            "scene_level_router",
            "one_group_horizon_router",
            "group_only_router",
            "group_horizon_router",
            "group_horizon_without_persistence_gate",
            "motion_groups_4",
            "motion_groups_8_reoptimized",
            "motion_groups_16",
            "shuffled_groups_8",
        ],
        default=None,
        help="Evaluate only the named routing variants; default evaluates all.",
    )
    args = get_combined_args(parser)

    seed = 2024 * args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    runtime = _load_runtime(args, model_parser, optimization_parser, pipeline_parser)
    dataset, _, gaussians, _, observed, train_views, _, representation = runtime
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    fit_frames = int(math.floor(state.shape[0] * args.fit_fraction))
    calibration_views = train_views[fit_frames:]
    if len(calibration_views) != state.shape[0] - fit_frames:
        raise RuntimeError("Observed-prefix calibration view mismatch")

    phase93 = torch.load(args.base_checkpoint, map_location="cpu")
    config = phase93["config"]
    if (
        config["observed_frames"] != args.expected_observed_frames
        or config["future_frames"] != args.expected_future_frames
        or config["future_frames_used_for_training_grouping_or_routing"] != 0
        or int(config["fit_frames"]) != fit_frames
    ):
        raise RuntimeError("Phase-93 checkpoint protocol mismatch")
    recorded_calibration_gcn = phase93.get("expert_checkpoints", {}).get(
        "calibration_gcn"
    )
    if not recorded_calibration_gcn or not Path(recorded_calibration_gcn).is_file():
        raise RuntimeError(
            "Verified Phase-93 checkpoint does not record an available "
            "calibration GCN"
        )
    # Scene checkpoints were created across several phases whose directory
    # naming conventions differ.  The source Phase-93 artifact is the
    # authoritative path; a runner-side guessed path is not scientific
    # evidence and can silently select the wrong calibration model.
    args.calibration_gcn_checkpoint = recorded_calibration_gcn
    motion_assignment = phase93["assignment"].cuda()
    main_weights = phase93["joint_weights"].cpu()

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )
    structured_group_data = _prepare_structured_groups(args, canonical, state[:fit_frames])
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
    epochs = 2 if args.mode == "smoke" else args.epochs
    history = _fit_random(
        args, calibration_random, state, canonical, fit_frames, epochs
    )
    history.extend(_fit_structured(
        args, calibration_structured, state, canonical, fit_frames, epochs,
        structured_group_data,
    ))
    calibration_frames = state.shape[0] - fit_frames
    calibration_candidates = _candidate_bank(
        calibration_random, calibration_gcn, calibration_structured,
        structured_labels, state[:fit_frames], canonical, args,
        calibration_frames, "observed_tail_calibration",
    )
    target = state[fit_frames:].permute(1, 0, 2)
    route_steps = 2 if args.mode == "smoke" else args.router_steps

    routes = {}
    route_history = {}
    assignments = {}
    scene_assignment, _, _ = build_soft_motion_groups(
        state[:fit_frames], canonical, 1, args.group_temperature
    )
    routes["scene_global"], route_history["scene_global"] = _fit_route(
        args, calibration_candidates, target, scene_assignment,
        calibration_views, runtime, background, 1, route_steps,
    )
    assignments["scene_global"] = scene_assignment.cpu()
    for groups in (1, 4, 8, 16):
        assignment, _, _ = build_soft_motion_groups(
            state[:fit_frames], canonical, groups, args.group_temperature
        )
        weights, items = _fit_route(
            args, calibration_candidates, target, assignment,
            calibration_views, runtime, background, 4, route_steps,
        )
        key = "motion_groups_%d" % groups
        routes[key] = weights
        route_history[key] = items
        assignments[key] = assignment.cpu()

    # Pure group-only control: retain the eight observed-only motion groups,
    # but share one route over the complete forecast horizon.  This isolates
    # spatial specialization from horizon specialization.
    routes["motion_groups_8_group_only"], route_history[
        "motion_groups_8_group_only"
    ] = _fit_route(
        args, calibration_candidates, target,
        assignments["motion_groups_8"].cuda(), calibration_views,
        runtime, background, 1, route_steps,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 167)
    permutation = torch.randperm(motion_assignment.shape[0], generator=generator)
    shuffled_assignment = motion_assignment.cpu()[permutation].cuda()
    routes["shuffled_groups_8"], route_history["shuffled_groups_8"] = _fit_route(
        args, calibration_candidates, target, shuffled_assignment,
        calibration_views, runtime, background, 4, route_steps,
    )
    assignments["shuffled_groups_8"] = shuffled_assignment.cpu()

    global_risk = _global_calibration_risk(
        args, calibration_candidates, calibration_views, runtime, background
    )
    fixed_index = int(torch.argmin(global_risk).item())
    fixed_name = EXPERTS[fixed_index]
    del calibration_random, calibration_gcn, calibration_structured
    del calibration_candidates
    torch.cuda.empty_cache()

    full_experts = _load_full_experts(args, state, canonical)
    final_candidates = _candidate_bank(
        full_experts[0], full_experts[1], full_experts[2], full_experts[3],
        state, canonical, args, args.expected_future_frames, "future_forecast",
    )
    uniform_assignment = torch.ones(final_candidates.shape[0], 1)
    uniform_weights = torch.full((1, 1, len(EXPERTS)), 1.0 / len(EXPERTS))
    no_persistence = main_weights.clone()
    no_persistence[..., 0] = 0.0
    no_persistence[..., 1:] /= no_persistence[..., 1:].sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0e-8)

    variants = {
        "uniform_mixture": (uniform_assignment, uniform_weights, 1),
        "scene_level_router": (
            assignments["scene_global"], routes["scene_global"], 1
        ),
        "one_group_horizon_router": (
            assignments["motion_groups_1"], routes["motion_groups_1"], 4
        ),
        "group_only_router": (
            assignments["motion_groups_8"],
            routes["motion_groups_8_group_only"], 1,
        ),
        "group_horizon_router": (
            phase93["assignment"], main_weights, int(config["calibration_bins"])
        ),
        "group_horizon_without_persistence_gate": (
            phase93["assignment"], no_persistence, int(config["calibration_bins"])
        ),
        "motion_groups_4": (assignments["motion_groups_4"], routes["motion_groups_4"], 4),
        "motion_groups_8_reoptimized": (
            assignments["motion_groups_8"], routes["motion_groups_8"], 4
        ),
        "motion_groups_16": (
            assignments["motion_groups_16"], routes["motion_groups_16"], 4
        ),
        "shuffled_groups_8": (
            assignments["shuffled_groups_8"], routes["shuffled_groups_8"], 4
        ),
    }
    evaluation_variants = getattr(args, "evaluation_variants", None)
    if evaluation_variants is not None:
        requested = set(evaluation_variants)
        variants = {name: value for name, value in variants.items() if name in requested}
    predictions = {
        name: _prediction(final_candidates, assignment, weights, bins)
        for name, (assignment, weights, bins) in variants.items()
    }
    checkpoint = {
        "method": "phase167_phase93_matched_causal_ablation",
        "config": {
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "fit_frames": fit_frames,
            "calibration_frames": calibration_frames,
            "future_frames_used_for_training_grouping_routing_or_selection": 0,
            "seed": args.seed,
            "experts": list(EXPERTS),
            "shuffled_control": "fixed row permutation preserving group masses",
            "group_count_ablation_changes_router_groups_only": True,
        },
        "assignments": assignments,
        "routes": routes,
        "main_phase93_assignment": phase93["assignment"],
        "main_phase93_weights": main_weights,
        "no_persistence_weights": no_persistence,
        "uniform_assignment": uniform_assignment,
        "uniform_weights": uniform_weights,
        "calibration_global_risk": global_risk,
        "calibration_selected_fixed_expert": fixed_name,
        "route_history": route_history,
        "expert_fit_history": history,
        "representation_checkpoint": representation,
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(checkpoint_path))
    reloaded = torch.load(str(checkpoint_path), map_location="cpu")
    reload_variants = {
        "uniform_mixture": (
            reloaded["uniform_assignment"], reloaded["uniform_weights"], 1
        ),
        "scene_level_router": (
            reloaded["assignments"]["scene_global"],
            reloaded["routes"]["scene_global"], 1,
        ),
        "one_group_horizon_router": (
            reloaded["assignments"]["motion_groups_1"],
            reloaded["routes"]["motion_groups_1"], 4,
        ),
        "group_only_router": (
            reloaded["assignments"]["motion_groups_8"],
            reloaded["routes"]["motion_groups_8_group_only"], 1,
        ),
        "group_horizon_router": (
            reloaded["main_phase93_assignment"],
            reloaded["main_phase93_weights"], int(config["calibration_bins"]),
        ),
        "group_horizon_without_persistence_gate": (
            reloaded["main_phase93_assignment"],
            reloaded["no_persistence_weights"], int(config["calibration_bins"]),
        ),
        "motion_groups_4": (
            reloaded["assignments"]["motion_groups_4"],
            reloaded["routes"]["motion_groups_4"], 4,
        ),
        "motion_groups_8_reoptimized": (
            reloaded["assignments"]["motion_groups_8"],
            reloaded["routes"]["motion_groups_8"], 4,
        ),
        "motion_groups_16": (
            reloaded["assignments"]["motion_groups_16"],
            reloaded["routes"]["motion_groups_16"], 4,
        ),
        "shuffled_groups_8": (
            reloaded["assignments"]["shuffled_groups_8"],
            reloaded["routes"]["shuffled_groups_8"], 4,
        ),
    }
    if evaluation_variants is not None:
        requested = set(evaluation_variants)
        reload_variants = {
            name: value for name, value in reload_variants.items() if name in requested
        }
    for name, (assignment, weights, bins) in reload_variants.items():
        value = _prediction(final_candidates, assignment, weights, bins)
        if not torch.equal(predictions[name], value):
            raise RuntimeError("Cold-reload mismatch for %s" % name)
    if args.mode == "smoke":
        print(json.dumps({
            "smoke": "passed", "cold_reload": True,
            "fixed_expert": fixed_name,
            "variants": sorted(predictions),
        }, indent=2))
        return

    baseline = _read_json(args.base_summary)
    expert_results = {name: baseline["results"][name] for name in EXPERTS}
    results = dict(expert_results)
    results["best_fixed_expert_calibration_selected"] = expert_results[fixed_name]
    for name, prediction in predictions.items():
        if name == "group_horizon_router":
            results[name] = baseline["results"]["joint_winner_bank"]
        else:
            results[name] = _render_and_measure(
                args, name, prediction, runtime, background
            )
    oracle = _oracle_from_expert_results(expert_results)
    summary = {
        "phase": 167,
        "method": checkpoint["method"],
        "scene": args.scene_name,
        "cold_reload": True,
        "protocol": checkpoint["config"],
        "results": results,
        "causal_best_fixed_expert": {
            "selected_on_observed_calibration_only": fixed_name,
            "calibration_rendered_risk": {
                name: float(global_risk[index])
                for index, name in enumerate(EXPERTS)
            },
        },
        "future_gt_oracle": oracle,
        "resources": {
            "ablation_fit_and_render_seconds": time.time() - started,
            "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            "checkpoint_bytes": os.path.getsize(str(checkpoint_path)),
            "router_parameters": {
                name: int(weights.numel())
                for name, weights in routes.items()
            },
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "base_checkpoint": args.base_checkpoint,
            "base_summary": args.base_summary,
        },
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(summary_path), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
