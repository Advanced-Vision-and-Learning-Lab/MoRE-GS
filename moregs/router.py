"""Joint differentiable rendered-risk router for the Phase-90 expert bank.

Phase 90 scored one group intervention at a time.  Those marginal decisions
did not compose after soft groups were mixed.  This causal follow-up freezes
the same expert family and jointly optimizes all group/horizon gates through
the renderer on the disjoint observed-prefix calibration tail.  The true
future is accessed only after a router checkpoint has been cold reloaded.
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
from options.gaussian_option import Gaussian_Options  # noqa: E402
from moregs.forecasting import _build_state, _load_runtime  # noqa: E402
from moregs.rendered_risk import (  # noqa: E402
    EXPERTS,
    _audit_trainable,
    _fit_bank,
    _forecast,
    _image_risk,
    _mix,
    _new_bank,
    _render_and_measure,
    _render_prediction,
)


class JointRenderedRouter(nn.Module):
    """Persistence gate plus one sparse non-persistence expert per cell."""

    def __init__(self, marginal_weights):
        super(JointRenderedRouter, self).__init__()
        selected = marginal_weights.argmax(dim=-1)
        initial_gate = torch.where(
            selected == 0,
            torch.full_like(selected, 0.05, dtype=torch.float32),
            torch.full_like(selected, 0.80, dtype=torch.float32),
        )
        self.gate_logits = nn.Parameter(torch.logit(initial_gate))
        expert_logits = torch.zeros(
            *selected.shape, len(EXPERTS) - 1, dtype=torch.float32
        )
        for expert in range(1, len(EXPERTS)):
            expert_logits[..., expert - 1] = torch.where(
                selected == expert,
                torch.full_like(initial_gate, 2.0),
                torch.zeros_like(initial_gate),
            )
        self.expert_logits = nn.Parameter(expert_logits)

    def forward(self):
        gate = torch.sigmoid(self.gate_logits)
        non_persistence = torch.softmax(self.expert_logits, dim=-1)
        return torch.cat([
            (1.0 - gate).unsqueeze(-1),
            gate.unsqueeze(-1) * non_persistence,
        ], dim=-1)


def _audit_router(router):
    groups = [
        {"name": name, "shape": list(parameter.shape),
         "parameters": int(parameter.numel())}
        for name, parameter in router.named_parameters()
    ]
    print(json.dumps({
        "joint_router_parameter_groups": groups,
        "joint_router_trainable_parameters": int(sum(
            item["parameters"] for item in groups
        )),
    }))


def _optimize_router(args, router, candidates, assignment, views, runtime,
                     background):
    _, pipeline, gaussians, _, _, _, _, _ = runtime
    candidates = candidates.detach()
    assignment = assignment.detach()
    time_value = torch.from_numpy(views[0].time).to(torch.float32).cuda()
    with torch.no_grad():
        _, _, _, _, weights_xyz, weights_rotation = gaussians(
            time_value, args.ckpt_iteration, return_weights=True
        )
        persistence_renders = []
        for horizon, view in enumerate(views):
            persistence_renders.append(_render_prediction(
                view, candidates[:, horizon, 0], gaussians, pipeline,
                background, weights_xyz, weights_rotation,
            ).detach())

    optimizer = torch.optim.AdamW(
        router.parameters(), lr=args.router_learning_rate,
        weight_decay=args.router_weight_decay,
    )
    history = []
    router.train()
    horizon_count = candidates.shape[1]
    for step in range(args.router_steps):
        horizon = (step * 7 + 3) % horizon_count
        group_weights = router()
        prediction, _ = _mix(
            candidates, assignment, group_weights, args.calibration_bins
        )
        rotation = F.normalize(prediction[:, horizon, 3:], dim=-1)
        prediction_horizon = torch.cat([
            prediction[:, horizon, :3], rotation
        ], dim=-1)
        rendering = _render_prediction(
            views[horizon], prediction_horizon, gaussians, pipeline,
            background, weights_xyz, weights_rotation,
        )
        gt = views[horizon].original_image[:3].cuda()
        rendered_risk = _image_risk(
            rendering, gt, persistence_renders[horizon], background, args
        )
        weights = router()
        # A scene-level router has one horizon bin, so there are no adjacent
        # bins over which to define a smoothness penalty.  Taking the mean of
        # that empty tensor produces NaN and invalidates the matched ablation.
        if weights.shape[1] > 1:
            smoothness = torch.mean((weights[:, 1:] - weights[:, :-1]) ** 2)
        else:
            smoothness = weights.new_zeros(())
        residual_mass = weights[..., 1:].sum(dim=-1).mean()
        expert_entropy = -torch.sum(
            weights[..., 1:] * torch.log(weights[..., 1:].clamp_min(1.0e-8)),
            dim=-1,
        ).mean()
        loss = (
            rendered_risk
            + args.router_smoothness_weight * smoothness
            + args.router_residual_weight * residual_mass
            + args.router_entropy_weight * expert_entropy
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite router loss at step %d" % step)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step == args.router_steps - 1 or step % 25 == 0:
            item = {
                "step": step,
                "horizon": horizon,
                "loss": float(loss.item()),
                "rendered_risk": float(rendered_risk.item()),
                "residual_mass": float(residual_mass.item()),
                "smoothness": float(smoothness.item()),
            }
            history.append(item)
            print(json.dumps(item))
    router.eval()
    return router().detach(), history


def main():
    parser = ArgumentParser(description="Phase 91 joint rendered router")
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
    parser.add_argument("--calibration_bins", default=4, type=int)
    parser.add_argument("--rendered_region_weight", default=1.0, type=float)
    parser.add_argument("--router_steps", default=200, type=int)
    parser.add_argument("--router_learning_rate", default=0.05, type=float)
    parser.add_argument("--router_weight_decay", default=0.0, type=float)
    parser.add_argument("--router_smoothness_weight", default=0.002, type=float)
    parser.add_argument("--router_residual_weight", default=0.0001, type=float)
    parser.add_argument("--router_entropy_weight", default=0.0001, type=float)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--expected_observed_frames", required=True, type=int)
    parser.add_argument("--expected_future_frames", required=True, type=int)
    parser.add_argument(
        "--expert_checkpoint", "--phase90_checkpoint", dest="expert_checkpoint", required=True
    )
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
    base = torch.load(args.expert_checkpoint, map_location="cpu")
    config = base["config"]
    if (
        config["future_frames_used_for_training_grouping_or_routing"] != 0
        or config["observed_frames"] != args.expected_observed_frames
        or config["future_frames"] != args.expected_future_frames
    ):
        raise RuntimeError("Phase-90 checkpoint protocol mismatch")
    fit_frames = int(config["fit_frames"])
    assignment = base["assignment"].cuda()
    rigidity = base["rigidity"].cuda()
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )

    calibration_bank = _new_bank(args, assignment, rigidity)
    _audit_trainable(calibration_bank, gaussians)
    calibration_epochs = 2 if args.mode == "smoke" else args.epochs
    calibration_history = _fit_bank(
        args, calibration_bank, state, canonical, fit_frames,
        calibration_epochs, "phase91_calibration",
    )
    candidates = _forecast(
        calibration_bank, state[:fit_frames], canonical, args,
        state.shape[0] - fit_frames,
    )
    router = JointRenderedRouter(base["rendered_weights"].float()).cuda()
    _audit_router(router)
    if args.mode == "smoke":
        args.router_steps = 2
    joint_weights, router_history = _optimize_router(
        args, router, candidates, assignment, train_views[fit_frames:],
        runtime, background,
    )
    checkpoint = {
        "method": "phase91_joint_differentiable_rendered_router",
        "joint_weights": joint_weights.cpu(),
        "router_state": router.state_dict(),
        "router_history": router_history,
        "calibration_history": calibration_history,
        "phase90_checkpoint": args.expert_checkpoint,
        "config": {
            "experts": list(EXPERTS),
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "fit_frames": fit_frames,
            "calibration_frames": int(state.shape[0] - fit_frames),
            "calibration_bins": args.calibration_bins,
            "router_steps": args.router_steps,
            "future_frames_used_for_router_training": 0,
            "calibration_expert_trained_on_fit_only": True,
            "router_jointly_optimized_through_renderer": True,
            "final_experts_frozen_from_phase90": True,
            "exact_persistence_path": True,
            "seed": args.seed,
        },
        "router_parameters": int(sum(p.numel() for p in router.parameters())),
        "train_seconds": time.time() - started,
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
        "representation_checkpoint": representation,
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(checkpoint_path))
    reloaded = torch.load(str(checkpoint_path), map_location="cuda")
    if reloaded["config"]["future_frames_used_for_router_training"] != 0:
        raise RuntimeError("Future leakage in router checkpoint")

    del calibration_bank, candidates, router
    torch.cuda.empty_cache()
    final_bank = _new_bank(args, assignment, rigidity)
    final_bank.load_state_dict(base["model"])
    final_bank.eval()
    for parameter in final_bank.parameters():
        parameter.requires_grad_(False)
    final_candidates = _forecast(final_bank, state, canonical, args)
    prediction, point_weights = _mix(
        final_candidates, assignment, reloaded["joint_weights"],
        args.calibration_bins,
    )
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Non-finite cold-reloaded forecast")
    if args.mode == "smoke":
        print(json.dumps({
            "smoke": "passed", "cold_reload": True,
            "router_parameters": checkpoint["router_parameters"],
            "mean_expert_weights": point_weights.mean((0, 1)).cpu().tolist(),
        }, indent=2))
        return

    result = _render_and_measure(
        args, "joint_rendered_router", prediction, runtime, background
    )
    summary = {
        "method": checkpoint["method"],
        "scene": args.scene_name,
        "cold_reload": True,
        "protocol": checkpoint["config"],
        "metrics": result["metrics"],
        "per_view": result["per_view"],
        "routing": {
            "expert_order": list(EXPERTS),
            "group_bin_weights": reloaded["joint_weights"].cpu().tolist(),
            "mean_point_weights": point_weights.mean((0, 1)).cpu().tolist(),
        },
        "resources": {
            "train_seconds": checkpoint["train_seconds"],
            "peak_gpu_allocated_mb": max(
                checkpoint["peak_gpu_allocated_mb"],
                torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            ),
            "router_parameters": checkpoint["router_parameters"],
            "frozen_expert_parameters": int(base["parameter_count"]),
            "router_checkpoint_bytes": os.path.getsize(str(checkpoint_path)),
            "frozen_expert_checkpoint_bytes": os.path.getsize(args.expert_checkpoint),
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "phase90_checkpoint": args.expert_checkpoint,
            "render_root": result["render_root"],
        },
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(summary_path), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
