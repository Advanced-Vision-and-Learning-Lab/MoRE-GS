"""Observed-prefix calibrated, regime-aware Gaussian trajectory forecasting.

The frozen GaussianPrediction keypoint representation is divided into soft
motion groups using observed trajectories only.  A compact shared model emits
three typed candidates: exact persistence, a group-coherent SE(3) trajectory,
and a local non-rigid trajectory.  A disjoint tail of the observed prefix
calibrates group/horizon expert weights using a lower confidence bound.  The
residual mass is constrained to be non-increasing with forecast horizon.
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
    _finite_state,
    _forecast_losses,
    _load_runtime,
    _temporal_mask,
)
from moregs.kinematic_residual import (  # noqa: E402
    _axis_angle_matrix,
    _axis_angle_quaternion,
    _quaternion_multiply,
)


def _weighted_group_mean(values, assignment):
    """values [N,...], assignment [N,G] -> [G,...]."""
    mass = assignment.sum(dim=0).clamp_min(1.0e-6)
    result = torch.einsum("ng,n...->g...", assignment, values)
    view = (mass.shape[0],) + (1,) * (values.ndim - 1)
    return result / mass.view(view)


def build_soft_motion_groups(state, canonical, groups=8, temperature=0.18,
                             iterations=30):
    """Deterministic soft k-means over observed motion and weak spatial cues."""
    xyz = state[:, :, :3]
    frames, nodes = xyz.shape[:2]
    scene_scale = canonical.std(dim=0, unbiased=False).mean().clamp_min(1.0e-4)
    samples = min(8, frames)
    sample_indices = torch.linspace(
        0, frames - 1, samples, device=xyz.device
    ).round().long()
    displacement = xyz[sample_indices] - xyz[:1]
    trajectory = displacement.permute(1, 0, 2).reshape(nodes, -1) / scene_scale
    velocity = xyz[1:] - xyz[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    jerk = acceleration[1:] - acceleration[:-1]
    speed = torch.linalg.vector_norm(velocity, dim=-1).mean(dim=0, keepdim=False)
    accel = torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=0) if acceleration.shape[0] else torch.zeros_like(speed)
    jerk_value = torch.linalg.vector_norm(jerk, dim=-1).mean(dim=0) if jerk.shape[0] else torch.zeros_like(speed)
    canonical_normalized = (
        (canonical - canonical.mean(dim=0, keepdim=True)) / scene_scale
    ) * 0.2
    descriptor = torch.cat([
        trajectory,
        canonical_normalized,
        (speed / scene_scale).unsqueeze(-1),
        (accel / scene_scale).unsqueeze(-1),
        (jerk_value / scene_scale).unsqueeze(-1),
    ], dim=-1)
    descriptor = (descriptor - descriptor.mean(dim=0, keepdim=True)) / (
        descriptor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-3)
    )

    group_count = min(int(groups), int(nodes))
    centers = [descriptor[torch.argmax(torch.sum(descriptor.square(), dim=-1))]]
    for _ in range(1, group_count):
        current = torch.stack(centers)
        distance = torch.cdist(descriptor, current).square() / float(descriptor.shape[-1])
        nearest = distance.min(dim=1).values
        centers.append(descriptor[torch.argmax(nearest)])
    centers = torch.stack(centers)
    for _ in range(iterations):
        distance = torch.cdist(descriptor, centers).square() / float(descriptor.shape[-1])
        assignment = torch.softmax(-distance / float(temperature), dim=-1)
        mass = assignment.sum(dim=0).clamp_min(1.0e-5)
        centers = torch.einsum("ng,nd->gd", assignment, descriptor) / mass[:, None]
    distance = torch.cdist(descriptor, centers).square() / float(descriptor.shape[-1])
    assignment = torch.softmax(-distance / float(temperature), dim=-1)
    assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    # Group rigidity is based on preservation of pairwise distance throughout
    # the observed prefix, not on a learned label or future motion.
    pair_distance = torch.cdist(xyz, xyz) / scene_scale
    reference = pair_distance[:1]
    change = torch.abs(pair_distance - reference).mean(dim=0)
    base_distance = reference[0]
    identity = torch.eye(nodes, device=xyz.device, dtype=xyz.dtype)
    rigidity = []
    for group in range(group_count):
        pair_weight = assignment[:, group, None] * assignment[None, :, group]
        pair_weight = pair_weight * (1.0 - identity)
        denominator = pair_weight.sum().clamp_min(1.0e-6)
        relative_change = (
            (change * pair_weight).sum() / denominator
        ) / ((base_distance * pair_weight).sum() / denominator).clamp_min(1.0e-4)
        rigidity.append(torch.exp(-5.0 * relative_change).clamp(0.0, 1.0))
    rigidity = torch.stack(rigidity)

    if velocity.shape[0] >= 4:
        # NumPy avoids a cuFFT initialization failure in the legacy
        # GaussianPrediction PyTorch/CUDA environment.
        velocity_cpu = velocity.detach().cpu().numpy()
        spectrum = np.abs(np.fft.rfft(velocity_cpu, axis=0)) ** 2
        non_dc = spectrum[1:].sum(axis=-1)
        periodicity_numpy = non_dc.max(axis=0) / np.maximum(non_dc.sum(axis=0), 1.0e-8)
        periodicity_point = torch.as_tensor(
            periodicity_numpy, device=xyz.device, dtype=xyz.dtype
        )
    else:
        periodicity_point = torch.zeros(nodes, device=xyz.device)
    entropy = -torch.sum(assignment * torch.log(assignment.clamp_min(1.0e-8)), dim=-1)
    entropy = entropy / math.log(float(max(group_count, 2)))
    diagnostics = {
        "groups": group_count,
        "effective_group_mass": assignment.sum(dim=0).detach().cpu().tolist(),
        "mean_assignment_entropy": float(entropy.mean().item()),
        "rigidity_by_group": rigidity.detach().cpu().tolist(),
        "mean_rigidity": float(rigidity.mean().item()),
        "mean_speed_normalized": float((speed / scene_scale).mean().item()),
        "mean_acceleration_normalized": float((accel / scene_scale).mean().item()),
        "mean_jerk_normalized": float((jerk_value / scene_scale).mean().item()),
        "periodicity_by_group": _weighted_group_mean(
            periodicity_point.unsqueeze(-1), assignment
        ).squeeze(-1).detach().cpu().tolist(),
        "mean_periodicity": float(periodicity_point.mean().item()),
        "future_frames_used": 0,
    }
    return assignment.detach(), rigidity.detach(), diagnostics


class RegimeAwareForecaster(nn.Module):
    """Shared rigid/local candidate generator over fixed soft motion groups."""

    def __init__(self, assignment, rigidity, sequence_length=10, hidden=32,
                 heads=8, feedforward=64, dropout=0.2, local_rank=8):
        super(RegimeAwareForecaster, self).__init__()
        self.sequence_length = int(sequence_length)
        self.groups = int(assignment.shape[1])
        self.hidden = int(hidden)
        self.register_buffer("assignment", assignment.float())
        self.register_buffer("rigidity", rigidity.float())

        self.group_input = nn.Linear(7, hidden)
        self.group_position = nn.Parameter(torch.zeros(sequence_length, hidden))
        self.group_token = nn.Embedding(self.groups, hidden)
        self.rigidity_projection = nn.Linear(1, hidden)
        group_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.group_encoder = nn.TransformerEncoder(group_layer, num_layers=1)

        self.local_input = nn.Linear(7, hidden)
        self.canonical_projection = nn.Linear(3, hidden)
        self.local_position = nn.Parameter(torch.zeros(sequence_length, hidden))
        local_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=1)
        self.mask_token = nn.Parameter(torch.zeros(hidden))

        self.horizon_projection = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.rigid_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 6)
        )
        self.local_head = nn.Sequential(
            nn.Linear(2 * hidden, local_rank), nn.GELU(), nn.Linear(local_rank, 7)
        )
        nn.init.normal_(self.group_position, std=0.02)
        nn.init.normal_(self.local_position, std=0.02)
        nn.init.normal_(self.group_token.weight, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.zeros_(self.rigid_head[-1].weight)
        nn.init.zeros_(self.rigid_head[-1].bias)
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)

    @staticmethod
    def _align(rotation, reference):
        sign = torch.where(
            torch.sum(rotation * reference, dim=-1, keepdim=True) < 0,
            -torch.ones_like(rotation[..., :1]),
            torch.ones_like(rotation[..., :1]),
        )
        return F.normalize(rotation * sign, dim=-1)

    def _augment_group_context(self, context):
        """Extension hook for dictionary-conditioned Phase-87 variants."""
        return context

    def _pool_group_context(self, descriptors):
        """Pool encoded temporal descriptors; subclasses may retain code structure."""
        return descriptors.mean(dim=1)

    def _group_features(self, sequence):
        xyz = sequence[:, :, :3]
        centroid = _weighted_group_mean(xyz, self.assignment)
        velocity = torch.cat([
            torch.zeros_like(centroid[:, :1]), centroid[:, 1:] - centroid[:, :-1]
        ], dim=1)
        difference = xyz[:, None, :, :] - centroid[None, :, :, :]
        radius_point = torch.sum(difference.square(), dim=-1)
        radius = torch.sqrt(
            torch.einsum("ng,ngl->gl", self.assignment, radius_point) /
            self.assignment.sum(dim=0).clamp_min(1.0e-6)[:, None]
        ).unsqueeze(-1)
        return torch.cat([centroid, velocity, radius], dim=-1), centroid

    def forward(self, sequence, canonical, horizons, horizon_scale,
                temporal_mask=None):
        nodes = sequence.shape[0]
        horizon_count = horizons.shape[0]
        scene_scale = canonical.std(dim=0, unbiased=False).mean().clamp_min(1.0e-4)
        horizon_features = DirectHorizonTransformer._horizon_features(
            horizons, horizon_scale
        )
        horizon_hidden = self.horizon_projection(horizon_features)

        group_features, centroid = self._group_features(sequence)
        group_mean = group_features.mean(dim=1, keepdim=True)
        group_std = group_features.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1.0e-4)
        group_features = (group_features - group_mean) / group_std
        indices = torch.arange(self.groups, device=sequence.device)
        group_hidden = self.group_input(group_features)
        group_hidden = group_hidden + self.group_position.unsqueeze(0)
        group_hidden = group_hidden + self.group_token(indices).unsqueeze(1)
        group_hidden = group_hidden + self.rigidity_projection(
            self.rigidity[:, None]
        ).unsqueeze(1)
        if temporal_mask is not None and bool(temporal_mask.any()):
            replacement = self.group_token(indices).unsqueeze(1)
            replacement = replacement + self.group_position.unsqueeze(0)
            replacement = replacement + self.rigidity_projection(
                self.rigidity[:, None]
            ).unsqueeze(1)
            group_hidden = torch.where(
                temporal_mask.view(1, -1, 1), replacement, group_hidden
            )
        group_descriptors = self.group_encoder(group_hidden)
        group_context = self._pool_group_context(group_descriptors)
        group_context = self._augment_group_context(group_context)
        group_joint = torch.cat([
            group_context[:, None, :].expand(-1, horizon_count, -1),
            horizon_hidden[None, :, :].expand(self.groups, -1, -1),
        ], dim=-1)
        transform = self.rigid_head(group_joint)

        mean = sequence.mean(dim=1, keepdim=True)
        std = sequence.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-4)
        normalized = (sequence - mean) / std
        canonical_normalized = (
            canonical - canonical.mean(dim=0, keepdim=True)
        ) / canonical.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-4)
        local_hidden = self.local_input(normalized)
        local_hidden = local_hidden + self.local_position.unsqueeze(0)
        local_hidden = local_hidden + self.canonical_projection(
            canonical_normalized
        ).unsqueeze(1)
        if temporal_mask is not None and bool(temporal_mask.any()):
            replacement = self.mask_token.view(1, 1, -1)
            replacement = replacement + self.local_position.unsqueeze(0)
            replacement = replacement + self.canonical_projection(
                canonical_normalized
            ).unsqueeze(1)
            local_hidden = torch.where(
                temporal_mask.view(1, -1, 1), replacement, local_hidden
            )
        local_context = self.local_encoder(local_hidden).mean(dim=1)
        local_joint = torch.cat([
            local_context[:, None, :].expand(-1, horizon_count, -1),
            horizon_hidden[None, :, :].expand(nodes, -1, -1),
        ], dim=-1)
        local_raw = self.local_head(local_joint)

        last = sequence[:, -1]
        persistence = last[:, None, :].expand(-1, horizon_count, -1)
        group_speed = torch.linalg.vector_norm(
            centroid[:, 1:] - centroid[:, :-1], dim=-1
        ).mean(dim=1)
        horizon = horizons.view(1, -1, 1)
        translation_bound = (
            1.5 * group_speed[:, None, None] * horizon
            + 0.0025 * scene_scale * horizon
        ).clamp(max=float(0.30 * scene_scale.item()))
        translation = torch.tanh(transform[..., :3]) * translation_bound
        rotation_bound = (0.03 * horizon).clamp(max=0.35)
        axis_angle = torch.tanh(transform[..., 3:]) * rotation_bound

        group_center = centroid[:, -1]
        relative = last[None, :, :3] - group_center[:, None, :]
        matrix = _axis_angle_matrix(axis_angle)
        rotated = torch.einsum("ghij,gnj->ghni", matrix, relative)
        group_xyz = (
            group_center[:, None, None, :] + rotated + translation[:, :, None, :]
        ).permute(2, 1, 0, 3)
        rigid_xyz = torch.sum(
            self.assignment[:, None, :, None] * group_xyz, dim=2
        )
        delta_quaternion = _axis_angle_quaternion(axis_angle)
        mixed_delta = torch.einsum("ng,ghq->nhq", self.assignment, delta_quaternion)
        mixed_delta = F.normalize(mixed_delta, dim=-1)
        rigid_rotation = _quaternion_multiply(
            mixed_delta, last[:, None, 3:].expand(-1, horizon_count, -1)
        )
        rigid_rotation = self._align(
            rigid_rotation, last[:, None, 3:]
        )
        rigid = torch.cat([rigid_xyz, rigid_rotation], dim=-1)

        point_speed = torch.linalg.vector_norm(
            sequence[:, 1:, :3] - sequence[:, :-1, :3], dim=-1
        ).mean(dim=1)
        local_bound = (
            1.5 * point_speed[:, None, None] * horizons.view(1, -1, 1)
            + 0.0025 * scene_scale * horizons.view(1, -1, 1)
        ).clamp(max=float(0.25 * scene_scale.item()))
        local_xyz = last[:, None, :3] + torch.tanh(local_raw[..., :3]) * local_bound
        local_rotation = last[:, None, 3:] + torch.tanh(local_raw[..., 3:]) * (
            0.35 * (horizons / float(max(horizon_scale, 1))).view(1, -1, 1)
        )
        local_rotation = self._align(local_rotation, last[:, None, 3:])
        local = torch.cat([local_xyz, local_rotation], dim=-1)
        candidates = torch.stack([persistence, rigid, local], dim=2)
        return candidates, transform


def _new_model(args, assignment, rigidity):
    return RegimeAwareForecaster(
        assignment=assignment,
        rigidity=rigidity,
        sequence_length=args.sequence_length,
        hidden=args.hidden,
        heads=args.heads,
        feedforward=args.feedforward,
        dropout=args.dropout,
        local_rank=args.local_rank,
    ).cuda()


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


def _candidate_point_error(candidates, target):
    xyz = torch.mean((candidates[..., :3] - target[:, :, None, :3]) ** 2, dim=-1)
    target_rotation = target[:, :, None, 3:]
    sign = torch.where(
        torch.sum(candidates[..., 3:] * target_rotation, dim=-1, keepdim=True) < 0,
        -torch.ones_like(target_rotation[..., :1]),
        torch.ones_like(target_rotation[..., :1]),
    )
    rotation = torch.mean(
        (candidates[..., 3:] - target_rotation * sign) ** 2, dim=-1
    )
    return xyz + rotation


def calibrate_experts(args, model, state, canonical, assignment, fit_frames):
    """Calibrate on a disjoint observed-prefix tail; never read true future."""
    samples = [[[] for _ in range(args.calibration_bins)] for _ in range(args.groups)]
    model.eval()
    with torch.no_grad():
        for cutoff in range(fit_frames, state.shape[0]):
            available = min(args.expected_future_frames, state.shape[0] - cutoff)
            if available <= 0 or cutoff < args.sequence_length:
                continue
            sequence = state[cutoff - args.sequence_length:cutoff].permute(1, 0, 2)
            target = state[cutoff:cutoff + available].permute(1, 0, 2)
            horizons = torch.arange(1, available + 1, device=state.device, dtype=state.dtype)
            candidates, _ = model(
                sequence, canonical, horizons, args.expected_future_frames
            )
            point_error = _candidate_point_error(candidates, target)
            group_error = torch.einsum(
                "ng,nhe->ghe", assignment, point_error
            ) / assignment.sum(dim=0).clamp_min(1.0e-6)[:, None, None]
            for horizon_index in range(available):
                bin_index = min(
                    args.calibration_bins - 1,
                    int(horizon_index * args.calibration_bins /
                        float(args.expected_future_frames)),
                )
                for group in range(args.groups):
                    samples[group][bin_index].append(
                        group_error[group, horizon_index].detach().cpu().numpy()
                    )

    weights = torch.zeros(
        args.groups, args.calibration_bins, 3, dtype=state.dtype
    )
    statistics = []
    for group in range(args.groups):
        previous_mass = 1.0
        for bin_index in range(args.calibration_bins):
            values = samples[group][bin_index]
            if not values:
                weights[group, bin_index, 0] = 1.0
                previous_mass = 0.0
                statistics.append({
                    "group": group, "bin": bin_index, "samples": 0,
                    "weights": [1.0, 0.0, 0.0], "reason": "no_calibration_support",
                })
                continue
            array = np.stack(values, axis=0)
            difference = array[:, :1] - array[:, 1:]
            mean = difference.mean(axis=0)
            if array.shape[0] > 1:
                standard_error = difference.std(axis=0, ddof=1) / math.sqrt(array.shape[0])
            else:
                standard_error = np.full(2, np.inf, dtype=np.float32)
            lower = mean - args.confidence_z * standard_error
            positive = np.maximum(lower, 0.0)
            baseline = max(float(array[:, 0].mean()), 1.0e-8)
            relative = positive / baseline
            raw_mass = min(args.maximum_residual_mass, float(relative.sum() * args.confidence_gain))
            mass = min(previous_mass, raw_mass)
            previous_mass = mass
            if positive.sum() > 0.0 and mass > 0.0:
                proportion = positive / positive.sum()
                selected = np.asarray([
                    1.0 - mass, mass * proportion[0], mass * proportion[1]
                ], dtype=np.float32)
            else:
                selected = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            weights[group, bin_index] = torch.from_numpy(selected)
            statistics.append({
                "group": group,
                "bin": bin_index,
                "samples": int(array.shape[0]),
                "mean_improvement_rigid_local": mean.tolist(),
                "standard_error_rigid_local": standard_error.tolist(),
                "lower_confidence_improvement_rigid_local": lower.tolist(),
                "baseline_error": baseline,
                "weights": selected.tolist(),
            })
    return weights.cuda(), statistics


def _mix_candidates(candidates, assignment, group_weights, calibration_bins,
                    future_frames):
    horizon_count = candidates.shape[1]
    point_weights = []
    for horizon_index in range(horizon_count):
        bin_index = min(
            calibration_bins - 1,
            int(horizon_index * calibration_bins / float(future_frames)),
        )
        value = torch.matmul(assignment, group_weights[:, bin_index])
        point_weights.append(value)
    point_weights = torch.stack(point_weights, dim=1)
    point_weights = point_weights / point_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    xyz = torch.sum(point_weights.unsqueeze(-1) * candidates[..., :3], dim=2)
    rotation = torch.sum(point_weights.unsqueeze(-1) * candidates[..., 3:], dim=2)
    reference = candidates[:, :, 0, 3:]
    sign = torch.where(
        torch.sum(rotation * reference, dim=-1, keepdim=True) < 0,
        -torch.ones_like(rotation[..., :1]),
        torch.ones_like(rotation[..., :1]),
    )
    rotation = F.normalize(rotation * sign, dim=-1)
    return torch.cat([xyz, rotation], dim=-1), point_weights


def train(args, model_parser, optimization_parser, pipeline_parser):
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    runtime = _load_runtime(args, model_parser, optimization_parser, pipeline_parser)
    _, _, gaussians, _, observed, _, _, representation_path = runtime
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    assignment, rigidity, diagnostics = build_soft_motion_groups(
        state, canonical, args.groups, args.group_temperature
    )
    if assignment.shape[1] != args.groups:
        raise RuntimeError("Requested group count exceeds node count")
    model = _new_model(args, assignment, rigidity)
    _audit_trainable(model, gaussians)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1.0e-5
    )
    fit_frames = int(math.floor(state.shape[0] * args.fit_fraction))
    fit_frames = max(args.sequence_length + 2, min(fit_frames, state.shape[0] - 2))
    max_horizon = min(
        args.max_train_horizon,
        args.expected_future_frames,
        fit_frames - args.sequence_length,
    )
    valid_cutoffs = list(range(args.sequence_length, fit_frames))
    history = []
    model.train()
    for epoch in range(args.epochs):
        cutoff = valid_cutoffs[(epoch * 17 + 11) % len(valid_cutoffs)]
        available = min(max_horizon, fit_frames - cutoff)
        sequence = state[cutoff - args.sequence_length:cutoff].permute(1, 0, 2)
        target = state[cutoff:cutoff + available].permute(1, 0, 2)
        horizons = torch.arange(1, available + 1, device=state.device, dtype=state.dtype)
        temporal_mask, mask_ratio = _temporal_mask(
            epoch, args.epochs, args.sequence_length, state.device
        )
        candidates, transform = model(
            sequence, canonical, horizons, args.expected_future_frames, temporal_mask
        )
        rigid_losses = _forecast_losses(
            candidates[:, :, 1], target, sequence,
            torch.ones(candidates.shape[0], available, 2, device=state.device),
        )
        local_losses = _forecast_losses(
            candidates[:, :, 2], target, sequence,
            torch.ones(candidates.shape[0], available, 2, device=state.device),
        )
        local_delta = candidates[:, :, 2, :3] - sequence[:, -1:, :3]
        group_delta = _weighted_group_mean(local_delta, assignment)
        smooth_delta = torch.einsum("ng,ghd->nhd", assignment, group_delta)
        local_smoothness = F.mse_loss(local_delta, smooth_delta)
        loss = (
            rigid_losses[0] + local_losses[0]
            + args.local_smoothness_weight * local_smoothness
            + args.transform_regularizer * torch.mean(transform.square())
        )
        auxiliary_loss = torch.zeros((), device=state.device)
        if hasattr(model, "auxiliary_loss"):
            auxiliary_loss = model.auxiliary_loss()
            auxiliary_weight = getattr(
                args, "auxiliary_weight",
                getattr(args, "typicality_weight", 0.0),
            )
            loss = loss + auxiliary_weight * auxiliary_loss
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
                "rigid_loss": float(rigid_losses[0].item()),
                "local_loss": float(local_losses[0].item()),
                "local_smoothness": float(local_smoothness.item()),
                "auxiliary_loss": float(auxiliary_loss.item()),
                "mask_ratio": float(mask_ratio),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(item)
            print(json.dumps(item))

    group_weights, calibration = calibrate_experts(
        args, model, state, canonical, assignment, fit_frames
    )
    method_name = getattr(model, "method_name", "phase86_regime_aware_forecaster")
    checkpoint = {
        "method": method_name,
        "model": model.state_dict(),
        "assignment": assignment.detach().cpu(),
        "rigidity": rigidity.detach().cpu(),
        "group_weights": group_weights.detach().cpu(),
        "regime_diagnostics": diagnostics,
        "calibration": calibration,
        "history": history,
        "config": {
            "sequence_length": args.sequence_length,
            "hidden": args.hidden,
            "heads": args.heads,
            "feedforward": args.feedforward,
            "dropout": args.dropout,
            "local_rank": args.local_rank,
            "groups": args.groups,
            "epochs": args.epochs,
            "fit_fraction": args.fit_fraction,
            "fit_frames": fit_frames,
            "calibration_frames": int(state.shape[0] - fit_frames),
            "calibration_bins": args.calibration_bins,
            "confidence_z": args.confidence_z,
            "confidence_gain": args.confidence_gain,
            "maximum_residual_mass": args.maximum_residual_mass,
            "observed_frames": args.expected_observed_frames,
            "future_frames": args.expected_future_frames,
            "future_frames_used": 0,
            "grouping_observed_prefix_only": True,
            "calibration_observed_prefix_only": True,
            "horizon_residual_mass_non_increasing": True,
            "exact_persistence_expert": True,
            "autoregressive": False,
            "gaussian_representation_trainable": False,
            "seed": args.seed,
        },
        "representation_checkpoint": representation_path,
        "train_seconds": time.time() - started,
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    if hasattr(model, "protocol_config"):
        checkpoint["config"].update(model.protocol_config())
    if hasattr(model, "extra_checkpoint"):
        checkpoint.update(model.extra_checkpoint())
    path = Path(args.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))

    reloaded = torch.load(str(path), map_location="cuda")
    reload_model = _new_model(
        args, reloaded["assignment"].cuda(), reloaded["rigidity"].cuda()
    )
    reload_model.load_state_dict(reloaded["model"])
    reload_model.eval()
    with torch.no_grad():
        horizons = torch.arange(
            1, args.expected_future_frames + 1, device=state.device, dtype=state.dtype
        )
        candidates, _ = reload_model(
            state[-args.sequence_length:].permute(1, 0, 2),
            canonical, horizons, args.expected_future_frames,
        )
        prediction, _ = _mix_candidates(
            candidates, reloaded["assignment"].cuda(),
            reloaded["group_weights"].cuda(), args.calibration_bins,
            args.expected_future_frames,
        )
    if not _finite_state(reloaded["model"]) or not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Cold-reloaded checkpoint is non-finite")
    print(json.dumps({
        "checkpoint": str(path),
        "bytes": os.path.getsize(str(path)),
        "parameter_count": checkpoint["parameter_count"],
        "fit_frames": fit_frames,
        "calibration_frames": int(state.shape[0] - fit_frames),
        "cold_reload": True,
    }, indent=2))


def evaluate_checkpoint(args, model_parser, optimization_parser, pipeline_parser):
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    runtime = _load_runtime(args, model_parser, optimization_parser, pipeline_parser)
    dataset, pipeline, gaussians, _, observed, _, test_views, representation_path = runtime
    state = _build_state(observed.kpts_xyz_train, observed.kpts_r_train)
    canonical = gaussians.get_superGaussians.detach()
    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    config = checkpoint["config"]
    required = (
        config["future_frames_used"] == 0
        and config["grouping_observed_prefix_only"]
        and config["calibration_observed_prefix_only"]
        and config["horizon_residual_mass_non_increasing"]
        and config["exact_persistence_expert"]
        and not config["autoregressive"]
        and not config["gaussian_representation_trainable"]
        and config["observed_frames"] == args.expected_observed_frames
        and config["future_frames"] == args.expected_future_frames
    )
    if not required:
        raise RuntimeError("Checkpoint protocol mismatch")
    assignment = checkpoint["assignment"].cuda()
    rigidity = checkpoint["rigidity"].cuda()
    group_weights = checkpoint["group_weights"].cuda()
    model = _new_model(args, assignment, rigidity)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    horizons = torch.arange(
        1, args.expected_future_frames + 1, device=state.device, dtype=state.dtype
    )
    torch.cuda.synchronize()
    forecast_started = time.time()
    with torch.no_grad():
        candidates, _ = model(
            state[-args.sequence_length:].permute(1, 0, 2),
            canonical, horizons, args.expected_future_frames,
        )
        prediction, point_weights = _mix_candidates(
            candidates, assignment, group_weights, config["calibration_bins"],
            args.expected_future_frames,
        )
    torch.cuda.synchronize()
    forecast_seconds = time.time() - forecast_started
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Non-finite forecast")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )
    render_started = time.time()
    eval_root, _ = render_kpts(
        args.model_path,
        args.experiment_name,
        "[metrics]phase86_regime_aware",
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
    psnr_values = np.asarray(list(per_view["PSNR"].values()))
    horizon_psnr = [
        float(psnr_values[index].mean())
        for index in np.array_split(np.arange(len(psnr_values)), 4)
    ]
    mean_weights = point_weights.mean(dim=(0, 1)).detach().cpu().tolist()
    final_weights = point_weights[:, -1].mean(dim=0).detach().cpu().tolist()
    summary = {
        "method": checkpoint.get("method", "phase86_regime_aware_forecaster"),
        "scene": args.scene_name,
        "development_diagnostic_not_untouched_promotion": True,
        "cold_reload": True,
        "protocol": config,
        "metrics": metrics,
        "horizon_psnr": horizon_psnr,
        "routing": {
            "expert_order": ["persistence", "rigid_se3", "local_nonrigid"],
            "mean_point_weights": mean_weights,
            "last_horizon_point_weights": final_weights,
            "group_horizon_bin_weights": group_weights.detach().cpu().tolist(),
        },
        "regime_diagnostics": checkpoint["regime_diagnostics"],
        "calibration": checkpoint["calibration"],
        "resources": {
            "train_seconds": checkpoint["train_seconds"],
            "train_peak_gpu_allocated_mb": checkpoint["peak_gpu_allocated_mb"],
            "forecast_head_seconds": forecast_seconds,
            "forecast_head_fps": args.expected_future_frames / forecast_seconds,
            "render_and_png_fps": args.expected_future_frames / render_seconds,
            "eval_peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            "parameters": checkpoint["parameter_count"],
            "checkpoint_bytes": os.path.getsize(args.checkpoint),
            "eval_seconds": time.time() - started,
        },
        "artifacts": {
            "checkpoint": args.checkpoint,
            "representation_checkpoint": representation_path,
            "render_root": eval_root,
        },
    }
    if hasattr(model, "dictionary_diagnostics"):
        summary["dictionary"] = model.dictionary_diagnostics()
    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    parser = ArgumentParser(description="Phase 86 regime-aware forecaster")
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
    parser.add_argument("--confidence_z", default=1.0, type=float)
    parser.add_argument("--confidence_gain", default=5.0, type=float)
    parser.add_argument("--maximum_residual_mass", default=0.95, type=float)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--expected_observed_frames", required=True, type=int)
    parser.add_argument("--expected_future_frames", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--experiment_name", default="moregs_grouping")
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
