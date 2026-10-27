"""Matched Fourier-coefficient versus random-basis forecasting control.

Both variants have exactly the same parameters and optimization.  The only
difference is the fixed orthonormal temporal basis: ordered DCT-II frequencies
or a seeded random orthonormal matrix.  Coefficients are divided into three
bands, encoded by a shared Transformer, decoded as persistence-anchored
residuals, and predicted for all horizons in parallel.
"""

from __future__ import print_function

import json
import math
import os
import sys
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
from moregs import forecasting as phase80  # noqa: E402


def _dct_basis(length):
    index = np.arange(length, dtype=np.float32)
    frequency = np.arange(length, dtype=np.float32)[:, None]
    basis = np.sqrt(2.0 / float(length)) * np.cos(
        math.pi / float(length) * (index[None, :] + 0.5) * frequency
    )
    basis[0] = math.sqrt(1.0 / float(length))
    return torch.from_numpy(basis.astype(np.float32))


def _random_basis(length, seed):
    generator = np.random.RandomState(seed)
    matrix = generator.normal(size=(length, length)).astype(np.float64)
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return torch.from_numpy((q * signs[None, :]).T.astype(np.float32))


class FrequencyBasisTransformer(nn.Module):
    def __init__(self, sequence_length=10, hidden=32, heads=8,
                 feedforward=64, dropout=0.2, basis_variant="dct", seed=42):
        super(FrequencyBasisTransformer, self).__init__()
        self.sequence_length = int(sequence_length)
        self.hidden = int(hidden)
        self.bands = 3
        if basis_variant == "dct":
            basis = _dct_basis(sequence_length)
        elif basis_variant == "random":
            basis = _random_basis(sequence_length, 2024 * int(seed) + 92)
        else:
            raise ValueError("Unknown basis variant %s" % basis_variant)
        self.register_buffer("temporal_basis", basis)
        boundaries = np.array_split(np.arange(sequence_length), self.bands)
        band_ids = torch.empty(sequence_length, dtype=torch.long)
        for band, indices in enumerate(boundaries):
            band_ids[torch.as_tensor(indices, dtype=torch.long)] = band
        self.register_buffer("band_ids", band_ids)

        self.input_projection = nn.Linear(7, hidden)
        self.canonical_projection = nn.Linear(3, hidden)
        self.coefficient_position = nn.Parameter(
            torch.zeros(sequence_length, hidden)
        )
        self.band_token = nn.Embedding(self.bands, hidden)
        self.mask_token = nn.Parameter(torch.zeros(hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.horizon_projection = nn.Sequential(
            nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * hidden, 2 * hidden), nn.GELU(),
            nn.Linear(2 * hidden, 7),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 2),
        )
        nn.init.normal_(self.coefficient_position, std=0.02)
        nn.init.normal_(self.band_token.weight, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -3.0)

    def forward(self, sequence, canonical, horizons, horizon_scale,
                temporal_mask=None):
        mean = sequence.mean(dim=1, keepdim=True)
        std = sequence.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-4)
        normalized = (sequence - mean) / std
        coefficients = torch.einsum(
            "kl,nld->nkd", self.temporal_basis, normalized
        )
        canonical_normalized = (
            canonical - canonical.mean(dim=0, keepdim=True)
        ) / canonical.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1.0e-4)
        hidden = self.input_projection(coefficients)
        hidden = hidden + self.coefficient_position.unsqueeze(0)
        hidden = hidden + self.band_token(self.band_ids).unsqueeze(0)
        hidden = hidden + self.canonical_projection(
            canonical_normalized
        ).unsqueeze(1)
        if temporal_mask is not None and bool(temporal_mask.any()):
            replacement = self.mask_token.view(1, 1, -1)
            replacement = replacement + self.coefficient_position.unsqueeze(0)
            replacement = replacement + self.band_token(self.band_ids).unsqueeze(0)
            replacement = replacement + self.canonical_projection(
                canonical_normalized
            ).unsqueeze(1)
            hidden = torch.where(
                temporal_mask.view(1, -1, 1), replacement, hidden
            )
        encoded = self.encoder(hidden)
        contexts = []
        for band in range(self.bands):
            contexts.append(encoded[:, self.band_ids == band].mean(dim=1))
        context = torch.stack(contexts, dim=1)

        horizon_features = phase80.DirectHorizonTransformer._horizon_features(
            horizons, horizon_scale
        )
        horizon_hidden = self.horizon_projection(horizon_features)
        context = context[:, None].expand(-1, horizons.shape[0], -1, -1)
        horizon_hidden = horizon_hidden[None, :, None].expand(
            context.shape[0], -1, self.bands, -1
        )
        joint = torch.cat([context, horizon_hidden], dim=-1)
        raw = self.decoder(joint) * std[:, 0, None, None, :]
        band_gates = torch.sigmoid(self.gate(joint))
        residual_xyz = torch.sum(
            band_gates[..., :1] * raw[..., :3], dim=2
        ) / math.sqrt(float(self.bands))
        residual_rotation = torch.sum(
            band_gates[..., 1:] * raw[..., 3:], dim=2
        ) / math.sqrt(float(self.bands))
        last = sequence[:, -1:].expand(-1, horizons.shape[0], -1)
        xyz = last[..., :3] + residual_xyz
        rotation = F.normalize(last[..., 3:] + residual_rotation, dim=-1)
        sign = torch.where(
            torch.sum(rotation * last[..., 3:], dim=-1, keepdim=True) < 0,
            -torch.ones_like(rotation[..., :1]),
            torch.ones_like(rotation[..., :1]),
        )
        rotation = rotation * sign
        effective_gates = band_gates.mean(dim=2)
        return torch.cat([xyz, rotation], dim=-1), effective_gates


def _new_model(args):
    return FrequencyBasisTransformer(
        sequence_length=args.sequence_length,
        hidden=args.hidden,
        heads=args.heads,
        feedforward=args.feedforward,
        dropout=args.dropout,
        basis_variant=args.basis_variant,
        seed=args.seed,
    ).cuda()


def _annotate_checkpoint(path, variant):
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["method"] = "phase92_%s_basis_multiband" % variant
    checkpoint["config"]["basis_variant"] = variant
    checkpoint["config"]["basis_is_fixed_orthonormal"] = True
    checkpoint["config"]["frequency_bands"] = 3
    torch.save(checkpoint, path)


def _annotate_summary(path, variant):
    with open(path, "r") as handle:
        summary = json.load(handle)
    summary["method"] = "phase92_%s_basis_multiband" % variant
    summary["protocol"]["basis_variant"] = variant
    summary["protocol"]["basis_is_fixed_orthonormal"] = True
    summary["protocol"]["frequency_bands"] = 3
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)


def main():
    parser = ArgumentParser(description="Phase 92 frequency basis control")
    optimization_parser = OptimizationParams(parser)
    model_parser = ModelParams(parser, sentinel=True)
    pipeline_parser = PipelineParams(parser)
    options = Gaussian_Options(parser)
    options.initial()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--basis_variant", choices=["dct", "random"], required=True)
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
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--expected_observed_frames", required=True, type=int)
    parser.add_argument("--expected_future_frames", required=True, type=int)
    parser.add_argument("--allow_repeated_future_times", action="store_true")
    parser.add_argument("--strict_fit_no_future_views", action="store_true")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--experiment_name", required=True)
    args = get_combined_args(parser)

    seed = 2024 * args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    phase80._new_model = _new_model
    if args.mode == "train":
        phase80.train(args, model_parser, optimization_parser, pipeline_parser)
        _annotate_checkpoint(args.checkpoint, args.basis_variant)
    else:
        if not args.summary:
            raise ValueError("--summary is required in eval mode")
        phase80.evaluate_checkpoint(
            args, model_parser, optimization_parser, pipeline_parser
        )
        _annotate_summary(args.summary, args.basis_variant)


if __name__ == "__main__":
    main()
