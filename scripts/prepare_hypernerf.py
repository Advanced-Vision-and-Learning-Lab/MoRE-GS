#!/usr/bin/env python3
"""Create a non-destructive, official-ID-only HyperNeRF forecasting view."""

import argparse
import json
import os
from pathlib import Path


def link(source: Path, output: Path, name: str) -> None:
    target = source / name
    destination = output / name
    if not target.exists():
        raise FileNotFoundError(target)
    if destination.is_symlink():
        if destination.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected symlink target: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    os.symlink(target, destination, target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--time-split", type=float, default=0.9)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    dataset = json.loads((source / "dataset.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())
    official_train = list(dataset.get("train_ids", []))
    official_val = list(dataset.get("val_ids", []))
    if not official_train:
        raise RuntimeError("The source has no official training IDs")
    if set(official_train) & set(official_val):
        raise RuntimeError("Official HyperNeRF train/validation IDs overlap")

    ordered = sorted(
        official_train,
        key=lambda name: (int(metadata[name]["warp_id"]), name),
    )
    split = int(len(ordered) * args.time_split)
    observed = ordered[:split]
    future = ordered[split:]
    if not observed or not future or set(observed) & set(future):
        raise RuntimeError("Invalid observed/future temporal split")
    if max(int(metadata[name]["warp_id"]) for name in observed) >= min(
        int(metadata[name]["warp_id"]) for name in future
    ):
        raise RuntimeError("Temporal prefix is not strictly ordered")

    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "camera",
        "rgb",
        "metadata.json",
        "scene.json",
        "points.npy",
        "points3D_downsample.ply",
    ):
        link(source, output, name)

    strict = dict(dataset)
    strict["ids"] = ordered
    strict["train_ids"] = observed
    strict["val_ids"] = future
    strict["time_normalization_max_warp"] = max(
        int(metadata[name]["warp_id"]) for name in ordered
    )
    observed_percent = int(round(100 * args.time_split))
    future_percent = 100 - observed_percent
    strict["moregs_policy"] = (
        "official train_ids only; first {} percent observed; final {} percent future"
        .format(observed_percent, future_percent)
    )
    (output / "dataset.json").write_text(json.dumps(strict, indent=2) + "\n")

    manifest = {
        "scene": args.scene,
        "source": str(source),
        "output": str(output),
        "policy": strict["moregs_policy"],
        "time_split": args.time_split,
        "official_train_ids": len(official_train),
        "excluded_official_validation_ids": len(official_val),
        "observed": len(observed),
        "future": len(future),
        "training_evaluation_overlap": 0,
        "observed_ids": observed,
        "future_ids": future,
    }
    (output / "MOREGS_SPLIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({
        key: value for key, value in manifest.items()
        if key not in ("observed_ids", "future_ids")
    }, sort_keys=True))


if __name__ == "__main__":
    main()
