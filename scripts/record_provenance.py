#!/usr/bin/env python3
"""Create or verify the immutable provenance manifest for one MoRE-GS run."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root, suffixes):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes):
        relative = str(path.relative_to(root)).encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git_output(root, arguments):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments), stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return ("unavailable: %s" % error).encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.model / ("chkpnt%d.pth" % args.iteration)
    if not checkpoint.is_file():
        raise RuntimeError("missing canonical checkpoint: %s" % checkpoint)
    release_root = Path(__file__).resolve().parents[1]
    dataset_files = {}
    for name in ("transforms_train.json", "transforms_test.json", "dataset.json", "metadata.json"):
        path = args.source / name
        if path.is_file():
            dataset_files[name] = sha256_file(path)
    cfg_path = args.model / "cfg_args"
    backbone_diff = git_output(args.backbone, ["diff", "--binary"])
    manifest = {
        "schema": "moregs_run_provenance_v1",
        "scene": args.scene,
        "forecast_seed": args.seed,
        "canonical_iteration": args.iteration,
        "canonical_checkpoint": {
            "resolved_path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "dataset_source": str(args.source.resolve()),
        "dataset_file_sha256": dataset_files,
        "cfg_args_sha256": sha256_file(cfg_path) if cfg_path.is_file() else None,
        "gaussianprediction": {
            "root": str(args.backbone.resolve()),
            "git_head": git_output(args.backbone, ["rev-parse", "HEAD"]).decode("utf-8", "replace").strip(),
            "git_diff_sha256": hashlib.sha256(backbone_diff).hexdigest(),
            "git_diff_bytes": len(backbone_diff),
        },
        "moregs_source_sha256": sha256_tree(
            release_root, set((".py", ".sh", ".json", ".yml", ".yaml"))
        ),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("provenance mismatch with existing manifest: %s" % args.output)
        print("provenance verified: %s" % args.output)
        return
    args.output.write_text(serialized, encoding="utf-8")
    print("provenance created: %s" % args.output)


if __name__ == "__main__":
    main()
