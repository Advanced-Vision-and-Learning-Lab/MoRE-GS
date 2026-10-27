#!/usr/bin/env python3
"""Validate that the release is source-only and internally consistent."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {"data", "datasets", "output", "outputs", "checkpoints", "runs"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".pyc"}
PRIVATE_MARKERS = (
    "/".join(("", "home", "grads", "")),
    "h" + "f_",
    "api" + "_key=",
    "access" + "_token=",
)


def main() -> None:
    violations = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts:
            continue
        if "__pycache__" in relative.parts:
            violations.append(f"generated Python cache: {relative}")
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            # Empty, documented output paths are not created in the release.
            violations.append(f"forbidden directory content: {relative}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden artifact: {relative}")
        if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
            violations.append(f"unexpected file larger than 10 MiB: {relative}")
        if path.is_file() and path.resolve() != Path(__file__).resolve() and path.suffix.lower() in {".py", ".sh", ".md", ".json", ".yml", ".yaml", ".toml"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for marker in PRIVATE_MARKERS:
                if marker in text:
                    violations.append(f"private path or credential marker {marker!r}: {relative}")

    for source in (ROOT / "moregs").glob("*.py"):
        ast.parse(source.read_text(), filename=str(source))

    dnerf = json.loads((ROOT / "configs/dnerf.json").read_text())
    hyper = json.loads((ROOT / "configs/hypernerf.json").read_text())
    for name, config in (("D-NeRF", dnerf), ("HyperNeRF", hyper)):
        assert config["experts"]["structured_hard_groups"] == 8, name
        assert config["router"]["soft_groups"] == 16, name
        assert config["router"]["horizon_bins"] == 4, name
        assert config["protocol"]["fit_fraction_within_observed"] == 0.75, name

    if violations:
        raise SystemExit("\n".join(violations))
    print("PASS: source-only MoRE-GS release; Python syntax and selected configs verified")


if __name__ == "__main__":
    main()
