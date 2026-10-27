#!/usr/bin/env python3
"""Fail early when the MoRE-GS CUDA or backbone environment is incomplete."""

from __future__ import print_function

import importlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKBONE = Path(
    os.environ.get(
        "GAUSSIAN_PREDICTION_ROOT",
        str(ROOT / "external" / "gaussian_prediction"),
    )
)


def main():
    report = {
        "python": sys.version.split()[0],
        "backbone": str(BACKBONE),
        "checks": {},
    }
    failures = []
    if not (BACKBONE / "train.py").is_file():
        failures.append("GaussianPrediction is not installed at {}".format(BACKBONE))
    else:
        sys.path.insert(0, str(BACKBONE))

    try:
        import torch
        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["checks"]["cuda_available"] = bool(torch.cuda.is_available())
        if not torch.cuda.is_available():
            failures.append("PyTorch cannot access CUDA device 0")
        else:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["compute_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as error:
        failures.append("PyTorch check failed: {}".format(error))

    for module in ("tinycudann", "diff_gaussian_rasterization", "simple_knn"):
        try:
            importlib.import_module(module)
            report["checks"][module] = True
        except Exception as error:
            report["checks"][module] = False
            failures.append("{} import failed: {}".format(module, error))

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("\n".join(failures))
    print("PASS: MoRE-GS CUDA and GaussianPrediction environment is ready")


if __name__ == "__main__":
    main()

