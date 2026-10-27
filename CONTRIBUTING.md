# Contributing

Bug reports and reproducibility reports are welcome. Please include:

- the dataset and scene;
- the exact chronological split and evaluation identities;
- the canonical checkpoint hash;
- the forecast seed and optimization budget;
- `moregs_provenance.json` and the cold-reload status;
- the GaussianPrediction revision and local diff hash.

Do not submit datasets, checkpoints, rendered outputs, credentials, or
machine-specific absolute paths. New causal comparisons must hold training
views, evaluation views, resolution, seed, reconstruction, and optimization
budget fixed. HyperNeRF experiments must exclude official validation IDs from
training, calibration, and forecasting.

Run `python scripts/validate_release.py` before opening a pull request.

