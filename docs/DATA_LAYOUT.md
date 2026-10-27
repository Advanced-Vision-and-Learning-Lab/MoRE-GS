# Data layout

No dataset is distributed with MoRE-GS.

## D-NeRF

Pass a scene directory containing `transforms_train.json` and its referenced
images directly to `scripts/run_dnerf_scene.sh`.

```text
/datasets/dnerf/lego/
  transforms_train.json
  transforms_test.json
  train/
  test/
```

## HyperNeRF

The original scene must contain official `train_ids` and `val_ids` plus the
camera, RGB, metadata, scene, and point-cloud files expected by
GaussianPrediction. Create a non-destructive prepared view:

```text
/datasets/hypernerf/cut-lemon-original/
/datasets/hypernerf/cut-lemon-moregs-90-10/
```

The preparation script uses symbolic links for large assets, writes a new
`dataset.json`, and records `MOREGS_SPLIT_MANIFEST.json`. It refuses official
train/validation overlap and refuses to overwrite unexpected paths.

