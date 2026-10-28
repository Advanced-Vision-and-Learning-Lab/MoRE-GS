<div align="center">

# MoRE-GS

### Motion-Regime Experts for Future Gaussian Splatting

**Shima Salehi · Joshua Peeples**  
Texas A&M University · WACV 2027

[Project page](https://advanced-vision-and-learning-lab.github.io/MoRE-GS-website/) ·
[Paper](https://advanced-vision-and-learning-lab.github.io/MoRE-GS-website/assets/MoRE-GS.pdf) ·
[Supplementary](https://advanced-vision-and-learning-lab.github.io/MoRE-GS-website/assets/MoRE-GS-supplementary.pdf)

</div>

MoRE-GS is a causal, render-aware framework for forecasting a withheld future
suffix of a dynamic 3D Gaussian scene. It freezes the Gaussian representation
reconstructed from the observed temporal prefix and routes among four
complementary control-keypoint motion experts: persistence, a coefficient-space
Transformer, the GaussianPrediction graph convolutional network, and a
rigidity-aware structured Transformer.

The routing table varies over soft motion groups and forecast-horizon bins. It
is calibrated through the frozen renderer on a chronological tail contained
entirely inside the observed prefix. Future images are opened only for final
evaluation after the experts and router have been frozen and cold-reloaded.

## Paper results

The selected 16-group model reports the following scene-unweighted means. The
cross-method rows are system-level comparisons using the protocols described in
the paper; matched causal controls are reported separately in the ablation.

| Dataset | Protocol | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| D-NeRF (8 scenes) | chronological 80/20 | **27.47** | **0.948** | **0.045** |
| HyperNeRF (4 scenes) | chronological 90/10 | **19.22** | 0.584 | **0.349** |

On D-NeRF, MoRE-GS improves mean PSNR by 0.82 dB over the local ODE-GS
reproduction. On HyperNeRF, it improves mean PSNR by 0.39 dB and reduces LPIPS
by 0.050, while ODE-GS retains higher mean SSIM. Full per-scene values and
comparison qualifications are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Repository layout

```text
MoRE-GS/
├── configs/                 selected D-NeRF and HyperNeRF settings
├── docs/                    protocol, data layout, results, provenance
├── moregs/                  experts, grouping, routing, and evaluation
├── patches/                 pinned GaussianPrediction compatibility patch
├── scripts/                 setup, preparation, run, and validation tools
├── CITATION.cff
├── environment.yml
└── pyproject.toml
```

Datasets, canonical reconstructions, checkpoints, rendered images, caches, and
machine-specific experiment directories are intentionally excluded.

## Selected configuration

- eight hard groups internal to the structured expert;
- sixteen soft groups used by the group–horizon router;
- four forecast-horizon bins;
- four experts, including persistence;
- a 256-parameter routing table (`16 × 4 × 4`).

Changing the soft-router group count does not retrain or resize the expert bank.
The hard and soft groupings serve different roles and need not have the same
cardinality.

## Installation

Create the pinned environment, install the GaussianPrediction backbone at the
recorded revision, and verify CUDA dependencies:

```bash
conda env create -f environment.yml
conda activate moregs
bash scripts/setup_gaussianprediction.sh
python scripts/check_environment.py
```

The environment targets Python 3.7, PyTorch 1.13.1, and CUDA 11.7. The setup
script prints the upstream CUDA-extension installation step because those
extensions must match the active CUDA toolchain.

## Data

Download D-NeRF or HyperNeRF from their official sources and keep the datasets
outside this repository. See [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md).

HyperNeRF scenes must first be converted into an official-training-ID-only
chronological view. The preparation script excludes the released validation
camera IDs and records an auditable split manifest:

```bash
python scripts/prepare_hypernerf.py \
  --source /path/to/original/cut-lemon \
  --output /path/to/prepared/cut-lemon \
  --scene cut-lemon \
  --time-split 0.9
```

## Run one scene

D-NeRF:

```bash
PYTHON_BIN=/path/to/python \
bash scripts/run_dnerf_scene.sh \
  /path/to/dnerf/lego \
  output/lego \
  lego \
  17
```

HyperNeRF:

```bash
PYTHON_BIN=/path/to/python \
bash scripts/run_hypernerf_scene.sh \
  /path/to/prepared/cut-lemon \
  output/cut-lemon \
  cut-lemon \
  42
```

Each runner reconstructs the observed prefix, trains full-prefix and fit-only
experts, calibrates the observed-only router, refits the selected 16-group
router, cold-reloads the final checkpoint, and writes a JSON summary in the
requested output directory.

For exact reproduction from an archived canonical checkpoint:

```bash
MOREGS_CANONICAL_CHECKPOINT=/path/to/chkpnt60000.pth \
PYTHON_BIN=/path/to/python \
bash scripts/run_dnerf_scene.sh /path/to/dnerf/lego output/lego lego 17
```

A newly trained canonical reconstruction is a fresh replication: different
keypoint identities and bindings can change the forecasting problem even when
the reconstruction quality is similar. Every run therefore records checkpoint,
dataset, source, and runtime hashes in `moregs_provenance.json`.

## Release validation

Before publishing or archiving the source release, run:

```bash
python scripts/validate_release.py
```

The audit rejects datasets, checkpoints, large generated artifacts, malformed
Python sources, and changes to the selected group and split contracts.

## Citation

```bibtex
@inproceedings{salehi2027moregs,
  title     = {{MoRE-GS}: Motion-Regime Experts for Future Gaussian Splatting},
  author    = {Salehi, Shima and Peeples, Joshua},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2027}
}
```

## License and acknowledgments

MoRE-GS builds on
[GaussianPrediction](https://github.com/BoMingZhao/GaussianPrediction) at the
revision recorded in `patches/GAUSSIANPREDICTION_COMMIT`. Upstream repositories
and their submodules retain their own licenses. This source release uses the
included research-only Gaussian-Splatting license; review all upstream terms
before redistribution or commercial use.

