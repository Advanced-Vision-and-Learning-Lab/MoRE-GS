# Paper results

These are the values reported in the WACV 2027 paper. Views are averaged within
each scene and scenes are then weighted equally.

## D-NeRF: chronological 80/20 forecasting

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| GaussianPrediction | 25.34 | 0.936 | 0.053 |
| MoGaF | 25.87 | 0.936 | 0.049 |
| ODE-GS | 26.64 | 0.942 | 0.057 |
| **MoRE-GS (G=16)** | **27.47** | **0.948** | **0.045** |

GaussianPrediction and ODE-GS are local reproductions with the same temporal
split and evaluation identities but method-specific representations and
training budgets. MoGaF values are taken from its paper. These rows are
system-level comparisons, not causal ablations.

## HyperNeRF: chronological 90/10 forecasting

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| ODE-GS | 18.83 | **0.596** | 0.399 |
| **MoRE-GS (G=16)** | **19.22** | 0.584 | **0.349** |

This four-scene real-capture study provides preliminary cross-dataset evidence.
MoRE-GS improves PSNR and LPIPS, while ODE-GS retains higher mean SSIM.

## Matched causal ablation

| Variant | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Persistence | 24.70 | 0.936 | 0.053 |
| Coefficient Transformer | 26.51 | 0.941 | 0.051 |
| GCN | 25.22 | 0.934 | 0.052 |
| Structured Transformer | 25.95 | 0.932 | 0.061 |
| Observed-only fixed expert | 26.61 | 0.942 | 0.051 |
| Uniform mixture | 26.66 | 0.944 | 0.045 |
| Without persistence gate | 26.85 | 0.944 | 0.048 |
| Shuffled groups | 26.92 | 0.943 | 0.050 |
| **MoRE-GS (G=16)** | **27.47** | **0.948** | **0.045** |

The ablation holds the frozen reconstruction, chronological split, forecasting
initialization, optimization budget, and evaluation identities fixed. Consult
the paper and supplement for the complete routing and group-count tables.

