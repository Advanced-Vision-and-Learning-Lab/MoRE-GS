# Causal forecasting protocol

## Temporal partitions

For D-NeRF, the first 80% of the time-ordered training sequence is observed
and the final 20% is the sealed future suffix. For HyperNeRF, only official
training IDs are retained; after chronological sorting, the first 90% is
observed and the last 10% is sealed. Official validation IDs are excluded from
both training and evaluation.

The observed prefix is split again for routing. Temporary experts use its
first 75%, and their predictions on the remaining 25% calibrate the router
through the frozen renderer. This 75/25 split is internal to the observed
prefix; it is not the dataset's observed/future split.

## Fit--calibrate--refit sequence

1. Reconstruct the dynamic Gaussian scene using only the observed prefix.
2. Freeze Gaussian geometry, appearance, opacity, deformation, renderer,
   control keypoints, and keypoint-to-Gaussian bindings.
3. Fit temporary experts on the first 75% of the observed prefix.
4. Forecast its disjoint observed tail and calibrate router weights in image
   space.
5. Discard temporary experts.
6. Independently fit deployment experts on the complete observed prefix.
7. Freeze and save the router, then cold-reload it.
8. Open the future suffix only to render final predictions and compute metrics.

## Final architecture

- persistence: copies the final observed keypoint pose;
- coefficient Transformer: seeded random orthonormal temporal basis followed
  by a one-layer Transformer and gated residual decoder;
- graph expert: the released GaussianPrediction GCN with autoregressive
  rollout;
- structured Transformer: eight hard motion groups, Kabsch-based rigidity
  classification, and rigid/non-rigid physical losses;
- router: sixteen observed-only soft groups, four horizon bins, four experts,
  and a separately parameterized persistence gate.

The hard groups belong only to the structured expert. The sixteen soft groups
belong only to the router. Their counts therefore need not match.

## Prohibited information

Future images, future motion states, future metrics, HyperNeRF validation
cameras, and future-dependent checkpoint or hyperparameter choices must not
influence reconstruction, grouping, expert fitting, routing, early stopping,
or model selection.

