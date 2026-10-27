# Source provenance

The release modules are clarity-preserving renames of the frozen research
implementation. Imports were changed from numbered research filenames to the
`moregs` package; the algorithms and checkpoint schemas were retained. The
SHA-256 values below identify the source files before packaging.

| Release module | Frozen research source | SHA-256 |
|---|---|---|
| `moregs/structured_expert.py` | `phase77_masked_group_transformer.py` | `c041588005452e056da26dcc29364b8390e6d478267b06cc0654fe0bafb1e8b2` |
| `moregs/forecasting.py` | `phase80_direct_multihorizon.py` | `3d0328f96bb993e02f3562776948d1f2a0ed041cf91ef4693d2281a85c0853dc` |
| `moregs/graph_expert.py` | `phase81_support_motion_router.py` | `0829d4ac0d9f3fda02d7a1027893cb1273c681e4cb23e15c2af9f72647de0c44` |
| `moregs/kinematic_residual.py` | `phase82_shared_kinematic_residual.py` | `6bdb956363db37c8183c0e9a92b725158c7fbc518c508077feb9c4045373f041` |
| `moregs/grouping.py` | `phase86_regime_aware_forecaster.py` | `f1772d48e3e6f80fcf5cd5ce82071c8211518151a1faacb52aaeb3df4a2527f5` |
| `moregs/rendered_risk.py` | `phase90_rendered_risk_moe.py` | `c589c34325bd8c5b4b75b7038a1156da7449783bd8a84b822498da04d8aca7ad` |
| `moregs/router.py` | `phase91_joint_rendered_router.py` | `2b4e0b1b37d6c2146dc5b363844e173434a4ac1293a1ca2d0b007ed555b34862` |
| `moregs/coefficient_expert.py` | `phase92_frequency_basis_control.py` | `151dcab1a2f8506d2379508b67b03a8680264e1ba1693e47ee8137c8c440152b` |
| `moregs/train.py` | `phase93_crossfit_winner_bank.py` | `99652a377623907f7c72acc0dc18e216714f76d12608a5d663b4254479ff8107` |
| `moregs/ablation.py` | `phase167_phase93_causal_ablation.py` | `cf60808c604bbd14d80ea48b915e18e16fd8d1c520b81cc0c76608226c452d48` |
| `scripts/prepare_hypernerf.py` | `prepare_phase224_gp_strict_hyper.py` | `0abe42fc9c877a1ff83ac48e2bc46c4fa315c2d9fe84cbd5c2c14dbbf6dc4269` |

Legacy phase identifiers remain in serialized checkpoint metadata so existing
audited checkpoints can be read without conversion. They are provenance tags,
not additional methods included in this release.

Public command-line flags use descriptive names. The legacy spellings remain
accepted as aliases:

| Module | Flag | Legacy alias |
|---|---|---|
| `moregs/ablation.py` | `--base_checkpoint`, `--base_summary` | `--phase93_checkpoint`, `--phase93_summary` |
| `moregs/router.py` | `--expert_checkpoint` | `--phase90_checkpoint` |

