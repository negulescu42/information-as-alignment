# Cell 25 — Truth-Maintenance Under Local State Evolution

## Status

- **Status:** `diagnostic_clean`
- **Paper use:** `diagnostic_support_only_until_controls_verified`
- **Known caveat:** Ordinary/outside control metrics were not present in this cell schema. Use this result as a truth-maintenance decomposition, not a standalone locality/control claim.

## Framing

This diagnostic separates catastrophic forgetting from controlled truth-maintenance. The question is not simply whether the system forgets. When local semantic state changes, contradicted or obsolete beliefs should weaken or retire. The key question is whether stable constraints and ordinary controls remain intact.

## Phase A aggregate trajectory

| Boundary | Accuracy | Δ from previous |
|---|---:|---:|
| After A / Onboarding | 0.956 | — |
| After B / Initiative | 0.948 | -0.008 |
| After C / Reorg | 0.867 | -0.081 |
| After D / Turnover | 0.850 | -0.017 |
| Total A→D | — | -0.106 |

## Reorg selectivity

- Affected mean B→C delta: `-0.1725`
- Unaffected mean B→C delta: `-0.01999999999999998`
- Selectivity, affected minus unaffected: `-0.1525`
- Interpretation: Reorg impact is selective: affected categories degrade more than unaffected categories. This supports controlled truth-maintenance: contradicted local state is weakened or retired while stable categories are comparatively preserved.

## Stable-category retention

- Stable-category mean BT(A→D): `-0.05666666666666664`
- Stable-category retained rate: `0.3333333333333333`

## Phase B survival

- Phase B aggregate first-to-final BT: `-0.0025000000000000577`
- Phase B final aggregate: `0.98`

## Ordinary controls

- Available in this cell schema: `False`
- Note: No ordinary/outside control metrics found in all_evals schema for this cell.

## Dissolutions

- Available: `True`
- Source: `mmlu_ibf_out/canonical_training_results.json`
- Note: Dissolution-like fields found in training-results JSON.

## Interpretation

This diagnostic separates catastrophic forgetting from controlled truth-maintenance. A degradation concentrated at revision boundaries is not automatically a failure; it may indicate that contradicted or obsolete local state is being retired. The key signal is whether unaffected categories and ordinary controls remain stable.
