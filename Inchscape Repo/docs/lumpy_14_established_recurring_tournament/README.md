# Lumpy 14 - Established Recurring A/B Tournament

## Scope

Notebook 14 freezes the July 2024 established recurring A/B population and excludes cold-start, new, developing, dormant, occasional, rare, and ABC-C SKUs.

- Recurring A: 215 SKUs, 213 with positive holdout demand.
- Recurring B: 110 SKUs, all with positive holdout demand.
- Combined: 325 SKUs, 323 with positive holdout demand.

The experiment uses a three-month operational gap throughout. Seven complete three-month rolling origins are available before the final cutoff. The first five tune model variants and the last two select one challenger per cohort.

## Models

The eight Notebook 11 families are tested with cutoff-safe calibration factors from 0.75 to 2.00:

- SBA Croston
- TSB
- Recent six-month mean
- Historical seasonal mean
- Three hurdle XGBoost variants
- Direct XGBoost Tweedie regression

The business ranking is below 70% WMAPE first, below 50% second, then below 100%, median WMAPE, portfolio WMAPE, and absolute bias.

## Locked Challengers

- Recurring A: Direct XGBoost Tweedie, scale 0.75.
- Recurring B: Hurdle XGBoost depth 2 square-root weighting, expected mode, scale 0.75.

On the required 18-month holdout, the A challenger improved the incumbent while the B challenger regressed. The champion-retention rule therefore accepts the A challenger and retains the Notebook 13 B forecast.

## Recommended Retained Result

| Cohort | Positive SKUs | Below 50% | Below 70% | Below 100% | Median WMAPE | Portfolio WMAPE | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recurring A | 213 | 36 | 111 | 178 | 68.6% | 76.7% | -37.0% |
| Recurring B | 110 | 14 | 40 | 80 | 78.3% | 82.1% | -41.0% |
| Combined | 323 | 50 | 151 | 258 | 71.4% | 77.8% | -37.8% |

Compared with Notebook 13 on the identical 325 SKUs, the retained result adds two SKUs below 50% and four below 70%, reduces median WMAPE from 72.1% to 71.4%, and improves bias from -42.0% to -37.8%. Four fewer SKUs are below 100%, so the change is a targeted improvement in the primary business thresholds rather than a universal improvement.

Recurring A now has a majority of positive-demand SKUs below 70%. Recurring B does not.

## Interpretation

The A result supports cohort-specific regression. The B result shows that two short validation blocks are not enough to guarantee that a calibrated challenger will transfer to the full 18-month horizon. B should remain on its incumbent model until a challenger wins a broader rolling validation study.

The final 18-month benchmark has now been inspected repeatedly across notebooks. Treat it as the current development benchmark, not a permanently untouched estimate of future performance.

## Primary Files

- `lumpy_14_rolling_origin_inventory.csv`
- `lumpy_14_tuned_variant_per_model.csv`
- `lumpy_14_validation_model_summary.csv`
- `lumpy_14_selected_champion_per_cohort.csv`
- `lumpy_14_champion_retention_decisions.csv`
- `lumpy_14_recommended_retained_summary.csv`
- `lumpy_14_recommended_retained_per_sku.csv`
- `lumpy_14_notebook13_comparison_same_skus.csv`

Large forecast and checkpoint files are written beneath `results/lumpy_14_established_recurring_tournament/` and are ignored by Git.
