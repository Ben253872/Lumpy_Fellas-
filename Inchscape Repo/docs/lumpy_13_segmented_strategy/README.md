# Lumpy 13 - Chronology-Safe Segmented Strategy

## Purpose

Notebook 13 reuses the completed Notebook 11 model forecasts and Notebook 12 cold-start candidates. It does not retrain the forecasting models.

The notebook separates SKUs into three operational strategies:

- `point_forecast_priority`: recurring, recent, established/developing A or B SKUs.
- `cautious_point_forecast`: intermediate cases where a point forecast is retained with a wider range.
- `inventory_policy_priority`: rare, dormant, stale, or cold-start SKUs where inventory controls and uncertainty ranges matter more than a precise point estimate.

## Chronology Correction

Notebook 11 selected each SKU model using complete development folds. Because the folds advance every three months but each test lasts 18 months, those development outcomes overlap the final holdout.

Notebook 13 allows an outcome to influence model selection only when all three months in that forecast block were present by the decision cutoff. Repeated forecasts for the same SKU, model, and calendar block are deduplicated using the latest eligible origin.

At the July 2024 final cutoff:

- 148,416 rows had previously been treated as development data.
- 99,800 rows had outcomes after the cutoff and were removed.
- Four calendar blocks were genuinely known.
- Two later known blocks could be used for chronology-safe policy validation.

This is a small validation sample, so the result is more defensible but still uncertain.

## Development Decision

`frequency_champion` won the chronology-safe development comparison:

- 165 of 508 positive-demand SKUs below 50% WMAPE.
- 217 below 70% WMAPE.
- Median SKU block WMAPE: 84.2%.

The policy selects a shared model for each demand-frequency tier. It avoids making a highly variable per-SKU model decision from only a few known blocks.

## Final All-SKU Result

- Complete coverage: 690 of 690 SKUs.
- Positive-demand SKUs: 642.
- Below 50% WMAPE: 55.
- Below 70% WMAPE: 171.
- Below 100% WMAPE: 361.
- Median SKU block WMAPE: 90.3%.
- Median SKU block MASE: 0.489.
- Portfolio block WMAPE: 102.5%.
- Bias: -21.0%.

The point-forecast-priority tier is materially stronger than the other routes:

- 285 positive-demand SKUs.
- 43 below 50% WMAPE.
- 132 below 70% WMAPE.
- Median SKU block WMAPE: 71.8%.

Rare, occasional, dormant, and cold-start SKUs remain unsuitable for precise point forecasts. Their output includes an empirical 80% range and an inventory-policy label.

## Primary Files

- `lumpy_13_chronology_audit.csv`
- `lumpy_13_development_policy_summary.csv`
- `lumpy_13_old_vs_strict_comparison.csv`
- `lumpy_13_established_holdout_policy_comparison.csv`
- `lumpy_13_strict_cold_method_development.csv`
- `lumpy_13_final_all_690_summary.csv`
- `lumpy_13_final_all_690_per_sku.csv`
- `lumpy_13_final_segment_summary.csv`
- `lumpy_13_strategy_assignment_all_690.csv`

Large block-level forecast files are written under `results/lumpy_13_segmented_strategy/` and are ignored by Git.
