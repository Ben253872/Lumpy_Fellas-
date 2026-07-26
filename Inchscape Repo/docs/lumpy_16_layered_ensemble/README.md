# Notebook 16: Layered Mixture Of Experts

Notebook 16 combines, rather than refits, the strongest cached candidates from Notebook 15. It is intentionally a fast second-stage experiment.

## Layers

- Expert layer: diverse classical, hurdle XGBoost, and direct Tweedie XGBoost forecasts across history windows.
- Routing agent: hard and probability-weighted model routing.
- Stacking agent: robust averages, non-negative linear/ElasticNet, random forest, histogram Poisson, and Tweedie XGBoost stacks.
- Correction agent: a conservative residual model on the positive linear stack.
- Confidence agent: estimates each SKU's probability of finishing below 70% WMAPE without excluding it.
- Overfit agent: rejects a strategy if its below-70 coverage falls by more than 15 percentage points from development to validation.

## Chronology

Origins 1-5 are used in leave-one-origin-out development. Origins 6-7 remain untouched until strategy validation. The locked strategy is then fitted on all seven known origins and challenged against the existing Notebook 14 champion on the required 18-month, 3-month-gap backtest.

## Selection

Selection remains aligned with the individual-SKU goal:

1. number of positive-demand SKUs below 70% WMAPE;
2. number below 50%;
3. number below 100%;
4. median WMAPE and portfolio diagnostics.

The incumbent is retained automatically unless the layered challenger wins under this ordering. Confidence bands never remove SKUs from the reported denominator.

## Main Outputs

- `lumpy_16_final_champion_comparison.csv`
- `lumpy_16_sku_coverage.csv`
- `lumpy_16_individual_sku_results.csv`
- `lumpy_16_locked_strategies.csv`
- `lumpy_16_overfit_agent_audit.csv`
- `lumpy_16_agent_decision_record.csv`
- `lumpy_16_actual_vs_forecast.png`
- `lumpy_16_sku_actual_vs_forecast_gallery.png`

Large row-level forecast files remain under `results/` and are ignored by Git.
