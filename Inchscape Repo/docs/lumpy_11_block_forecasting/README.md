# Lumpy 11 - Three-Month Block Forecasting

This experiment forecasts lumpy collision-parts demand directly in three-month blocks and keeps all SKUs in the output.

## Policies

- Required: 18-month horizon, 3-month operational gap
- Diagnostics: 9- and 18-month horizons with 1-, 3-, and 6-month gaps

Each policy uses up to six chronological folds. Earlier folds select a model for each SKU; the latest fold is untouched until final scoring.

## Selection priority

1. Most positive-demand SKUs below 50% aligned block WMAPE
2. Most below 70%
3. Most below 100%
4. Median SKU block WMAPE

Zero is included only as an audit baseline and cannot be selected. Eligible candidates include block hurdle XGBoost, direct Tweedie XGBoost, SBA, TSB, recent mean, and historical month-of-year demand.

## Primary outputs

- `lumpy_11_policy_holdout_summary.csv`
- `lumpy_11_policy_holdout_per_sku.csv`
- `lumpy_11_monthly_rolling_holdout_per_sku.csv`
- `lumpy_11_required_policy_model_summary.csv`
- `lumpy_11_development_selected_model_per_sku.csv`
- `lumpy_11_recipe_validation_scores.csv`

The aligned block metric is the primary experimental score. The monthly rolling three-month output preserves comparability with earlier notebooks.
