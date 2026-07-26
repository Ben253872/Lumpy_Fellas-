# Lumpy 10 - Internal Hybrid Tuning

This experiment removes external features and tests a focused Croston/XGBoost hybrid for individual lumpy SKUs.

## Primary result

Use `lumpy_10_untouched_holdout_summary.csv` as the decision table. Earlier folds select each SKU's model; the latest fold is held out and scored once.

## Tuning dimensions

- XGBoost depth 1, 2, and 3
- shorter and longer boosting schedules
- unweighted, square-root, and fully balanced occurrence classification
- log-positive-size and Tweedie-positive-size regression
- expected value, soft gating, and hard gating
- calibration multipliers
- SBA, TSB, and recent-mean blends
- a bias-aware validation objective that penalizes underforecast more heavily

## Main review files

- `lumpy_10_untouched_holdout_summary.csv`
- `lumpy_10_untouched_holdout_per_sku.csv`
- `lumpy_10_development_selected_model_per_sku.csv`
- `lumpy_10_development_model_summary.csv`
- `lumpy_10_recipe_validation_scores.csv`
- `lumpy_10_all_fold_sku_model_scores.csv`

The all-fold best-model file is descriptive only. It must not replace the untouched holdout result when reporting expected performance.
