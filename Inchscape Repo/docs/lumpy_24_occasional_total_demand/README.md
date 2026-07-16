# Occasional Direct 18-Month Total Demand

Notebook 24 predicts each established occasional SKU's complete 18-month demand total using out-of-fold cross-sectional models, then distributes that total across the six official blocks using uniform or hierarchical seasonal timing.

Promotion is based on the stated business objective: a challenger must increase the number of validation SKUs below 70% or 50% WMAPE. Median WMAPE ranks candidates but cannot promote a model when no additional SKU crosses either target threshold.

## Result

The direct-total approach did not improve the threshold counts for occasional A, B, or C. All three Notebook 22 incumbents were therefore retained before final evaluation. The candidate diagnostics remain useful evidence that predicting an 18-month total does not, by itself, solve sparse demand placement at individual-SKU level.

Main outputs:

- `lumpy_24_locked_strategies.json`
- `lumpy_24_total_model_validation.csv`
- `lumpy_24_official_summary.csv`
- `lumpy_24_individual_sku_results.csv`
- `lumpy_24_diagnostic_total_ceiling.csv`
- `lumpy_24_actual_vs_forecast.png`
