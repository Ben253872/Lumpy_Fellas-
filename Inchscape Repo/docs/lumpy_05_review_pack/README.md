# Lumpy 05 Review Pack

Run `notebooks/lumpy_05_future_forecasting.ipynb` to refresh this folder.

This folder is the small, shareable output pack. The large raw notebook outputs are written to `results/lumpy_05_future_forecasting/` and ignored by Git.

Expected files:

- `lumpy_05_best_models_by_scope.csv`: the winning model for the full-history lumpy scope and the short flag-only lumpy scope.
- `lumpy_05_model_comparison_all_models.csv`: all tested models ranked by the 3-month rolling monthly-total WMAPE.
- `lumpy_05_future_monthly_by_scope.csv`: 18-month future forecast totals by month and scope.
- `lumpy_05_future_monthly_side_by_side.csv`: full-vs-short monthly future forecast totals side by side.
- `lumpy_05_full_history_all_skus_wmape.csv`: every full-history lumpy SKU with backtest actuals, forecast totals, SKU WMAPE, bias, positive-month counts, and 18-month future forecast totals.
- `lumpy_05_short_history_all_skus_wmape.csv`: every short flag-only lumpy SKU with the same SKU-level review fields.
- `lumpy_05_sku_wmape_file_summary.csv`: row counts and valid-WMAPE counts for the two all-SKU review files.
- `lumpy_05_top_50_common_skus_by_wmape.csv`: the 50 common SKUs with the lowest average SKU WMAPE across both scopes.
- `lumpy_05_top_50_sku_backtest_and_future_monthly.csv`: monthly backtest rows plus the 18-month future forecast rows for those 50 SKUs, for SKU lookup/review.
- `lumpy_05_final_comparison_summary.csv`: final audit summary for forecast dates, horizon length, selected model, and forecast totals.
