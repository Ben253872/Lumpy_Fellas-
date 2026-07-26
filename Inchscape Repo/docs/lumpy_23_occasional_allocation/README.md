# Occasional Aggregate And Empirical-Bayes Allocation

Notebook 23 forecasts occasional demand at cohort, family, and subfamily level before allocating it to individual SKUs with empirical-Bayes shrinkage and optional occurrence budgeting.

Configurations are selected on the strict historical six-block horizon and compared with Notebook 22 before final outcomes are evaluated.

Main outputs:

- `lumpy_23_locked_strategies.json`
- `lumpy_23_allocation_validation.csv`
- `lumpy_23_official_summary.csv`
- `lumpy_23_individual_sku_results.csv`
- `lumpy_23_diagnostic_allocation_ceiling.csv`
- `lumpy_23_actual_vs_forecast.png`
