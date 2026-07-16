# Notebook 18: Established Recurring C Transfer Tournament

> **Superseded by Notebook 20.** This experiment passed a 3-month model horizon and therefore evaluated only the first block of the supplied 18-month test period. Its final C statistics must not be used as 18-month results.

This notebook freezes A/B after Notebook 17 and moves to the like-for-like established recurring ABC-units C population.

There are 35 C SKUs at the frozen classification cutoff. Because this is a small cohort, shared XGBoost structures are tested using both pooled A+B+C training and C-only training. Tuned classical models, robust ensembles, and personalised online expert memory are included.

Origins 1-5 tune recipes and memory. Origins 6-7 lock the strategy. The final evaluation remains the required 18-month horizon with a 3-month information gap and 3-month block WMAPE.

Main outputs:

- `lumpy_18_c_final_summary.csv`
- `lumpy_18_c_individual_sku_results.csv`
- `lumpy_18_locked_c_strategy.csv`
- `lumpy_18_strategy_validation.csv`
- `lumpy_18_validation_single_models.csv`
- `lumpy_18_c_actual_vs_forecast.png`
- `lumpy_18_c_sku_gallery.png`
