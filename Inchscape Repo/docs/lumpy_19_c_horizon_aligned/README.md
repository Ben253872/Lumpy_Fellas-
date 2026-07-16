# Notebook 19: C Horizon-Aligned Calibration

> **Superseded by Notebook 20.** Although this experiment supplied 18 months of test data, it passed a 3-month model horizon and constructed only the first block. Its apparent improvement is withdrawn and must not be reported as an 18-month result.

Notebook 19 corrects the C calibration mismatch by selecting models on a historical 18-month validation horizon rather than isolated 3-month origins.

The validation forecast covers May 2023 through October 2024 using information through January 2023. The final forecast covers November 2024 through April 2026 using information through July 2024. Both use six 3-month blocks and a 3-month information gap.

C-only, pooled-transfer, hurdle, direct Tweedie, SBA, TSB, recent-demand, and robust ensemble candidates are compared on the historical horizon. The locked strategy is applied once to the final horizon. Final outcomes never switch the champion.

Main outputs:

- `lumpy_19_locked_strategy.json`
- `lumpy_19_horizon_validation_candidates.csv`
- `lumpy_19_horizon_validation_ensembles.csv`
- `lumpy_19_c_final_summary.csv`
- `lumpy_19_c_individual_sku_results.csv`
- `lumpy_19_diagnostic_oracle_ceiling.csv`
- `lumpy_19_c_actual_vs_forecast.png`
- `lumpy_19_c_sku_gallery.png`
