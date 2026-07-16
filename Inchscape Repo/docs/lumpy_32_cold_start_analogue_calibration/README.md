# Lumpy 32 - Cold-Start Analogue Calibration

This experiment targets the 40 genuine cold-start SKUs with no demand history at the official information cutoff.

It reuses Notebook 12's chronology-safe pseudo-cold analogue forecasts and tests calibrated forecast levels plus sparse block-placement rules. The candidate is selected using historical pseudo-cold cases and locked before the genuine cold-start horizon is evaluated.

Primary outputs:

- `lumpy_32_validation_candidates.csv`
- `lumpy_32_locked_strategy.csv`
- `lumpy_32_official_comparison.csv`
- `lumpy_32_individual_sku_results.csv`
- `lumpy_32_actual_vs_forecast.png`

## Result

Historical pseudo-cold validation locked the subfamily mean, retaining all six
blocks and applying a 0.30 scale. On the 40 genuine cold starts, this reduced
median WMAPE from 117.83% to 92.65% and portfolio WMAPE from 137.11% to 93.35%.
The number below 100% increased from 12 to 23.

It did not improve the primary threshold: below-70 performance fell from 8 to
3 SKUs and below-50 fell from 4 to 0. The challenger is therefore not promoted
over Notebook 12. The split result indicates domain shift between simulated
cold starts and genuine launch SKUs. The next experiment should train on
launch-like historical cohorts and estimate occurrence separately from demand
size, while preserving the same official horizon and metric contract.
