# Established Rare Post-Event Quantity Calibration

Notebook 30 keeps Notebook 29's group-prior event timing and tests quantity corrections only after an event block has been selected.

The calibration library includes validation-derived pooled scaling, historical positive-size caps, and shrinkage toward each SKU's median, mean and upper-quartile positive size. Zero forecasts remain zero. Notebook 29 remains the incumbent unless below-70 or below-50 validation coverage improves.

## Result

No quantity transform improved the validation below-70 or below-50 counts for rare A, B or C. All three Notebook 29 incumbents were retained, so the final result is unchanged: 0 of 33 positive SKUs below 50%, 4 below 70%, 11 below 100%, median SKU WMAPE 110.62%, and portfolio WMAPE 146.61%.

This is a useful negative result. Once Notebook 29's event blocks are fixed, scaling, capping and historical-size shrinkage do not move more SKUs across the target thresholds. Further rare-SKU work should return to occurrence information or probabilistic inventory decisions rather than add more quantity calibration.

Main outputs:

- `lumpy_30_locked_strategies.json`
- `lumpy_30_validation_candidates.csv`
- `lumpy_30_official_summary.csv`
- `lumpy_30_individual_sku_results.csv`
- `lumpy_30_event_diagnostics.csv`
- `lumpy_30_actual_vs_forecast.png`
