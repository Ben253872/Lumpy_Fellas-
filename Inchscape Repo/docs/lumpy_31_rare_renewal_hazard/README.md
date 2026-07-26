# Established Rare Renewal Hazard

Notebook 31 is the final planned point-forecast experiment for the 56 established rare SKUs. It models time to the next demand event as a pooled renewal process.

Monthly age-dependent hazards are estimated globally and at family/subfamily level, with SKU shrinkage. A state distribution is propagated through the three-month information gap before forecasting six three-month event probabilities. Expected, top-one, top-two and horizon-gated forecasts compete with Notebook 29.

Main outputs:

- `lumpy_31_locked_strategies.json`
- `lumpy_31_validation_candidates.csv`
- `lumpy_31_official_summary.csv`
- `lumpy_31_individual_sku_results.csv`
- `lumpy_31_event_diagnostics.csv`
- `lumpy_31_probability_calibration.csv`
- `lumpy_31_actual_vs_forecast.png`

## Result

The historically locked renewal strategy produced 1/33 positive SKUs below 50%,
5/33 below 70%, and 7/33 below 100% on the official horizon. Median WMAPE was
162.50% and portfolio WMAPE was 205.01%.

Compared with Notebook 29, this is a small improvement below 50% (0 to 1) and
below 70% (4 to 5), but a material deterioration below 100% (11 to 7), median
WMAPE (110.62% to 162.50%), event precision, and portfolio WMAPE. The experiment
therefore does not justify another point-forecast layer for established rare
SKUs. Notebook 29 remains the more balanced operational benchmark; renewal
probabilities may still be useful later for stock-policy or confidence-range
decisions rather than as the point forecast.
