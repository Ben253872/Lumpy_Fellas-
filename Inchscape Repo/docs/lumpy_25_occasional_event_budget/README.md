# Established Occasional Calibrated Hurdle And Event Budget

Notebook 25 extends the four-stage intermittent-demand architecture for the 116 established occasional SKUs.

It tests cross-fitted sigmoid and isotonic occurrence calibration, ML and historically shrunk positive-size estimates, top-two/top-three/adaptive event budgets, probability normalisation, and cautious blends with the Notebook 22 incumbent.

Every candidate uses the official 18-month horizon, three-month information gap, and six non-overlapping three-month blocks. Strategies are selected on the historical horizon and locked before final evaluation. Promotion requires more validation SKUs below 70% or 50% WMAPE.

## Result

Historical validation promoted a calibrated/event-budget challenger for each A/B/C cohort. On the untouched final horizon, the combined occasional result improved from 3 to 4 positive SKUs below 70% WMAPE and from 13 to 21 below 100%. Median SKU WMAPE improved from 200.82% to 154.31%, portfolio WMAPE from 192.13% to 147.45%, and bias from +63.87% to -19.57%.

The improvement is meaningful but well short of the coverage target. Occasional A improved from 0 to 2 SKUs below 70%; B moved from 2 to 1; C remained at 1. The diagnostic per-SKU oracle is deliberately ineligible, but its much stronger ceiling shows that these SKUs require a learnable routing rule rather than one cohort-wide recipe.

Main outputs:

- `lumpy_25_locked_strategies.json`
- `lumpy_25_validation_candidates.csv`
- `lumpy_25_stage_attribution.csv`
- `lumpy_25_official_summary.csv`
- `lumpy_25_individual_sku_results.csv`
- `lumpy_25_diagnostic_oracle_ceiling.csv`
- `lumpy_25_actual_vs_forecast.png`
