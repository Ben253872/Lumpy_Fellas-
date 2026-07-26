# Established Rare Group Occurrence Priors

Notebook 29 tests whether cutoff-safe family and subfamily occurrence history improves event timing for the 56 established rare SKUs.

Smoothed month-of-year group occurrence rates reshape each SKU's six block probabilities while preserving total probability mass. Family, subfamily and blended timing priors compete across multiple strengths, top-one/top-two rules and positive-size treatments. Notebook 28 remains the incumbent unless validation threshold counts improve.

## Result

Group occurrence priors replaced Notebook 28 for rare A and C; rare B retained its incumbent. On the final horizon, below-70 coverage improved from 3 to 4 positive SKUs and below-100 coverage from 7 to 11. Median SKU WMAPE improved from 112.50% to 110.62%, bias from -54.90% to -23.68%, event recall from 25.0% to 28.8%, and event precision from 19.4% to 33.3%. False-positive blocks fell from 54 to 30.

The trade-off is volume: rare A overforecast, causing portfolio WMAPE to worsen from 132.45% to 146.61%. No SKU finished below 50%. Group timing is therefore useful, but the next experiment should calibrate forecast quantity after event placement rather than add more timing models.

Main outputs:

- `lumpy_29_locked_strategies.json`
- `lumpy_29_validation_candidates.csv`
- `lumpy_29_official_summary.csv`
- `lumpy_29_individual_sku_results.csv`
- `lumpy_29_event_diagnostics.csv`
- `lumpy_29_actual_vs_forecast.png`
