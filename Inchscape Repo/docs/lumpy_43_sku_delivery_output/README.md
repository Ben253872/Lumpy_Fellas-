# Lumpy 43 - SKU Delivery Output

Notebook 43 combines Notebook 42's selective actionable recommendations with dormant governance rows and produces queryable 690-SKU and 637-SKU delivery tables. The tables contain classifications, selected sources, six forecast/actual blocks, 18-month totals, evaluation metrics, bias and reliability flags.

The output is an official backtest delivery artifact, not a newly trained future forecast.

## Result

The full delivery contains 690 rows and 33 columns; the actionable delivery
contains 637 rows. Reliability assignment yields 63 forecast-led SKUs, 143
forecast-plus-review SKUs, 166 manual-review-with-forecast SKUs, 265
exception-policy SKUs and 53 dormant lifecycle-review SKUs.

The tables preserve all six forecast and actual blocks, 18-month totals,
individual WMAPE, diagnostic evaluation MASE, bias and champion provenance.
Actual-dependent metrics and reliability tiers are evaluation outputs and must
not be used as future model features.
