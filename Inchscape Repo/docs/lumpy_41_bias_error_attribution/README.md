# Lumpy 41 - Bias Concentration And Error Attribution

Notebook 41 audits Notebook 40's 637-SKU actionable champion portfolio. It attributes bias and absolute error by SKU concentration, segment, volume quartile and block-level event mechanism without modifying forecasts.

## Result

The top 10% of positive SKUs contribute 40.98% of actionable absolute error and
39.47% of actual volume. The highest actual-volume quartile contributes -3,326
units of signed error, while the lowest quartile overforecasts by 660 units.

Matched events with underestimated positive quantity contribute 4,828 absolute
error units. Missed events contribute 679 and false events 1,668. The dominant
problem is therefore positive-quantity underestimation on higher-volume SKUs,
not simply failure to predict whether demand occurs.

Recurring SKUs contribute 56.10% of total absolute error, followed by occasional
12.76%, new 12.08%, cold start 9.05%, developing 6.95% and rare 3.07%.

Notebook 42 should test cutoff-safe high-predicted-volume calibration using
expanding historical folds. It must not derive scale factors from official actuals.
