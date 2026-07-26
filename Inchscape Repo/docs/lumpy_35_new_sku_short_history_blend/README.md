# Lumpy 35 - New-SKU Short-History And Analogue Blend

Notebook 35 targets the 39 new SKUs with 1-11 months of cutoff-safe history. It compares eight frozen short-history models, five metadata analogue forecasts and weighted blends across five historical development folds before evaluating the official fold.

## Result

Historical validation locked a blend of 75% TSB short-history forecast and 25%
subfamily analogue forecast. On the official 39 new SKUs, below-70 coverage
improved from 3 to 13 SKUs. Median WMAPE improved from 85.00% to 74.84% and
portfolio WMAPE improved from 97.71% to 79.93%. Below-50 remained 2 and
below-100 remained 31.

Notebook 35 is promoted as the new-SKU champion. Forecast bias remains strongly
negative at -49.75%, so the next focused experiment should calibrate the blend's
level or route higher-volume new SKUs without changing the locked horizon,
three-month gap, block structure, or metric.
