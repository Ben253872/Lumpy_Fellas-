# Lumpy 42 - Historical Forecast-Quantile Calibration

Notebook 42 tests expanding-window, segment-specific forecast-quantile calibration learned from historical TSB forecasts and transferred once to Notebook 40's actionable champions. Official actuals are never used to fit or select scale factors.

## Result

Segment-specific historical selection retained no calibration for occasional and
rare, selected calibration challengers for developing, new and recurring, and
had no transferable cold-start mapping. On official evaluation, only new SKUs
passed the SKU-level promotion rule: below-50 improved from 2 to 6 and below-70
from 16 to 25, while below-100 remained 31.

The selective recommendation retains Notebook 40 for every other segment. Across
597 positive actionable SKUs it improves below-50 coverage from 59 to 63 and
below-70 from 197 to 206, with below-100 unchanged at 372. Portfolio WMAPE
improves from 86.75% to 85.58% and bias from -36.70% to -33.97%. Median WMAPE
changes slightly from 82.64% to 82.79%.

The all-segment calibrated challenger is rejected because calibration transfer
damaged recurring and developing thresholds. The recommended output is
`lumpy_42_recommended_actionable_forecasts.csv`.
