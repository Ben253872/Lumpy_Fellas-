# Lumpy 38 - Developing-SKU High-Volume Routing

Notebook 38 retains Notebook 37's locked hurdle/analogue blend and tests cutoff-safe high-volume routing across recent demand, positive size, acceleration and predicted horizon total.

## Result

Historical validation selected the upper quartile by predicted 18-month horizon
total and applied a 1.50 scale only to that group. On the 24 positive official
developing SKUs, below-50 coverage improved from 2 to 3, while below-70 remained
12 and below-100 remained 16. Median WMAPE remained 69.46% and bias improved from
-39.65% to -30.31%. Portfolio WMAPE worsened slightly from 84.17% to 84.69%.

Notebook 38 is promoted under the agreed SKU-level priority: below-70 is tied and
below-50 improves. It is a targeted individual-SKU improvement, not a portfolio
accuracy improvement. Further developing-SKU tuning should stop unless richer
operational signals become available.
