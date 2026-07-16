# Lumpy 34 - Cold-Start Positive-Quantity Calibration

Notebook 34 retains the genuine-launch occurrence approach from Notebook 33 and tests weighted median, mean and upper-tail positive demand sizes. Selection remains leave-one-launch-out and precedes the official 40-SKU evaluation.

## Result

Historical validation locked a 20-neighbour expected-value hurdle using weighted
median positive size and a 1.15 scale. On the official cold starts it improved
median WMAPE from 97.97% to 92.46%, portfolio WMAPE from 90.44% to 88.19%, and
below-100 coverage from 21 to 23 SKUs.

The primary below-70 count fell from 10 to 9, below-50 remained 1, and bias
worsened slightly from -39.25% to -41.13%. Notebook 34 is therefore not promoted.
Notebook 33 remains the cold-start champion. Quantity calibration alone has not
resolved the remaining high-volume launch underforecast, so the next segment to
address is the 39 new SKUs with short but usable demand histories.
