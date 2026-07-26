# Lumpy 33 - Genuine Launch-Cohort Hurdle

Notebook 33 forecasts the 40 official cold-start SKUs from genuine historical product launches. It separates block occurrence probability from positive demand size and locks the strategy with leave-one-launch-out validation.

The official contract remains an 18-month horizon after a three-month information gap, scored as six three-month blocks with pooled SKU WMAPE.

## Result

Twenty-eight genuine historical launches were eligible without crossing into
the official evaluation period. Leave-one-launch-out validation locked a
20-neighbour expected-value hurdle forecast at a 0.75 scale.

On the 40 official cold starts, Notebook 33 improved the below-70 count from 8
to 10 and the below-100 count from 12 to 21. Median WMAPE improved from 117.83%
to 97.97%, while portfolio WMAPE improved from 137.11% to 90.44%. Below-50 fell
from 4 to 1, and the forecast remains under-biased (470.79 forecast units versus
775 actual units).

Notebook 33 is promoted as the broader cold-start benchmark because the primary
below-70 count and the supporting breadth measures improved. The next focused
experiment should preserve its occurrence placement while calibrating demand
size for higher-volume launches.
