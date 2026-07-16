# Lumpy 39 - Dormant-SKU Reactivation And Inventory Policy

Notebook 39 treats the 53 dormant SKUs as an expanding-window reactivation hurdle. It estimates block occurrence and positive size, preserves the official point-forecast contract, and adds review-oriented inventory-policy tiers.

## Result

Historical selection locked a recency-pooled expected-value hurdle with smoothing
15, a 50% individual-size blend and a 0.75 scale. On the 45 positive official
dormant SKUs, median WMAPE improved from 283.33% to 212.48%, portfolio WMAPE from
309.40% to 204.37%, and bias from +191.23% to +67.07%.

Individual thresholds deteriorated: below-50 fell from 1 to 0, below-70 from 2
to 0, and below-100 from 4 to 1. Notebook 39 is therefore not promoted as the
point forecast; Notebook 13 remains the dormant benchmark.

The policy output is diagnostic only. Forty-one SKUs entered safety-stock review
and 12 entered monitor-reactivation, but the lower tier had a higher observed
reactivation rate. The model captures the high overall reactivation rate without
ranking SKUs reliably enough for automated inventory action. A future dormant
experiment should use product supersession, stock availability, fitment, quotes,
orders or other operational reactivation signals rather than further demand-only
point-forecast tuning.
