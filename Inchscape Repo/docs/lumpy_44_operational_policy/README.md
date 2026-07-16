# Lumpy 44 - Operational Forecast Policy

Notebook 44 maps Notebook 43 reliability tiers into planning action, review cadence, inventory treatment, ownership and underforecast escalation. The recommendations are review-ready starting policies, not cost-optimised automated stock rules.

## Result

The operational table assigns 63 SKUs to forecast-led replenishment, 143 to
forecast plus monthly review, 166 to manual approval with forecast guidance,
265 to exception or inventory policy, and 53 dormant SKUs to lifecycle review.

There are 199 underforecast escalations: 125 in manual-review-with-forecast and
74 in exception policy. The largest work queues are recurring manual-review SKUs
(86 escalations), occasional exception-policy SKUs (31), recurring exception
SKUs (22), and occasional manual-review SKUs (16).

These queues organise review effort. They do not calculate safety stock or
purchase quantities without service-level, lead-time, cost and order-constraint
inputs.
