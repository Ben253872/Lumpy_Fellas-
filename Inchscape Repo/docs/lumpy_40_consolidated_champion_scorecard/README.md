# Lumpy 40 - Consolidated Champion Scorecard

Notebook 40 reconciles all 690 SKUs onto the official six-block contract, routes each segment to its latest compatible champion, and publishes both full-governance and dormant-excluded actionable scorecards.

Notebook 21 is the authoritative recurring source. It resolves the 16-positive-SKU discrepancy caused by comparing Notebook 19's single-horizon recurring-C subset with later six-block segment results.

## Result

Coverage reconciles exactly to 690 SKUs, 4,140 three-month block rows, 642
positive SKUs and 9,111 actual units. Against Notebook 13, the consolidated
champions improve all-690 below-50 coverage from 55 to 60, below-70 from 171 to
199, and below-100 from 361 to 376. Median WMAPE improves from 90.33% to 86.46%
and portfolio WMAPE from 102.52% to 91.35%. Bias becomes more negative, moving
from -20.97% to -31.99%.

The actionable portfolio excluding dormant contains 637 SKUs and 597 positive
SKUs. Champion routing improves below-70 coverage from 169 to 197 and below-100
from 357 to 372. Median WMAPE improves from 85.54% to 82.64% and portfolio WMAPE
from 98.16% to 86.75%. Bias changes from -25.44% to -36.70%, which remains an
important planning caveat.

Dormant SKUs remain present in the governance output but are excluded from the
actionable scorecard and should receive manual lifecycle/reactivation review.
