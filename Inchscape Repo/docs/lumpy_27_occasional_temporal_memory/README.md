# Established Occasional Temporal Model Memory

Notebook 27 tests whether a hurdle strategy that worked for an occasional SKU in an earlier 18-month horizon remains useful in a later unseen 18-month horizon.

Three historical horizons support two walk-forward transfers before final selection. Rules compare raw WMAPE and within-horizon ranks, recency decay, and shrinkage between each SKU's history and its A/B/C cohort. All horizons retain the three-month information gap and six three-month evaluation blocks.

## Result

Temporal SKU memory improved the summed below-70 count over cohort-only memory in historical walk-forward evaluation for A, B, and C, so a memory rule was locked for every cohort. On the final horizon, Notebook 27 tied Notebook 25 at 4 positive SKUs below 70%, improved below-50 coverage from 1 to 2 and below-100 coverage from 21 to 22, reduced median SKU WMAPE from 154.31% to 152.75%, and reduced portfolio WMAPE from 147.45% to 145.87%.

This is a valid but small improvement. Exact best-candidate persistence was only 26.1% from Horizon 1 to 2 and 13.8% from Horizon 2 to 3, confirming that individual strategy winners are unstable. C gained below-70 coverage, while A and B lost coverage, so temporal memory should be treated as a cautious challenger rather than evidence that occasional demand is solved.

Main outputs:

- `lumpy_27_walk_forward_validation.csv`
- `lumpy_27_memory_rule_summary.csv`
- `lumpy_27_candidate_persistence.csv`
- `lumpy_27_locked_memory_rules.json`
- `lumpy_27_official_summary.csv`
- `lumpy_27_individual_sku_results.csv`
- `lumpy_27_actual_vs_forecast.png`
