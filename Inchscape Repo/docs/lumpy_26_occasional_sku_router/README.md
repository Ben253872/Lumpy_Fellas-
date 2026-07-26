# Established Occasional SKU-Level Router

Notebook 26 tests whether cutoff-safe SKU characteristics can route each of the 116 established occasional SKUs to a better Notebook 25 hurdle strategy.

Eligible methods are a grouped out-of-fold Random Forest candidate-error model and a peer nearest-neighbour router that excludes the SKU being scored. Same-SKU historical champion selection is retained only as an explicitly optimistic diagnostic.

The forecast contract remains an 18-month horizon, three-month information gap, six non-overlapping three-month blocks, and per-SKU pooled WMAPE. A router is selected on historical validation and locked before final evaluation.

## Result

Historical validation retained Notebook 25 for occasional A and B and promoted the 20-neighbour mean router for C. On the final horizon, the routed experiment improved median SKU WMAPE from 154.31% to 133.82%, portfolio WMAPE from 147.45% to 138.54%, and below-100 coverage from 21 to 22 positive SKUs. However, below-70 coverage fell from 4 to 3 because the C validation gain did not transfer: C moved from 1 to 0 below 70%.

Notebook 26 therefore does not replace Notebook 25 as the operational champion. The result shows that peer routing improves broad error distribution but is not stable enough for the primary below-70 objective. The ineligible same-SKU diagnostic was substantially stronger, suggesting that temporal model memory may be more useful than cross-SKU similarity, but it needs multiple historical horizons for an honest test.

Main outputs:

- `lumpy_26_candidate_library.csv`
- `lumpy_26_router_validation.csv`
- `lumpy_26_locked_routers.json`
- `lumpy_26_official_summary.csv`
- `lumpy_26_individual_sku_results.csv`
- `lumpy_26_actual_vs_forecast.png`
