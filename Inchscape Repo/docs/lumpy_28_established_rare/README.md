# Established Rare Single-Event Hurdle

Notebook 28 is the first focused experiment for the 56 established rare SKUs. It treats rare demand as an event-occurrence and conditional-size problem.

Rare-only and established-transfer hurdle models test calibrated expected value, top-one and top-two event placement, complete-horizon event gating, positive-size shrinkage, classical intermittent-demand references, and an explicit no-event baseline.

Every candidate uses the official 18-month horizon, three-month information gap and six three-month blocks. Strategies are selected on the historical horizon and locked before final evaluation. Event recall, precision and false-positive blocks are reported alongside WMAPE and MASE.

## Result

Only 33 of the 56 rare SKUs had positive demand in the final 18-month horizon. The locked strategies placed 0 below 50%, 3 below 70%, and 7 below 100%, with median SKU WMAPE of 112.50% and portfolio WMAPE of 132.45%.

Event placement is the dominant limitation. Across 52 actual positive blocks, the models correctly identified 13: event recall was 25.0%, precision was 19.4%, and 54 non-demand blocks received a positive forecast. Rare C produced all three below-70 results. Rare B's historical validation selected an effective no-event policy and forecast no demand in the final horizon, where 3 SKUs and 4 blocks were actually positive.

This establishes the first rare-SKU baseline. The next improvement should target pooled family/subfamily occurrence evidence or inventory-probability outputs rather than simply increasing positive-size regression depth.

Main outputs:

- `lumpy_28_locked_strategies.json`
- `lumpy_28_validation_candidates.csv`
- `lumpy_28_official_summary.csv`
- `lumpy_28_individual_sku_results.csv`
- `lumpy_28_event_diagnostics.csv`
- `lumpy_28_actual_vs_forecast.png`
