# Official Established Recurring A/B/C Scorecard

Notebook 21 is the source of truth for the requested configuration.

- cutoff: July 2024
- gap: August through October 2024
- horizon: November 2024 through April 2026
- blocks: six non-overlapping 3-month forecasts
- primary metric: per-SKU pooled WMAPE across all six blocks
- official model sources: validation-locked Notebook 17 challengers for A/B and corrected validation-locked Notebook 20 for C

Different classifications may use different forecasting models. They are compared only through this common evaluation contract.

Post-final champion switching is prohibited. The scorecard does not use Notebook 17's final-retention output.

The future 9-month horizon with a 1-month gap is registered as a separate challenger and is not yet an official comparable A/B/C result.

Main outputs:

- `lumpy_21_official_abc_summary.csv`
- `lumpy_21_official_individual_sku_results.csv`
- `lumpy_21_official_three_month_blocks.csv`
- `lumpy_21_secondary_block_average.csv`
- `lumpy_21_configuration_registry.csv`
- `lumpy_21_official_actual_vs_forecast.png`
