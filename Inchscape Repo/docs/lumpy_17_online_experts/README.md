# Notebook 17: Personalised Online Expert Memory

Notebook 17 is the final established recurring A/B experiment. It treats every base forecast as an expert and learns a separate, chronology-safe expert portfolio for each SKU.

## Experimental Layers

- Personal memory: exponentially weighted normalised error by SKU and expert.
- Hierarchical shrinkage: short SKU histories shrink toward cohort-level expert performance.
- Peer borrowing: nearest-neighbour SKUs contribute model-performance evidence.
- Change detection: detected level shifts shorten the memory automatically.
- Specialist concentration: top-2, top-4, and all-expert portfolios are tested.
- Champion blending: personalised memory can be blended back toward the Notebook 16 champion.

## Chronology

Origins 1-2 initialise memory. Origins 3-5 tune 324 configurations prequentially, meaning every forecast uses only lower-numbered origins. Origins 6-7 validate the top 20 sequentially. The required final 18-month forecast uses all seven known origins.

## Retention

The personalised challenger is compared with the current Notebook 16 champion for the same SKUs, dates, gap, horizon, and metric. Below-70 SKU count remains the first decision criterion, followed by below-50 count and median WMAPE.

## Main Outputs

- `lumpy_17_final_champion_comparison.csv`
- `lumpy_17_sku_coverage.csv`
- `lumpy_17_individual_sku_results.csv`
- `lumpy_17_locked_memory_configs.csv`
- `lumpy_17_validation_and_overfit_audit.csv`
- `lumpy_17_sku_expert_memory.csv`
- `lumpy_17_actual_vs_forecast.png`
- `lumpy_17_sku_actual_vs_forecast_gallery.png`

Large row-level forecasts remain under the ignored `results/` directory.
