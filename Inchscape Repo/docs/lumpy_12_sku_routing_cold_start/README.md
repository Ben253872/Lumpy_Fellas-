# Lumpy 12 - SKU Routing And Cold Starts

This experiment reuses notebook 11 forecasts and adds cutoff-safe SKU classification, development-trained model routing, and peer-based cold-start forecasting.

## Routing classifications

- Lifecycle: cold start, new, developing, established, dormant
- Demand frequency: rare, occasional, recurring
- Recency: recent, stale, dormant
- Positive-demand size pattern
- Unit and value ABC tiers
- Potential stock constraint, clearly labelled as a screening flag

## Cold starts

Peer methods use subfamily and material-description similarity. Development folds simulate cold starts by excluding each target SKU's demand profile from the peer pool. The selected method is then tested once on the 40 genuine cold starts.

The backtest assumes product master descriptions are known before launch. Transactional future demand, stock, revenue, and price are never used as target features.

## Primary outputs

- `lumpy_12_sku_classification_all_690.csv`
- `lumpy_12_router_holdout_summary.csv`
- `lumpy_12_cold_start_method_development_summary.csv`
- `lumpy_12_real_cold_start_holdout_summary.csv`
- `lumpy_12_final_all_690_holdout_summary.csv`
- `lumpy_12_final_all_690_holdout_per_sku.csv`
- `lumpy_12_final_segment_summary.csv`
- `lumpy_12_coverage_audit.csv`
