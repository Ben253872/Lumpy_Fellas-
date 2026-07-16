# Notebook 49 - Controlled Feature Group Ablation

Tests feature groups admitted by Notebook 48 on rolling historical folds. Candidates preserve the required 18-month horizon, three-month gap and six three-month block per-SKU WMAPE shape. The final holdout is not used for selection.

## Outcome

Historical selection promoted three locked challengers for final evaluation:

- Recurring: demand plus commercial and stock.
- New: demand plus commercial and stock.
- Developing: demand plus stock and product.

Occasional and rare SKUs did not improve enough to proceed.
