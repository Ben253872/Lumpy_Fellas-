# Notebook 48 - Rolling Feature Importance

Uses rolling historical folds and a common Random Forest diagnostic to test feature usefulness by SKU segment.

## Outcome

- Demand history is the strongest feature group.
- Commercial, stock and selected product fields show material importance in some segments.
- External and calendar features fail the 0.001 log-MAE materiality gate and do not proceed.
- Importance is diagnostic evidence only; Notebook 49 tests group-level forecast impact.
