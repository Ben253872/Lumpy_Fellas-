# Notebook 50 - Locked Feature Challenger Holdout

Retrains only the feature recipes promoted on Notebook 49 historical folds and evaluates them once on the untouched final 18-month holdout. The current Notebook 42 recommendation remains the baseline; replacement decisions are exported as experimental evidence.

## Outcome

No challenger passed the aligned holdout gate. All comparisons use the same frozen Notebook 42 SKU membership, targets, six blocks and positive-SKU denominator.

- Recurring below 70%: 42.18% baseline versus 39.11% challenger.
- New below 70%: 64.10% baseline versus 30.77% challenger.
- Developing below 70%: 50.00% baseline versus 12.50% challenger.

The current Notebook 42 recommendation remains unchanged. The next useful feature work requires genuinely leading SKU-level signals such as fitment, supersession, quotations/orders, backorders or lifecycle dates; current macro and static fields should not be added.
