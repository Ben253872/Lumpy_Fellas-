# Lumpy 15 - Full Established Recurring A/B Optimisation

## Scope

Notebook 15 remains on the 325 established recurring A/B SKUs and expands the Notebook 14 search before considering another lifecycle group.

It tests:

- 13 structural XGBoost configurations;
- direct Tweedie powers 1.20, 1.35, and 1.50;
- hurdle depth, estimator, learning-rate, regularisation, occurrence-weight, and positive-size alternatives;
- 18, 24, 30, 36, and 48-month history windows;
- 455 structural rolling-origin fits;
- tuned SBA alpha and TSB alpha/beta combinations;
- recent-demand windows from 3 to 18 months;
- expected, soft-gated, hard-gated, calibrated, and baseline-blended recipes;
- size/volatility subgroup routing;
- pairwise model ensembles;
- Ridge, random-forest, histogram-gradient-boosting, and XGBoost correction regressors.

The first five three-month origins tune model recipes. The final two known origins select the A and B strategies. The required final benchmark remains an 18-month horizon with a three-month operational gap.

## Validation Winners

Size/volatility subgroup routing won validation for both cohorts:

- Recurring A: 148 of 215 below 70% and 98 below 50%.
- Recurring B: 56 of 104 positive-demand SKUs below 70% and 38 below 50%.

The correction regressors did not beat the routed or ensemble approaches. Their low aggregate bias did not translate into more individually useful forecasts.

## Final Retention Decision

Neither locked challenger beat the current Notebook 14 incumbent on the 18-month benchmark:

- Recurring A tied at 111 below 70%, but fell from 36 to 35 below 50% and had a worse median WMAPE.
- Recurring B fell from 40 to 36 below 70%, from 14 to 13 below 50%, and had a worse median WMAPE.

Both Notebook 14 incumbents are therefore retained.

| Cohort | Positive SKUs | Below 50% | Below 70% | Below 100% | Median WMAPE | Portfolio WMAPE | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recurring A | 213 | 36 | 111 | 178 | 68.6% | 76.7% | -37.0% |
| Recurring B | 110 | 14 | 40 | 80 | 78.3% | 82.1% | -41.0% |
| Combined | 323 | 50 | 151 | 258 | 71.4% | 77.8% | -37.8% |

## Diagnostic Oracle Ceiling

The oracle is descriptive only: it chooses the best final candidate separately for each SKU after seeing actual demand. It cannot be used operationally. It shows whether the candidate pool contains enough model diversity for a better router to reach the target.

| Cohort | Positive SKUs | Oracle below 70% | Oracle share | Oracle median WMAPE |
|---|---:|---:|---:|---:|
| Recurring A | 213 | 152 | 71.4% | 57.1% |
| Recurring B | 110 | 57 | 51.8% | 66.7% |
| Combined | 323 | 209 | 64.7% | 60.7% |

Even perfect hindsight cannot produce 75% below 70% from the current 180 tuned demand-history candidates. The remaining gap is therefore not just hyperparameter tuning or model routing.

## Interpretation

Further searches over the same demand-only features are unlikely to deliver a defensible step change and risk fitting the repeatedly inspected benchmark. A/B should remain the focus, but the next experiment needs new cutoff-safe signal: stock availability and replenishment, product supersession, vehicle fitment/parc, lifecycle dates, order or quote indicators, backorders, and other operational leading indicators.

## Primary Files

- `lumpy_15_tuning_all_candidates.csv`
- `lumpy_15_validation_tuned_trials.csv`
- `lumpy_15_subgroup_choices.csv`
- `lumpy_15_ensemble_choices.csv`
- `lumpy_15_correction_choices.csv`
- `lumpy_15_validation_strategy_summary.csv`
- `lumpy_15_locked_strategy_per_cohort.csv`
- `lumpy_15_retention_decisions.csv`
- `lumpy_15_recommended_summary.csv`
- `lumpy_15_diagnostic_oracle_ceiling.csv`

Large component checkpoints and block forecasts are ignored by Git.
