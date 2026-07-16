# Notebook 47 - Feature Quality And Eligibility

Audits missingness, effective SKU coverage, uniqueness, cutoff availability and publication timing before modelling.

## Outcome

- Constant and unusable fields were excluded, including Brand, Channel, Region, Country, Family at the cutoff, and five all-missing external series.
- Subfamily, Material Description and Currency passed with 16.15% effective SKU-level missingness.
- Commercial and stock fields are historical-only; 25 macro fields are eligible only with a publication-safe lag.
- Eligibility is necessary but does not prove forecast value. Notebooks 48-50 perform that test.
