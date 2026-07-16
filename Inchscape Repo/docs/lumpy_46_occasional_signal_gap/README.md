# Lumpy 46 - Occasional Signal Gap And Value Of Information

Notebook 46 compares the occasional champion with the ineligible candidate oracle, inventories the completed demand-only search, audits available leading signals, and publishes a concrete data-acquisition specification and modelling go/no-go decision.

## Result

The current champion places 4/103 positive occasional SKUs below 70% WMAPE,
while the ineligible per-SKU candidate oracle places 84/103 below 70% and 83/103
below 50%. This represents diagnostic headroom for 80 SKUs, but it cannot be
deployed because the oracle uses official outcomes to select each candidate.

The completed demand-only search contains 30 candidate strategies, 12 router
configurations and three router families across Notebooks 22-27. Stock history
and product hierarchy are available and already tested. Supersession, vehicle
fitment, quote/order pipeline, backorder/lost-sales, supplier lead-time and
lifecycle-date histories are absent.

The decision is no further demand-only occasional model. The next modelling gate
is at least one cutoff-safe leading-signal history, with supersession chain first
and vehicle fitment/parc second. A field-level acquisition specification is saved
in `lumpy_46_data_acquisition_spec.csv`.
