from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def calibrated_routes(
    forecasts: pd.DataFrame,
    features: pd.DataFrame,
    global_scales: Iterable[float],
    route_quantiles: Iterable[float],
    low_scales: Iterable[float],
    high_scales: Iterable[float],
    route_features: Iterable[str] = ("recent_6m_total",),
) -> pd.DataFrame:
    keys = [column for column in ("fold_id", "sku_id") if column in forecasts.columns]
    route_features = tuple(route_features)
    feature_columns = keys + list(route_features)
    enriched = forecasts.merge(features[feature_columns].drop_duplicates(keys), on=keys, how="left", validate="many_to_one")
    output = []
    for scale in global_scales:
        candidate = enriched.copy(); candidate["forecast"] *= float(scale); candidate["candidate_id"] = f"global_scale_{scale:.2f}"; output.append(candidate)
    fold_key = "fold_id" if "fold_id" in enriched.columns else None
    for route_feature in route_features:
        for quantile in route_quantiles:
            if fold_key:
                cutoff = enriched.groupby(fold_key)[route_feature].transform(lambda values: values.quantile(quantile))
            else:
                cutoff = pd.Series(enriched[route_feature].quantile(quantile), index=enriched.index)
            high = enriched[route_feature].gt(cutoff)
            for low_scale in low_scales:
                for high_scale in high_scales:
                    candidate = enriched.copy()
                    candidate["forecast"] *= high.map({True: float(high_scale), False: float(low_scale)})
                    candidate["candidate_id"] = f"{route_feature}_q{quantile:.2f}__low_{low_scale:.2f}__high_{high_scale:.2f}"
                    output.append(candidate)
    return pd.concat(output, ignore_index=True)
