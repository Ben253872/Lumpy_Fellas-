from __future__ import annotations

import numpy as np
import pandas as pd


def horizon_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ("fold_id", "segment", "sku_id") if column in forecasts.columns]
    return forecasts.groupby(keys, as_index=False).agg(actual_total=("target", "sum"), forecast_total=("forecast", "sum"))


def fit_quantile_model(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment, group in history.groupby("segment", sort=False):
        cuts = group.forecast_total.quantile([0.25, 0.5, 0.75]).to_numpy(float)
        quartile = np.searchsorted(cuts, group.forecast_total.to_numpy(float), side="right") + 1
        fitted = group.assign(quartile=quartile)
        factors = {}
        global_factor = group.actual_total.sum() / max(group.forecast_total.sum(), 1e-9)
        for number in range(1, 5):
            selected = fitted.loc[fitted.quartile.eq(number)]
            factors[number] = selected.actual_total.sum() / max(selected.forecast_total.sum(), 1e-9) if len(selected) else global_factor
        rows.append({"segment": segment, "q25": cuts[0], "q50": cuts[1], "q75": cuts[2], **{f"raw_factor_q{number}": factors[number] for number in range(1, 5)}})
    return pd.DataFrame(rows)


def apply_quantile_model(
    forecasts: pd.DataFrame,
    model: pd.DataFrame,
    strength: float,
    lower_cap: float = 0.75,
    upper_cap: float = 1.5,
) -> pd.DataFrame:
    result = forecasts.copy()
    totals = horizon_table(result)
    totals = totals.merge(model, on="segment", how="left", validate="many_to_one")
    totals["quartile"] = 1 + (totals.forecast_total.gt(totals.q25)).astype(int) + (totals.forecast_total.gt(totals.q50)).astype(int) + (totals.forecast_total.gt(totals.q75)).astype(int)
    totals["raw_factor"] = [row[f"raw_factor_q{int(row.quartile)}"] for _, row in totals.iterrows()]
    totals["calibration_factor"] = (1.0 + float(strength) * (totals.raw_factor - 1.0)).clip(lower_cap, upper_cap)
    keys = [column for column in ("fold_id", "segment", "sku_id") if column in result.columns]
    result = result.merge(totals[keys + ["quartile", "calibration_factor"]], on=keys, how="left", validate="many_to_one")
    result["forecast"] = result.forecast * result.calibration_factor.fillna(1.0)
    return result
