from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd


EXTERNAL_FEATURES = {
    "Bank_Lending_Rate", "Building_Permits", "CPI_Housing_Utilities", "CPI_Transportation",
    "Car_Registrations", "Consumer_Credit", "Core_Consumer_Prices", "Core_Inflation_Rate",
    "Deposit_Interest_Rate", "Employed_Persons", "Employment_Rate", "Food_Inflation",
    "Housing_Index", "Import_Prices", "Imports", "Industrial_Production_Mom", "Inflation_Rate",
    "Inflation_Rate_MoM", "Interest_Rate", "Labor_Force_Participation_Rate",
    "Leading_Economic_Index", "Minimum_Wages", "Private_Sector_Credit", "Producer_Prices",
    "Producer_Prices_Change", "Retail_Sales_MoM", "Total_Vehicle_Sales", "Unemployed_Persons",
    "Unemployment_Rate", "Wages",
}
EVALUATION_OR_TARGET = {"demand"}
IDENTIFIERS = {"sku_id"}
HISTORICAL_ONLY = {"REVENUE", "COST", "TOTAL_PROFIT", "UNIT_PRICE", "UNIT_COST", "STOCK_END_MONTH", "STOCK_START_MONTH", "NEW_ENTRY_STOCK"}
STATIC_FEATURES = {"COUNTRY_BRAND_CHANNEL", "Country", "Brand", "Channel", "FAMILY_DESCRIPTION", "SUBFAMILY_DESCRIPTION", "MATERIAL_DESCRIPTION", "CURRENCY", "REGION"}


def _hash_series(series: pd.Series) -> str:
    normalized = series.astype("string").fillna("<NA>")
    return hashlib.sha1(pd.util.hash_pandas_object(normalized, index=False).values.tobytes()).hexdigest()


def quality_table(frame: pd.DataFrame, cutoff: pd.Timestamp, date_column: str = "month") -> pd.DataFrame:
    dated = frame.copy()
    dated[date_column] = pd.to_datetime(dated[date_column])
    cutoff_frame = dated.loc[dated[date_column].le(pd.Timestamp(cutoff))]
    hashes: dict[str, list[str]] = {}
    rows = []
    for column in dated.columns:
        series = dated[column]
        cutoff_series = cutoff_frame[column]
        raw_missing_pct = 100 * float(series.isna().mean())
        raw_cutoff_missing_pct = 100 * float(cutoff_series.isna().mean())
        if column in STATIC_FEATURES:
            effective = dated.sort_values(date_column).groupby("sku_id")[column].last()
            cutoff_effective = cutoff_frame.sort_values(date_column).groupby("sku_id")[column].last()
            non_null = effective.dropna()
            cutoff_non_null = cutoff_effective.dropna()
            missing_pct = 100 * float(effective.isna().mean())
            cutoff_missing_pct = 100 * float(cutoff_effective.isna().mean())
        else:
            non_null = series.dropna()
            cutoff_non_null = cutoff_series.dropna()
            missing_pct = raw_missing_pct
            cutoff_missing_pct = raw_cutoff_missing_pct
        unique = int(non_null.nunique(dropna=True))
        dominant = float(non_null.value_counts(normalize=True, dropna=True).iloc[0]) if len(non_null) else np.nan
        digest = _hash_series(series)
        hashes.setdefault(digest, []).append(column)
        varies_by_sku = False
        varies_by_month = False
        if column not in {"sku_id", date_column} and len(non_null):
            varies_by_sku = bool(dated.groupby(date_column)[column].nunique(dropna=True).max() > 1)
            varies_by_month = bool(dated.groupby("sku_id")[column].nunique(dropna=True).max() > 1)
        available_dates = dated.loc[series.notna(), date_column]
        rows.append({"feature": column, "dtype": str(series.dtype), "raw_missing_pct": raw_missing_pct, "raw_cutoff_missing_pct": raw_cutoff_missing_pct, "missing_pct": missing_pct, "cutoff_missing_pct": cutoff_missing_pct, "unique_values": unique, "cutoff_unique_values": int(cutoff_non_null.nunique(dropna=True)), "dominant_value_pct": 100*dominant if pd.notna(dominant) else np.nan, "varies_by_sku": varies_by_sku, "varies_by_month": varies_by_month, "first_available_month": available_dates.min() if len(available_dates) else pd.NaT, "last_available_month": available_dates.max() if len(available_dates) else pd.NaT, "series_hash": digest})
    result = pd.DataFrame(rows)
    duplicate_map = {column: ", ".join(item for item in columns if item != column) for columns in hashes.values() if len(columns)>1 for column in columns}
    result["duplicate_of"] = result.feature.map(duplicate_map).fillna("")
    return classify_features(result)


def classify_features(quality: pd.DataFrame) -> pd.DataFrame:
    result = quality.copy()
    statuses = []
    reasons = []
    for row in result.itertuples():
        feature = row.feature
        if feature in IDENTIFIERS:
            status, reason = "identifier", "Identifier, not a model feature."
        elif feature in EVALUATION_OR_TARGET:
            status, reason = "target", "Forecast target; never usable as a predictor for the same period."
        elif feature == "month":
            status, reason = "eligible_transform", "Use cutoff-safe calendar transforms, not raw future outcomes."
        elif row.unique_values <= 1 or row.cutoff_unique_values <= 1:
            status, reason = "constant", "One or fewer observed values at the official cutoff."
        elif row.cutoff_missing_pct >= 80:
            status, reason = "insufficient_coverage", "At least 80% missing at the official cutoff."
        elif row.dominant_value_pct >= 99:
            status, reason = "near_constant", "At least 99% of observed rows share one value."
        elif row.duplicate_of:
            status, reason = "redundant", f"Exact duplicate of: {row.duplicate_of}."
        elif feature in EXTERNAL_FEATURES:
            status, reason = "external_lag_required", "External context; only publication-safe historical lags may be used."
        elif feature in HISTORICAL_ONLY:
            status, reason = "historical_aggregation_only", "Use lagged/cutoff aggregates only; future values are unknown."
        elif row.cutoff_missing_pct > 20:
            status, reason = "impute_cautiously", "Eligible only with fold-safe imputation and missingness indicators."
        else:
            status, reason = "eligible", "Adequate cutoff coverage and variation."
        statuses.append(status); reasons.append(reason)
    result["eligibility_status"] = statuses
    result["eligibility_reason"] = reasons
    result["eligible_for_importance"] = result.eligibility_status.isin(["eligible", "eligible_transform", "historical_aggregation_only", "external_lag_required", "impute_cautiously"])
    return result
