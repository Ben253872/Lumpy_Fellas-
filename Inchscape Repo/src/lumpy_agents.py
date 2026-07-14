"""Lightweight review agents for lumpy forecasting outputs.

These are deliberately separated from the model build. They read saved CSV
outputs and return compact, rule-based recommendations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _status_from_wmape(wmape_percent: float | int | None) -> str:
    if pd.isna(wmape_percent):
        return "no_actual_demand_to_score"
    if wmape_percent < 50:
        return "strong"
    if wmape_percent < 70:
        return "usable"
    if wmape_percent < 100:
        return "watch"
    return "weak"


def load_lumpy_outputs(root: Path) -> dict[str, pd.DataFrame]:
    table_dir = Path(root) / "results" / "lumpy_outputs" / "tables"
    files = {
        "model_data": "lumpy_model_data.csv",
        "splits": "lumpy_backtest_splits.csv",
        "forecasts": "lumpy_backtest_forecasts.csv",
        "model_summary": "lumpy_model_summary.csv",
        "monthly_totals": "lumpy_monthly_total_results.csv",
        "best_model": "lumpy_best_model_by_window.csv",
        "external_inventory": "lumpy_selected_external_features.csv",
    }
    outputs = {}
    for key, filename in files.items():
        path = table_dir / filename
        if path.exists():
            outputs[key] = pd.read_csv(path)
        else:
            outputs[key] = pd.DataFrame()
    return outputs


def run_data_quality_agent(model_data: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    if model_data.empty:
        return pd.DataFrame(
            [
                {
                    "agent": "Data Quality Agent",
                    "status": "missing_model_data",
                    "finding": "No lumpy model data output was found.",
                    "recommendation": "Run lumpy_01_data_check or the build notebook first.",
                }
            ]
        )

    data = model_data.copy()
    data["month"] = pd.to_datetime(data["month"], errors="coerce")
    data["demand"] = pd.to_numeric(data["demand"], errors="coerce").fillna(0.0)
    zero_share = float(data["demand"].eq(0).mean())
    positive_rows = int(data["demand"].gt(0).sum())
    sku_count = int(data["sku_id"].nunique())
    month_count = int(data["month"].nunique())

    if positive_rows == 0:
        status = "blocked"
        recommendation = "No positive lumpy demand exists in the model frame; do not train demand models."
    elif zero_share > 0.95:
        status = "sparse"
        recommendation = "SKU-month demand is extremely sparse; prefer aggregate monthly planning and occurrence-gated SKU allocation."
    else:
        status = "usable"
        recommendation = "Data is sparse but modelable; continue with aggregate and hurdle comparisons."

    return pd.DataFrame(
        [
            {
                "agent": "Data Quality Agent",
                "status": status,
                "rows": len(data),
                "sku_count": sku_count,
                "month_count": month_count,
                "first_month": data["month"].min(),
                "last_month": data["month"].max(),
                "positive_rows": positive_rows,
                "zero_row_share": zero_share,
                "fold_count": len(splits),
                "finding": f"{zero_share:.1%} of SKU-month rows are zero.",
                "recommendation": recommendation,
            }
        ]
    )


def run_model_selection_agent(model_summary: pd.DataFrame) -> pd.DataFrame:
    if model_summary.empty:
        return pd.DataFrame(
            [
                {
                    "agent": "Model Selection Agent",
                    "status": "missing_model_summary",
                    "finding": "No model summary output was found.",
                    "recommendation": "Run the build notebook before reviewing model choice.",
                }
            ]
        )

    rows = []
    for window_label, window in model_summary.groupby("window_label", dropna=False):
        ranked = window.sort_values(["wmape_percent", "model"]).reset_index(drop=True)
        best = ranked.iloc[0]
        runner_up = ranked.iloc[1] if len(ranked) > 1 else None
        gap = (
            runner_up["wmape_percent"] - best["wmape_percent"]
            if runner_up is not None and pd.notna(runner_up["wmape_percent"]) and pd.notna(best["wmape_percent"])
            else np.nan
        )
        status = _status_from_wmape(best["wmape_percent"])
        if best["model"] == "Zero Forecast":
            recommendation = "Zero is winning; current SKU-month signal is weak. Prioritize aggregate demand or business-rule stocking review before more modelling."
        elif best["model"] == "Aggregate Allocation":
            recommendation = "Keep aggregate allocation. Improve allocation signals before adding more model complexity."
        elif best["model"] == "Hurdle Random Forest":
            recommendation = "Keep hurdle RF if scikit-learn is installed and rf_status is fit; otherwise treat it as a placeholder fallback."
        else:
            recommendation = "Keep as benchmark pressure; only promote if it beats zero and aggregate allocation by a clear margin."

        rows.append(
            {
                "agent": "Model Selection Agent",
                "window_label": window_label,
                "status": status,
                "best_model": best["model"],
                "best_wmape_percent": best["wmape_percent"],
                "runner_up_model": runner_up["model"] if runner_up is not None else pd.NA,
                "runner_up_gap_points": gap,
                "finding": f"{best['model']} leads this window with {best['wmape_percent']:.1f}% WMAPE.",
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows)


def run_monthly_total_agent(monthly_totals: pd.DataFrame) -> pd.DataFrame:
    if monthly_totals.empty:
        return pd.DataFrame(
            [
                {
                    "agent": "Monthly Total Agent",
                    "status": "missing_monthly_totals",
                    "finding": "No monthly total output was found.",
                    "recommendation": "Run the build notebook before judging aggregate demand.",
                }
            ]
        )

    data = monthly_totals.copy()
    data["absolute_error"] = pd.to_numeric(data["absolute_error"], errors="coerce")
    data["actual"] = pd.to_numeric(data["actual"], errors="coerce")
    summary = (
        data.groupby(["window_label", "model"], as_index=False)
        .agg(actual_total=("actual", "sum"), error_total=("absolute_error", "sum"))
    )
    summary["monthly_total_wmape_percent"] = np.where(
        summary["actual_total"].gt(0),
        100 * summary["error_total"] / summary["actual_total"],
        np.nan,
    )
    rows = []
    for window_label, window in summary.groupby("window_label", dropna=False):
        best = window.sort_values(["monthly_total_wmape_percent", "model"]).iloc[0]
        status = _status_from_wmape(best["monthly_total_wmape_percent"])
        rows.append(
            {
                "agent": "Monthly Total Agent",
                "window_label": window_label,
                "status": status,
                "best_monthly_total_model": best["model"],
                "monthly_total_wmape_percent": best["monthly_total_wmape_percent"],
                "finding": f"{best['model']} is best on monthly total demand.",
                "recommendation": (
                    "Use monthly-total planning as the primary lens if it is materially better than SKU-month WMAPE."
                    if status in {"strong", "usable"}
                    else "Monthly total demand is still noisy; avoid over-claiming model precision."
                ),
            }
        )
    return pd.DataFrame(rows)


def run_occurrence_agent(forecasts: pd.DataFrame) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame(
            [
                {
                    "agent": "Occurrence Agent",
                    "status": "missing_forecasts",
                    "finding": "No forecast output was found.",
                    "recommendation": "Run the build notebook before reviewing positive-demand capture.",
                }
            ]
        )

    data = forecasts.copy()
    data["demand"] = pd.to_numeric(data["demand"], errors="coerce").fillna(0.0)
    data["forecast"] = pd.to_numeric(data["forecast"], errors="coerce").fillna(0.0)
    data["actual_positive"] = data["demand"].gt(0)
    data["forecast_positive"] = data["forecast"].gt(0)
    data["missed_positive"] = data["actual_positive"] & ~data["forecast_positive"]
    data["false_positive"] = ~data["actual_positive"] & data["forecast_positive"]

    summary = (
        data.groupby(["window_label", "model"], as_index=False)
        .agg(
            actual_positive_rows=("actual_positive", "sum"),
            forecast_positive_rows=("forecast_positive", "sum"),
            missed_positive_rows=("missed_positive", "sum"),
            false_positive_rows=("false_positive", "sum"),
            rows=("demand", "size"),
        )
    )
    summary["positive_recall_proxy"] = np.where(
        summary["actual_positive_rows"].gt(0),
        1 - summary["missed_positive_rows"] / summary["actual_positive_rows"],
        np.nan,
    )
    summary["false_positive_share"] = summary["false_positive_rows"] / summary["rows"]
    summary["agent"] = "Occurrence Agent"
    summary["status"] = np.where(
        summary["positive_recall_proxy"].ge(0.5),
        "capturing_some_positives",
        "missing_positives",
    )
    summary["recommendation"] = np.where(
        summary["model"].eq("Zero Forecast"),
        "Keep zero as a control only.",
        "Review whether this model captures positives without creating too many false positives.",
    )
    return summary[
        [
            "agent",
            "window_label",
            "model",
            "status",
            "actual_positive_rows",
            "forecast_positive_rows",
            "missed_positive_rows",
            "false_positive_rows",
            "positive_recall_proxy",
            "false_positive_share",
            "recommendation",
        ]
    ]


def run_external_feature_agent(external_inventory: pd.DataFrame) -> pd.DataFrame:
    if external_inventory.empty:
        return pd.DataFrame(
            [
                {
                    "agent": "External Feature Agent",
                    "status": "no_external_features",
                    "finding": "No selected external features were recorded.",
                    "recommendation": "Run with calendar-only first, then add selected external signals only if they improve validation.",
                }
            ]
        )
    known_count = int(external_inventory["usage"].astype(str).str.contains("known_ahead").sum())
    lagged_count = int(len(external_inventory) - known_count)
    return pd.DataFrame(
        [
            {
                "agent": "External Feature Agent",
                "status": "selected_features_available",
                "selected_feature_count": len(external_inventory),
                "known_ahead_feature_count": known_count,
                "lagged_feature_count": lagged_count,
                "finding": f"{len(external_inventory)} selected external features are available.",
                "recommendation": "Keep the selected set small. Compare against calendar-only before adding more weather or annual context.",
            }
        ]
    )


def run_all_agents(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "data_quality": run_data_quality_agent(outputs.get("model_data", pd.DataFrame()), outputs.get("splits", pd.DataFrame())),
        "model_selection": run_model_selection_agent(outputs.get("model_summary", pd.DataFrame())),
        "monthly_total": run_monthly_total_agent(outputs.get("monthly_totals", pd.DataFrame())),
        "occurrence": run_occurrence_agent(outputs.get("forecasts", pd.DataFrame())),
        "external_features": run_external_feature_agent(outputs.get("external_inventory", pd.DataFrame())),
    }


def write_agent_reports(root: Path, reports: dict[str, pd.DataFrame]) -> dict[str, Path]:
    table_dir = Path(root) / "results" / "lumpy_outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, report in reports.items():
        path = table_dir / f"lumpy_agent_{name}.csv"
        report.to_csv(path, index=False)
        paths[name] = path
    return paths
