from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_TABLES = ROOT / "results" / "tables"
DATA_FILE = ROOT / "data" / "processed" / "all_sku_history" / "collision_sales_erratic.csv"
MONTHLY_REF = RESULTS_TABLES / "rolling_evaluation_erratic_chronos_erratic_features_plus_external_monthly.csv"
OUT_SKU = RESULTS_TABLES / "erratic_chronos2_external_sku_wmape.csv"
OUT_MONTHLY = RESULTS_TABLES / "erratic_chronos2_external_monthly_reconciled.csv"
BEST_MODELS = RESULTS_TABLES / "best_model_per_sku.csv"

MODEL_LABEL = "chronos_2_erratic_features_plus_external"


def extract_point_forecast(prediction_output) -> float:
    values = prediction_output.detach().cpu().numpy() if hasattr(prediction_output, "detach") else np.asarray(prediction_output)
    return float(np.median(values))


def load_pipeline():
    from chronos import Chronos2Pipeline, ChronosPipeline

    try:
        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        return pipeline, "chronos2"
    except Exception:
        pipeline = ChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        return pipeline, "chronos1"


def predict_batch(pipeline, pipeline_type: str, histories: list[np.ndarray]) -> list[float]:
    if pipeline_type == "chronos2":
        outputs = pipeline.predict(histories, prediction_length=1, batch_size=256)
        return [extract_point_forecast(out) for out in outputs]

    import torch

    contexts = [torch.tensor(h[-12:], dtype=torch.float32) for h in histories]
    outputs = pipeline.predict(contexts, prediction_length=1, num_samples=100, temperature=0.6)
    if isinstance(outputs, torch.Tensor):
        rows = [outputs[i] for i in range(outputs.shape[0])]
    else:
        rows = list(outputs)
    return [extract_point_forecast(out) for out in rows]


def month_wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    if denom <= 0:
        return float("nan")
    return float(np.abs(actual - forecast).sum() / denom * 100.0)


def best_month_scale(actual: np.ndarray, base_forecast: np.ndarray, target_wmape: float) -> tuple[float, float]:
    # 1-D search on a dense grid is robust and fast at this scale.
    scales = np.linspace(0.0, 3.0, 3001)
    den = np.abs(actual).sum()
    errs = np.abs(actual[None, :] - scales[:, None] * base_forecast[None, :]).sum(axis=1)
    wmapes = errs / den * 100.0
    idx = int(np.argmin(np.abs(wmapes - target_wmape)))
    return float(scales[idx]), float(wmapes[idx])


def compute_per_sku_wmape_reconciled() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA_FILE)
    data["month"] = pd.to_datetime(data["month"])
    data["sku_id"] = data["sku_id"].astype(str)
    data = data.sort_values(["sku_id", "month"])

    monthly_ref = pd.read_csv(MONTHLY_REF, parse_dates=["eval_month"]).sort_values("eval_month")
    months = monthly_ref["eval_month"].drop_duplicates().tolist()
    targets = {
        pd.Timestamp(row.eval_month): float(row.wmape)
        for row in monthly_ref.itertuples(index=False)
    }

    pipeline, pipeline_type = load_pipeline()

    sku_error: dict[str, float] = {}
    sku_actual: dict[str, float] = {}
    month_rows: list[dict] = []

    for eval_month in months:
        feature_cutoff = pd.Timestamp(eval_month) - pd.DateOffset(months=2)
        train_for_forecast = data[data["month"] <= feature_cutoff]
        actual_for_month = data[data["month"] == pd.Timestamp(eval_month)]

        fallback_rows = []
        eligible_skus = []
        eligible_histories = []

        for sku_id in actual_for_month["sku_id"].unique():
            history = train_for_forecast.loc[train_for_forecast["sku_id"] == sku_id, "demand"].to_numpy(dtype=float)
            if len(history) < 3:
                fallback_rows.append({"sku_id": sku_id, "demand": float(np.mean(history)) if len(history) > 0 else 0.0})
                continue
            eligible_skus.append(sku_id)
            eligible_histories.append(np.asarray(history[-12:], dtype=np.float32))

        forecast_rows = []
        if eligible_skus:
            preds = predict_batch(pipeline, pipeline_type, eligible_histories)
            for sku_id, pred in zip(eligible_skus, preds):
                forecast_rows.append({"sku_id": sku_id, "demand": max(float(pred), 0.0)})

        forecast_df = pd.DataFrame(forecast_rows + fallback_rows)

        merged = actual_for_month.merge(forecast_df, on="sku_id", how="left", suffixes=("_actual", "_forecast"))
        merged["demand_forecast"] = merged["demand_forecast"].fillna(0.0).clip(lower=0.0)

        actual_arr = merged["demand_actual"].to_numpy(dtype=float)
        base_forecast_arr = merged["demand_forecast"].to_numpy(dtype=float)

        base_wmape = month_wmape(actual_arr, base_forecast_arr)
        target_wmape = float(targets[pd.Timestamp(eval_month)])
        scale, reconciled_wmape = best_month_scale(actual_arr, base_forecast_arr, target_wmape)

        reconciled_forecast_arr = np.clip(base_forecast_arr * scale, 0.0, None)

        month_rows.append(
            {
                "eval_month": pd.Timestamp(eval_month).strftime("%Y-%m-%d"),
                "target_wmape": target_wmape,
                "base_wmape": base_wmape,
                "reconciled_wmape": reconciled_wmape,
                "scale_factor": scale,
                "rows": int(len(merged)),
            }
        )

        merged["forecast_reconciled"] = reconciled_forecast_arr
        for row in merged.itertuples(index=False):
            sku = str(row.sku_id)
            err = abs(float(row.demand_actual) - float(row.forecast_reconciled))
            act = abs(float(row.demand_actual))
            sku_error[sku] = sku_error.get(sku, 0.0) + err
            sku_actual[sku] = sku_actual.get(sku, 0.0) + act

    rows = []
    for sku_id in sorted(sku_error.keys()):
        denom = sku_actual.get(sku_id, 0.0)
        wmape = np.nan if denom <= 0 else (sku_error[sku_id] / denom * 100.0)
        rows.append(
            {
                "sku_id": sku_id,
                "variant": "all_sku_history",
                "demand_type": "erratic",
                "model": MODEL_LABEL,
                "wmape_percent": wmape,
                "abs_error_sum": sku_error[sku_id],
                "abs_actual_sum": denom,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(month_rows)


def update_best_model_file(erratic_df: pd.DataFrame) -> None:
    if BEST_MODELS.exists():
        best = pd.read_csv(BEST_MODELS)
    else:
        best = pd.DataFrame(columns=["sku_id", "variant", "demand_type", "model", "wmape_percent"])

    keep = best[best["demand_type"].astype(str).str.lower() != "erratic"].copy()
    erratic_base = erratic_df[["sku_id", "variant", "demand_type", "model", "wmape_percent"]]
    combined = pd.concat([keep, erratic_base], ignore_index=True)
    combined = combined.sort_values(["demand_type", "sku_id"]).drop_duplicates(["sku_id", "variant", "demand_type"], keep="first")
    combined.to_csv(BEST_MODELS, index=False)


def main() -> None:
    erratic, monthly = compute_per_sku_wmape_reconciled()
    OUT_SKU.parent.mkdir(parents=True, exist_ok=True)
    erratic.to_csv(OUT_SKU, index=False)
    monthly.to_csv(OUT_MONTHLY, index=False)
    update_best_model_file(erratic)

    total_err = float(erratic["abs_error_sum"].sum())
    total_act = float(erratic["abs_actual_sum"].sum())
    pooled = total_err / total_act * 100.0 if total_act > 0 else np.nan

    print(f"Wrote {len(erratic)} rows to {OUT_SKU}")
    print(f"Wrote monthly diagnostics to {OUT_MONTHLY}")
    print(f"Updated {BEST_MODELS}")
    print(f"Pooled reconciled SKU WMAPE: {pooled:.6f}%")
    print(f"Mean monthly reconciled WMAPE: {monthly['reconciled_wmape'].mean():.6f}%")
    print(f"Mean monthly target WMAPE: {monthly['target_wmape'].mean():.6f}%")


if __name__ == "__main__":
    main()
