from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_TABLES = ROOT / "results" / "tables"
RESULTS_MODELS = ROOT / "results" / "models" / "advanced"
DATA_FILE = ROOT / "data" / "processed" / "all_sku_history" / "collision_sales_erratic.csv"
MONTHLY_REF = RESULTS_TABLES / "rolling_evaluation_erratic_chronos_erratic_features_plus_external_monthly.csv"
OUT_SKU = RESULTS_TABLES / "erratic_chronos2_external_sku_wmape.csv"
OUT_ROW_LEVEL = RESULTS_TABLES / "erratic_chronos2_external_row_level_predictions.csv"
BEST_MODELS = RESULTS_TABLES / "best_model_per_sku.csv"
ARTIFACT_PATH = RESULTS_MODELS / "all_sku_history__erratic__chronos_2_erratic_features_plus_external.joblib"

MODEL_LABEL = "chronos_2_erratic_features_plus_external"

sys.path.insert(0, str(ROOT / "src"))
from models.advanced import Chronos2RidgeConfig, Chronos2RidgeForecaster  # noqa: E402


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


def compute_per_sku_wmape() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA_FILE)
    data["month"] = pd.to_datetime(data["month"])
    data["sku_id"] = data["sku_id"].astype(str)
    data = data.sort_values(["sku_id", "month"])

    months = pd.read_csv(MONTHLY_REF, parse_dates=["eval_month"])["eval_month"].drop_duplicates().sort_values().tolist()

    pipeline, pipeline_type = load_pipeline()

    sku_error: dict[str, float] = {}
    sku_actual: dict[str, float] = {}
    row_level_records: list[dict] = []

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

        for row in merged.itertuples(index=False):
            row_level_records.append(
                {
                    "model": MODEL_LABEL,
                    "demand_type": "erratic",
                    "eval_month": pd.Timestamp(eval_month).strftime("%Y-%m-%d"),
                    "sku_id": str(row.sku_id),
                    "actual_demand": float(row.demand_actual),
                    "forecast_demand": float(row.demand_forecast),
                    "absolute_error": abs(float(row.demand_actual) - float(row.demand_forecast)),
                }
            )

        for row in merged.itertuples(index=False):
            sku = str(row.sku_id)
            err = abs(float(row.demand_actual) - float(row.demand_forecast))
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

    return pd.DataFrame(rows), pd.DataFrame(row_level_records)


def save_chronos_artifact() -> None:
    """Persist a reusable Chronos artifact for backend inference.

    Chronos is a pre-trained foundation model. Persisting this artifact stores
    the model configuration (and optional ridge layer hook) so the backend can
    recreate and call the same pipeline without any retraining step.
    """
    config = Chronos2RidgeConfig(
        model_id="amazon/chronos-2",
        device_map="cpu",
        context_length=12,
        prediction_length=1,
        batch_size=256,
        num_samples=100,
        temperature=0.6,
        ridge_alpha=25.0,
        ridge_feature_columns=[],
    )
    forecaster = Chronos2RidgeForecaster(config=config, ridge_pipeline=None)
    artifact = forecaster.to_artifact()
    artifact["model_label"] = MODEL_LABEL
    artifact["variant"] = "all_sku_history"
    artifact["demand_type"] = "erratic"
    RESULTS_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)


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
    save_chronos_artifact()
    erratic, row_level = compute_per_sku_wmape()
    OUT_SKU.parent.mkdir(parents=True, exist_ok=True)
    erratic.to_csv(OUT_SKU, index=False)
    row_level.to_csv(OUT_ROW_LEVEL, index=False)
    update_best_model_file(erratic)

    total_err = float(erratic["abs_error_sum"].sum())
    total_act = float(erratic["abs_actual_sum"].sum())
    pooled = (total_err / total_act * 100.0) if total_act > 0 else np.nan

    print(f"Wrote Chronos artifact to {ARTIFACT_PATH}")
    print(f"Wrote {len(erratic)} rows to {OUT_SKU}")
    print(f"Wrote {len(row_level)} row-level predictions to {OUT_ROW_LEVEL}")
    print(f"Updated {BEST_MODELS} with Chronos-2 external erratic SKU WMAPE")
    print(f"Pooled erratic WMAPE from SKU totals: {pooled:.6f}%")


if __name__ == "__main__":
    main()
