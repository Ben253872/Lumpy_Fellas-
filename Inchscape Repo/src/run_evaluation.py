"""Unified rolling evaluation runner using pooled WMAPE as the primary metric.

This script replaces model/dataset-specific run scripts by exposing one CLI:
- Select dataset directory under data/processed/
- Optionally filter demand types
- Select any mix of tree models and Chronos models
- Evaluate with a fixed forecast lag and rolling monthly test window

Primary metric:
- pooled_wmape = sum(abs(actual - forecast)) / sum(abs(actual)) * 100

Also writes per-SKU pooled WMAPE across the whole evaluation window.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from models.advanced import lumpy_hurdle_forecast, tree_forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified rolling evaluation (pooled WMAPE)")
    parser.add_argument(
        "--dataset",
        default="all_sku_history",
        help="Folder under data/processed to evaluate (default: all_sku_history)",
    )
    parser.add_argument(
        "--demand-types",
        default="all",
        help="Comma-separated demand types to keep (e.g. erratic,lumpy). Default: all",
    )
    parser.add_argument(
        "--models",
        default="xgboost,lightgbm,random_forest,lumpy_hurdle",
        help=(
            "Comma-separated model list. Supported tokens: xgboost, lightgbm, random_forest, "
            "lumpy_hurdle, chronos:<huggingface_model_id>"
        ),
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=18,
        help="Number of last months to evaluate (default: 18)",
    )
    parser.add_argument(
        "--lag-months",
        type=int,
        default=2,
        help="Forecast lag in months (default: 2)",
    )
    parser.add_argument(
        "--max-months",
        type=int,
        default=0,
        help="Optional cap on evaluated months for quick checks (0 means all selected months)",
    )
    parser.add_argument(
        "--output-prefix",
        default="evaluation",
        help="Prefix for result files in results/tables (default: evaluation)",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Chronos 2 batch size")
    parser.add_argument("--num-samples", type=int, default=100, help="Chronos 1 sample count")
    parser.add_argument("--temperature", type=float, default=0.6, help="Chronos 1 temperature")
    return parser.parse_args()


def _normalise_model_name(model_token: str) -> str:
    return model_token.replace("/", "_").replace(":", "__")


def load_processed_data(repo_root: Path, dataset: str) -> pd.DataFrame:
    data_dir = repo_root / "data" / "processed" / dataset
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    chunks: list[pd.DataFrame] = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if not {"sku_id", "month", "demand"}.issubset(df.columns):
            continue
        if "demand_type" not in df.columns:
            df["demand_type"] = csv_file.stem.replace("collision_sales_", "")
        chunks.append(df)

    if not chunks:
        raise ValueError(f"No valid evaluation tables with sku_id/month/demand in {data_dir}")

    data = pd.concat(chunks, ignore_index=True)
    data["month"] = pd.to_datetime(data["month"])
    data = data.sort_values(["sku_id", "month"]).reset_index(drop=True)
    return data


def parse_models(models_arg: str) -> list[str]:
    models = [token.strip() for token in models_arg.split(",") if token.strip()]
    if not models:
        raise ValueError("At least one model must be provided in --models")
    return models


def _load_chronos_pipeline(model_id: str):
    try:
        from chronos import Chronos2Pipeline, ChronosPipeline
    except ImportError as exc:
        raise RuntimeError("chronos-forecasting is not installed") from exc

    try:
        pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")
        return pipeline, "chronos2"
    except Exception:
        pipeline = ChronosPipeline.from_pretrained(model_id, device_map="cpu")
        return pipeline, "chronos1"


def _extract_point_forecast(prediction_output) -> float:
    values = prediction_output.detach().cpu().numpy() if hasattr(prediction_output, "detach") else np.asarray(prediction_output)
    return float(np.median(values))


def _chronos_predict_one_month(
    pipeline,
    pipeline_type: str,
    train_for_forecast: pd.DataFrame,
    actual_for_month: pd.DataFrame,
    batch_size: int,
    num_samples: int,
    temperature: float,
) -> pd.DataFrame:
    histories: list[np.ndarray] = []
    skus: list[str] = []
    fallback_rows: list[dict] = []

    for sku_id in actual_for_month["sku_id"].unique():
        history = train_for_forecast.loc[train_for_forecast["sku_id"] == sku_id, "demand"].to_numpy(dtype=float)
        if len(history) < 3:
            fallback_rows.append(
                {
                    "sku_id": sku_id,
                    "month": actual_for_month["month"].iloc[0],
                    "demand": float(np.mean(history)) if len(history) > 0 else 0.0,
                }
            )
            continue
        skus.append(sku_id)
        histories.append(np.asarray(history[-12:], dtype=np.float32))

    predicted_rows: list[dict] = []
    if skus:
        if pipeline_type == "chronos2":
            outputs = pipeline.predict(histories, prediction_length=1, batch_size=batch_size)
            preds = [_extract_point_forecast(out) for out in outputs]
        else:
            import torch

            contexts = [torch.tensor(h[-12:], dtype=torch.float32) for h in histories]
            outputs = pipeline.predict(
                contexts,
                prediction_length=1,
                num_samples=num_samples,
                temperature=temperature,
            )
            if hasattr(outputs, "shape"):
                rows = [outputs[i] for i in range(outputs.shape[0])]
            else:
                rows = list(outputs)
            preds = [_extract_point_forecast(out) for out in rows]

        for sku_id, pred in zip(skus, preds):
            predicted_rows.append(
                {
                    "sku_id": sku_id,
                    "month": actual_for_month["month"].iloc[0],
                    "demand": max(float(pred), 0.0),
                }
            )

    return pd.DataFrame(predicted_rows + fallback_rows)


def forecast_one_month(
    model_token: str,
    train_for_forecast: pd.DataFrame,
    actual_for_month: pd.DataFrame,
    chronos_cache: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    forecast_df = pd.DataFrame({
        "sku_id": actual_for_month["sku_id"].unique(),
        "month": actual_for_month["month"].iloc[0],
    })

    if model_token in {"xgboost", "lightgbm", "random_forest"}:
        return tree_forecast(train_for_forecast, forecast_df, model_type=model_token)

    if model_token == "lumpy_hurdle":
        return lumpy_hurdle_forecast(train_for_forecast, forecast_df)

    if model_token.startswith("chronos:"):
        model_id = model_token.split(":", 1)[1].strip()
        if not model_id:
            raise ValueError("chronos model token must be chronos:<huggingface_model_id>")

        if model_id not in chronos_cache:
            pipeline, pipeline_type = _load_chronos_pipeline(model_id)
            chronos_cache[model_id] = {"pipeline": pipeline, "pipeline_type": pipeline_type}

        entry = chronos_cache[model_id]
        return _chronos_predict_one_month(
            pipeline=entry["pipeline"],
            pipeline_type=entry["pipeline_type"],
            train_for_forecast=train_for_forecast,
            actual_for_month=actual_for_month,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            temperature=args.temperature,
        )

    raise ValueError(f"Unsupported model token: {model_token}")


def evaluate_model(
    model_token: str,
    data: pd.DataFrame,
    test_months: list[pd.Timestamp],
    lag_months: int,
    chronos_cache: dict,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    sku_stats: dict[str, dict[str, float]] = {}

    for month_idx, eval_month in enumerate(test_months, start=1):
        feature_cutoff = eval_month - pd.DateOffset(months=lag_months)
        train_for_forecast = data[data["month"] <= feature_cutoff].copy()
        actual_for_month = data[data["month"] == eval_month].copy()

        if train_for_forecast.empty or actual_for_month.empty:
            continue

        preds = forecast_one_month(model_token, train_for_forecast, actual_for_month, chronos_cache, args)
        if preds.empty:
            continue

        merged = actual_for_month.merge(
            preds[["sku_id", "month", "demand"]],
            on=["sku_id", "month"],
            how="left",
            suffixes=("_actual", "_forecast"),
        )
        merged["demand_forecast"] = merged["demand_forecast"].fillna(0.0).clip(lower=0.0)

        abs_error = (merged["demand_actual"] - merged["demand_forecast"]).abs()
        abs_actual = merged["demand_actual"].abs()

        month_pooled = (abs_error.sum() / abs_actual.sum() * 100) if abs_actual.sum() > 0 else np.nan
        rows.append(
            {
                "model": model_token,
                "eval_month": eval_month,
                "month_num": month_idx,
                "pooled_wmape": month_pooled,
                "abs_error_sum": float(abs_error.sum()),
                "abs_actual_sum": float(abs_actual.sum()),
                "evaluation_rows": int(len(merged)),
                "unique_skus": int(merged["sku_id"].nunique()),
            }
        )

        for sku_id, group in merged.groupby("sku_id"):
            sku_abs_error = float((group["demand_actual"] - group["demand_forecast"]).abs().sum())
            sku_abs_actual = float(group["demand_actual"].abs().sum())
            if sku_id not in sku_stats:
                sku_stats[sku_id] = {"abs_error_sum": 0.0, "abs_actual_sum": 0.0}
            sku_stats[sku_id]["abs_error_sum"] += sku_abs_error
            sku_stats[sku_id]["abs_actual_sum"] += sku_abs_actual

    monthly_df = pd.DataFrame(rows)
    return monthly_df, sku_stats


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).parent.parent

    data = load_processed_data(repo_root, args.dataset)
    if args.demand_types.lower() != "all":
        keep = {value.strip().lower() for value in args.demand_types.split(",") if value.strip()}
        data = data[data["demand_type"].str.lower().isin(keep)].copy()

    all_months = sorted(data["month"].dropna().unique())
    if len(all_months) < args.test_months + args.lag_months:
        raise ValueError(
            "Not enough monthly history for requested test window and lag. "
            f"Need at least {args.test_months + args.lag_months} unique months, got {len(all_months)}."
        )

    test_months = all_months[-args.test_months :]
    if args.max_months and args.max_months > 0:
        test_months = test_months[: args.max_months]

    models = parse_models(args.models)
    chronos_cache: dict = {}

    print("=" * 88)
    print("UNIFIED ROLLING EVALUATION")
    print("=" * 88)
    print(f"Dataset: {args.dataset}")
    print(f"Demand types: {args.demand_types}")
    print(f"Models: {', '.join(models)}")
    print(f"Lag months: {args.lag_months}")
    print(f"Test months: {len(test_months)} ({test_months[0].strftime('%Y-%m')} to {test_months[-1].strftime('%Y-%m')})")
    print(f"Rows in scope: {len(data):,}")

    all_monthly: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    all_sku_rows: list[dict] = []

    for model_token in models:
        print("\n" + "-" * 88)
        print(f"Evaluating {model_token}")
        print("-" * 88)

        monthly_df, sku_stats = evaluate_model(
            model_token=model_token,
            data=data,
            test_months=test_months,
            lag_months=args.lag_months,
            chronos_cache=chronos_cache,
            args=args,
        )

        if monthly_df.empty:
            print("No evaluation rows produced.")
            continue

        total_abs_error = float(monthly_df["abs_error_sum"].sum())
        total_abs_actual = float(monthly_df["abs_actual_sum"].sum())
        pooled_wmape = (total_abs_error / total_abs_actual * 100) if total_abs_actual > 0 else np.nan

        summary_rows.append(
            {
                "model": model_token,
                "pooled_wmape": pooled_wmape,
                "num_months": int(monthly_df["eval_month"].nunique()),
                "num_rows": int(monthly_df["evaluation_rows"].sum()),
                "num_skus": int(monthly_df["unique_skus"].max()),
                "abs_error_sum": total_abs_error,
                "abs_actual_sum": total_abs_actual,
            }
        )

        for sku_id, stats in sku_stats.items():
            denom = stats["abs_actual_sum"]
            sku_wmape = float(stats["abs_error_sum"] / denom * 100) if denom > 0 else np.nan
            all_sku_rows.append(
                {
                    "model": model_token,
                    "sku_id": sku_id,
                    "sku_wmape": sku_wmape,
                    "abs_error_sum": stats["abs_error_sum"],
                    "abs_actual_sum": stats["abs_actual_sum"],
                }
            )

        all_monthly.append(monthly_df)
        print(f"pooled_wmape = {pooled_wmape:.2f}%")

    results_dir = repo_root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    if all_monthly:
        monthly_out = pd.concat(all_monthly, ignore_index=True)
        monthly_out.to_csv(results_dir / f"{args.output_prefix}_monthly.csv", index=False)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("pooled_wmape", ascending=True)
        summary_df.to_csv(results_dir / f"{args.output_prefix}_summary.csv", index=False)
        print("\nSummary:")
        print(summary_df[["model", "pooled_wmape", "num_months"]].to_string(index=False))

    if all_sku_rows:
        sku_df = pd.DataFrame(all_sku_rows)
        sku_df.to_csv(results_dir / f"{args.output_prefix}_sku_wmape.csv", index=False)

    print("\nSaved files:")
    print(f"- results/tables/{args.output_prefix}_summary.csv")
    print(f"- results/tables/{args.output_prefix}_monthly.csv")
    print(f"- results/tables/{args.output_prefix}_sku_wmape.csv")


if __name__ == "__main__":
    main()
