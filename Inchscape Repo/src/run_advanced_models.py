"""Train, evaluate, and persist advanced forecasting models across all dataset slices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.advanced import _encoded_features, engineer_time_features, lumpy_hurdle_forecast, tree_forecast  # noqa: E402
from models.benchmarks import wmape  # noqa: E402


DEMAND_TYPES = ("smooth", "intermittent", "erratic", "lumpy")
VARIANTS = ("collision_flag_only", "all_sku_history")
MODEL_NAMES = ("xgboost", "lightgbm", "random_forest", "lumpy_hurdle")


def holdout_validation(data: pd.DataFrame, model_name: str, evaluation_months: int) -> pd.DataFrame:
    """Evaluate one advanced model on the final N months (faster than rolling-origin)."""
    months = sorted(pd.to_datetime(data["month"]).unique())
    if len(months) <= evaluation_months:
        return pd.DataFrame(columns=["sku_id", "month", "demand", "forecast"])
    split_month = months[-evaluation_months]
    train = data.loc[data["month"] < split_month].copy()
    actual = data.loc[data["month"] >= split_month].copy()
    if actual.empty or train.empty:
        return pd.DataFrame(columns=["sku_id", "month", "demand", "forecast"])
    if model_name == "lumpy_hurdle":
        predicted = lumpy_hurdle_forecast(train, actual)
    else:
        predicted = tree_forecast(train, actual, model_name)
    return actual[["sku_id", "month", "demand"]].merge(
        predicted.rename(columns={"demand": "forecast"}), on=["sku_id", "month"], how="left"
    ).assign(forecast=lambda frame: frame["forecast"].fillna(0.0))


def _build_tree_estimator(model_type: str):
    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=50,
            learning_rate=0.1,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        )
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(n_estimators=50, learning_rate=0.1, random_state=42, verbose=-1)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1)
    raise ValueError("Unsupported tree model type")


def fit_tree_artifact(data: pd.DataFrame, model_type: str) -> dict:
    """Train one full-history tree model and return an artifact dictionary for persistence."""
    engineered, features = _encoded_features(engineer_time_features(data))
    model = _build_tree_estimator(model_type)
    model.fit(engineered[features].fillna(0), engineered["demand"])
    return {
        "artifact_type": "tree",
        "model_name": model_type,
        "features": features,
        "model": model,
    }


def fit_lumpy_hurdle_artifact(data: pd.DataFrame, threshold: float = 0.5, weight_factor: float = 1.0) -> dict:
    """Train the full-history two-stage model and return an artifact dictionary for persistence."""
    import lightgbm as lgb

    engineered, features = _encoded_features(engineer_time_features(data))
    x_train = engineered[features].fillna(0)
    target = engineered["demand"]
    occurred = target.gt(0).astype(int)
    weight = (len(occurred) - occurred.sum()) / max(occurred.sum(), 1) * weight_factor
    classifier = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        scale_pos_weight=weight,
    ).fit(x_train, occurred)
    positive = target.gt(0)
    regressor = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1).fit(
        x_train.loc[positive], target.loc[positive]
    )
    return {
        "artifact_type": "lumpy_hurdle",
        "model_name": "lumpy_hurdle",
        "features": features,
        "threshold": threshold,
        "weight_factor": weight_factor,
        "classifier": classifier,
        "regressor": regressor,
    }


def load_dataset(variant: str, demand_type: str) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / variant / f"collision_sales_{demand_type}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {path}")
    return pd.read_csv(path, parse_dates=["month"])


def save_results_figure(metrics: pd.DataFrame, figure_path: Path) -> None:
    dataset_order = [f"{variant}:{demand}" for variant in VARIANTS for demand in DEMAND_TYPES]
    pivot = (
        metrics.assign(dataset=lambda frame: frame["variant"] + ":" + frame["demand_type"].str.lower())
        .pivot(index="model", columns="dataset", values="wmape_percent")
        .reindex(index=list(MODEL_NAMES), columns=dataset_order)
    )

    fig, axis = plt.subplots(figsize=(14, 5.5))
    image = axis.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    axis.set_title("Advanced model WMAPE (%) across 8 processed datasets")
    axis.set_xlabel("Dataset (variant:demand type)")
    axis.set_ylabel("Model")
    axis.set_xticks(range(len(pivot.columns)), labels=list(pivot.columns), rotation=35, ha="right")
    axis.set_yticks(range(len(pivot.index)), labels=list(pivot.index))
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            value = pivot.iat[row_idx, col_idx]
            text = "nan" if pd.isna(value) else f"{value:.2f}"
            axis.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="WMAPE (%)")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-months", type=int, default=6, help="Number of final months used for holdout evaluation")
    args = parser.parse_args()

    model_dir = ROOT / "results" / "models" / "advanced"
    table_dir = ROOT / "results" / "tables"
    figure_dir = ROOT / "results" / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    forecast_frames: list[pd.DataFrame] = []

    for variant in VARIANTS:
        for demand_type in DEMAND_TYPES:
            data = load_dataset(variant, demand_type)

            for model_name in MODEL_NAMES:
                evaluation = holdout_validation(data, model_name, evaluation_months=args.evaluation_months)
                metric = wmape(evaluation["month"], evaluation["demand"], evaluation["forecast"], evaluation["sku_id"]) if not evaluation.empty else np.nan
                forecast_frames.append(
                    evaluation.assign(
                        variant=variant,
                        demand_type=demand_type.title(),
                        model=model_name,
                    )
                )
                metrics_rows.append(
                    {
                        "variant": variant,
                        "demand_type": demand_type.title(),
                        "model": model_name,
                        "evaluation_months": args.evaluation_months,
                        "wmape_percent": metric,
                        "forecast_rows": int(len(evaluation)),
                    }
                )

                if model_name == "lumpy_hurdle":
                    artifact = fit_lumpy_hurdle_artifact(data)
                else:
                    artifact = fit_tree_artifact(data, model_name)
                artifact["variant"] = variant
                artifact["demand_type"] = demand_type
                artifact["trained_through_month"] = str(pd.to_datetime(data["month"]).max().date())
                artifact["training_rows"] = int(len(data))
                artifact_path = model_dir / f"{variant}__{demand_type}__{model_name}.joblib"
                joblib.dump(artifact, artifact_path)
                print(f"{variant:20} {demand_type:13} {model_name:14} complete")

    metrics = pd.DataFrame(metrics_rows).sort_values(["variant", "demand_type", "wmape_percent"])
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    metrics_path = table_dir / "advanced_metrics_all_datasets.csv"
    forecasts_path = table_dir / "advanced_forecasts_all_datasets.csv"
    figure_path = figure_dir / "advanced_model_wmape_all_datasets.png"
    metrics.to_csv(metrics_path, index=False)
    forecasts.to_csv(forecasts_path, index=False)
    save_results_figure(metrics, figure_path)

    summary = {
        "datasets_run": len(VARIANTS) * len(DEMAND_TYPES),
        "models_per_dataset": len(MODEL_NAMES),
        "evaluation_months": args.evaluation_months,
        "model_artifacts_saved": len(list(model_dir.glob("*.joblib"))),
        "metrics_table": str(metrics_path.relative_to(ROOT)),
        "forecasts_table": str(forecasts_path.relative_to(ROOT)),
        "figure": str(figure_path.relative_to(ROOT)),
    }
    summary_path = model_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
