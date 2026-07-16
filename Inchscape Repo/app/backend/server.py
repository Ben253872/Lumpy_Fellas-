from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "app" / "frontend"
RESULTS_DIR = ROOT / "results"
PROCESSED_DIR = ROOT / "data" / "processed"
VARIANTS = ("all_sku_history", "collision_flag_only")
FUTURE_HORIZON_MONTHS = 18

sys.path.insert(0, str(ROOT / "src"))
from models.advanced import _encoded_features, engineer_time_features  # noqa: E402


class ForecastPoint(BaseModel):
    month: str
    actual_demand: float
    forecast: float


class FutureForecastPoint(BaseModel):
    month: str
    forecast: float


class VariantResult(BaseModel):
    variant: str
    demand_type: str
    selected_model: str | None
    wmape_percent: float | None
    prescribed_forecast: ForecastPoint | None
    forecast_history: list[ForecastPoint]
    future_predictions: list[FutureForecastPoint]
    future_is_flat: bool


class SkuResponse(BaseModel):
    sku_id: str
    results: list[VariantResult]


class ForecastService:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.metrics = self._load_metrics()
        self.forecasts = self._load_forecasts()
        self.best_models = self._build_best_model_lookup()
        self.profiles = self._load_profiles()
        self.dataset_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self.artifact_cache: dict[tuple[str, str, str], dict] = {}

    def _load_metrics(self) -> pd.DataFrame:
        path = self.results_dir / "tables" / "advanced_metrics_all_datasets.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        metrics = pd.read_csv(path)
        metrics["demand_type"] = metrics["demand_type"].astype(str).str.title()
        return metrics

    def _load_forecasts(self) -> pd.DataFrame:
        path = self.results_dir / "tables" / "advanced_forecasts_all_datasets.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing forecasts file: {path}")
        forecasts = pd.read_csv(path, parse_dates=["month"])
        forecasts["demand_type"] = forecasts["demand_type"].astype(str).str.title()
        forecasts["sku_id"] = forecasts["sku_id"].astype(str)
        return forecasts

    def _build_best_model_lookup(self) -> dict[tuple[str, str], dict[str, float | str]]:
        valid = self.metrics.dropna(subset=["wmape_percent"]).copy()
        chosen = (
            valid.sort_values(["variant", "demand_type", "wmape_percent"])  # lowest WMAPE wins
            .groupby(["variant", "demand_type"], as_index=False)
            .first()
        )
        best: dict[tuple[str, str], dict[str, float | str]] = {}
        for row in chosen.to_dict("records"):
            best[(row["variant"], row["demand_type"])] = {
                "model": str(row["model"]),
                "wmape_percent": float(row["wmape_percent"]),
            }
        return best

    def _load_profiles(self) -> dict[str, pd.DataFrame]:
        profiles: dict[str, pd.DataFrame] = {}
        for variant in VARIANTS:
            path = self.results_dir / variant / "tables" / "sku_demand_profile.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing SKU profile file: {path}")
            profile = pd.read_csv(path)
            profile["sku_id"] = profile["sku_id"].astype(str)
            profile["demand_type"] = profile["demand_type"].astype(str).str.title()
            profiles[variant] = profile.set_index("sku_id")
        return profiles

    def _dataset_for_variant_type(self, variant: str, demand_type: str) -> pd.DataFrame:
        key = (variant, demand_type)
        if key in self.dataset_cache:
            return self.dataset_cache[key]
        path = PROCESSED_DIR / variant / f"collision_sales_{demand_type.lower()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed demand file: {path}")
        data = pd.read_csv(path, parse_dates=["month"])
        data["sku_id"] = data["sku_id"].astype(str)
        self.dataset_cache[key] = data
        return data

    def _artifact_for_assignment(self, variant: str, demand_type: str, model_name: str) -> dict:
        key = (variant, demand_type, model_name)
        if key in self.artifact_cache:
            return self.artifact_cache[key]
        path = self.results_dir / "models" / "advanced" / f"{variant}__{demand_type.lower()}__{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        artifact = joblib.load(path)
        self.artifact_cache[key] = artifact
        return artifact

    @staticmethod
    def _predict_from_artifact(artifact: dict, features_row: pd.DataFrame) -> float:
        if artifact.get("artifact_type") == "lumpy_hurdle":
            threshold = float(artifact.get("threshold", 0.5))
            probability = float(artifact["classifier"].predict_proba(features_row)[:, 1][0])
            if probability < threshold:
                return 0.0
            prediction = float(artifact["regressor"].predict(features_row)[0])
            return max(0.0, prediction)
        prediction = float(artifact["model"].predict(features_row)[0])
        return max(0.0, prediction)

    def _forecast_future(
        self,
        sku_id: str,
        variant: str,
        demand_type: str,
        model_name: str,
        horizon_months: int = FUTURE_HORIZON_MONTHS,
    ) -> list[FutureForecastPoint]:
        data = self._dataset_for_variant_type(variant, demand_type)
        sku_history = data.loc[data["sku_id"] == sku_id, ["sku_id", "month", "demand"]].copy().sort_values("month")
        if sku_history.empty:
            return []

        artifact = self._artifact_for_assignment(variant, demand_type, model_name)
        feature_names = artifact["features"]
        running = sku_history.copy()
        future_points: list[FutureForecastPoint] = []

        for _ in range(horizon_months):
            next_month = pd.Timestamp(running["month"].max()) + pd.DateOffset(months=1)
            pending = pd.DataFrame([{"sku_id": sku_id, "month": next_month, "demand": np.nan}])
            candidate = pd.concat([running, pending], ignore_index=True)
            engineered, _ = _encoded_features(engineer_time_features(candidate))
            row = engineered.tail(1)
            features = row.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)
            prediction = self._predict_from_artifact(artifact, features)
            running = pd.concat(
                [
                    running,
                    pd.DataFrame([{"sku_id": sku_id, "month": next_month, "demand": prediction}]),
                ],
                ignore_index=True,
            )
            future_points.append(FutureForecastPoint(month=next_month.strftime("%Y-%m-%d"), forecast=float(prediction)))

        return future_points

    def lookup_sku(self, sku_id: str) -> SkuResponse:
        sku = str(sku_id).strip()
        if not sku:
            raise ValueError("SKU cannot be empty")

        variant_results: list[VariantResult] = []

        for variant in VARIANTS:
            profile = self.profiles[variant]
            if sku not in profile.index:
                continue

            demand_type = str(profile.at[sku, "demand_type"])
            assignment = self.best_models.get((variant, demand_type))
            selected_model = None if assignment is None else str(assignment["model"])
            wmape_value = None if assignment is None else float(assignment["wmape_percent"])

            history: list[ForecastPoint] = []
            prescribed: ForecastPoint | None = None
            future_predictions: list[FutureForecastPoint] = []
            future_is_flat = False

            if selected_model is not None:
                rows = self.forecasts.loc[
                    (self.forecasts["sku_id"] == sku)
                    & (self.forecasts["variant"] == variant)
                    & (self.forecasts["demand_type"] == demand_type)
                    & (self.forecasts["model"] == selected_model)
                ].sort_values("month")

                history = [
                    ForecastPoint(
                        month=row["month"].strftime("%Y-%m-%d"),
                        actual_demand=float(row["demand"]),
                        forecast=float(row["forecast"]),
                    )
                    for _, row in rows.iterrows()
                ]
                if history:
                    prescribed = history[-1]
                future_predictions = self._forecast_future(
                    sku_id=sku,
                    variant=variant,
                    demand_type=demand_type,
                    model_name=selected_model,
                )
                if future_predictions:
                    values = [point.forecast for point in future_predictions]
                    future_is_flat = bool(max(values) - min(values) < 1e-9)

            variant_results.append(
                VariantResult(
                    variant=variant,
                    demand_type=demand_type,
                    selected_model=selected_model,
                    wmape_percent=wmape_value,
                    prescribed_forecast=prescribed,
                    forecast_history=history,
                    future_predictions=future_predictions,
                    future_is_flat=future_is_flat,
                )
            )

        if not variant_results:
            raise KeyError(f"SKU not found in any processed variant: {sku}")

        return SkuResponse(sku_id=sku, results=variant_results)


service = ForecastService(RESULTS_DIR)
app = FastAPI(title="Inchscape SKU Forecast Explorer", version="1.0.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sku/{sku_id}", response_model=SkuResponse)
def get_sku(sku_id: str) -> SkuResponse:
    try:
        return service.lookup_sku(sku_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
