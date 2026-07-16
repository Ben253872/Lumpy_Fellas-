from __future__ import annotations

import numpy as np
import pandas as pd


def compose_rare_forecast(
    frame: pd.DataFrame,
    probability: np.ndarray,
    size: np.ndarray,
    mode: str,
    scale: float = 1.0,
    horizon_threshold: float = 0.0,
) -> np.ndarray:
    working = frame[["sku_id", "block_number"]].copy()
    working["probability"] = np.asarray(probability, dtype=float).clip(0.0, 1.0)
    working["size"] = np.asarray(size, dtype=float).clip(0.0)
    if mode == "zero":
        values = np.zeros(len(working))
    elif mode == "expected":
        values = working.probability * working["size"]
    else:
        working["rank"] = working.groupby("sku_id").probability.rank(method="first", ascending=False)
        top_count = 2 if "top2" in mode else 1
        selected = working["rank"].le(top_count)
        if mode in {"top1_expected", "top2_expected"}:
            values = np.where(selected, working.probability * working["size"], 0.0)
        elif mode in {"top1_full", "top2_full"}:
            values = np.where(selected, working["size"], 0.0)
        elif mode in {"horizon_gate_expected", "horizon_gate_full"}:
            no_event = working.groupby("sku_id").probability.transform(lambda values: float(np.prod(1.0 - values)))
            horizon_probability = 1.0 - no_event
            gate = selected & horizon_probability.ge(float(horizon_threshold))
            base = working.probability * working["size"] if mode.endswith("expected") else working["size"]
            values = np.where(gate, base, 0.0)
        elif mode == "normalised_one":
            total = working.groupby("sku_id").probability.transform("sum").replace(0.0, np.nan)
            adjusted = (working.probability / total).fillna(0.0).clip(0.0, 1.0)
            values = adjusted * working["size"]
        else:
            raise ValueError(f"Unknown rare-event mode: {mode}")
    cap = frame["cap"].to_numpy(float) if "cap" in frame else np.repeat(np.inf, len(frame))
    return np.minimum(np.maximum(0.0, np.asarray(values, dtype=float) * float(scale)), cap)


def rare_recipe_grid() -> pd.DataFrame:
    rows = [{"mode": "zero", "scale": 1.0, "horizon_threshold": 0.0}]
    for mode in ("expected", "top1_expected", "top1_full", "top2_expected", "top2_full", "normalised_one"):
        for scale in (0.5, 0.75, 1.0, 1.25):
            rows.append({"mode": mode, "scale": scale, "horizon_threshold": 0.0})
    for mode in ("horizon_gate_expected", "horizon_gate_full"):
        for threshold in (0.25, 0.50, 0.75):
            for scale in (0.5, 0.75, 1.0):
                rows.append({"mode": mode, "scale": scale, "horizon_threshold": threshold})
    result = pd.DataFrame(rows)
    result["recipe_id"] = result.apply(
        lambda row: f"{row['mode']}__h{row['horizon_threshold']:.2f}__s{row['scale']:.2f}", axis=1
    )
    return result.drop_duplicates("recipe_id").reset_index(drop=True)


def occurrence_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame.target.gt(0)
    predicted = frame.forecast.gt(0)
    true_positive = int((actual & predicted).sum())
    actual_positive = int(actual.sum())
    predicted_positive = int(predicted.sum())
    return {
        "actual_positive_blocks": actual_positive,
        "forecast_positive_blocks": predicted_positive,
        "true_positive_blocks": true_positive,
        "event_recall": true_positive / actual_positive if actual_positive else np.nan,
        "event_precision": true_positive / predicted_positive if predicted_positive else np.nan,
        "false_positive_blocks": int((~actual & predicted).sum()),
    }
