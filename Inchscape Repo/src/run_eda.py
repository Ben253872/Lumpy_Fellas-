"""Create CSV analysis tables and labelled figures from processed collision sales."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEMAND_TYPES = ["Smooth", "Intermittent", "Erratic", "Lumpy"]


def read_segments(processed_dir: Path) -> pd.DataFrame:
    frames = []
    for demand_type in DEMAND_TYPES:
        filename = f"collision_sales_{demand_type.lower()}.csv"
        path = processed_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run `python src/prepare_data.py` first.")
        frames.append(pd.read_csv(path, parse_dates=["month"]))
    return pd.concat(frames, ignore_index=True)


def save_figure(name: str, figure_dir: Path) -> None:
    plt.tight_layout()
    plt.savefig(figure_dir / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["collision_flag_only", "all_sku_history"], default="collision_flag_only")
    args = parser.parse_args()
    processed_dir = ROOT / "data" / "processed" / args.variant
    table_dir = ROOT / "results" / args.variant / "tables"
    figure_dir = ROOT / "results" / args.variant / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    sales = read_segments(processed_dir)

    profile = pd.read_csv(table_dir / "sku_demand_profile.csv")
    type_summary = (
        profile.loc[profile["demand_type"].isin(DEMAND_TYPES)]
        .groupby("demand_type", as_index=False)
        .agg(
            number_of_skus=("sku_id", "nunique"),
            total_demand=("total_demand", "sum"),
            median_zero_month_share=("zero_month_share", "median"),
            median_average_demand_interval=("average_demand_interval", "median"),
        )
    )
    type_summary["sku_share"] = type_summary["number_of_skus"] / type_summary["number_of_skus"].sum()
    type_summary["demand_share"] = type_summary["total_demand"] / type_summary["total_demand"].sum()
    type_summary = type_summary.set_index("demand_type").reindex(DEMAND_TYPES).reset_index()
    type_summary.to_csv(table_dir / "demand_type_summary.csv", index=False)

    monthly_total = (
        sales.groupby("month", as_index=False)
        .agg(total_demand=("demand", "sum"), active_skus=("demand", lambda values: (values > 0).sum()))
        .sort_values("month")
    )
    monthly_total.to_csv(table_dir / "monthly_total_demand.csv", index=False)
    monthly_by_type = (
        sales.groupby(["month", "demand_type"], as_index=False)
        .agg(total_demand=("demand", "sum"), active_skus=("demand", lambda values: (values > 0).sum()))
        .sort_values(["month", "demand_type"])
    )
    monthly_by_type.to_csv(table_dir / "monthly_demand_by_type.csv", index=False)

    examples = (
        profile.loc[profile["demand_type"].isin(DEMAND_TYPES)]
        .sort_values(["demand_type", "total_demand"], ascending=[True, False])
        .groupby("demand_type", as_index=False)
        .head(2)
        [["sku_id", "demand_type", "total_demand", "total_months", "months_with_demand"]]
    )
    examples.to_csv(table_dir / "example_skus.csv", index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(monthly_total["month"], monthly_total["total_demand"], color="#1f77b4", linewidth=2)
    plt.title("Total monthly demand for collision parts")
    plt.xlabel("Month")
    plt.ylabel("Units demanded")
    plt.grid(axis="y", alpha=0.25)
    save_figure("total_monthly_collision_demand.png", figure_dir)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(type_summary["demand_type"], type_summary["number_of_skus"], color="#4c78a8")
    plt.title("Collision SKUs by demand type")
    plt.xlabel("Demand type")
    plt.ylabel("Number of SKUs")
    for bar, value in zip(bars, type_summary["number_of_skus"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.0f}", ha="center", va="bottom")
    save_figure("collision_skus_by_demand_type.png", figure_dir)

    plt.figure(figsize=(11, 5))
    for demand_type in DEMAND_TYPES:
        subset = monthly_by_type.loc[monthly_by_type["demand_type"].eq(demand_type)]
        plt.plot(subset["month"], subset["total_demand"], linewidth=1.8, label=demand_type)
    plt.title("Monthly collision demand by demand type")
    plt.xlabel("Month")
    plt.ylabel("Units demanded")
    plt.legend(title="Demand type", ncols=2)
    plt.grid(axis="y", alpha=0.25)
    save_figure("monthly_collision_demand_by_type.png", figure_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, demand_type in zip(axes.ravel(), DEMAND_TYPES):
        selected = examples.loc[examples["demand_type"].eq(demand_type), "sku_id"]
        for sku_id in selected:
            series = sales.loc[sales["sku_id"].eq(sku_id)].sort_values("month")
            axis.plot(series["month"], series["demand"], marker="o", markersize=2.5, label=sku_id)
        axis.set_title(f"{demand_type} demand examples")
        axis.set_xlabel("Month")
        axis.set_ylabel("Units demanded")
        axis.legend(title="SKU", fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    save_figure("demand_type_examples.png", figure_dir)

    print(f"Created {len(list(table_dir.glob('*.csv')))} CSV tables and {len(list(figure_dir.glob('*.png')))} figures.")


if __name__ == "__main__":
    main()
