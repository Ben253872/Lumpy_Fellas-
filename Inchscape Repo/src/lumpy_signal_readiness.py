from __future__ import annotations

import pandas as pd


def readiness_table(columns: list[str]) -> pd.DataFrame:
    available = set(columns)
    requirements = [
        ("stock_history", ["STOCK_END_MONTH", "STOCK_START_MONTH", "NEW_ENTRY_STOCK"], "distinguish observed zero demand from potential stock constraint", "already_tested"),
        ("product_hierarchy", ["FAMILY_DESCRIPTION", "SUBFAMILY_DESCRIPTION", "MATERIAL_DESCRIPTION"], "peer grouping and product similarity", "already_tested"),
        ("supersession_chain", ["SUPERSESSION_FROM", "SUPERSESSION_TO"], "transfer demand between replaced and replacement parts", "required_new_signal"),
        ("vehicle_fitment", ["VEHICLE_MODEL", "MODEL_YEAR", "FITMENT"], "link collision exposure to compatible parts", "required_new_signal"),
        ("quote_order_pipeline", ["QUOTE_DATE", "OPEN_ORDER_QTY"], "observe demand before invoiced sales", "required_new_signal"),
        ("backorder_lost_sales", ["BACKORDER_QTY", "LOST_SALES_QTY"], "separate no demand from unfulfilled demand", "required_new_signal"),
        ("lead_time_supplier", ["SUPPLIER_LEAD_TIME", "SUPPLIER_AVAILABILITY"], "convert uncertainty into replenishment policy", "required_new_signal"),
        ("lifecycle_dates", ["LAUNCH_DATE", "DISCONTINUE_DATE"], "identify launch, decline and obsolescence transitions", "required_new_signal"),
    ]
    rows = []
    for signal, required, purpose, recommendation in requirements:
        present = [column for column in required if column in available]
        rows.append({"signal": signal, "required_columns": ", ".join(required), "present_columns": ", ".join(present), "column_coverage": len(present)/len(required), "purpose": purpose, "recommendation": recommendation, "ready": len(present)==len(required)})
    return pd.DataFrame(rows)


def oracle_gap(champion_below_70: int, oracle_below_70: int, positive_skus: int) -> pd.DataFrame:
    return pd.DataFrame([{"positive_skus": positive_skus, "champion_below_70": champion_below_70, "oracle_below_70": oracle_below_70, "absolute_headroom": oracle_below_70-champion_below_70, "champion_share": champion_below_70/positive_skus, "oracle_share": oracle_below_70/positive_skus}])
