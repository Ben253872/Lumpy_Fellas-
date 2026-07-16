import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import lumpy_temporal_memory as memory


def sample_errors():
    return pd.DataFrame(
        {
            "horizon_id": ["h1"] * 4 + ["h2"] * 4,
            "sku_id": [1, 1, 2, 2] * 2,
            "candidate_id": ["a", "b", "a", "b"] * 2,
            "wmape": [10, 50, 20, 40, 15, 60, 25, 35],
        }
    )


def test_temporal_selection_prefers_persistent_candidate():
    cohorts = pd.DataFrame({"sku_id": [1, 2], "cohort": ["x", "x"]})
    selected = memory.temporal_selections(sample_errors(), cohorts, ["h1", "h2"])
    assert selected.set_index("sku_id").selected_candidate_id.to_dict() == {1: "a", 2: "a"}


def test_zero_individual_weight_uses_cohort_evidence():
    cohorts = pd.DataFrame({"sku_id": [1, 2], "cohort": ["x", "x"]})
    selected = memory.temporal_selections(sample_errors(), cohorts, ["h1"], individual_weight=0.0)
    assert selected.selected_candidate_id.nunique() == 1


def test_persistence_table_counts_same_winner():
    result = memory.persistence_table(sample_errors())
    assert result.same_exact_winner.iloc[0] == 2
