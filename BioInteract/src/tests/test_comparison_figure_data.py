"""Guard the Table 1 / Figure 1 baseline-comparison correspondence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.generate_comparison_figures import SPLIT_AUROC, SPLIT_METHODS  # noqa: E402


EXPECTED_METHODS = [
    "DeepDTA",
    "GraphDTA",
    "AttentionDTA",
    "MolTrans",
    "TransformerCPI",
    "DrugBAN",
    "BioInteract",
]
EXPECTED_AUROC = np.array(
    [
        [0.878, 0.783, 0.592],
        [0.893, 0.815, 0.621],
        [0.900, 0.838, 0.643],
        [0.907, 0.856, 0.668],
        [0.910, 0.862, 0.672],
        [0.915, 0.874, 0.695],
        [0.904, 0.930, 0.733],
    ]
)


def test_identifier_split_panel_represents_every_table_1_method() -> None:
    """Panel C must contain all Table 1 baseline rows and AUROC values."""
    assert SPLIT_METHODS == EXPECTED_METHODS
    np.testing.assert_allclose(SPLIT_AUROC, EXPECTED_AUROC)


def test_exported_identifier_split_metadata_matches_table_1() -> None:
    """The release metadata must match the rendered Figure 1 Panel C values."""
    metadata_path = PROJECT_ROOT / "results" / "figure_data" / "fig_comparison_splits.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["methods"] == EXPECTED_METHODS
    np.testing.assert_allclose(metadata["auroc_matrix"], EXPECTED_AUROC)
