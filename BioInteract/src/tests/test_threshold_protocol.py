"""Regression tests for validation-only threshold selection."""

import ast
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import classification_metrics, select_f1_threshold  # noqa: E402


ENTRY_POINTS = (
    "src/cli/train.py",
    "src/cli/evaluate.py",
    "src/cli/evaluate_splits.py",
    "src/experiments/run_split.py",
    "src/experiments/run_split_v2.py",
    "src/experiments/run_split_v3.py",
    "src/experiments/run_split_final.py",
)


def test_threshold_selection_is_explicit_and_validation_only():
    validation_labels = np.array([0, 0, 1, 1])
    validation_scores = np.array([0.10, 0.35, 0.60, 0.90])
    test_labels = np.array([0, 1, 0, 1])
    test_scores = np.array([0.20, 0.40, 0.55, 0.95])

    threshold = select_f1_threshold(validation_labels, validation_scores)
    metrics = classification_metrics(test_labels, test_scores, threshold=threshold)

    assert metrics["threshold"] == threshold
    with pytest.raises(ValueError, match="validation-selected threshold"):
        classification_metrics(test_labels, test_scores)


def test_fixed_threshold_is_not_reoptimised_on_test_labels():
    validation_threshold = 0.80
    test_labels = np.array([0, 1, 1, 0])
    test_scores = np.array([0.20, 0.70, 0.90, 0.85])

    metrics = classification_metrics(
        test_labels, test_scores, threshold=validation_threshold
    )

    assert metrics["threshold"] == validation_threshold
    assert metrics["F1"] == pytest.approx(0.5)


@pytest.mark.parametrize("relative_path", ENTRY_POINTS)
def test_entry_points_freeze_validation_threshold_before_metric_reporting(
    relative_path,
):
    """Each supported entry point must pass, not reselect, its threshold."""
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    metric_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "classification_metrics"
    ]

    assert metric_calls
    assert "select_f1_threshold" in source
    assert "validation_threshold" in source
    assert all(
        any(keyword.arg == "threshold" for keyword in call.keywords)
        for call in metric_calls
    )
