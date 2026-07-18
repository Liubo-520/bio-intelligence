"""Unit contracts for canonical entity-split metric artifacts."""

from pathlib import Path
import sys

import numpy as np
import pytest


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from revision.recompute_entity_split_metrics import (  # noqa: E402
    build_artifact_status,
    build_split_record,
)


def test_amp_reevaluation_status_is_explicitly_noncanonical():
    status = build_artifact_status({"training": {"amp": True}})

    assert status["canonical"] is False
    assert status["inference_precision"] == "CUDA AMP autocast (float16)"
    assert "full-precision" in status["reason"]


def test_split_record_freezes_validation_threshold_for_test_metrics():
    """Threshold-dependent test scores must use the supplied validation value."""
    validation_threshold = 0.60
    test_labels = np.array([0, 0, 0, 0, 1])
    test_predictions = np.array([0.20, 0.40, 0.55, 0.65, 0.75])

    record = build_split_record(
        public_name="Random",
        split_key="random",
        checkpoint_path="BioInteract/checkpoints/best_random.pt",
        checkpoint_sha256="checkpoint-sha",
        input_counts={"train": 70, "validation": 10, "test": 5},
        validation_threshold=validation_threshold,
        validation_csv="random_validation_predictions.csv",
        test_csv="random_test_predictions.csv",
        test_labels=test_labels,
        test_predictions=test_predictions,
    )

    assert set(record) == {
        "public_name",
        "split_key",
        "checkpoint",
        "input_counts",
        "validation",
        "test",
    }
    assert record["validation"]["f1_threshold"] == pytest.approx(validation_threshold)
    assert record["validation"]["predictions_csv"] == "random_validation_predictions.csv"
    assert set(record["test"]["threshold_free_metrics"]) == {"AUROC", "AUPRC"}
    assert set(record["test"]["threshold_dependent_metrics"]) == {
        "F1", "Precision", "Recall", "threshold"
    }
    assert record["test"]["threshold_dependent_metrics"]["threshold"] == pytest.approx(
        validation_threshold
    )
    assert record["test"]["threshold_dependent_metrics"]["F1"] == pytest.approx(2 / 3)
