"""Unit contracts for canonical entity-split metric artifacts."""

from pathlib import Path
import inspect
import sys

import numpy as np
import pytest
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from revision.recompute_entity_split_metrics import (  # noqa: E402
    build_artifact_status,
    build_split_record,
    collect_full_precision_predictions,
)


class _FixedLogitModel(torch.nn.Module):
    def forward(self, drug_batch, *args):
        return drug_batch


def test_full_precision_collector_does_not_depend_on_cli_evaluator_or_amp():
    script_source = inspect.getsource(sys.modules[build_artifact_status.__module__])
    assert "src.cli.evaluate" not in script_source
    assert "autocast(" not in script_source

    model = _FixedLogitModel()
    model.train()
    batch = {
        "drug_batch": torch.tensor([-2.0, 2.0]),
        "esm2_embedding": torch.zeros(2, 1, 1),
        "physicochemical": torch.zeros(2, 1, 4),
        "domain_labels": torch.zeros(2, 1, dtype=torch.long),
        "protein_mask": torch.ones(2, 1, dtype=torch.bool),
        "label": torch.tensor([[0.0], [1.0]]),
        "drug_ids": ["drug-0", "drug-1"],
        "target_ids": ["target-0", "target-1"],
    }

    predictions, labels, drug_ids, target_ids = collect_full_precision_predictions(
        model, [batch], "cpu"
    )

    assert model.training is False
    assert predictions == pytest.approx([0.11920292, 0.88079708])
    assert labels.tolist() == [0.0, 1.0]
    assert drug_ids == ["drug-0", "drug-1"]
    assert target_ids == ["target-0", "target-1"]


def test_source_config_file_hash_does_not_shadow_checkpoint_config_hash_helper():
    script_source = inspect.getsource(sys.modules[build_artifact_status.__module__])

    assert "source_config, config_sha256 = _load_source_config()" not in script_source


def test_status_is_canonical_only_for_checkpoint_config_full_precision_protocol():
    status = build_artifact_status(
        model_config_source="checkpoint['config']",
        inference_mode="torch.inference_mode (full precision)",
    )

    assert status["canonical"] is True
    assert status["inference_precision"] == "full precision"
    assert "current-release" in status["reason"]

    wrong_protocol = build_artifact_status(
        model_config_source="default.yaml model",
        inference_mode="CUDA AMP",
    )
    assert wrong_protocol["canonical"] is False


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
        model_config={"predictor": {"task": "classification"}},
        model_config_sha256="architecture-sha",
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
    assert record["checkpoint"]["model_config_source"] == "checkpoint['config']"
    assert record["checkpoint"]["model_config_sha256"] == "architecture-sha"
    assert record["checkpoint"]["model_config"] == {
        "predictor": {"task": "classification"}
    }
    assert record["validation"]["predictions_csv"] == "random_validation_predictions.csv"
    assert set(record["test"]["threshold_free_metrics"]) == {"AUROC", "AUPRC"}
    assert set(record["test"]["threshold_dependent_metrics"]) == {
        "F1", "Precision", "Recall", "threshold"
    }
    assert record["test"]["threshold_dependent_metrics"]["threshold"] == pytest.approx(
        validation_threshold
    )
    assert record["test"]["threshold_dependent_metrics"]["F1"] == pytest.approx(2 / 3)
