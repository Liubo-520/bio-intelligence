"""Regression contracts for promoting canonical entity-split artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from revision.promote_canonical_entity_metrics import promote_artifacts


def _write_csv(path: Path, scores: list[float]) -> None:
    pd.DataFrame(
        {
            "drug_id": ["D0", "D1", "D2", "D3"],
            "target_id": ["T0", "T1", "T2", "T3"],
            "label": [0, 0, 1, 1],
            "prediction": scores,
        }
    ).to_csv(path, index=False)


def test_promote_artifacts_copies_predictions_and_uses_canonical_metrics(tmp_path):
    source_dir = tmp_path / "canonical"
    source_dir.mkdir()
    csv_name = "random_test_predictions.csv"
    _write_csv(source_dir / csv_name, [0.1, 0.2, 0.8, 0.9])
    artifact_path = source_dir / "original_entity_split_metrics.json"
    artifact_path.write_text(
        json.dumps(
            {
                "artifact": "original_entity_split_metrics",
                "status": {"canonical": True},
                "seed": 42,
                "source_config": {"path": "BioInteract/configs/default.yaml", "sha256": "config"},
                "inputs": {"input.csv": {"sha256": "input"}},
                "splits": [
                    {
                        "public_name": "Random",
                        "split_key": "random",
                        "checkpoint": {"path": "BioInteract/checkpoints/best_random.pt", "sha256": "checkpoint"},
                        "input_counts": {"test": 4, "test_positive": 2},
                        "validation": {"f1_threshold": 0.7, "predictions_csv": "random_validation_predictions.csv"},
                        "test": {
                            "predictions_csv": csv_name,
                            "threshold_free_metrics": {"AUROC": 1.0, "AUPRC": 1.0},
                            "threshold_dependent_metrics": {
                                "F1": 1.0,
                                "Precision": 1.0,
                                "Recall": 1.0,
                                "threshold": 0.7,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result_dir = tmp_path / "BioInteract" / "results"

    summary = promote_artifacts(
        artifact_path=artifact_path,
        results_dir=result_dir,
        bootstrap_replicates=20,
    )

    result_metrics = json.loads((result_dir / "test_random.json").read_text())
    copied_predictions = pd.read_csv(result_dir / "figure_data" / "predictions_random.csv")
    persisted_summary = json.loads(
        (result_dir / "figure_data" / "prediction_summary.json").read_text()
    )

    assert result_metrics == {
        "AUROC": 1.0,
        "AUPRC": 1.0,
        "F1": 1.0,
        "Precision": 1.0,
        "Recall": 1.0,
        "threshold": 0.7,
    }
    assert copied_predictions["prediction"].tolist() == [0.1, 0.2, 0.8, 0.9]
    assert summary["canonical"] is True
    assert persisted_summary["bootstrap_replicates"] == 20
    assert persisted_summary["splits"]["random"]["reported_metrics"] == result_metrics
    assert persisted_summary["splits"]["random"]["canonical_source"] == {
        "artifact": "original_entity_split_metrics.json",
        "checkpoint": "BioInteract/checkpoints/best_random.pt",
    }


def test_promote_artifacts_rejects_noncanonical_input(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps({"status": {"canonical": False}, "splits": []}))

    try:
        promote_artifacts(artifact_path, tmp_path / "results", bootstrap_replicates=5)
    except ValueError as error:
        assert "canonical" in str(error)
    else:
        raise AssertionError("noncanonical artifact must not be promoted")
