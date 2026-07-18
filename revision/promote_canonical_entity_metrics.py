"""Promote the audited entity-split artifact into public result files.

The sole eligible input is ``revision/analysis/original_entity_split_metrics.json``
when its status is marked canonical.  This avoids a second inference pass and
keeps every public metric, prediction CSV, and bootstrap interval traceable to
the same full-precision checkpoint evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = WORKSPACE_ROOT / "revision" / "analysis" / "original_entity_split_metrics.json"
DEFAULT_RESULTS = WORKSPACE_ROOT / "BioInteract" / "results"

SPLIT_FILES = {
    "random": ("test_random.json", "predictions_random.csv", "Random"),
    "cold_target": ("test_cold_target.json", "predictions_cold_target.csv", "Target-ID-held-out"),
    "cold_drug": ("test_cold_drug.json", "predictions_cold_drug.csv", "Drug-ID-held-out"),
}


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _bootstrap_interval(labels: np.ndarray, scores: np.ndarray, metric: str, replicates: int) -> list[float]:
    rng = np.random.default_rng(42)
    evaluator = roc_auc_score if metric == "AUROC" else average_precision_score
    estimates = []
    for _ in range(replicates):
        indices = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[indices]
        if sampled_labels.min() != sampled_labels.max():
            estimates.append(float(evaluator(sampled_labels, scores[indices])))
    if not estimates:
        raise ValueError("bootstrap contained no valid two-class replicates")
    return [float(value) for value in np.percentile(estimates, [2.5, 97.5])]


def _reported_metrics(record: dict[str, Any]) -> dict[str, float]:
    threshold_free = record["test"]["threshold_free_metrics"]
    threshold_dependent = record["test"]["threshold_dependent_metrics"]
    return {
        "AUROC": float(threshold_free["AUROC"]),
        "AUPRC": float(threshold_free["AUPRC"]),
        "F1": float(threshold_dependent["F1"]),
        "Precision": float(threshold_dependent["Precision"]),
        "Recall": float(threshold_dependent["Recall"]),
        "threshold": float(threshold_dependent["threshold"]),
    }


def promote_artifacts(
    artifact_path: Path,
    results_dir: Path,
    *,
    bootstrap_replicates: int = 600,
) -> dict[str, Any]:
    """Write public outputs from one audited, canonical artifact only."""
    artifact_path = Path(artifact_path)
    results_dir = Path(results_dir)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not artifact.get("status", {}).get("canonical"):
        raise ValueError("only a canonical entity-split artifact may be promoted")

    figure_data_dir = results_dir / "figure_data"
    summary: dict[str, Any] = {
        "artifact": "current_release_entity_split_prediction_summary",
        "canonical": True,
        "canonical_artifact": "revision/analysis/original_entity_split_metrics.json",
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": 42,
        "source_config": artifact["source_config"],
        "inputs": artifact["inputs"],
        "splits": {},
    }

    for record in artifact["splits"]:
        split_key = record["split_key"]
        if split_key not in SPLIT_FILES:
            raise ValueError(f"unsupported split key: {split_key}")
        result_filename, prediction_filename, display_name = SPLIT_FILES[split_key]
        reported_metrics = _reported_metrics(record)
        source_csv = artifact_path.parent / record["test"]["predictions_csv"]
        prediction_frame = pd.read_csv(source_csv)
        required_columns = {"drug_id", "target_id", "label", "prediction"}
        if set(prediction_frame.columns) != required_columns:
            raise ValueError(f"unexpected prediction columns in {source_csv.name}")
        labels = prediction_frame["label"].to_numpy(dtype=int)
        predictions = prediction_frame["prediction"].to_numpy(dtype=float)
        if len(prediction_frame) != record["input_counts"]["test"]:
            raise ValueError(f"test-count mismatch in {source_csv.name}")
        if int(labels.sum()) != record["input_counts"]["test_positive"]:
            raise ValueError(f"positive-count mismatch in {source_csv.name}")
        recomputed = {
            "AUROC": float(roc_auc_score(labels, predictions)),
            "AUPRC": float(average_precision_score(labels, predictions)),
        }
        for metric, value in recomputed.items():
            if not np.isclose(value, reported_metrics[metric], rtol=0, atol=1e-12):
                raise ValueError(f"{split_key} {metric} does not match the canonical CSV")

        prediction_target = figure_data_dir / prediction_filename
        prediction_target.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(prediction_target, index=False)
        _json_dump(results_dir / result_filename, reported_metrics)
        summary["splits"][split_key] = {
            "display_name": display_name,
            "n_test": int(len(prediction_frame)),
            "positive_count": int(labels.sum()),
            "prediction_csv": f"BioInteract/results/figure_data/{prediction_filename}",
            "reported_metrics": reported_metrics,
            "auroc_interval": _bootstrap_interval(labels, predictions, "AUROC", bootstrap_replicates),
            "auprc_interval": _bootstrap_interval(labels, predictions, "AUPRC", bootstrap_replicates),
            "validation": record["validation"],
            "canonical_source": {
                "artifact": artifact_path.name,
                "checkpoint": record["checkpoint"]["path"],
            },
            "checkpoint": record["checkpoint"],
        }

    _json_dump(figure_data_dir / "prediction_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote canonical entity-split metrics")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--bootstrap-replicates", type=int, default=600)
    args = parser.parse_args()
    summary = promote_artifacts(
        args.artifact, args.results_dir, bootstrap_replicates=args.bootstrap_replicates
    )
    print(json.dumps({"canonical": summary["canonical"], "splits": summary["splits"]}, indent=2))


if __name__ == "__main__":
    main()
