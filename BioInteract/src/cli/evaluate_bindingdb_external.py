"""Evaluate the frozen BioInteract random checkpoint on a curated BindingDB cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.bindingdb_external import (
    FROZEN_RANDOM_THRESHOLD,
    external_metrics,
    passes_primary_gate,
    run_frozen_inference,
)
from src.utils.paths import resolve_project_path


FROZEN_CHECKPOINT = "checkpoints/best_random.pt"


def _portable_project_path(path: Path) -> str:
    """Represent an artifact path relative to the released BioInteract root."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact is outside the BioInteract project: {path}") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed external-blind-evaluation command-line contract."""
    parser = argparse.ArgumentParser(
        description="Blindly evaluate frozen best_random BioInteract on curated BindingDB pairs"
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--esm-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default=FROZEN_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=FROZEN_RANDOM_THRESHOLD)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--summary-path",
        default="results/external_validation/bindingdb_external_summary.json",
    )
    parser.add_argument(
        "--manifest-path",
        default="results/external_validation/bindingdb_external_manifest.json",
    )
    return parser


def _canonical_probability_hash(predictions: pd.DataFrame) -> str:
    subset = predictions.loc[:, ["drug_id", "target_id", "label", "probability", "frozen_prediction"]]
    payload = subset.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_curation_manifest(out_dir: Path) -> dict[str, object]:
    path = out_dir / "curation_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing curation manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"source_url", "source_version", "archive", "counts"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Curation manifest is incomplete: missing {missing}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Evaluate the fixed protocol and persist summary/manifest artifacts."""
    args = build_parser().parse_args(argv)
    if args.checkpoint != FROZEN_CHECKPOINT:
        raise ValueError(f"External primary evaluation requires {FROZEN_CHECKPOINT}")
    if args.threshold != FROZEN_RANDOM_THRESHOLD:
        raise ValueError(
            "External primary evaluation requires the frozen Davis validation threshold "
            f"{FROZEN_RANDOM_THRESHOLD}"
        )

    pairs_path = resolve_project_path(args.pairs)
    esm_cache = resolve_project_path(args.esm_cache)
    out_dir = resolve_project_path(args.out_dir)
    checkpoint = resolve_project_path(args.checkpoint)
    summary_path = resolve_project_path(args.summary_path)
    manifest_path = resolve_project_path(args.manifest_path)
    pairs = pd.read_csv(pairs_path)
    curation_manifest = _load_curation_manifest(out_dir)
    predictions = run_frozen_inference(
        pairs,
        esm_cache,
        checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    predictions_path = out_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    probability_hash = _canonical_probability_hash(predictions)
    fingerprint_path = out_dir / "evaluation_fingerprint.json"
    prior_hash = None
    if fingerprint_path.is_file():
        prior_hash = json.loads(fingerprint_path.read_text(encoding="utf-8")).get(
            "probability_sha256"
        )
    rerun_identical = prior_hash == probability_hash if prior_hash is not None else False
    _write_json(fingerprint_path, {"probability_sha256": probability_hash})

    labels = predictions["label"].to_numpy(dtype=int)
    probabilities = predictions["probability"].to_numpy(dtype=float)
    positive_pairs = int(labels.sum())
    negative_pairs = int(len(labels) - positive_pairs)
    has_both_classes = bool({0, 1}.issubset(set(labels.tolist())))
    metrics = (
        external_metrics(labels, probabilities, args.threshold)
        if has_both_classes else None
    )
    summary: dict[str, object] = {
        "protocol": "strict_bindingdb_kd_only_double_novel_v1",
        "checkpoint": FROZEN_CHECKPOINT,
        "threshold": FROZEN_RANDOM_THRESHOLD,
        "total_pairs": int(len(predictions)),
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "unique_drugs": int(predictions["drug_id"].nunique()),
        "unique_targets": int(predictions["target_id"].nunique()),
        "all_esm_valid": True,
        "manifest_complete": True,
        "rerun_identical": rerun_identical,
        "has_both_classes": has_both_classes,
        "prediction_probability_sha256": probability_hash,
        "metrics": metrics,
    }
    summary["primary_gate_passed"] = passes_primary_gate(summary)
    result_manifest: dict[str, object] = {
        "protocol": summary["protocol"],
        "curation_manifest": curation_manifest,
        "checkpoint": {
            "path": FROZEN_CHECKPOINT,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "prediction_artifact": {
            "path": _portable_project_path(predictions_path),
            "probability_sha256": probability_hash,
        },
        "summary_path": _portable_project_path(summary_path),
    }
    _write_json(summary_path, summary)
    _write_json(manifest_path, result_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
