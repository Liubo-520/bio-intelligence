"""Regenerate canonical entity-split metrics from archived checkpoints only.

This script intentionally performs inference only.  For every archived split it
recreates the seed-42 partition, chooses the F1 decision threshold from the
validation predictions, and applies that frozen threshold to test predictions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BIOINTERACT_ROOT = WORKSPACE_ROOT / "BioInteract"
if str(BIOINTERACT_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOINTERACT_ROOT))

from src.cli.evaluate import _collect_predictions, load_dataset_raw  # noqa: E402
from src.data.dataset import DTIDataset, collate_dti  # noqa: E402
from src.data.split import get_split_fn  # noqa: E402
from src.models.biointeract import BioInteract  # noqa: E402
from src.utils.metrics import classification_metrics, select_f1_threshold  # noqa: E402


CONFIG_PATH = BIOINTERACT_ROOT / "configs" / "default.yaml"
CHECKPOINTS_DIR = BIOINTERACT_ROOT / "checkpoints"
DATA_DIR = BIOINTERACT_ROOT / "data" / "raw" / "davis"
EMBEDDING_DIR = BIOINTERACT_ROOT / "data" / "esm2_embeddings"
SEED = 42

SPLITS = (
    {
        "public_name": "Random",
        "split_key": "random",
        "checkpoint": "best_random.pt",
        "artifact_stem": "random",
    },
    {
        "public_name": "Target-ID-held-out",
        "split_key": "cold_target",
        "checkpoint": "best_cold_target.pt",
        "artifact_stem": "target_id_held_out",
    },
    {
        "public_name": "Drug-ID-held-out",
        "split_key": "cold_drug",
        "checkpoint": "best_cold_drug.pt",
        "artifact_stem": "drug_id_held_out",
    },
)


def sha256_file(path: Path) -> str:
    """Return a content SHA-256 for one file without retaining it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> tuple[str, int, int]:
    """Hash relative names and file bytes in a directory deterministically."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    total_bytes = 0
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
    return digest.hexdigest(), len(files), total_bytes


def portable_path(path: Path) -> str:
    """Return an artifact path relative to this workspace."""
    return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()


def build_split_record(
    *,
    public_name: str,
    split_key: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    input_counts: dict[str, int],
    validation_threshold: float,
    validation_csv: str,
    test_csv: str,
    test_labels: np.ndarray,
    test_predictions: np.ndarray,
) -> dict[str, Any]:
    """Build the serializable metric record for one frozen-threshold split."""
    metrics = classification_metrics(
        np.asarray(test_labels),
        np.asarray(test_predictions),
        threshold=float(validation_threshold),
    )
    return {
        "public_name": public_name,
        "split_key": split_key,
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256,
        },
        "input_counts": {key: int(value) for key, value in input_counts.items()},
        "validation": {
            "f1_threshold": float(validation_threshold),
            "predictions_csv": validation_csv,
        },
        "test": {
            "predictions_csv": test_csv,
            "threshold_free_metrics": {
                "AUROC": float(metrics["AUROC"]),
                "AUPRC": float(metrics["AUPRC"]),
            },
            "threshold_dependent_metrics": {
                "F1": float(metrics["F1"]),
                "Precision": float(metrics["Precision"]),
                "Recall": float(metrics["Recall"]),
                "threshold": float(metrics["threshold"]),
            },
        },
    }


def _prediction_frame(
    drug_ids: list[str],
    target_ids: list[str],
    labels: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "drug_id": drug_ids,
            "target_id": target_ids,
            "label": np.asarray(labels).astype(int),
            "prediction": np.asarray(predictions, dtype=float),
        }
    )


def _dataset_kwargs(config: dict[str, Any], drug_smiles: dict, target_sequences: dict) -> dict[str, Any]:
    return {
        "drug_smiles": drug_smiles,
        "target_sequences": target_sequences,
        "esm2_cache_dir": config["data"].get("esm2_cache_dir", "data/esm2_embeddings"),
        "max_protein_len": config["data"].get("max_protein_len", 1200),
        "use_domain_features": config["model"]["target_encoder"].get(
            "use_domain_features", True
        ),
        "task": config["model"]["predictor"].get("task", "classification"),
    }


def _runtime_metadata(requested_device: str, resolved_device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(resolved_device)
        if cuda_available and resolved_device.type == "cuda"
        else None,
    }


def _resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def _load_source_config() -> tuple[dict[str, Any], str]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["training"]["seed"] != SEED:
        raise ValueError(f"Expected archived seed {SEED}, found {config['training']['seed']}.")
    if config["model"]["predictor"].get("task") != "classification":
        raise ValueError("This canonical artifact is defined for classification checkpoints.")
    return config, sha256_file(CONFIG_PATH)


def _input_metadata() -> dict[str, Any]:
    embedding_sha256, embedding_file_count, embedding_total_bytes = sha256_directory(
        EMBEDDING_DIR
    )
    inputs = {}
    for filename in ("interactions.csv", "drug_smiles.csv", "target_sequences.csv"):
        path = DATA_DIR / filename
        inputs[portable_path(path)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    inputs[portable_path(EMBEDDING_DIR)] = {
        "sha256": embedding_sha256,
        "file_count": embedding_file_count,
        "bytes": embedding_total_bytes,
    }
    return inputs


def recompute_entity_split_metrics(output_dir: Path, device: str) -> dict[str, Any]:
    """Run archived-checkpoint inference and write the canonical JSON artifact."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    portable_path(output_dir)
    resolved_device = _resolve_device(device)
    source_config, config_sha256 = _load_source_config()
    interactions, drug_smiles, target_sequences = load_dataset_raw("davis")
    input_metadata = _input_metadata()
    records = []

    for split_metadata in SPLITS:
        config = copy.deepcopy(source_config)
        config["data"]["split"] = split_metadata["split_key"]
        split_fn = get_split_fn(config["data"]["split"])
        train_df, validation_df, test_df = split_fn(
            interactions,
            val_ratio=config["data"].get("val_ratio", 0.1),
            test_ratio=config["data"].get("test_ratio", 0.2),
            seed=config["training"]["seed"],
        )

        dataset_kwargs = _dataset_kwargs(config, drug_smiles, target_sequences)
        validation_dataset = DTIDataset(validation_df, **dataset_kwargs)
        test_dataset = DTIDataset(test_df, **dataset_kwargs)
        batch_size = config["training"]["batch_size"]
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_dti,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_dti,
            num_workers=0,
        )

        checkpoint_path = CHECKPOINTS_DIR / split_metadata["checkpoint"]
        model = BioInteract(config["model"]).to(resolved_device)
        checkpoint = torch.load(
            checkpoint_path, map_location=resolved_device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        validation_predictions, validation_labels, validation_drug_ids, validation_target_ids = _collect_predictions(
            model, validation_loader, config, resolved_device
        )
        validation_threshold = select_f1_threshold(
            validation_labels, validation_predictions
        )
        test_predictions, test_labels, test_drug_ids, test_target_ids = _collect_predictions(
            model, test_loader, config, resolved_device
        )

        stem = split_metadata["artifact_stem"]
        validation_filename = f"{stem}_validation_predictions.csv"
        test_filename = f"{stem}_test_predictions.csv"
        _prediction_frame(
            validation_drug_ids,
            validation_target_ids,
            validation_labels,
            validation_predictions,
        ).to_csv(output_dir / validation_filename, index=False)
        _prediction_frame(
            test_drug_ids, test_target_ids, test_labels, test_predictions
        ).to_csv(output_dir / test_filename, index=False)

        input_counts = {
            "all_interactions": len(interactions),
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
            "validation_positive": int(np.sum(validation_labels)),
            "test_positive": int(np.sum(test_labels)),
        }
        records.append(
            build_split_record(
                public_name=split_metadata["public_name"],
                split_key=split_metadata["split_key"],
                checkpoint_path=portable_path(checkpoint_path),
                checkpoint_sha256=sha256_file(checkpoint_path),
                input_counts=input_counts,
                validation_threshold=validation_threshold,
                validation_csv=validation_filename,
                test_csv=test_filename,
                test_labels=test_labels,
                test_predictions=test_predictions,
            )
        )
        del checkpoint, model, validation_dataset, test_dataset
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    command = (
        "python revision/recompute_entity_split_metrics.py "
        f"--output-dir {portable_path(output_dir)} --device {device}"
    )
    artifact = {
        "artifact": "original_entity_split_metrics",
        "protocol": {
            "partitioning": "archived seed-42 70/10/20 entity split",
            "threshold_selection": "F1 threshold selected once from validation predictions",
            "test_evaluation": "frozen validation threshold applied to test predictions",
        },
        "supersedes_historical": {
            "statement": "This artifact supersedes historical test-selected-threshold metrics.",
            "retired_files": [
                "BioInteract/results/test_random.json",
                "BioInteract/results/test_cold_target.json",
                "BioInteract/results/test_cold_drug.json",
            ],
            "historical_prediction_files_not_overwritten": [
                "BioInteract/results/figure_data/predictions_random.csv",
                "BioInteract/results/figure_data/predictions_cold_target.csv",
                "BioInteract/results/figure_data/predictions_cold_drug.csv",
            ],
        },
        "seed": SEED,
        "source_config": {
            "path": portable_path(CONFIG_PATH),
            "sha256": config_sha256,
        },
        "inputs": input_metadata,
        "command": command,
        "runtime": _runtime_metadata(device, resolved_device),
        "splits": records,
    }
    artifact_path = output_dir / "original_entity_split_metrics.json"
    with artifact_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate canonical original entity-split metric artifacts."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("revision/analysis"),
        help="Workspace-relative directory for JSON and prediction CSV artifacts.",
    )
    parser.add_argument(
        "--device", default="cuda", help="Torch device for no-training inference.")
    args = parser.parse_args()
    artifact = recompute_entity_split_metrics(args.output_dir, args.device)
    print(json.dumps({"artifact": artifact["artifact"], "splits": artifact["splits"]}, indent=2))


if __name__ == "__main__":
    main()
