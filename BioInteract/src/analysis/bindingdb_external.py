"""Strict curation and evaluation helpers for the BindingDB external cohort.

The module intentionally accepts only the narrow Kd-only, exact-identity protocol
defined in ``docs/superpowers/specs/2026-07-19-bindingdb-external-validation-design.md``.
It never selects a threshold or model using external labels.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable
import zipfile

# CUDA >= 10.2 requires this process-wide setting before importing torch when
# deterministic cuBLAS matrix multiplication is requested below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DTIDataset, collate_dti
from src.models.biointeract import BioInteract
from src.utils.metrics import classification_metrics


REQUIRED_BINDINGDB_COLUMNS = (
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "Kd (nM)",
    "Number of Protein Chains in Target",
    "BindingDB Target Chain Sequence",
)
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
KD_THRESHOLD_NM = 30.0
FROZEN_RANDOM_THRESHOLD = 0.5959881544113159


def canonicalize_smiles(smiles: object) -> tuple[str, str] | None:
    """Return canonical isomeric SMILES and InChIKey, or ``None`` if unparsable."""
    if smiles is None or pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    RDLogger.DisableLog("rdApp.error")
    try:
        molecule = Chem.MolFromSmiles(text)
    finally:
        RDLogger.EnableLog("rdApp.error")
    if molecule is None:
        return None
    return (
        Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True),
        Chem.MolToInchiKey(molecule),
    )


def normalize_sequence(sequence: object) -> str | None:
    """Return a valid single-chain amino-acid sequence, otherwise ``None``."""
    if sequence is None or pd.isna(sequence):
        return None
    normalized = "".join(str(sequence).split()).upper()
    if not normalized or not set(normalized).issubset(STANDARD_AMINO_ACIDS):
        return None
    return normalized


def parse_exact_kd(value: object) -> float | None:
    """Parse a positive uncensored numeric Kd value in nM."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not re.fullmatch(r"(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
        return None
    numeric = float(text)
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _is_single_chain(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(re.fullmatch(r"1(?:\.0+)?", str(value).strip()))


def _stable_identifier(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _canonical_bindingdb_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize documented legacy and 2026-07 BindingDB headers for one TSV chunk."""
    available = set(frame.columns)
    column_map: dict[str, str] = {}
    for canonical in (
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "Kd (nM)",
    ):
        if canonical not in available:
            raise ValueError(f"BindingDB frame is missing required column: {canonical}")
        column_map[canonical] = canonical

    chain_count = "Number of Protein Chains in Target"
    if chain_count not in available:
        candidates = sorted(
            column for column in available
            if column.startswith("Number of Protein Chains in Target")
        )
        if not candidates:
            raise ValueError(f"BindingDB frame is missing required column: {chain_count}")
        column_map[chain_count] = candidates[0]
    else:
        column_map[chain_count] = chain_count

    sequence = "BindingDB Target Chain Sequence"
    if sequence not in available:
        candidates = sorted(
            column for column in available
            if column == "BindingDB Target Chain Sequence 1"
        )
        if not candidates:
            raise ValueError(f"BindingDB frame is missing required column: {sequence}")
        column_map[sequence] = candidates[0]
    else:
        column_map[sequence] = sequence

    rename_map = {
        source: canonical
        for canonical, source in column_map.items()
        if source != canonical
    }
    return frame.rename(columns=rename_map)


def curate_measurements(
    frame: pd.DataFrame,
    davis_smiles: set[str],
    davis_sequences: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Curate strict, threshold-consistent, double-novel BindingDB pairs.

    ``davis_smiles`` must contain canonical isomeric SMILES, and
    ``davis_sequences`` must contain normalized full sequences. Both sets are
    compared against every eligible BindingDB measurement before replicate
    reconciliation.
    """
    eligible_frame, audit = _filter_eligible_measurements(
        frame, davis_smiles, davis_sequences
    )
    pairs, audit = _reconcile_eligible_measurements(eligible_frame, audit)
    return pairs, dict(audit)


def _filter_eligible_measurements(
    frame: pd.DataFrame,
    davis_smiles: set[str],
    davis_sequences: set[str],
) -> tuple[pd.DataFrame, Counter[str]]:
    """Filter one BindingDB frame without reconciling cross-chunk replicates."""
    frame = _canonical_bindingdb_columns(frame)
    audit: Counter[str] = Counter(input_rows=int(len(frame)))
    eligible: list[dict[str, object]] = []

    for row in frame.loc[:, REQUIRED_BINDINGDB_COLUMNS].itertuples(index=False):
        reaction_set_id, smiles, kd, chain_count, sequence = row
        if not _is_single_chain(chain_count):
            audit["excluded_multichain"] += 1
            continue
        parsed_kd = parse_exact_kd(kd)
        if parsed_kd is None:
            audit["excluded_censored_kd"] += 1
            continue
        canonical_ligand = canonicalize_smiles(smiles)
        if canonical_ligand is None:
            audit["excluded_invalid_smiles"] += 1
            continue
        canonical_smiles, inchi_key = canonical_ligand
        if canonical_smiles in davis_smiles:
            audit["excluded_davis_ligand"] += 1
            continue
        normalized_sequence = normalize_sequence(sequence)
        if normalized_sequence is None or len(normalized_sequence) > 1200:
            audit["excluded_invalid_sequence"] += 1
            continue
        if normalized_sequence in davis_sequences:
            audit["excluded_davis_target"] += 1
            continue
        eligible.append(
            {
                "canonical_smiles": canonical_smiles,
                "inchi_key": inchi_key,
                "sequence": normalized_sequence,
                "kd_nM": parsed_kd,
                "reaction_set_id": str(reaction_set_id),
            }
        )

    return pd.DataFrame(
        eligible,
        columns=[
            "canonical_smiles", "inchi_key", "sequence", "kd_nM", "reaction_set_id",
        ],
    ), audit


def _reconcile_eligible_measurements(
    eligible_frame: pd.DataFrame,
    audit: Counter[str],
) -> tuple[pd.DataFrame, Counter[str]]:
    """Collapse valid measurements into only threshold-consistent pair medians."""
    if eligible_frame.empty:
        pairs = pd.DataFrame(
            columns=[
                "drug_id", "target_id", "canonical_smiles", "inchi_key", "sequence",
                "kd_nM", "label", "replicate_count", "reaction_set_ids",
            ]
        )
        audit["eligible_measurements_before_replicate_reconciliation"] = 0
        audit["retained_pairs"] = 0
        audit["retained_positive_pairs"] = 0
        audit["retained_negative_pairs"] = 0
        audit["retained_unique_drugs"] = 0
        audit["retained_unique_targets"] = 0
        return pairs, audit

    pair_rows: list[dict[str, object]] = []
    for (canonical_smiles, sequence), group in eligible_frame.groupby(
        ["canonical_smiles", "sequence"], sort=True
    ):
        labels = (group["kd_nM"] < KD_THRESHOLD_NM).astype(int)
        if labels.nunique() != 1:
            audit["excluded_conflicting_pair"] += 1
            continue
        median_kd = float(group["kd_nM"].median())
        pair_rows.append(
            {
                "drug_id": _stable_identifier("BDBD", canonical_smiles),
                "target_id": _stable_identifier("BDBT", sequence),
                "canonical_smiles": canonical_smiles,
                "inchi_key": group["inchi_key"].iloc[0],
                "sequence": sequence,
                "kd_nM": median_kd,
                "label": int(median_kd < KD_THRESHOLD_NM),
                "replicate_count": int(len(group)),
                "reaction_set_ids": ";".join(sorted(group["reaction_set_id"])),
            }
        )

    pairs = pd.DataFrame(pair_rows).sort_values(
        ["drug_id", "target_id"], kind="stable"
    ).reset_index(drop=True) if pair_rows else pd.DataFrame()
    if pairs.empty:
        pairs = pd.DataFrame(
            columns=[
                "drug_id", "target_id", "canonical_smiles", "inchi_key", "sequence",
                "kd_nM", "label", "replicate_count", "reaction_set_ids",
            ]
        )

    audit["eligible_measurements_before_replicate_reconciliation"] = len(eligible_frame)
    audit["retained_pairs"] = len(pairs)
    audit["retained_positive_pairs"] = int(pairs["label"].sum())
    audit["retained_negative_pairs"] = int(len(pairs) - pairs["label"].sum())
    audit["retained_unique_drugs"] = int(pairs["drug_id"].nunique())
    audit["retained_unique_targets"] = int(pairs["target_id"].nunique())
    return pairs, audit


def canonicalize_smiles_iter(smiles_values: Iterable[object]) -> set[str]:
    """Canonicalize a collection of ligand structures and drop unparsable entries."""
    canonical: set[str] = set()
    for smiles in smiles_values:
        result = canonicalize_smiles(smiles)
        if result is not None:
            canonical.add(result[0])
    return canonical


def load_davis_entity_sets(davis_dir: str | Path) -> tuple[set[str], set[str]]:
    """Load canonical ligand and normalized target identity sets from Davis."""
    data_dir = Path(davis_dir)
    drug_frame = pd.read_csv(data_dir / "drug_smiles.csv")
    target_frame = pd.read_csv(data_dir / "target_sequences.csv")
    if "smiles" not in drug_frame or "sequence" not in target_frame:
        raise ValueError("Davis source tables must contain smiles and sequence columns")
    sequences = {
        sequence
        for raw_sequence in target_frame["sequence"]
        if (sequence := normalize_sequence(raw_sequence)) is not None
    }
    return canonicalize_smiles_iter(drug_frame["smiles"]), sequences


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_targets_fasta(pairs: pd.DataFrame, path: Path) -> None:
    unique_targets = pairs[["target_id", "sequence"]].drop_duplicates().sort_values(
        "target_id", kind="stable"
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in unique_targets.itertuples(index=False):
            handle.write(f">{row.target_id}\n{row.sequence}\n")


def curate_archive(
    archive: str | Path,
    out_dir: str | Path,
    source_url: str,
    source_version: str,
    davis_dir: str | Path,
    chunksize: int = 200_000,
) -> dict[str, object]:
    """Curate one fixed BindingDB ZIP snapshot and write local-only artifacts."""
    archive_path = Path(archive)
    output_dir = Path(out_dir)
    if not archive_path.is_file():
        raise FileNotFoundError(f"BindingDB archive not found: {archive_path}")
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    davis_smiles, davis_sequences = load_davis_entity_sets(davis_dir)
    audit: Counter[str] = Counter()
    eligible_frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(archive_path) as zipped:
        tsv_members = [
            member for member in zipped.namelist()
            if member.lower().endswith((".tsv", ".txt")) and not member.endswith("/")
        ]
        if len(tsv_members) != 1:
            raise ValueError(
                f"Expected exactly one TSV data member in {archive_path.name}, found {tsv_members}"
            )
        with zipped.open(tsv_members[0]) as raw_tsv:
            for chunk in pd.read_csv(
                raw_tsv,
                sep="\t",
                dtype=str,
                chunksize=chunksize,
                low_memory=False,
            ):
                eligible, chunk_audit = _filter_eligible_measurements(
                    chunk, davis_smiles, davis_sequences
                )
                audit.update(chunk_audit)
                if not eligible.empty:
                    eligible_frames.append(eligible)

    columns = ["canonical_smiles", "inchi_key", "sequence", "kd_nM", "reaction_set_id"]
    eligible_frame = (
        pd.concat(eligible_frames, ignore_index=True)
        if eligible_frames else pd.DataFrame(columns=columns)
    )
    pairs, audit = _reconcile_eligible_measurements(eligible_frame, audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "pairs.csv"
    targets_path = output_dir / "targets.fasta"
    audit_path = output_dir / "curation_audit.json"
    pairs.to_csv(pairs_path, index=False)
    _write_targets_fasta(pairs, targets_path)
    manifest = {
        "protocol": "strict_bindingdb_kd_only_double_novel_v1",
        "source_url": source_url,
        "source_version": source_version,
        "curated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "filename": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256_file(archive_path),
            "tsv_member": tsv_members[0],
        },
        "counts": dict(audit),
        # The released protocol always excludes identities against this Davis
        # collection. Keep the manifest portable rather than recording a local
        # workstation path passed by a caller or test fixture.
        "davis_identity_reference": "data/raw/davis",
        "output": {
            "pairs_csv": pairs_path.name,
            "targets_fasta": targets_path.name,
        },
    }
    audit_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def validate_esm_cache(
    pairs: pd.DataFrame,
    cache_dir: str | Path,
    esm2_dim: int = 640,
) -> dict[str, int]:
    """Require a real, model-compatible ESM tensor for every external target."""
    required = {"target_id", "sequence"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"pairs must contain {sorted(required)}")
    root = Path(cache_dir)
    unique_targets = pairs.loc[:, ["target_id", "sequence"]].drop_duplicates()
    for target in unique_targets.itertuples(index=False):
        tensor_path = root / f"{target.target_id}.pt"
        if not tensor_path.is_file():
            raise FileNotFoundError(f"Missing ESM-2 embedding for {target.target_id}: {tensor_path}")
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"ESM-2 embedding for {target.target_id} must be a rank-2 tensor")
        if tensor.shape[1] != esm2_dim:
            raise ValueError(
                f"ESM-2 embedding for {target.target_id} has feature dimension "
                f"{tensor.shape[1]}, expected {esm2_dim}"
            )
        if tensor.shape[0] < len(target.sequence):
            raise ValueError(
                f"ESM-2 embedding for {target.target_id} has {tensor.shape[0]} residues, "
                f"but the curated sequence has {len(target.sequence)}"
            )
    return {"checked_targets": int(len(unique_targets))}


def _validate_binary_arrays(labels: np.ndarray, probabilities: np.ndarray) -> None:
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.size != probabilities.size:
        raise ValueError("labels and probabilities must be one-dimensional arrays of equal length")
    if labels.size == 0 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("external metrics require both binary outcome classes")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")


def external_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    bootstrap_seed: int = 42,
    bootstrap_replicates: int = 600,
) -> dict[str, object]:
    """Compute external metrics using a frozen threshold and stratified bootstrap."""
    y_true = np.asarray(labels, dtype=int).reshape(-1)
    y_probability = np.asarray(probabilities, dtype=float).reshape(-1)
    _validate_binary_arrays(y_true, y_probability)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")

    point = classification_metrics(y_true, y_probability, threshold=threshold)
    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    generator = np.random.default_rng(bootstrap_seed)
    auroc_samples: list[float] = []
    auprc_samples: list[float] = []
    for _ in range(bootstrap_replicates):
        sample_indices = np.concatenate(
            [
                generator.choice(positive_indices, size=len(positive_indices), replace=True),
                generator.choice(negative_indices, size=len(negative_indices), replace=True),
            ]
        )
        sample_labels = y_true[sample_indices]
        sample_probabilities = y_probability[sample_indices]
        auroc_samples.append(float(roc_auc_score(sample_labels, sample_probabilities)))
        auprc_samples.append(float(average_precision_score(sample_labels, sample_probabilities)))

    return {
        "AUROC": float(point["AUROC"]),
        "AUPRC": float(point["AUPRC"]),
        "F1": float(point["F1"]),
        "Precision": float(point["Precision"]),
        "Recall": float(point["Recall"]),
        "threshold": float(threshold),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_replicates": int(bootstrap_replicates),
        "AUROC_95CI": [float(value) for value in np.percentile(auroc_samples, [2.5, 97.5])],
        "AUPRC_95CI": [float(value) for value in np.percentile(auprc_samples, [2.5, 97.5])],
    }


def passes_primary_gate(summary: dict[str, object]) -> bool:
    """Return whether every prespecified condition permits manuscript inclusion."""
    return (
        int(summary.get("positive_pairs", 0)) >= 100
        and int(summary.get("negative_pairs", 0)) >= 100
        and summary.get("all_esm_valid") is True
        and summary.get("manifest_complete") is True
        and summary.get("rerun_identical") is True
    )


def checkpoint_model_config(checkpoint_path: str | Path) -> dict:
    """Read the frozen model configuration embedded in a released checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain an embedded model configuration")
    model_config = config.get("model", config)
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError("checkpoint model configuration is empty")
    return model_config


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return device


def _configure_deterministic_inference(device: str) -> None:
    """Set deterministic flags before reconstructing a frozen inference model."""
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_frozen_inference(
    pairs: pd.DataFrame,
    esm_cache_dir: str | Path,
    checkpoint_path: str | Path,
    device: str = "auto",
    batch_size: int = 64,
    threshold: float = FROZEN_RANDOM_THRESHOLD,
) -> pd.DataFrame:
    """Return full-precision probabilities from a fixed checkpoint and threshold."""
    required = {"drug_id", "target_id", "canonical_smiles", "sequence", "label"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"curated pairs are missing required columns: {missing}")
    if pairs.empty:
        raise ValueError("cannot run frozen inference on an empty cohort")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    resolved_device = _resolve_device(device)
    _configure_deterministic_inference(resolved_device)
    validate_esm_cache(pairs, esm_cache_dir)
    model_config = checkpoint_model_config(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain model_state_dict")

    dataset_frame = pairs.loc[:, ["drug_id", "target_id", "label"]].copy()
    dataset = DTIDataset(
        dataset_frame,
        drug_smiles=dict(zip(pairs["drug_id"], pairs["canonical_smiles"])),
        target_sequences=dict(zip(pairs["target_id"], pairs["sequence"])),
        esm2_cache_dir=str(esm_cache_dir),
        max_protein_len=1200,
        use_domain_features=model_config.get("target_encoder", {}).get(
            "use_domain_features", True
        ),
        esm2_dim=model_config.get("target_encoder", {}).get("esm2_dim", 640),
        task=model_config.get("predictor", {}).get("task", "classification"),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dti,
        num_workers=0,
    )
    model = BioInteract(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probabilities: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            probability = model.predict_proba(
                batch["drug_batch"].to(resolved_device),
                batch["esm2_embedding"].to(resolved_device),
                batch["physicochemical"].to(resolved_device),
                batch["domain_labels"].to(resolved_device),
                batch["protein_mask"].to(resolved_device),
                morgan_fp=batch["morgan_fp"].to(resolved_device),
            )
            probabilities.extend(probability.detach().cpu().numpy().reshape(-1).astype(float))

    output = pairs.copy().reset_index(drop=True)
    output["probability"] = probabilities
    output["frozen_prediction"] = (output["probability"] >= threshold).astype(int)
    return output
