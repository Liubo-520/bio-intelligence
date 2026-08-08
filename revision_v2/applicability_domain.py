"""Applicability-domain analysis for the frozen external BindingDB evaluation.

Reviewer 1 asked whether the limited external transfer reflects chemical
novelty, target dissimilarity, assay/label heterogeneity, or calibration
failure of the frozen Davis checkpoint. This script quantifies each of those
four candidate explanations from the archived curated cohort and its frozen
prediction file, and emits the per-bin tables used in the manuscript and in
the web-server applicability-domain readout.

Distance to the Davis training distribution is measured on two axes:

* ligand axis - maximum Tanimoto similarity of a non-chiral radius-2,
  1024-bit Morgan fingerprint against all 68 Davis compounds;
* target axis - maximum global sequence identity against all 442 Davis
  targets, using the same exhaustive Biopython global alignment as the
  within-Davis strict cold-target audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "BioInteract"

LIGAND_BINS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.0001)]
TARGET_BINS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.0001)]



def _aligner() -> Align.PairwiseAligner:
    """Return the global aligner configured exactly as in the cold-target audit."""
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5
    return aligner


def global_identity(first: str, second: str, aligner: Align.PairwiseAligner) -> float:
    """Return exact-residue matches divided by full global-alignment length."""
    alignment = aligner.align(first, second)[0]
    matches = sum(
        1
        for left, right in zip(alignment[0], alignment[1])
        if left == right and left != "-"
    )
    return matches / alignment.length if alignment.length else 0.0


def morgan_fingerprints(smiles_values: list[str]) -> list:
    """Return non-chiral radius-2 1024-bit Morgan fingerprints."""
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, includeChirality=False
    )
    fingerprints = []
    for smiles in smiles_values:
        molecule = Chem.MolFromSmiles(smiles)
        fingerprints.append(None if molecule is None else generator.GetFingerprint(molecule))
    RDLogger.EnableLog("rdApp.*")
    return fingerprints


def max_ligand_similarity(external_smiles: list[str], davis_smiles: list[str]) -> np.ndarray:
    """Return each external ligand's maximum Tanimoto similarity to Davis."""
    davis = [fingerprint for fingerprint in morgan_fingerprints(davis_smiles) if fingerprint]
    values = []
    for fingerprint in morgan_fingerprints(external_smiles):
        values.append(
            float(np.max(BulkTanimotoSimilarity(fingerprint, davis))) if fingerprint else np.nan
        )
    return np.asarray(values, dtype=float)


def max_target_identity(external: dict[str, str], davis: dict[str, str]) -> dict[str, float]:
    """Return each external target's maximum global identity to a Davis target.

    Every external sequence is aligned against every Davis target, so the
    reported maximum uses the same identity definition and the same exhaustive
    search as the within-Davis strict cold-target audit.
    """
    aligner = _aligner()
    davis_sequences = list(davis.values())
    identity: dict[str, float] = {}
    for index, (target_id, sequence) in enumerate(external.items(), start=1):
        identity[target_id] = max(
            global_identity(sequence, candidate, aligner) for candidate in davis_sequences
        )
        if index % 25 == 0:
            print(f"  aligned {index}/{len(external)} external targets", flush=True)
    return identity


def bin_label(lower: float, upper: float, last: bool) -> str:
    """Return the printable half-open interval label for one bin."""
    return f"[{lower:.1f}, {upper:.1f}{']' if last else ')'}".replace("1.0001", "1.0")


def stratify(frame: pd.DataFrame, column: str, bins: list[tuple[float, float]]) -> list[dict]:
    """Return per-bin counts and metrics for one applicability-domain axis."""
    rows = []
    for index, (lower, upper) in enumerate(bins):
        subset = frame[(frame[column] >= lower) & (frame[column] < upper)]
        positives = int(subset["label"].sum())
        record = {
            "bin": bin_label(lower, min(upper, 1.0), index == len(bins) - 1),
            "pairs": int(len(subset)),
            "positives": positives,
            "unique_entities": int(
                subset["drug_id" if column == "ligand_similarity" else "target_id"].nunique()
            ),
            "mean_predicted_probability": float(subset["probability"].mean()) if len(subset) else None,
        }
        if positives and positives < len(subset):
            record["AUROC"] = float(roc_auc_score(subset["label"], subset["probability"]))
            record["AUPRC"] = float(average_precision_score(subset["label"], subset["probability"]))
        else:
            record["AUROC"] = None
            record["AUPRC"] = None
        rows.append(record)
    return rows


def calibration(frame: pd.DataFrame, threshold: float) -> dict:
    """Compare frozen-checkpoint score behaviour with the observed prevalence."""
    return {
        "frozen_threshold": float(threshold),
        "observed_positive_prevalence": float(frame["label"].mean()),
        "mean_predicted_probability": float(frame["probability"].mean()),
        "median_predicted_probability": float(frame["probability"].median()),
        "fraction_above_threshold": float((frame["probability"] >= threshold).mean()),
        "mean_probability_positives": float(frame.loc[frame["label"] == 1, "probability"].mean()),
        "mean_probability_negatives": float(frame.loc[frame["label"] == 0, "probability"].mean()),
        "probability_quantiles": {
            str(q): float(frame["probability"].quantile(q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
    }


def label_heterogeneity(frame: pd.DataFrame) -> dict:
    """Summarise replicate disagreement as a proxy for assay heterogeneity."""
    replicated = frame[frame["replicate_count"] > 1]
    return {
        "pairs_with_replicate_measurements": int(len(replicated)),
        "median_replicate_count": float(replicated["replicate_count"].median()) if len(replicated) else None,
        "distinct_source_records": int(
            frame["reaction_set_ids"].str.split(";").map(len).sum()
        ),
        "note": (
            "Pairs whose replicate measurements straddled the 30 nM boundary were "
            "removed during curation; the count of such conflicts is reported in the "
            "curation waterfall."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Compute and persist the applicability-domain report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Frozen BindingDB predictions CSV")
    parser.add_argument("--davis-dir", default=str(PROJECT / "data" / "raw" / "davis"))
    parser.add_argument("--out", default="revision_v2/analysis/applicability_domain.json")
    parser.add_argument("--per-entity-out", default="revision_v2/analysis/applicability_domain_pairs.csv")
    parser.add_argument("--threshold", type=float, default=0.5959881544113159)
    args = parser.parse_args(argv)

    predictions = pd.read_csv(args.predictions)
    davis_dir = Path(args.davis_dir)
    davis_drugs = pd.read_csv(davis_dir / "drug_smiles.csv")
    davis_targets = pd.read_csv(davis_dir / "target_sequences.csv")

    external_ligands = (
        predictions[["drug_id", "canonical_smiles"]].drop_duplicates().reset_index(drop=True)
    )
    print(f"scoring {len(external_ligands)} external ligands against Davis chemistry", flush=True)
    external_ligands["ligand_similarity"] = max_ligand_similarity(
        external_ligands["canonical_smiles"].tolist(), davis_drugs["smiles"].tolist()
    )

    external_targets = (
        predictions[["target_id", "sequence"]].drop_duplicates().set_index("target_id")["sequence"].to_dict()
    )
    davis_sequences = dict(zip(davis_targets["target_id"], davis_targets["sequence"]))
    print(f"aligning {len(external_targets)} external targets against Davis targets", flush=True)
    identity = max_target_identity(external_targets, davis_sequences)

    frame = predictions.merge(
        external_ligands[["drug_id", "ligand_similarity"]], on="drug_id", how="left"
    )
    frame["target_identity"] = frame["target_id"].map(identity)
    frame.to_csv(ROOT / args.per_entity_out, index=False)

    in_domain = frame[(frame["ligand_similarity"] >= 0.3) & (frame["target_identity"] >= 0.3)]
    report = {
        "cohort": {
            "pairs": int(len(frame)),
            "positives": int(frame["label"].sum()),
            "unique_ligands": int(frame["drug_id"].nunique()),
            "unique_targets": int(frame["target_id"].nunique()),
        },
        "ligand_similarity_to_davis": {
            "definition": "Maximum Tanimoto similarity of non-chiral radius-2, 1024-bit Morgan fingerprints against all 68 Davis compounds.",
            "median": float(external_ligands["ligand_similarity"].median()),
            "minimum": float(external_ligands["ligand_similarity"].min()),
            "maximum": float(external_ligands["ligand_similarity"].max()),
            "fraction_below_0.3": float((external_ligands["ligand_similarity"] < 0.3).mean()),
            "bins": stratify(frame, "ligand_similarity", LIGAND_BINS),
        },
        "target_identity_to_davis": {
            "definition": "Maximum global sequence identity against all 442 Davis targets; exhaustive Biopython global alignment (match 1, mismatch 0, gap open -1, gap extend -0.5).",
            "median": float(np.median(list(identity.values()))),
            "minimum": float(np.min(list(identity.values()))),
            "maximum": float(np.max(list(identity.values()))),
            "fraction_below_0.3": float(np.mean(np.asarray(list(identity.values())) < 0.3)),
            "bins": stratify(frame, "target_identity", TARGET_BINS),
        },
        "joint_in_domain_subset": {
            "definition": "Pairs whose ligand similarity and target identity to Davis are both at least 0.3.",
            "pairs": int(len(in_domain)),
            "positives": int(in_domain["label"].sum()),
            "AUROC": float(roc_auc_score(in_domain["label"], in_domain["probability"]))
            if in_domain["label"].nunique() == 2 else None,
            "AUPRC": float(average_precision_score(in_domain["label"], in_domain["probability"]))
            if in_domain["label"].nunique() == 2 else None,
        },
        "calibration": calibration(frame, args.threshold),
        "label_heterogeneity": label_heterogeneity(frame),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
