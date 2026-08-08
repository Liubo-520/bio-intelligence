"""Bundle the Davis training distribution used by the web-server domain check.

Reviewer 1 asked that the interactive server tell a user whether a submitted
ligand and target lie close to or far from the Davis training distribution.
The check needs the training entities themselves, so this script writes a
compact reference file containing the 68 Davis Morgan fingerprints (as bit
indices) and the 442 Davis target sequences, together with the applicability
thresholds and the external-cohort evidence that justifies them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "BioInteract"

# Bin edges shared with the external applicability-domain analysis, so that the
# server readout and the manuscript tables use one definition.
LIGAND_NEAR = 0.40
LIGAND_FAR = 0.30
TARGET_NEAR = 0.40
TARGET_FAR = 0.30


def morgan_bits(smiles: str) -> list[int]:
    """Return the on-bit indices of a non-chiral radius-2, 1024-bit fingerprint."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, includeChirality=False
    )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse {smiles}")
    return sorted(generator.GetFingerprint(molecule).GetOnBits())


def main(argv: list[str] | None = None) -> None:
    """Write the reference bundle consumed by the Gradio application."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--davis-dir", default=str(PROJECT / "data" / "raw" / "davis"))
    parser.add_argument(
        "--out", default=str(PROJECT / "hf_space" / "examples" / "davis_applicability_reference.json")
    )
    parser.add_argument("--external-report", default=str(ROOT / "revision_v2" / "analysis" / "applicability_domain.json"))
    args = parser.parse_args(argv)

    RDLogger.DisableLog("rdApp.*")
    davis_dir = Path(args.davis_dir)
    drugs = pd.read_csv(davis_dir / "drug_smiles.csv")
    targets = pd.read_csv(davis_dir / "target_sequences.csv")

    external = {}
    external_path = Path(args.external_report)
    if external_path.exists():
        report = json.loads(external_path.read_text(encoding="utf-8"))
        external = {
            "cohort": report["cohort"],
            "ligand_bins": report["ligand_similarity_to_davis"]["bins"],
            "target_bins": report["target_identity_to_davis"]["bins"],
            "joint_in_domain_subset": report["joint_in_domain_subset"],
        }

    bundle = {
        "dataset": "Davis kinase inhibitor benchmark",
        "ligand_fingerprint": {
            "definition": "Non-chiral radius-2, 1024-bit Morgan fingerprint; similarity is Tanimoto.",
            "on_bits": {row.drug_id: morgan_bits(row.smiles) for row in drugs.itertuples(index=False)},
        },
        "target_sequences": dict(zip(targets["target_id"], targets["sequence"])),
        "thresholds": {
            "ligand_near": LIGAND_NEAR,
            "ligand_far": LIGAND_FAR,
            "target_near": TARGET_NEAR,
            "target_far": TARGET_FAR,
            "rationale": (
                "Bins match the external BindingDB applicability-domain analysis: "
                "a submitted entity is reported as inside the Davis domain when its "
                "nearest-training-neighbour score is at least 0.40, as borderline "
                "between 0.30 and 0.40, and as outside below 0.30."
            ),
        },
        "external_evidence": external,
        "provenance": {
            "drug_table_sha256": hashlib.sha256(
                (davis_dir / "drug_smiles.csv").read_bytes()
            ).hexdigest(),
            "target_table_sha256": hashlib.sha256(
                (davis_dir / "target_sequences.csv").read_bytes()
            ).hexdigest(),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle), encoding="utf-8")
    print(
        f"wrote {out_path} with {len(bundle['ligand_fingerprint']['on_bits'])} ligands "
        f"and {len(bundle['target_sequences'])} targets"
    )


if __name__ == "__main__":
    main()
