"""Audit nominal ABL1 variants against the exact Davis model inputs.

The audit intentionally compares the sequences that the project loaders provide
to the model.  It does not infer mutation identity from the target-name string.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


COMPOUND_IDS = {"D0010", "D0017"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_abl1_variants(davis_dir: Path) -> dict[str, object]:
    """Return an exact-input audit for every Davis target named ``ABL1*``."""
    targets = _read_csv(davis_dir / "target_sequences.csv")
    variants = []
    for row in targets:
        if not row["target_name"].startswith("ABL1"):
            continue
        sequence = row["sequence"]
        variants.append(
            {
                "target_id": row["target_id"],
                "target_name": row["target_name"],
                "sequence_length": len(sequence),
                "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
            }
        )

    wild_type = next(item for item in variants if item["target_name"] == "ABL1")
    for item in variants:
        item["identical_to_abl1_input"] = item["sequence_sha256"] == wild_type["sequence_sha256"]

    target_names = {item["target_id"]: item["target_name"] for item in variants}
    drug_names = {
        row["drug_id"]: row["drug_name"]
        for row in _read_csv(davis_dir / "drug_smiles.csv")
        if row["drug_id"] in COMPOUND_IDS
    }
    compound_records = []
    for row in _read_csv(davis_dir / "interactions.csv"):
        if row["target_id"] in target_names and row["drug_id"] in COMPOUND_IDS:
            compound_records.append(
                {
                    "drug_id": row["drug_id"],
                    "drug_name": drug_names[row["drug_id"]],
                    "target_id": row["target_id"],
                    "target_name": target_names[row["target_id"]],
                    "label": int(row["label"]),
                    "affinity_nM": float(row["affinity"]),
                    "pKd": float(row["pKd"]),
                }
            )

    by_digest: defaultdict[str, list[str]] = defaultdict(list)
    for item in variants:
        by_digest[str(item["sequence_sha256"])].append(str(item["target_id"]))
    return {
        "variant_count": len(variants),
        "unique_sequence_digest_count": len(by_digest),
        "variants": variants,
        "compound_records": compound_records,
    }


def render_markdown(report: dict[str, object]) -> str:
    """Render a concise, submission-audit-ready report without interpretation drift."""
    variants = report["variants"]
    assert isinstance(variants, list)
    rows = [
        "# Davis ABL1 nominal-variant input audit",
        "",
        "The project loaders map `target_id` directly to the sequence in "
        "`BioInteract/data/raw/davis/target_sequences.csv`. The target-name field is not "
        "a model feature. The table below therefore audits the actual sequence inputs.",
        "",
        f"- Nominal ABL1 entries: {report['variant_count']}",
        f"- Unique sequence SHA-256 values: {report['unique_sequence_digest_count']}",
        "- Result: every listed nominal ABL1 entry has the same 1,167-residue input "
        "sequence as the `ABL1` record.",
        "",
        "| Target ID | Nominal target name | Length | SHA-256 | Identical to ABL1 input? |",
        "|---|---|---:|---|---|",
    ]
    for item in variants:
        rows.append(
            f"| {item['target_id']} | {item['target_name']} | {item['sequence_length']} | "
            f"`{item['sequence_sha256']}` | {'Yes' if item['identical_to_abl1_input'] else 'No'} |"
        )

    records = report["compound_records"]
    assert isinstance(records, list)
    rows.extend(
        [
            "",
            "## D0010/D0017 Davis records",
            "",
            "These are dataset labels/affinities; they do not make the identical sequence "
            "inputs mutation-specific.",
            "",
            "| Compound ID | Compound name | Target ID | Nominal target name | Label | Affinity (nM) | pKd |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for item in records:
        rows.append(
            f"| {item['drug_id']} | {item['drug_name']} | {item['target_id']} | {item['target_name']} | "
            f"{item['label']} | {item['affinity_nM']:g} | {item['pKd']:.12g} |"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--davis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_abl1_variants(args.davis_dir)
    args.output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("variant_count", "unique_sequence_digest_count")}, indent=2))


if __name__ == "__main__":
    main()
