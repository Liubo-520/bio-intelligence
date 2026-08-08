"""Reproducible row-level accounting for the BindingDB external cohort.

Reviewer 2 asked why a database with millions of records yields roughly two
thousand evaluation pairs. This script re-reads a BindingDB TSV snapshot and
reports every filter as an ordered waterfall, separating the two effects that
the previous single ``excluded_censored_kd`` counter merged: rows that report
no Kd at all (they report Ki, IC50, or EC50 instead) and rows whose Kd is
inequality-qualified. Filters are applied one at a time to a shrinking row set,
so the counts sum exactly to the input row total.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "BioInteract"
for path in (str(PROJECT),):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.analysis.bindingdb_external import (  # noqa: E402
    canonicalize_smiles,
    load_davis_entity_sets,
    normalize_sequence,
    parse_exact_kd,
)

STAGES = [
    ("input_rows", "Rows in the snapshot"),
    ("no_kd_reported", "No Kd value reported (Ki/IC50/EC50 rows)"),
    ("censored_kd", "Kd present but inequality-qualified or unparsable"),
    ("multichain_target", "Target is not a single protein chain"),
    ("invalid_smiles", "Ligand SMILES not parsable by RDKit"),
    ("invalid_or_long_sequence", "Non-standard residues or sequence > 1200 aa"),
    ("davis_ligand_overlap", "Ligand identical to a Davis compound"),
    ("davis_target_overlap", "Sequence identical to a Davis target"),
    ("eligible_measurements", "Eligible measurements before replicate merge"),
]


def _is_single_chain(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(re.fullmatch(r"1(?:\.0+)?", str(value).strip()))


def _has_kd_text(value: object) -> bool:
    return not (value is None or pd.isna(value) or not str(value).strip())


def audit_archive(archive: Path, davis_dir: Path, chunksize: int,
                  skip_structure_checks: bool) -> dict:
    """Return the ordered filter waterfall for one BindingDB ZIP snapshot."""
    davis_smiles, davis_sequences = (
        (set(), set()) if skip_structure_checks else load_davis_entity_sets(davis_dir)
    )
    counts: Counter[str] = Counter()
    ligand_keys: set[str] = set()
    sequence_keys: set[str] = set()

    with zipfile.ZipFile(archive) as zipped:
        members = [
            name for name in zipped.namelist()
            if name.lower().endswith((".tsv", ".txt")) and not name.endswith("/")
        ]
        if len(members) != 1:
            raise ValueError(f"expected one TSV member in {archive.name}, found {members}")
        with zipped.open(members[0]) as handle:
            for chunk in pd.read_csv(
                handle, sep="\t", dtype=str, chunksize=chunksize, low_memory=False,
                on_bad_lines="skip", quoting=3,
            ):
                counts["input_rows"] += len(chunk)
                chain_column = next(
                    column for column in chunk.columns
                    if column.startswith("Number of Protein Chains in Target")
                )
                sequence_column = next(
                    column for column in chunk.columns
                    if column.startswith("BindingDB Target Chain Sequence")
                )
                for row in chunk.loc[
                    :, ["Ligand SMILES", "Kd (nM)", chain_column, sequence_column]
                ].itertuples(index=False):
                    smiles, kd, chains, sequence = row
                    if not _has_kd_text(kd):
                        counts["no_kd_reported"] += 1
                        continue
                    if parse_exact_kd(kd) is None:
                        counts["censored_kd"] += 1
                        continue
                    if not _is_single_chain(chains):
                        counts["multichain_target"] += 1
                        continue
                    if skip_structure_checks:
                        counts["eligible_measurements"] += 1
                        continue
                    canonical = canonicalize_smiles(smiles)
                    if canonical is None:
                        counts["invalid_smiles"] += 1
                        continue
                    normalized = normalize_sequence(sequence)
                    if normalized is None or len(normalized) > 1200:
                        counts["invalid_or_long_sequence"] += 1
                        continue
                    if canonical[0] in davis_smiles:
                        counts["davis_ligand_overlap"] += 1
                        continue
                    if normalized in davis_sequences:
                        counts["davis_target_overlap"] += 1
                        continue
                    counts["eligible_measurements"] += 1
                    ligand_keys.add(canonical[0])
                    sequence_keys.add(normalized)

    waterfall = []
    remaining = counts["input_rows"]
    for key, label in STAGES[1:-1]:
        removed = int(counts[key])
        remaining -= removed
        waterfall.append(
            {"filter": key, "description": label, "rows_removed": removed,
             "rows_remaining": int(remaining)}
        )
    return {
        "archive": archive.name,
        "input_rows": int(counts["input_rows"]),
        "waterfall": waterfall,
        "eligible_measurements": int(counts["eligible_measurements"]),
        "eligible_unique_ligands": len(ligand_keys),
        "eligible_unique_sequences": len(sequence_keys),
        "structure_checks_applied": not skip_structure_checks,
    }


def main(argv: list[str] | None = None) -> None:
    """Audit one snapshot and append the result to a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--out", default="revision_v2/analysis/bindingdb_waterfall.json")
    parser.add_argument("--davis-dir", default=str(PROJECT / "data" / "raw" / "davis"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--skip-structure-checks",
        action="store_true",
        help="Count only the cheap measurement-quality filters (fast scale probe)",
    )
    args = parser.parse_args(argv)

    report = audit_archive(
        Path(args.archive), Path(args.davis_dir), args.chunksize, args.skip_structure_checks
    )
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    key = f"{report['archive']}{'' if report['structure_checks_applied'] else ':probe'}"
    existing[key] = report
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
