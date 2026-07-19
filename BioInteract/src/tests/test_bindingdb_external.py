"""Tests for the prespecified strict BindingDB external-validation protocol."""

from pathlib import Path
import sys
import zipfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.bindingdb_external import (  # noqa: E402
    canonicalize_smiles,
    curate_archive,
    curate_measurements,
    load_davis_entity_sets,
)


def record(smiles, sequence, kd, chains="1", reaction_id="rs-1"):
    """Build the subset of documented BindingDB TSV columns needed for curation."""
    return {
        "BindingDB Reactant_set_id": reaction_id,
        "Ligand SMILES": smiles,
        "Kd (nM)": kd,
        "Number of Protein Chains in Target": chains,
        "BindingDB Target Chain Sequence": sequence,
    }


def test_curate_measurements_keeps_only_exact_kd_single_chain_double_novel_consistent_pairs():
    frame = pd.DataFrame(
        [
            record("CCO", "AAAC", "10", reaction_id="rs-1"),
            record("CCO", "AAAC", "20", reaction_id="rs-2"),
            record("CCC", "GGGG", "50", reaction_id="rs-3"),
            record("CCC", "GGGG", "10", reaction_id="rs-4"),
            record("CCN", "CCCC", ">10", reaction_id="rs-5"),
            record("CCCl", "DDDD", "20", chains="2", reaction_id="rs-6"),
            record("CCBr", "EEEE", "40", reaction_id="rs-7"),
        ]
    )

    davis_ligand = canonicalize_smiles("CCBr")
    assert davis_ligand is not None
    pairs, audit = curate_measurements(
        frame, {davis_ligand[0]}, {"ZZZZ"}
    )

    assert pairs[["kd_nM", "label"]].to_dict("records") == [
        {"kd_nM": 15.0, "label": 1}
    ]
    assert audit["excluded_conflicting_pair"] == 1
    assert audit["excluded_censored_kd"] == 1
    assert audit["excluded_multichain"] == 1
    assert audit["excluded_davis_ligand"] == 1


def test_curate_measurements_excludes_exact_davis_sequence_even_when_target_ids_differ():
    frame = pd.DataFrame([record("CCO", "MPEPTIDE", "100")])

    pairs, audit = curate_measurements(frame, set(), {"MPEPTIDE"})

    assert pairs.empty
    assert audit["excluded_davis_target"] == 1


def test_load_davis_entity_sets_canonicalizes_ligands_and_normalizes_sequences(tmp_path):
    pd.DataFrame({"drug_id": ["D1"], "smiles": ["OCC"]}).to_csv(
        tmp_path / "drug_smiles.csv", index=False
    )
    pd.DataFrame({"target_id": ["T1"], "sequence": ["mpep tide"]}).to_csv(
        tmp_path / "target_sequences.csv", index=False
    )

    smiles, sequences = load_davis_entity_sets(tmp_path)

    assert smiles == {canonicalize_smiles("CCO")[0]}
    assert sequences == {"MPEPTIDE"}


def test_curate_archive_writes_pairs_fasta_and_complete_manifest(tmp_path):
    davis_dir = tmp_path / "davis"
    davis_dir.mkdir()
    pd.DataFrame({"drug_id": ["D1"], "smiles": ["CCBr"]}).to_csv(
        davis_dir / "drug_smiles.csv", index=False
    )
    pd.DataFrame({"target_id": ["T1"], "sequence": ["MPEPTIDE"]}).to_csv(
        davis_dir / "target_sequences.csv", index=False
    )
    archive = tmp_path / "bindingdb.zip"
    fixture = pd.DataFrame(
        [
            record("CCO", "AAAA", "10", reaction_id="rs-1"),
            record("CCO", "AAAA", "20", reaction_id="rs-2"),
            record("CCBr", "CCCC", "5", reaction_id="rs-3"),
        ]
    )
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("BindingDB_fixture.tsv", fixture.to_csv(sep="\t", index=False))

    manifest = curate_archive(
        archive=archive,
        out_dir=tmp_path / "out",
        source_url="https://example.org/BindingDB_fixture.zip",
        source_version="fixture",
        davis_dir=davis_dir,
        chunksize=1,
    )

    assert manifest["source_version"] == "fixture"
    assert manifest["counts"]["retained_pairs"] == 1
    assert (tmp_path / "out" / "pairs.csv").exists()
    assert (tmp_path / "out" / "targets.fasta").read_text(encoding="utf-8") == \
        ">BDBT_63c1dd951ffedf6f\nAAAA\n"


def test_curation_cli_requires_fixed_source_provenance_arguments():
    from src.cli.curate_bindingdb_external import build_parser

    args = build_parser().parse_args(
        [
            "--archive", "data/external/bindingdb.zip",
            "--out-dir", "data/external/curated",
            "--source-url", "https://example.org/bindingdb.zip",
            "--source-version", "BindingDB_fixture",
        ]
    )

    assert args.chunksize == 200_000
    assert args.davis_dir == "data/raw/davis"
