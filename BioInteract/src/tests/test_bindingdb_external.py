"""Tests for the prespecified strict BindingDB external-validation protocol."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.bindingdb_external import (  # noqa: E402
    canonicalize_smiles,
    curate_measurements,
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
            record("CCC", "BBBB", "50", reaction_id="rs-3"),
            record("CCC", "BBBB", "10", reaction_id="rs-4"),
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
