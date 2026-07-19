"""Tests for the prespecified strict BindingDB external-validation protocol."""

from pathlib import Path
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.bindingdb_external import (  # noqa: E402
    canonicalize_smiles,
    checkpoint_model_config,
    curate_archive,
    curate_measurements,
    external_metrics,
    load_davis_entity_sets,
    passes_primary_gate,
    run_frozen_inference,
    validate_esm_cache,
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
    assert audit["excluded_conflicting_measurements"] == 2
    assert audit["threshold_consistent_measurements"] == 2
    assert audit["retained_pairs_with_replicates"] == 1
    assert audit["replicate_measurements_collapsed"] == 1
    assert audit["excluded_censored_kd"] == 1
    assert audit["excluded_multichain"] == 1
    assert audit["excluded_davis_ligand"] == 1


def test_curate_measurements_excludes_exact_davis_sequence_even_when_target_ids_differ():
    frame = pd.DataFrame([record("CCO", "MPEPTIDE", "100")])

    pairs, audit = curate_measurements(frame, set(), {"MPEPTIDE"})

    assert pairs.empty
    assert audit["excluded_davis_target"] == 1


def test_curate_measurements_accepts_documented_202607_single_chain_column_names():
    frame = pd.DataFrame([record("CCO", "AAAA", "10")]).rename(
        columns={
            "Number of Protein Chains in Target": (
                "Number of Protein Chains in Target (>1 implies a multichain complex)"
            ),
            "BindingDB Target Chain Sequence": "BindingDB Target Chain Sequence 1",
        }
    )

    pairs, audit = curate_measurements(frame, set(), set())

    assert len(pairs) == 1
    assert pairs.loc[0, "label"] == 1
    assert audit["retained_pairs"] == 1


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
    assert manifest["davis_identity_reference"] == "data/raw/davis"
    assert str(davis_dir) not in str(manifest)
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


def test_validate_esm_cache_rejects_missing_or_wrong_dimensional_embeddings(tmp_path):
    pairs = pd.DataFrame({"target_id": ["BDBT_A"], "sequence": ["AAAA"]})

    with pytest.raises(FileNotFoundError):
        validate_esm_cache(pairs, tmp_path)

    torch.save(torch.zeros(4, 639), tmp_path / "BDBT_A.pt")
    with pytest.raises(ValueError, match="640"):
        validate_esm_cache(pairs, tmp_path)

    torch.save(torch.zeros(4, 640), tmp_path / "BDBT_A.pt")
    assert validate_esm_cache(pairs, tmp_path) == {"checked_targets": 1}


def test_external_metrics_uses_frozen_threshold_and_deterministic_bootstrap():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.7, 0.9])

    first = external_metrics(
        labels, probabilities, 0.5959881544, bootstrap_replicates=20
    )
    second = external_metrics(
        labels, probabilities, 0.5959881544, bootstrap_replicates=20
    )

    assert first == second
    assert first["threshold"] == 0.5959881544
    assert first["F1"] == pytest.approx(0.8)
    assert first["AUROC_95CI"] == [1.0, 1.0]
    assert first["AUPRC_95CI"] == [1.0, 1.0]


def test_primary_gate_requires_counts_manifest_esm_and_deterministic_rerun():
    summary = {
        "positive_pairs": 100,
        "negative_pairs": 100,
        "all_esm_valid": True,
        "manifest_complete": True,
        "rerun_identical": True,
    }

    assert passes_primary_gate(summary)
    summary["negative_pairs"] = 99
    assert not passes_primary_gate(summary)


def test_checkpoint_model_config_is_loaded_without_external_reselection(tmp_path):
    expected = {"predictor": {"task": "classification"}}
    checkpoint = {"config": {"model": expected}}
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    assert checkpoint_model_config(path) == expected


def test_run_frozen_inference_uses_checkpoint_and_emits_probabilities(tmp_path):
    pairs = pd.DataFrame(
        {
            "drug_id": ["BDBD_test"],
            "target_id": ["BDBT_test"],
            "canonical_smiles": ["CCO"],
            "sequence": ["AAAA"],
            "label": [1],
        }
    )
    torch.save(torch.zeros(4, 640), tmp_path / "BDBT_test.pt")
    checkpoint = PROJECT_ROOT / "checkpoints" / "best_random.pt"

    predictions = run_frozen_inference(
        pairs, tmp_path, checkpoint, device="cpu", batch_size=1
    )

    assert predictions["probability"].between(0.0, 1.0).all()
    assert predictions["frozen_prediction"].tolist() in ([0], [1])


def test_evaluation_cli_pins_the_random_checkpoint_and_validation_threshold():
    from src.cli.evaluate_bindingdb_external import build_parser

    args = build_parser().parse_args(
        [
            "--pairs", "data/external/curated/pairs.csv",
            "--esm-cache", "data/external/curated/esm2",
            "--out-dir", "data/external/curated",
        ]
    )

    assert args.checkpoint == "checkpoints/best_random.pt"
    assert args.threshold == pytest.approx(0.5959881544113159)


def test_evaluation_manifest_paths_are_project_relative():
    from src.cli.evaluate_bindingdb_external import _portable_project_path

    assert _portable_project_path(PROJECT_ROOT / "results" / "example.json") == (
        "results/example.json"
    )


def test_evaluation_cli_sets_cublas_workspace_before_loading_torch():
    env = os.environ.copy()
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import src.cli.evaluate_bindingdb_external; "
            "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == ":4096:8"
