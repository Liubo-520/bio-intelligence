# BindingDB Kd-only External Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an auditable strict double-novel BindingDB Kd-only blind evaluation of the frozen BioInteract random-split checkpoint, and revise the submission only if the prespecified cohort passes all gates.

**Architecture:** A pure curation module turns a fixed BindingDB TSV archive and the released Davis inputs into one derived, local-only pair table plus a provenance manifest. A separate evaluator validates ESM tensors, reconstructs the checkpoint configuration, produces full-precision probabilities, and computes frozen-threshold metrics and deterministic bootstrap intervals. The manuscript/rebuttal update consumes only a verified summary JSON, never intermediate console output.

**Tech Stack:** Python 3.10, pandas, RDKit, PyTorch, fair-esm, scikit-learn, pytest, LaTeX/BibTeX, PowerShell, Git/GitHub/Zenodo.

## Global Constraints

- Follow [the approved design](../specs/2026-07-19-bindingdb-external-validation-design.md) exactly: BindingDB Kd only; `Kd < 30 nM`; strict double-novel identity exclusion against all released Davis entities; frozen `best_random.pt`; frozen threshold `0.5959881544`.
- Retain only exact numeric Kd values and valid single-chain protein records with a sequence of 1--1,200 standard amino-acid residues; reject censored values, non-Kd assays, complexes, malformed structures, and inconsistent replicate labels.
- No model retraining, fine-tuning, threshold selection, calibration, sampling of negatives, or external-label-informed choices.
- Require at least 100 positive and 100 negative post-filtered pairs, complete provenance, valid ESM tensors, and byte-identical deterministic rerun metrics before modifying any scientific claim.
- Store BindingDB archives, derived pair-level data, ESM cache, and prediction CSVs under `BioInteract/data/external/` and keep them untracked; commit only code, tests, manifests, aggregate summaries, manuscript sources, and release metadata.
- Preserve unrelated dirty files. Do not stage `.env`, templates, old revision copies, raw downloads, caches, or user-generated `tmp/` files.

---

### Task 1: Encode the strict curation contract in failing tests

**Files:**
- Create: `BioInteract/src/tests/test_bindingdb_external.py`
- Create: `BioInteract/src/analysis/bindingdb_external.py`

**Interfaces:**
- Consumes: in-memory BindingDB-like `pandas.DataFrame` and Davis ligand/sequence sets.
- Produces: `curate_measurements(frame, davis_smiles, davis_sequences) -> tuple[pd.DataFrame, dict]` and `canonicalize_smiles(smiles) -> tuple[str, str] | None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_curate_measurements_keeps_only_exact_kd_single_chain_double_novel_consistent_pairs():
    frame = pd.DataFrame([
        record("CCO", "AAAC", "10"),
        record("CCO", "AAAC", "20"),
        record("CCC", "BBBB", "50"),
        record("CCC", "BBBB", "10"),
        record("CCN", "CCCC", ">10"),
        record("CCCl", "DDDD", "20", chains="2"),
        record("CCBr", "EEEE", "40"),
    ])
    pairs, audit = curate_measurements(frame, {canonicalize_smiles("CCBr")[0]}, {"ZZZZ"})
    assert pairs[["kd_nM", "label"]].to_dict("records") == [{"kd_nM": 15.0, "label": 1}]
    assert audit["excluded_conflicting_pair"] == 1
    assert audit["excluded_censored_kd"] == 1
    assert audit["excluded_multichain"] == 1
    assert audit["excluded_davis_ligand"] == 1

def test_curate_measurements_excludes_exact_davis_sequence_even_when_target_ids_differ():
    frame = pd.DataFrame([record("CCO", "MPEPTIDE", "100")])
    pairs, audit = curate_measurements(frame, set(), {"MPEPTIDE"})
    assert pairs.empty
    assert audit["excluded_davis_target"] == 1
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest BioInteract/src/tests/test_bindingdb_external.py -q`

Expected: import failure because `src.analysis.bindingdb_external` does not exist.

- [ ] **Step 3: Commit the red test only**

Run: `git add BioInteract/src/tests/test_bindingdb_external.py && git commit -m "test: define BindingDB external curation contract"`

Expected: one test-only commit with no production implementation.

### Task 2: Implement deterministic BindingDB curation and provenance

**Files:**
- Create: `BioInteract/src/analysis/bindingdb_external.py`
- Create: `BioInteract/src/cli/curate_bindingdb_external.py`
- Modify: `.gitignore`
- Test: `BioInteract/src/tests/test_bindingdb_external.py`

**Interfaces:**
- Consumes: `--archive BioInteract/data/external/BindingDB_All_202607_tsv.zip`, `--out-dir BioInteract/data/external/bindingdb_202607`, and released `BioInteract/data/raw/davis/` tables.
- Produces: untracked `pairs.csv`, `targets.fasta`, and `curation_audit.json`; tracked `BioInteract/results/external_validation/bindingdb_manifest_template.json` after the run supplies finalized hashes and counts.

- [ ] **Step 1: Implement the pure functions required by the tests**

```python
def canonicalize_smiles(smiles: object) -> tuple[str, str] | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return canonical, Chem.MolToInchiKey(mol)

def parse_exact_kd(value: object) -> float | None:
    text = str(value).strip()
    if not re.fullmatch(r"(?:0|[1-9]\\d*)(?:\\.\\d+)?(?:[eE][+-]?\\d+)?", text):
        return None
    numeric = float(text)
    return numeric if math.isfinite(numeric) and numeric > 0 else None

def curate_measurements(frame: pd.DataFrame, davis_smiles: set[str], davis_sequences: set[str]) -> tuple[pd.DataFrame, dict]:
    """Return threshold-consistent, single-chain, double-novel pair medians and exclusion counts."""
```

The implementation must process `Kd (nM)`, `Ligand SMILES`, `BindingDB Target Chain Sequence`, `Number of Protein Chains in Target`, and `BindingDB Reactant_set_id`; normalize sequences to uppercase; accept only `ACDEFGHIKLMNPQRSTVWY`; derive `drug_id`/`target_id` from stable SHA-256 prefixes; and group by canonical ligand SMILES plus full sequence before applying the median Kd.

- [ ] **Step 2: Implement the streaming CLI and manifest writer**

```python
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--source-url", required=True)
parser.add_argument("--source-version", required=True)
parser.add_argument("--chunksize", type=int, default=200_000)
```

Read the ZIP's sole TSV member with `pd.read_csv(..., sep="\\t", dtype=str, chunksize=args.chunksize, low_memory=False)`, aggregate audit counts over chunks, run global pair reconciliation after concatenation, save only local derived artifacts, and write archive URL/version/UTC timestamp/SHA-256/file size/column names/counts to JSON.

- [ ] **Step 3: Prevent accidental raw-data commits**

Append exactly these ignore lines:

```gitignore
BioInteract/data/external/
BioInteract/results/external_validation/*.csv
BioInteract/results/external_validation/*.pt
```

- [ ] **Step 4: Run GREEN tests**

Run: `python -m pytest BioInteract/src/tests/test_bindingdb_external.py -q`

Expected: all curation tests pass.

- [ ] **Step 5: Commit curation code and tests**

Run: `git add .gitignore BioInteract/src/analysis/bindingdb_external.py BioInteract/src/cli/curate_bindingdb_external.py BioInteract/src/tests/test_bindingdb_external.py && git commit -m "feat: add strict BindingDB external curation"`

### Task 3: Encode frozen inference, ESM validation, and metrics in tests

**Files:**
- Modify: `BioInteract/src/tests/test_bindingdb_external.py`
- Modify: `BioInteract/src/analysis/bindingdb_external.py`

**Interfaces:**
- Consumes: curated pair table, ESM directory, frozen probability array, fixed threshold.
- Produces: `validate_esm_cache(pairs, cache_dir, esm2_dim=640) -> None`, `external_metrics(labels, probabilities, threshold, bootstrap_seed=42, bootstrap_replicates=600) -> dict`, and `passes_primary_gate(summary) -> bool`.

- [ ] **Step 1: Write failing evaluation tests**

```python
def test_validate_esm_cache_rejects_missing_or_wrong_dimensional_embeddings(tmp_path):
    pairs = pd.DataFrame({"target_id": ["BDBT_A"], "sequence": ["AAAA"]})
    with pytest.raises(FileNotFoundError):
        validate_esm_cache(pairs, tmp_path)
    torch.save(torch.zeros(4, 639), tmp_path / "BDBT_A.pt")
    with pytest.raises(ValueError, match="640"):
        validate_esm_cache(pairs, tmp_path)

def test_external_metrics_uses_frozen_threshold_and_deterministic_bootstrap():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.7, 0.9])
    first = external_metrics(labels, probabilities, 0.5959881544, bootstrap_replicates=20)
    second = external_metrics(labels, probabilities, 0.5959881544, bootstrap_replicates=20)
    assert first == second
    assert first["threshold"] == 0.5959881544
    assert first["F1"] == pytest.approx(0.8)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest BioInteract/src/tests/test_bindingdb_external.py -q`

Expected: failure because the validation/metrics interfaces do not yet exist.

- [ ] **Step 3: Implement minimal strict validation and metrics**

Use `torch.load(..., weights_only=True)` and require an exact `(sequence_length, 640)` or longer-first-dimension tensor that can be truncated to sequence length. `external_metrics` must fail for one-class data, call `classification_metrics` with the provided threshold, and use stratified bootstrap sampling with `np.random.default_rng(42)` and 600 successful replicate metrics. Persist 2.5th/97.5th percentiles for AUROC and AUPRC. `passes_primary_gate` must require `positive_pairs >= 100`, `negative_pairs >= 100`, `all_esm_valid is True`, `manifest_complete is True`, and `rerun_identical is True`.

- [ ] **Step 4: Run GREEN tests and commit**

Run: `python -m pytest BioInteract/src/tests/test_bindingdb_external.py -q`

Expected: all tests pass.

Run: `git add BioInteract/src/analysis/bindingdb_external.py BioInteract/src/tests/test_bindingdb_external.py && git commit -m "feat: validate frozen external evaluation inputs"`

### Task 4: Implement frozen checkpoint inference and result verification

**Files:**
- Create: `BioInteract/src/cli/evaluate_bindingdb_external.py`
- Modify: `BioInteract/src/tests/test_bindingdb_external.py`
- Modify: `BioInteract/src/analysis/bindingdb_external.py`

**Interfaces:**
- Consumes: `pairs.csv`, ESM cache, `BioInteract/checkpoints/best_random.pt`.
- Produces: local `predictions.csv`, tracked aggregate `BioInteract/results/external_validation/bindingdb_external_summary.json`, and tracked `BioInteract/results/external_validation/bindingdb_external_manifest.json`.

- [ ] **Step 1: Write the failing checkpoint configuration test**

```python
def test_checkpoint_model_config_is_used_without_external_reselection(tmp_path):
    checkpoint = {"config": {"model": {"predictor": {"task": "classification"}}}}
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    assert checkpoint_model_config(path) == checkpoint["config"]["model"]
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest BioInteract/src/tests/test_bindingdb_external.py -q`

Expected: failure because `checkpoint_model_config` is not implemented.

- [ ] **Step 3: Implement full-precision inference**

```python
def checkpoint_model_config(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    return config["model"] if "model" in config else config
```

The CLI must reconstruct `BioInteract(checkpoint_model_config(checkpoint))`, load only `model_state_dict`, use `model.predict_proba(...)` with no autocast, and call `torch.use_deterministic_algorithms(True)` before CUDA model construction. It must not invoke `select_validation_threshold`. It writes only after validating all ESM tensors and joins probabilities back to the exact curated pair rows.

- [ ] **Step 4: Run GREEN tests**

Run: `python -m pytest BioInteract/src/tests/test_bindingdb_external.py -q`

Expected: all tests pass.

- [ ] **Step 5: Execute a small synthetic smoke test and commit**

Run: `python -m src.cli.evaluate_bindingdb_external --help`

Expected: exits 0 and documents the fixed checkpoint/threshold defaults.

Run: `git add BioInteract/src/analysis/bindingdb_external.py BioInteract/src/cli/evaluate_bindingdb_external.py BioInteract/src/tests/test_bindingdb_external.py && git commit -m "feat: add frozen BindingDB blind evaluator"`

### Task 5: Run the prespecified external experiment and enforce gates

**Files:**
- Create (untracked): `BioInteract/data/external/BindingDB_BindingDB_Articles_202607_tsv.zip`
- Create (untracked): `BioInteract/data/external/bindingdb_articles_202607/`
- Create: `BioInteract/results/external_validation/bindingdb_external_manifest.json`
- Create: `BioInteract/results/external_validation/bindingdb_external_summary.json`
- Create: `revision/analysis/bindingdb_external_validation_report.md`

**Interfaces:**
- Consumes: fixed archive, curation/evaluation CLIs, ESM extractor, frozen checkpoint.
- Produces: one passing aggregate result or one transparent feasibility report.

- [x] **Step 1: Download and fingerprint the dated BindingDB snapshot**

Run: `Invoke-WebRequest -Uri 'https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202607_tsv.zip' -OutFile 'BioInteract/data/external/BindingDB_BindingDB_Articles_202607_tsv.zip'`

Expected: nonempty ZIP; record the exact SHA-256 and file size through the curation manifest.

- [x] **Step 2: Curate before generating any external ESM embeddings**

Run: `Set-Location BioInteract; python -m src.cli.curate_bindingdb_external --archive data/external/BindingDB_BindingDB_Articles_202607_tsv.zip --out-dir data/external/bindingdb_articles_202607 --source-url https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202607_tsv.zip --source-version BindingDB_BindingDB_Articles_202607`

Expected: curation JSON reports every exclusion class and the final strict pair/positive/negative counts.

- [x] **Step 3: Stop or extract ESM according to the gate**

If either class has fewer than 100 pairs, write the feasibility report and skip all remaining experiment/revision steps. Otherwise run:

`Set-Location BioInteract; python -m src.tools.extract_esm2 --fasta data/external/bindingdb_articles_202607/targets.fasta --output_dir data/external/bindingdb_articles_202607/esm2_t30_150M --model esm2_t30_150M_UR50D --max_length 1200 --device cuda --skip_existing`

Expected: one 640-dimensional tensor for every curated target.

- [x] **Step 4: Evaluate twice and compare complete artifacts**

Run twice with the same paths, using `--device auto`; compare the summary JSON and the SHA-256 of probability columns. Set `rerun_identical` only when both are identical.

Expected: either a gate-passing summary with metrics/intervals or a report explaining the exact failed condition; no modified manuscript yet.

- [ ] **Step 5: Perform independent artifact checks and commit scientific artifacts**

Run: `python -m pytest BioInteract/src/tests/test_bindingdb_external.py -q`

Run: `git add BioInteract/results/external_validation/*.json revision/analysis/bindingdb_external_validation_report.md && git commit -m "results: record strict BindingDB external validation"`

### Task 6: Revise manuscript, response, and release only after a passing gate

**Files:**
- Modify: `submission_revision/manuscript_clean.tex`
- Modify: `submission_revision/manuscript_marked.tex`
- Modify: `submission_revision/supporting_information.tex`
- Modify: `submission_revision/response_to_reviewers.tex`
- Modify: `submission_revision/references.bib`
- Modify: `submission_revision/consistency_audit_report.md`
- Modify: `revision/build_release_archive.py`
- Modify: `RELEASE_NOTES.md`, `.zenodo.json`, `CITATION.cff`
- Create: updated clean/marked/SI/response PDFs and a new reproducibility audit

**Interfaces:**
- Consumes: verified `bindingdb_external_summary.json` and report only.
- Produces: a submission package whose every external-validation claim maps to an actual artifact and a v1.0.2 release only after final PDF/provenance checks.

- [ ] **Step 1: Write failing static scientific-claim tests**

Create `revision/analysis/test_bindingdb_submission_claims.py` that reads the four TeX files and asserts: (a) the source is BindingDB; (b) it says Kd-only and `30 nM`; (c) it states strict double-novel filtering and frozen random checkpoint/threshold; (d) each displayed number matches the summary JSON; (e) it contains the limitation that this does not establish general DTI performance; and (f) the Reviewer 2 Comment 3 response states where the new material appears.

- [ ] **Step 2: Run the claim test and verify RED**

Run: `python -m pytest revision/analysis/test_bindingdb_submission_claims.py -q`

Expected: failure because no passing external-validation text yet exists.

- [ ] **Step 3: Make the minimal manuscript and response updates**

Add one Methods paragraph, one Results paragraph/table row, one SI methods/results table, and one Reviewer 2 Comment 3 response. Use only actual final values from the verified JSON; do not modify Davis results or make structural/attention claims from BindingDB.

- [ ] **Step 4: Compile, verify, and commit the submission revision**

Run the existing TeX build sequence for clean manuscript, marked manuscript, SI, and response letter; run the claim test plus all curation tests; visually inspect changed PDF pages; and commit only reviewed sources/PDFs/figures.

- [ ] **Step 5: Publish v1.0.2 only after all gates pass**

Update the release archive allowlist for code/tests/aggregate manifests while excluding raw BindingDB, derived pairs, ESM tensors, and predictions. Build/verify archive hash, create GitHub Release `v1.0.2`, mint/upload the matching Zenodo version, then write returned DOI/version/hash into all public files. Stage and commit only factual release metadata. Do not alter v1.0.1.

## Plan self-review

- Design coverage: Tasks 1--2 implement measurement eligibility, identity novelty, replicate reconciliation, provenance, and raw-data exclusion. Tasks 3--4 lock ESM, checkpoint, threshold, metrics, confidence intervals, and determinism. Task 5 enforces the 100/100 stop rule. Task 6 makes writing/release strictly conditional on verified artifacts.
- Placeholder scan: no undecided source, label rule, checkpoint, threshold, gate, output path, or verification command remains.
- Interface consistency: curation writes `pairs.csv`/`targets.fasta`; ESM consumes `targets.fasta`; evaluation consumes `pairs.csv` plus ESM tensors; revision consumes only aggregate JSON/report.
