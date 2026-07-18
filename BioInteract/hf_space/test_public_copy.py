"""Scope checks for the publicly released BioInteract code and Space."""

import json
from pathlib import Path


SPACE = Path(__file__).resolve().parent
PROJECT = SPACE.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def test_space_retains_only_the_custom_demonstration_workflow():
    app = _read(SPACE / "app.py")

    required = (
        "binary high-affinity interaction classification",
        "model-native atom-residue attention attribution",
        "high-affinity-class classifier score (sigmoid output)",
        "512-residue",
        "1,200-residue",
        "not numerically equivalent",
        "unknown-domain representation",
        "initialised during application startup",
        "target-id-held-out",
        "not physical contacts",
    )
    for phrase in required:
        assert phrase in app

    prohibited = (
        "_fixed_cases",
        "show_case_study",
        "case studies",
        "high-affinity-class probability",
        "far from threshold",
        "near threshold",
        "confidence",
        "probability",
        "ax.axvline(0.5",
        "threshold",
        "initialises on the first request",
        "binding score",
    )
    for phrase in prohibited:
        assert phrase not in app


def test_readmes_disclose_supported_task_limits_and_runner_names():
    space_readme = _read(SPACE / "readme.md")
    root_readme = _read(PROJECT / "readme.md")

    required_scope = (
        "binary high-affinity interaction classification",
        "model-native atom-residue attention attribution",
        "classifier score",
        "512-residue",
        "1,200-residue",
        "not numerically equivalent",
        "unknown-domain representation",
        "target-id-held-out",
        "not physical contacts",
    )
    for document in (space_readme, root_readme):
        for phrase in required_scope:
            assert phrase in document

    assert "src.experiments.run_split_final" in root_readme
    assert "revision/run_strict_split_evaluation.py" in root_readme
    for document in (space_readme, root_readme):
        for phrase in ("case studies", "binding probability", "hotspot", "binding-pocket mapping"):
            assert phrase not in document


def test_shipped_example_is_global_statistics_only_and_claim_safe():
    payload = json.loads((SPACE / "examples" / "interpretability_report.json").read_text(encoding="utf-8"))
    assert set(payload) == {"global_stats"}
    assert payload["global_stats"]

    public_json = json.dumps(payload).lower()
    for phrase in (
        "abl1",
        "case_studies",
        "hotspot",
        "pocket",
        "resistance",
        "mutant",
        "contact",
        "probability",
    ):
        assert phrase not in public_json


def test_public_operational_sources_do_not_emit_unsupported_case_or_structure_claims():
    files = (
        PROJECT / "configs" / "default.yaml",
        PROJECT / "src" / "cli" / "interpret.py",
        PROJECT / "src" / "data" / "protein_feat.py",
        PROJECT / "src" / "data" / "split.py",
        PROJECT / "src" / "analysis" / "generate_figures.py",
        PROJECT / "src" / "analysis" / "generate_comparison_figures.py",
    )
    public_source = "\n".join(_read(path) for path in files)

    assert "configured domain-label channel" in public_source
    assert "none" in public_source
    assert "target-id-held-out" in public_source
    for phrase in (
        "case study",
        "case_stud",
        "hotspot",
        "binding pocket",
        "binding-site",
        "pdb-validated",
        "resistance",
        "mutant",
        "cold-target",
        "predicted binding probability",
        "high-confidence prediction",
    ):
        assert phrase not in public_source
