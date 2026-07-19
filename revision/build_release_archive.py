"""Build the portable BioInteract Zenodo archive.

The archive deliberately includes the released source, configurations,
checkpoints, Davis inputs/ESM cache, canonical metrics, strict-split artifacts,
and final reproducibility documents.  It excludes local environments,
credentials, cache directories, and retired pair-specific/ablation artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PREFIX = "biointeract-v1.0.3"

FILES = (
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "RELEASE_NOTES.md",
    "BioInteract/README.md",
    "BioInteract/requirements.txt",
    "BioInteract/requirements-audit.txt",
    "BioInteract/upload_to_hf.py",
    "BioInteract/hf_space/app.py",
    "BioInteract/hf_space/README.md",
    "BioInteract/hf_space/requirements.txt",
    "BioInteract/hf_space/test_public_copy.py",
    "BioInteract/hf_space/configs/default.yaml",
    "BioInteract/src/analysis/bindingdb_external.py",
    "BioInteract/src/cli/curate_bindingdb_external.py",
    "BioInteract/src/cli/evaluate_bindingdb_external.py",
    "BioInteract/src/tests/test_bindingdb_external.py",
    "revision/recompute_entity_split_metrics.py",
    "revision/build_release_archive.py",
    "revision/promote_canonical_entity_metrics.py",
    "revision/run_strict_split_evaluation.py",
    "revision/reviewer_controls.py",
    "revision/revision_analysis.py",
    "revision/analysis/original_entity_split_metrics.json",
    "revision/analysis/random_validation_predictions.csv",
    "revision/analysis/random_test_predictions.csv",
    "revision/analysis/target_id_held_out_validation_predictions.csv",
    "revision/analysis/target_id_held_out_test_predictions.csv",
    "revision/analysis/drug_id_held_out_validation_predictions.csv",
    "revision/analysis/drug_id_held_out_test_predictions.csv",
    "revision/analysis/reproducibility_manifest.md",
    "revision/analysis/strict_split_metrics.json",
    "revision/analysis/split_diagnostics.json",
    "revision/analysis/reviewer_controls.json",
    "revision/analysis/sequence_grouped_cold_target.pt",
    "revision/analysis/sequence_grouped_cold_both.pt",
    "revision/analysis/test_entity_split_metrics.py",
    "revision/analysis/test_promote_canonical_entity_metrics.py",
    "revision/analysis/test_reviewer_controls.py",
    "revision/analysis/test_split_diagnostics.py",
    "revision/analysis/abl1_variant_audit.md",
    "revision/analysis/abl1_variant_audit.py",
    "revision/analysis/test_abl1_variant_audit.py",
    "revision/analysis/bindingdb_external_validation_report.md",
    "revision/analysis/test_bindingdb_submission_claims.py",
    "BioInteract/src/tests/test_public_figure_provenance.py",
    "submission_revision/manuscript_clean.tex",
    "submission_revision/manuscript_clean.pdf",
    "submission_revision/manuscript_marked.tex",
    "submission_revision/manuscript_marked.pdf",
    "submission_revision/supporting_information.tex",
    "submission_revision/supporting_information.pdf",
    "submission_revision/response_to_reviewers.tex",
    "submission_revision/response_to_reviewers.pdf",
    "submission_revision/references.bib",
    "submission_revision/wlscirep.cls",
    "submission_revision/jabbrv.sty",
    "submission_revision/jabbrv-ltwa-all.ldf",
    "submission_revision/jabbrv-ltwa-en.ldf",
    "submission_revision/naturemag-doi.bst",
    "docs/superpowers/specs/2026-07-19-bindingdb-external-validation-design.md",
    "docs/superpowers/plans/2026-07-19-bindingdb-external-validation.md",
)

TREES = (
    "BioInteract/src",
    "BioInteract/configs",
    "BioInteract/data/raw",
    "BioInteract/data/esm2_embeddings",
    "BioInteract/checkpoints",
    "BioInteract/hf_space/src",
    "submission_revision/figures",
)

SELECTED_RESULTS = (
    "BioInteract/results/test_random.json",
    "BioInteract/results/test_cold_target.json",
    "BioInteract/results/test_cold_drug.json",
    "BioInteract/results/figure_data/attention_distribution.json",
    "BioInteract/results/figure_data/attention_distribution.npz",
    "BioInteract/results/figure_data/fig2_performance.json",
    "BioInteract/results/figure_data/fig4_training.json",
    "BioInteract/results/figure_data/fig5_attention_sparsity.json",
    "BioInteract/results/figure_data/prediction_summary.json",
    "BioInteract/results/figure_data/predictions_random.csv",
    "BioInteract/results/figure_data/predictions_cold_target.csv",
    "BioInteract/results/figure_data/predictions_cold_drug.csv",
    "BioInteract/results/figure_data/training_curves.json",
    "BioInteract/results/external_validation/bindingdb_external_manifest.json",
    "BioInteract/results/external_validation/bindingdb_external_summary.json",
)

EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".git", ".env"}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files(*, include_esm_cache: bool = True) -> list[Path]:
    selected: set[Path] = set()
    for rel in (*FILES, *SELECTED_RESULTS):
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Required release file is missing: {path}")
        selected.add(path)
    for rel in TREES:
        if rel == "BioInteract/data/esm2_embeddings" and not include_esm_cache:
            continue
        directory = ROOT / rel
        if not directory.is_dir():
            raise FileNotFoundError(f"Required release directory is missing: {directory}")
        for path in directory.rglob("*"):
            if path.is_file() and not any(part in EXCLUDED_PARTS for part in path.parts):
                selected.add(path)
    return sorted(selected, key=lambda path: path.relative_to(ROOT).as_posix())


def build_archive(output: Path, *, include_esm_cache: bool = True) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = release_files(include_esm_cache=include_esm_cache)
    if output.exists():
        output.unlink()
    hashes: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, f"{ARCHIVE_PREFIX}/{relative}")
            hashes.append(f"{file_digest(path)}  {relative}")
        if not include_esm_cache:
            zenodo_readme = ROOT / "revision" / "ZENODO_ARCHIVE_README.md"
            archive.write(zenodo_readme, f"{ARCHIVE_PREFIX}/ZENODO_ARCHIVE_README.md")
            hashes.append(f"{file_digest(zenodo_readme)}  ZENODO_ARCHIVE_README.md")
        archive.writestr(f"{ARCHIVE_PREFIX}/SHA256SUMS.txt", "\n".join(hashes) + "\n")
    return len(files), output.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BioInteract Zenodo release ZIP.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tmp" / "zenodo_release" / "biointeract-v1.0.3.zip",
        help="Output ZIP path (default: tmp/zenodo_release/biointeract-v1.0.3.zip).",
    )
    parser.add_argument(
        "--omit-esm-cache",
        action="store_true",
        help="Build the lean Zenodo package and document Git-LFS cache retrieval.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    count, size = build_archive(output, include_esm_cache=not args.omit_esm_cache)
    print(f"Created {output} with {count} files ({size / 1024**2:.1f} MiB).")


if __name__ == "__main__":
    main()
