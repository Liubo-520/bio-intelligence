"""Emit the LaTeX paragraph and table for the full-release BindingDB evaluation.

The numbers come straight from the frozen-evaluation summary written by
``src.cli.evaluate_bindingdb_external``, so the manuscript cannot drift from the
recorded run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    """Write the LaTeX fragment substituted at ``%%FULLDB-RESULT%%``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default=str(ROOT / "BioInteract" / "results" / "external_validation"
                    / "bindingdb_all_202608_summary.json"),
    )
    parser.add_argument(
        "--small-summary",
        default=str(ROOT / "BioInteract" / "results" / "external_validation"
                    / "bindingdb_external_summary.json"),
    )
    parser.add_argument("--out", default=str(ROOT / "revision_v2" / "analysis" / "fulldb_result.tex"))
    args = parser.parse_args(argv)

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    small = json.loads(Path(args.small_summary).read_text(encoding="utf-8"))
    metrics, small_metrics = summary["metrics"], small["metrics"]

    def interval(record: dict, key: str) -> str:
        low, high = record[f"{key}_95CI"]
        return f"{record[key]:.3f} [{low:.3f}, {high:.3f}]"

    direction = (
        "confirms" if abs(metrics["AUROC"] - small_metrics["AUROC"]) < 0.05 else "revises"
    )

    # Thousands separators must use LaTeX-safe braces without touching decimals.
    def n(value: int) -> str:
        return f"{value:,}".replace(",", "{,}")

    fragment = (
        f"The frozen checkpoint achieved AUROC $={metrics['AUROC']:.3f}$ "
        f"(95\\% CI {metrics['AUROC_95CI'][0]:.3f}--{metrics['AUROC_95CI'][1]:.3f}) and "
        f"AUPRC $={metrics['AUPRC']:.3f}$ "
        f"(95\\% CI {metrics['AUPRC_95CI'][0]:.3f}--{metrics['AUPRC_95CI'][1]:.3f}) on this "
        f"cohort, against a positive prevalence of "
        f"{summary['positive_pairs'] / summary['total_pairs']:.3f}.  At the frozen Davis "
        f"threshold, F1 was {metrics['F1']:.3f}, precision was {metrics['Precision']:.3f} "
        f"and recall was {metrics['Recall']:.3f}.  A cohort twenty times larger therefore "
        f"{direction} the conclusion drawn from the pre-specified evaluation: this "
        "Davis-trained classifier retains only weak ranking information outside its "
        "training distribution, and its transferred decision threshold is not usable "
        "without recalibration.\n\n"
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Frozen external evaluation on both BindingDB cohorts. Both use the same\n"
        "  \\texttt{best\\_random} checkpoint, the same frozen Davis threshold of\n"
        "  0.5959881544, and the same exact-$K_d$ double-novel protocol; they differ only\n"
        "  in the source snapshot. Intervals are 600-replicate pair-stratified bootstrap\n"
        "  95\\% confidence intervals with seed 42.}\n"
        "\\label{tab:bindingdb_external}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "\\textbf{Cohort} & \\textbf{Pairs / positives} & \\textbf{AUROC} & "
        "\\textbf{AUPRC} & \\textbf{F1} & \\textbf{Recall} \\\\\n\\midrule\n"
        f"Curated articles, 202607 (pre-specified) & {n(small['total_pairs'])} / "
        f"{n(small['positive_pairs'])} & {interval(small_metrics, 'AUROC')} & "
        f"{interval(small_metrics, 'AUPRC')} & {small_metrics['F1']:.3f} & "
        f"{small_metrics['Recall']:.3f} \\\\\n"
        f"Complete release, 202608 (verification) & {n(summary['total_pairs'])} / "
        f"{n(summary['positive_pairs'])} & {interval(metrics, 'AUROC')} & "
        f"{interval(metrics, 'AUPRC')} & {metrics['F1']:.3f} & "
        f"{metrics['Recall']:.3f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    )
    Path(args.out).write_text(fragment, encoding="utf-8")
    print(fragment)


if __name__ == "__main__":
    main()
