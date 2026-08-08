"""Emit the matched-protocol LaTeX table and provenance table from the run log.

The manuscript table and the Supporting Information provenance table are both
generated from ``baseline_suite.json`` so that no value is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The main-text table reports the single-variable ladder that answers the
# reviewer's ESM-2 question: the same molecular graph encoder throughout, first
# without a protein language model, then with one, then with cross-attention
# replacing concatenation. The remaining architectures were run on the random
# split only as a breadth check and are reported in the Supporting Information.
LADDER_ROWS = [
    ("GraphDTA", "Graph + sequence CNN", "No", r"GraphDTA\cite{nguyen2021graphdta}"),
    ("ESM2-GraphConcat", "Graph + PLM, concatenation", "Yes", "ESM2-GraphConcat"),
    ("BioInteract", "Graph + PLM, cross-attention", "Yes",
     r"\textbf{BioInteract (ours)}"),
]
EXTRA_ROWS = [
    ("DeepDTA", "Sequence CNN", "No", r"DeepDTA\cite{ozturk2018deepdta}"),
    ("ESM2-Bilinear", "Graph + PLM, bilinear attention", "Yes",
     r"ESM2-Bilinear\cite{bai2023drugban}"),
]
MODEL_ROWS = LADDER_ROWS + EXTRA_ROWS
PROTOCOLS = ["random", "target_id_held_out", "drug_id_held_out"]


def _cell(value: float | None, best: bool) -> str:
    if value is None:
        return "---"
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if best else text


def main(argv: list[str] | None = None) -> None:
    """Write both LaTeX fragments to the output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=str(ROOT / "revision_v2" / "analysis" / "baseline_suite.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "revision_v2" / "analysis"))
    args = parser.parse_args(argv)

    runs = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    lookup = {(run["model"], run["protocol"]): run for run in runs}
    available = [p for p in PROTOCOLS if any(k[1] == p for k in lookup)]

    def render(rows: list[tuple[str, str, str, str]], protocols: list[str]) -> str:
        """Render one table body, bolding the best value within each column."""
        best: dict[tuple[str, str], float] = {}
        for protocol in protocols:
            for metric in ("AUROC", "AUPRC"):
                values = [
                    lookup[(name, protocol)]["metrics"][metric]
                    for name, *_ in rows if (name, protocol) in lookup
                ]
                if values:
                    best[(protocol, metric)] = max(values)
        lines = []
        for name, paradigm, uses_esm, label in rows:
            cells = [label, paradigm, uses_esm]
            for protocol in protocols:
                record = lookup.get((name, protocol))
                for metric in ("AUROC", "AUPRC"):
                    value = record["metrics"][metric] if record else None
                    cells.append(_cell(value, value is not None
                                       and abs(value - best[(protocol, metric)]) < 1e-12))
            lines.append(" & ".join(cells) + r" \\")
        return "\n".join(lines)

    table_body = render(LADDER_ROWS, available)
    # The Supporting Information table repeats the ladder so that the breadth
    # check can be read against it on the protocol they share. It carries no
    # bibliography, so citation macros are stripped from that copy.
    breadth_body = re.sub(r"\\cite\{[^}]*\}", "", render(LADDER_ROWS + EXTRA_ROWS, ["random"]))

    # The provenance table documents exactly the runs the manuscript reports, so
    # it is filtered to the models that appear in Table 1 or Table S15.
    reported = {name for name, *_ in MODEL_ROWS}
    provenance = []
    for run in sorted((r for r in runs if r["model"] in reported),
                      key=lambda r: (PROTOCOLS.index(r["protocol"])
                                     if r["protocol"] in PROTOCOLS else 99, r["model"])):
        provenance.append(
            " & ".join([
                run["model"],
                run["protocol"].replace("_", r"\_"),
                f"{run['trainable_parameters']:,}".replace(",", "{,}"),
                str(run["epochs_completed"]),
                str(run["best_epoch"]),
                f"{run['best_validation_auroc']:.4f}",
                f"{run['metrics']['threshold_from_validation']:.4f}",
                f"{run['metrics']['AUROC']:.4f}",
                f"{run['metrics']['AUPRC']:.4f}",
            ]) + r" \\"
        )

    out_dir = Path(args.out_dir)
    (out_dir / "matched_table_body.tex").write_text(table_body + "\n", encoding="utf-8")
    (out_dir / "matched_breadth_body.tex").write_text(breadth_body + "\n", encoding="utf-8")
    (out_dir / "matched_provenance_body.tex").write_text(
        "\n".join(provenance) + "\n", encoding="utf-8"
    )
    print(f"protocols rendered: {available}")
    print("--- main-text ladder ---")
    print(table_body)
    print("--- Supporting Information breadth check (random split) ---")
    print(breadth_body)


if __name__ == "__main__":
    main()
