"""Emit manuscript_clean.tex and manuscript_marked.tex from one source file.

``manuscript_source.tex`` marks every block that is new or substantively
revised in this round with ``%%REV-START`` / ``%%REV-END`` comment sentinels.
The clean build simply drops the sentinels; the marked build replaces them with
red colour groups and adds the reader note the journal expects on a marked
revision. Keeping one source removes any possibility of the two files drifting
apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "analysis"
START = "%%REV-START"
END = "%%REV-END"
MARKED_PREAMBLE = (
    "\\usepackage{xcolor}\n"
    "\\newcommand{\\rev}[1]{\\textcolor{red}{#1}}\n"
)
MARKED_NOTE = (
    "\\noindent\\textcolor{red}{\\textit{Marked revision: material that is new or "
    "substantively revised in this round is shown in red. Analyses, tables and "
    "figures removed in this round are identified in the point-by-point "
    "response.}}\\par\\medskip\n"
)


def build(source: str, marked: bool) -> str:
    """Return the clean or marked LaTeX body for the shared source text."""
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped == START:
            if marked:
                lines.append("\\begingroup\\color{red}")
            continue
        if stripped == END:
            if marked:
                lines.append("\\endgroup")
            continue
        lines.append(line)
    text = "\n".join(lines) + "\n"
    if marked:
        text = text.replace(
            "\\graphicspath{{figures/}}", "\\graphicspath{{figures/}}\n" + MARKED_PREAMBLE, 1
        )
        text = text.replace("\\thispagestyle{empty}", "\\thispagestyle{empty}\n" + MARKED_NOTE, 1)
    return text


def main(argv: list[str] | None = None) -> None:
    """Write both manuscript variants next to the source file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(HERE / "manuscript_source.tex"))
    parser.add_argument("--clean", default=str(HERE / "manuscript_clean.tex"))
    parser.add_argument("--marked", default=str(HERE / "manuscript_marked.tex"))
    args = parser.parse_args(argv)

    source = Path(args.source).read_text(encoding="utf-8")
    # Table bodies are substituted from the analysis artifacts rather than
    # transcribed, so the manuscript can never drift from the recorded runs.
    for marker, artifact in (
        ("%%MATCHED-ROWS%%", ANALYSIS / "matched_table_body.tex"),
        ("%%FULLDB-RESULT%%", ANALYSIS / "fulldb_result.tex"),
    ):
        if marker in source:
            source = source.replace(marker, artifact.read_text(encoding="utf-8").rstrip())
    if source.count(START) != source.count(END):
        raise ValueError("unbalanced %%REV-START / %%REV-END markers")
    for placeholder in ("PLACEHOLDER",):
        if placeholder in source:
            raise ValueError(f"source still contains an unresolved {placeholder}")
    Path(args.clean).write_text(build(source, marked=False), encoding="utf-8")
    Path(args.marked).write_text(build(source, marked=True), encoding="utf-8")
    print(f"wrote {args.clean} and {args.marked} ({source.count(START)} marked blocks)")


if __name__ == "__main__":
    main()
