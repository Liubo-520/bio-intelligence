"""Render the figures that changed in this revision round.

Three figures are produced:

``fig_comparison``
    Reviewer 3 asked that the separate random-split panel and the split-by-split
    panel be merged, since they repeated the same values. This renders one
    two-panel figure with AUROC and AUPRC as grouped bars over all three
    identifier-based protocols, from the matched-protocol baseline suite.

``fig_identity_bins``
    Reviewer 1 asked for the identity-stratified cold-target results with the
    supporting counts, and Reviewer 3 asked that the bin-count histogram be
    replaced by the metrics themselves. This single panel plots AUROC and AUPRC
    per identity stratum and annotates the target, pair and positive counts.

``fig_webserver``
    Reviewers 1 and 3 both reported that the web-interface attribution panel was
    unreadable. This composes the captured interface screenshot with vector
    re-renderings of the two attribution views returned for the same input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "revision_v2" / "analysis"

PALETTE = {
    "ink": "#20303c",
    "teal": "#2f7f79",
    "gold": "#d89a34",
    "brick": "#b55a44",
    "slate": "#5f6f7b",
    "sky": "#5c9ead",
    "sand": "#c9b48a",
}
SPLIT_COLOURS = (PALETTE["teal"], PALETTE["gold"], PALETTE["brick"])
PROTOCOL_LABELS = {
    "random": "Random",
    "target_id_held_out": "Target-ID-held-out",
    "drug_id_held_out": "Drug-ID-held-out",
}
# Figure 1 mirrors main-text Table 1, which reports the single-variable ladder
# only. The additional architectures were run on the random split alone and are
# tabulated in the Supporting Information rather than plotted here, so that the
# figure has no ragged columns.
MODEL_ORDER = ["GraphDTA", "ESM2-GraphConcat", "BioInteract"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.edgecolor": PALETTE["ink"],
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _panel(ax, label: str, x: float = -0.09, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=PALETTE["ink"], va="bottom", clip_on=False)


def _soften(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#d9dfdf", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"wrote {output_dir / stem}.pdf")


def render_comparison(output_dir: Path, suite_path: Path) -> None:
    """Plot matched-protocol AUROC and AUPRC for every model and protocol."""
    runs = json.loads(suite_path.read_text(encoding="utf-8"))
    lookup = {(run["model"], run["protocol"]): run["metrics"] for run in runs}
    protocols = [key for key in PROTOCOL_LABELS if any(k[1] == key for k in lookup)]
    models = [name for name in MODEL_ORDER if any(k[0] == name for k in lookup)]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0))
    fig.subplots_adjust(hspace=0.42)
    positions = np.arange(len(models))
    width = 0.8 / max(len(protocols), 1)

    for axis_index, (ax, metric) in enumerate(zip(axes, ("AUROC", "AUPRC"))):
        for index, protocol in enumerate(protocols):
            values = [lookup.get((model, protocol), {}).get(metric, np.nan) for model in models]
            offset = (index - (len(protocols) - 1) / 2) * width
            bars = ax.bar(positions + offset, values, width * 0.92,
                          color=SPLIT_COLOURS[index % len(SPLIT_COLOURS)],
                          label=PROTOCOL_LABELS[protocol],
                          edgecolor="white", linewidth=0.6)
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                            ha="center", va="bottom", fontsize=6.6, rotation=90,
                            color=PALETTE["ink"])
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"$\\bf{{{m}}}$".replace("-", "\\text{-}") if m == "BioInteract" else m for m in models],
            fontsize=9,
        )
        ax.set_ylabel(f"Test {metric}")
        ax.set_ylim(0, 1.16 if metric == "AUROC" else 0.86)
        _soften(ax)
        _panel(ax, chr(ord("A") + axis_index))
        if axis_index == 0:
            ax.legend(frameon=False, ncol=len(protocols), loc="upper center",
                      bbox_to_anchor=(0.5, 1.16))
    _save(fig, output_dir, "fig_comparison")


def render_identity_bins(output_dir: Path, controls_path: Path) -> None:
    """Plot cold-target AUROC and AUPRC per sequence-identity stratum."""
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    section = controls["sequence_grouped_cold_target"]
    strata = section["strict_model_identity_stratification"]
    target_counts = section["full_length_identity"]["bins"]
    order = ["<0.40", "0.40--<0.60", "0.60--<0.80", ">=0.80"]
    labels = ["< 0.40", "0.40--0.60", "0.60--0.80", "$\\geq$ 0.80"]

    auroc = [strata[key]["metrics"]["AUROC"] for key in order]
    auprc = [strata[key]["metrics"]["AUPRC"] for key in order]
    pairs = [strata[key]["test_pairs"] for key in order]
    positives = [strata[key]["positive_pairs"] for key in order]
    targets = [target_counts[key] for key in order]
    prevalence = sum(positives) / sum(pairs)

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    fig.subplots_adjust(bottom=0.34)
    positions = np.arange(len(order))
    width = 0.36
    for offset, values, colour, label in (
        (-width / 2, auroc, PALETTE["teal"], "AUROC"),
        (width / 2, auprc, PALETTE["gold"], "AUPRC"),
    ):
        bars = ax.bar(positions + offset, values, width, color=colour, label=label,
                      edgecolor="white", linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color=PALETTE["ink"])

    ax.axhline(prevalence, color=PALETTE["slate"], linestyle="--", linewidth=1.0)
    ax.text(-0.42, prevalence + 0.014,
            f"positive prevalence {prevalence:.3f}", fontsize=7.6,
            color=PALETTE["slate"], ha="left")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Maximum global sequence identity to any training target", labelpad=58)
    ax.set_ylabel("Test metric")
    ax.set_ylim(0, 1.06)
    for position, (n_target, n_pair, n_pos) in enumerate(zip(targets, pairs, positives)):
        ax.annotate(
            f"{n_target} targets\n{n_pair:,} pairs\n{n_pos} positives",
            xy=(position, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -20), textcoords="offset points",
            ha="center", va="top", fontsize=7.8, color=PALETTE["slate"],
        )
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    _soften(ax)
    _save(fig, output_dir, "fig_identity_bins")


def render_webserver(output_dir: Path, screenshot: Path, attribution: Path) -> None:
    """Compose the interface screenshot with vector attribution re-renderings."""
    import matplotlib.image as mpimg

    payload = json.loads(attribution.read_text(encoding="utf-8"))
    interaction = np.asarray(payload["interaction_map"], dtype=float)
    residue_labels = payload["residue_labels"]
    top_labels = [entry[0] for entry in payload["top_residues"]]
    top_scores = [float(entry[1]) for entry in payload["top_residues"]]

    fig = plt.figure(figsize=(13.2, 7.6))
    grid = gridspec.GridSpec(2, 2, width_ratios=[1.0, 1.34], height_ratios=[1.0, 1.0],
                             wspace=0.14, hspace=0.62)

    screenshot_ax = fig.add_subplot(grid[:, 0])
    screenshot_ax.imshow(mpimg.imread(screenshot))
    screenshot_ax.set_axis_off()
    _panel(screenshot_ax, "A", x=0.0, y=1.005)

    heat_ax = fig.add_subplot(grid[0, 1])
    image = heat_ax.imshow(interaction, aspect="auto", cmap="Blues", interpolation="nearest")
    step = max(1, len(residue_labels) // 20)
    heat_ax.set_xticks(np.arange(0, len(residue_labels), step))
    heat_ax.set_xticklabels(residue_labels[::step], rotation=90, fontsize=6.4)
    heat_ax.set_yticks(np.arange(0, interaction.shape[0], 4))
    heat_ax.set_yticklabels([f"a{i + 1}" for i in range(0, interaction.shape[0], 4)], fontsize=6.8)
    heat_ax.set_xlabel("Protein residue", labelpad=3, fontsize=9)
    heat_ax.set_ylabel("Drug atom", labelpad=3, fontsize=9)
    heat_ax.set_title("Atom--residue attention attribution", pad=7, fontsize=10)
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.035, pad=0.015)
    colorbar.set_label("Attention weight", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    _panel(heat_ax, "B", x=-0.085, y=1.06)

    bar_ax = fig.add_subplot(grid[1, 1])
    bars = bar_ax.barh(top_labels[::-1], top_scores[::-1], color=PALETTE["sky"],
                       edgecolor=PALETTE["ink"], linewidth=0.5, height=0.66)
    bars[-1].set_color(PALETTE["teal"])
    for bar, score in zip(bars, top_scores[::-1]):
        bar_ax.text(score + 0.018, bar.get_y() + bar.get_height() / 2, f"{score:.3f}",
                    va="center", fontsize=7.6, color=PALETTE["ink"])
    bar_ax.set_xlim(0, 1.16)
    bar_ax.tick_params(labelsize=8)
    bar_ax.set_xlabel("Normalised residue attribution score", fontsize=9)
    bar_ax.set_title("Top 10 residue attributions", pad=7, fontsize=10)
    _soften(bar_ax, "x")
    _panel(bar_ax, "C", x=-0.085, y=1.06)

    _save(fig, output_dir, "fig_webserver")


def main(argv: list[str] | None = None) -> None:
    """Render the requested figures into the submission figure directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "submission_revision_v2" / "figures"))
    parser.add_argument("--suite", default=str(ANALYSIS / "baseline_suite.json"))
    parser.add_argument("--controls", default=str(ROOT / "revision" / "analysis" / "reviewer_controls.json"))
    parser.add_argument("--screenshot", default=str(ANALYSIS / "webserver_screenshot_crop.png"))
    parser.add_argument("--attribution", default=str(ANALYSIS / "webserver_attribution.json"))
    parser.add_argument("--only", nargs="*", default=["comparison", "identity", "webserver"])
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if "comparison" in args.only:
        render_comparison(output_dir, Path(args.suite))
    if "identity" in args.only:
        render_identity_bins(output_dir, Path(args.controls))
    if "webserver" in args.only:
        render_webserver(output_dir, Path(args.screenshot), Path(args.attribution))


if __name__ == "__main__":
    main()
