"""Render the manuscript's Davis baseline-comparison figure.

The baseline rows retain the first-submission experiment values. The BioInteract
row is updated from the current canonical entity-split evaluation artifacts so
that every row in Table 1 is represented in Figure 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.figure_common import (
    PALETTE,
    configure_matplotlib,
    panel_label,
    save_figure,
    soften_axes,
    write_manifest,
)


configure_matplotlib()

METHODS_ALL = [
    "DeepDTA\n(2018)",
    "GraphDTA\n(2021)",
    "AttentionDTA\n(2019)",
    "MolTrans\n(2021)",
    "TransformerCPI\n(2020)",
    "DrugBAN\n(2023)",
    "BioInteract\n(Ours)",
]
RANDOM_VALUES = {
    "AUROC": [0.878, 0.893, 0.900, 0.907, 0.910, 0.915, 0.904],
    "AUPRC": [0.352, 0.403, 0.425, 0.480, 0.492, 0.530, 0.560],
}
SPLIT_METHODS = [
    "DeepDTA",
    "GraphDTA",
    "AttentionDTA",
    "MolTrans",
    "TransformerCPI",
    "DrugBAN",
    "BioInteract",
]
SPLIT_AUROC = np.array(
    [
        [0.878, 0.783, 0.592],
        [0.893, 0.815, 0.621],
        [0.900, 0.838, 0.643],
        [0.907, 0.856, 0.668],
        [0.910, 0.862, 0.672],
        [0.915, 0.874, 0.695],
        [0.904, 0.930, 0.733],
    ]
)


def _colours(count: int) -> list[str]:
    values = [PALETTE["mist"]] * count
    values[-1] = PALETTE["gold"]
    return values


def render_random_split() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.15, wspace=0.20)
    y = np.arange(len(METHODS_ALL))
    for index, metric in enumerate(("AUROC", "AUPRC")):
        ax = axes[index]
        values = RANDOM_VALUES[metric]
        bars = ax.barh(y, values, color=_colours(len(values)), edgecolor="white")
        bars[-1].set_edgecolor(PALETTE["ink"])
        bars[-1].set_linewidth(1.3)
        for bar, value in zip(bars, values):
            ax.text(
                value + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                color=PALETTE["ink"],
                fontsize=9.5,
                fontweight="bold" if value == values[-1] else "normal",
            )
        ax.set_xlabel(metric)
        ax.set_xlim((0.82, 0.95) if metric == "AUROC" else (0.30, 0.66))
        ax.set_yticks(y)
        if index == 0:
            ax.set_yticklabels(METHODS_ALL)
        else:
            ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_title(f"Random-split {metric}", pad=12)
        soften_axes(ax, "x")
        panel_label(ax, chr(ord("A") + index))
    fig.suptitle(
        "Comparison with baseline DTI models on the Davis benchmark",
        x=0.52,
        y=0.98,
        fontsize=14,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    save_figure(
        fig,
        "fig_comparison_random",
        metadata={
            "title": "Baseline comparison on the random split",
            "methods": METHODS_ALL,
            "metrics": RANDOM_VALUES,
        },
    )


def render_identifier_splits() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    fig.subplots_adjust(top=0.85, bottom=0.14)
    image = ax.imshow(SPLIT_AUROC, cmap=plt.cm.YlGnBu, aspect="auto", vmin=0.55, vmax=0.95)
    for row in range(SPLIT_AUROC.shape[0]):
        for col in range(SPLIT_AUROC.shape[1]):
            value = SPLIT_AUROC[row, col]
            ax.text(
                col,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.82 else PALETTE["ink"],
                fontsize=10,
                fontweight="bold" if row == len(SPLIT_METHODS) - 1 else "normal",
            )
    ax.set_xticks(np.arange(3), ["Random", "Target-ID-held-out", "Drug-ID-held-out"])
    ax.set_yticks(np.arange(len(SPLIT_METHODS)), SPLIT_METHODS)
    ax.set_title("AUROC across identifier-based splitting protocols", pad=14)
    panel_label(ax, "C")
    ax.add_patch(
        Rectangle(
            (-0.5, len(SPLIT_METHODS) - 1.5),
            3,
            1,
            fill=False,
            edgecolor=PALETTE["gold"],
            linewidth=2.0,
        )
    )
    colourbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colourbar.set_label("AUROC")
    save_figure(
        fig,
        "fig_comparison_splits",
        metadata={
            "title": "AUROC comparison across identifier-based splits",
            "methods": SPLIT_METHODS,
            "splits": ["random", "target_id_held_out", "drug_id_held_out"],
            "auroc_matrix": SPLIT_AUROC,
        },
    )


def main() -> None:
    render_random_split()
    render_identifier_splits()
    write_manifest(["fig_comparison_random", "fig_comparison_splits"])


if __name__ == "__main__":
    main()
