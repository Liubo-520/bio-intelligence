"""Render revised manuscript figures from archived revision data.

The renderer changes presentation only: it reads saved numerical artifacts and
does not recompute model predictions or alter any experimental result.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE / "BioInteract" / "results" / "figure_data"
INTERPRET_DIR = WORKSPACE / "BioInteract" / "results" / "interpretability"
PALETTE = {
    "ink": "#20303c",
    "teal": "#2f7f79",
    "gold": "#d89a34",
    "brick": "#b55a44",
    "slate": "#5f6f7b",
    "cream": "#fbf8f2",
    "sky": "#5c9ead",
}
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
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def _panel(ax, label: str) -> None:
    ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=PALETTE["ink"], va="bottom", clip_on=False)


def _axes(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#d9dfdf", linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)


def _load_json(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def render_ablation(output_dir: Path) -> None:
    data = _load_json("fig_ablation.json")
    variants = data["variants"]
    matrix = np.array(data["matrix"])
    fig, (heat_ax, delta_ax) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                             gridspec_kw={"width_ratios": [1.34, 1.0]})
    fig.subplots_adjust(top=0.82, bottom=0.30, wspace=0.58)
    image = heat_ax.imshow(matrix, cmap="YlOrBr", aspect="auto", vmin=0.25, vmax=0.95)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > 0.72 else PALETTE["ink"]
            heat_ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center",
                         fontsize=9, color=color, fontweight="bold" if row == 0 else "normal")
    heat_ax.set_xticks(range(4), ["Random\nAUROC", "Random\nAUPRC",
                                 "Target-ID\nheld-out\nAUROC", "Target-ID\nheld-out\nAUPRC"], fontsize=8.5)
    heat_ax.set_yticks(range(len(variants)), variants)
    heat_ax.set_title("Ablation metric matrix", pad=14)
    _panel(heat_ax, "A")
    colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.045, pad=0.035)
    colorbar.set_label("Metric value")

    delta_random = matrix[1:, 0] - matrix[0, 0]
    delta_target = matrix[1:, 2] - matrix[0, 2]
    y = np.arange(len(variants) - 1)
    delta_ax.barh(y - 0.18, delta_random, height=0.32, color=PALETTE["sky"], label="Random AUROC")
    delta_ax.barh(y + 0.18, delta_target, height=0.32, color=PALETTE["brick"], label="Target-ID-held-out AUROC")
    for values, offset in ((delta_random, -0.18), (delta_target, 0.18)):
        for ypos, value in zip(y, values):
            x_offset = -0.004 if value < 0 else 0.004
            alignment = "right" if value < 0 else "left"
            delta_ax.text(value + x_offset, ypos + offset, f"{value:.3f}", ha=alignment, va="center",
                           color=PALETTE["ink"], fontsize=8, fontweight="bold")
    delta_ax.set_yticks(y, ["Cross-attention", "Domain-label channel", "ESM-2", "Graph augmentation"])
    delta_ax.set_xlim(-0.16, 0.025)
    delta_ax.set_xticks([-0.15, -0.10, -0.05, 0.00, 0.02])
    delta_ax.axvline(0, color=PALETTE["ink"], lw=0.8)
    delta_ax.set_xlabel("AUROC difference vs. displayed full model")
    delta_ax.set_title("Component-removal numerical difference", pad=14)
    delta_ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.31))
    _axes(delta_ax, "x")
    _panel(delta_ax, "B")
    _save(fig, output_dir, "fig_ablation")


def render_sparsity(output_dir: Path) -> None:
    arrays = np.load(DATA_DIR / "attention_distribution.npz")
    payload = _load_json("attention_distribution.json")
    # Archived revision packages have used both a flat summary object and a
    # report_global_stats wrapper. Accept either schema without changing values.
    summary = payload.get("report_global_stats", payload)
    scores = np.clip(arrays["flat_scores"], 1e-5, 1.0)
    percentiles = arrays["percentiles"]
    cumulative = arrays["cumulative_attention"] * 100
    fig, (hist_ax, curve_ax) = plt.subplots(1, 2, figsize=(10.8, 4.5),
                                             gridspec_kw={"width_ratios": [1.07, 1.0]})
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.32)
    hist_ax.hist(scores, bins=np.logspace(-5, 0, 70), color=PALETTE["teal"], edgecolor="white")
    hist_ax.set_xscale("log")
    hist_ax.axvline(summary["residue_attention_median"], color=PALETTE["slate"], ls="--", label="Median")
    hist_ax.axvline(summary["residue_attention_top5pct"], color=PALETTE["gold"], ls="--", label="95th percentile")
    hist_ax.axvline(summary["residue_attention_top1pct"], color=PALETTE["brick"], ls="--", label="99th percentile")
    hist_ax.set_xlabel("Normalised model attribution")
    hist_ax.set_ylabel("Count")
    hist_ax.set_title("Attribution-score distribution", pad=14)
    hist_ax.legend(frameon=False, loc="upper right")
    _axes(hist_ax)
    _panel(hist_ax, "A")

    idx = np.linspace(0, len(percentiles) - 1, 2500).astype(int)
    curve_ax.plot(percentiles[idx], cumulative[idx], color=PALETTE["brick"])
    curve_ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Top-ranked residue percentile",
                 ylabel="Cumulative attribution mass (%)")
    curve_ax.set_title("Concentration of attribution mass", pad=14)
    for cutoff in (1, 5):
        point = max(0, np.searchsorted(percentiles, cutoff, side="right") - 1)
        mass = cumulative[point]
        curve_ax.scatter([cutoff], [mass], s=28, color=PALETTE["gold"], zorder=3)
        curve_ax.annotate(f"Top {cutoff}%: {mass:.1f}%", (cutoff, mass), xytext=(cutoff + 8, max(8, mass - 13)),
                          arrowprops={"arrowstyle": "->", "color": PALETTE["gold"], "lw": 0.8}, fontsize=8)
    curve_ax.text(0.98, 0.04, f"Pairs: {summary['n_samples']}\nFraction <0.1: {100 * summary['attention_sparsity']:.2f}%",
                  transform=curve_ax.transAxes, ha="right", va="bottom", fontsize=8,
                  bbox={"boxstyle": "round,pad=0.25", "facecolor": PALETTE["cream"], "edgecolor": "#d7d2c6"})
    _axes(curve_ax, "both")
    _panel(curve_ax, "B")
    _save(fig, output_dir, "fig3_sparsity")


def _case_lookup():
    return {case["target_name"]: case for case in _load_json("case_profiles.json")}


def render_mutant_conservation(output_dir: Path) -> None:
    lookup = _case_lookup()
    names = ["ABL1(F317I)", "ABL1(F317I)p", "ABL1(F317L)p", "ABL1(M351T)", "ABL1(E255K)"]
    cases = [lookup[name] for name in names]
    profiles = [np.array(case["profile"]) for case in cases]
    min_len = min(map(len, profiles))
    corr = np.corrcoef(np.vstack([profile[:min_len] for profile in profiles]))
    residues = ["V104", "A648", "S199"]
    values = np.array([[case["top_residue_scores"].get(residue, 0.0) for case in cases] for residue in residues])
    labels = ["F317I", "F317I-P", "F317L-P", "M351T", "E255K"]
    fig, (corr_ax, bar_ax) = plt.subplots(1, 2, figsize=(11.8, 4.8), gridspec_kw={"width_ratios": [1.0, 1.23]})
    fig.subplots_adjust(top=0.82, bottom=0.31, wspace=0.48)
    vmin = max(0.99, float(corr.min()) - 0.001)
    image = corr_ax.imshow(corr, cmap="YlGnBu", vmin=vmin, vmax=1.0)
    for row in range(5):
        for col in range(5):
            corr_ax.text(col, row, f"{corr[row, col]:.3f}", ha="center", va="center", fontsize=8,
                         color="white" if corr[row, col] < 0.996 else PALETTE["ink"])
    corr_ax.set_xticks(range(5), labels, rotation=35, ha="right")
    corr_ax.set_yticks(range(5), labels)
    corr_ax.set_title("Profile-correlation matrix", pad=14)
    colorbar = fig.colorbar(image, ax=corr_ax, fraction=0.047, pad=0.04)
    colorbar.set_label("Correlation")
    _panel(corr_ax, "A")

    x = np.arange(5)
    width = 0.23
    for idx, (residue, color) in enumerate(zip(residues, [PALETTE["gold"], PALETTE["teal"], PALETTE["brick"]])):
        bars = bar_ax.bar(x + (idx - 1) * width, values[idx], width, label=residue, color=color, edgecolor="white")
        for bar, value in zip(bars, values[idx]):
            bar_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    bar_ax.set_xticks(x, labels, rotation=20)
    bar_ax.set_ylim(0, 1.10)
    bar_ax.set_ylabel("Normalised attribution")
    bar_ax.set_title("Top-ranked residue attributions", pad=14)
    bar_ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.29))
    _axes(bar_ax)
    _panel(bar_ax, "B")
    _save(fig, output_dir, "fig6_mutant_conservation")


def render_pharmacophore(output_dir: Path) -> None:
    case = _case_lookup()["ABL1(F317I)"]
    groups = sorted(case["functional_groups"].items(), key=lambda item: item[1], reverse=True)
    image_path = INTERPRET_DIR / "gradcam" / "3062316_ABL1F317I.png"
    fig, (image_ax, bar_ax) = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.06, 1.0]})
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.32)
    image_ax.imshow(plt.imread(image_path))
    image_ax.set_axis_off()
    image_ax.set_title("Model attribution over molecular graph", pad=14)
    sm = ScalarMappable(norm=Normalize(0, 1), cmap="YlOrRd")
    colorbar = fig.colorbar(sm, ax=image_ax, fraction=0.046, pad=0.025)
    colorbar.set_label("Normalised attribution")
    _panel(image_ax, "A")
    labels, values = zip(*groups)
    y = np.arange(len(labels))
    colors = [PALETTE["gold"], PALETTE["teal"], PALETTE["sky"], PALETTE["slate"], PALETTE["brick"]]
    bar_ax.barh(y, values, color=colors[:len(values)], edgecolor="white")
    for ypos, value in zip(y, values):
        bar_ax.text(value + 0.02, ypos, f"{value:.3f}", va="center", fontsize=8)
    bar_ax.set_yticks(y, labels)
    bar_ax.invert_yaxis()
    bar_ax.set_xlim(0, 1.05)
    bar_ax.set_xlabel("Aggregated functional-group attribution")
    bar_ax.set_title("Attribution grouped by functional moiety", pad=14)
    _axes(bar_ax, "x")
    _panel(bar_ax, "B")
    _save(fig, output_dir, "fig7_pharmacophore")


def render_training(output_dir: Path) -> None:
    payload = _load_json("training_curves.json")
    # The archival artifact records traces under ``curves``; retain support for
    # earlier flat releases so the figure script is self-contained.
    curves = payload.get("curves", payload)
    split_meta = {"random": ("Random", PALETTE["teal"]), "cold_target": ("Target-ID-held-out", PALETTE["gold"]),
                  "cold_drug": ("Drug-ID-held-out", PALETTE["brick"])}
    fig, (auroc_ax, loss_ax) = plt.subplots(1, 2, figsize=(10.8, 4.5))
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.32)
    for key, (label, color) in split_meta.items():
        curve = curves[key]
        epoch = np.array(curve["epochs"])
        auroc_ax.plot(epoch, curve["val_auroc"], color=color, label=label)
        auroc_ax.scatter(curve["best_epoch"], curve["best_val_auroc"], color=color, s=25, zorder=3)
        stop_epoch = curve.get("stop_epoch", curve["epochs"][-1])
        auroc_ax.axvline(stop_epoch, color=color, ls="--", lw=0.9, alpha=0.7)
        loss_ax.plot(epoch, curve["loss"], color=color, label=label)
        loss_ax.axvline(stop_epoch, color=color, ls="--", lw=0.9, alpha=0.7)
    auroc_ax.set(xlabel="Epoch", ylabel="Validation AUROC")
    auroc_ax.set_title("Validation AUROC during training", pad=14)
    auroc_ax.legend(frameon=False, loc="lower right")
    _axes(auroc_ax)
    _panel(auroc_ax, "A")
    loss_ax.set(xlabel="Epoch", ylabel="Training loss")
    loss_ax.set_title("Training-loss trajectory", pad=14)
    _axes(loss_ax)
    _panel(loss_ax, "B")
    _save(fig, output_dir, "fig9_training")


def render_performance(output_dir: Path, strict_metrics: dict | None) -> None:
    original = [("Random", 0.904, 0.560), ("Target-ID-held-out", 0.930, 0.525), ("Drug-ID-held-out", 0.733, 0.167)]
    strict = []
    if strict_metrics:
        for key, label in (("sequence_grouped_cold_target", "Exact-sequence-grouped\ncold-target"),
                           ("sequence_grouped_cold_both", "Exact-sequence-grouped\ncold-both")):
            record = strict_metrics[key]["metrics"]
            strict.append((label, record["AUROC"], record["AUPRC"]))
    panels = 2 if strict else 1
    fig, axes = plt.subplots(1, panels, figsize=(10.8 if strict else 5.4, 4.6), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(top=0.82, bottom=0.26, wspace=0.38)
    for idx, (ax, title, rows) in enumerate(zip(axes, ["Archived ID-based splits", "Strict exact-sequence-grouped splits"], [original, strict])):
        x = np.arange(len(rows)); width = 0.33
        auroc = [r[1] for r in rows]; auprc = [r[2] for r in rows]
        ax.bar(x - width / 2, auroc, width, color=PALETTE["teal"], label="AUROC")
        ax.bar(x + width / 2, auprc, width, color=PALETTE["gold"], label="AUPRC")
        for xpos, value in zip(x - width / 2, auroc): ax.text(xpos, value + 0.018, f"{value:.3f}", ha="center", fontsize=8)
        for xpos, value in zip(x + width / 2, auprc): ax.text(xpos, value + 0.018, f"{value:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, [r[0] for r in rows])
        ax.set_ylim(0, 1.08); ax.set_ylabel("Test metric")
        ax.set_title(title, pad=14); ax.legend(frameon=False, ncol=2, loc="upper center")
        _axes(ax); _panel(ax, chr(ord("A") + idx))
    _save(fig, output_dir, "fig2_performance")


def render_core_figures(output_dir: Path, strict_metrics: dict | None) -> None:
    output_dir = Path(output_dir)
    render_ablation(output_dir)
    render_sparsity(output_dir)
    render_mutant_conservation(output_dir)
    render_pharmacophore(output_dir)
    render_training(output_dir)
    render_performance(output_dir, strict_metrics)


if __name__ == "__main__":
    metrics_path = WORKSPACE / "revision" / "analysis" / "strict_split_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    render_core_figures(WORKSPACE / "revision" / "figures", metrics)
