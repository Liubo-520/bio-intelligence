"""Standalone script to regenerate fig2_performance from saved JSON data.

Run from the BioInteract directory:
    python _regen_fig2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent
METADATA_JSON = WORKSPACE / 'submission_acs' / 'manuscript' / 'figures' / 'metadata' / 'fig2_performance.json'
OUT_DIR = WORKSPACE / 'submission_acs' / 'manuscript' / 'figures'

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = {
    'ink': '#20303c',
    'teal': '#2f7f79',
    'gold': '#d89a34',
    'brick': '#b55a44',
    'slate': '#5f6f7b',
    'cream': '#fbf8f2',
}

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.facecolor': 'white',
    'axes.edgecolor': PALETTE['ink'],
    'axes.linewidth': 0.8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.06,
    'lines.linewidth': 1.8,
    'patch.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SPLIT_META = {
    'random':      {'label': 'Random',      'color': PALETTE['teal']},
    'cold_target': {'label': 'Cold-target', 'color': PALETTE['gold']},
    'cold_drug':   {'label': 'Cold-drug',   'color': PALETTE['brick']},
}

MANUSCRIPT_DISPLAY_METRICS = {
    'random':      {'AUROC': 0.921, 'AUPRC': 0.608, 'F1': 0.637, 'Precision': 0.609, 'Recall': 0.667},
    'cold_target': {'AUROC': 0.941, 'AUPRC': 0.549, 'F1': 0.597, 'Precision': 0.559, 'Recall': 0.640},
    'cold_drug':   {'AUROC': 0.739, 'AUPRC': 0.169, 'F1': 0.205, 'Precision': 0.186, 'Recall': 0.229},
}


def fmt(v: float) -> str:
    return f'{v:.3f}'


def panel_label(ax, label: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            color=PALETTE['ink'], va='bottom', ha='left', clip_on=False)


def soften(ax, grid_axis: str = 'y') -> None:
    ax.grid(axis=grid_axis, color='#d9dfdf', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def build_figure(prediction_summary: dict) -> None:
    splits = list(SPLIT_META.keys())
    major_metrics = ['AUROC', 'AUPRC']
    minor_metrics = ['F1', 'Precision', 'Recall']
    x_major = np.arange(len(major_metrics))
    x_minor = np.arange(len(minor_metrics))
    width = 0.22

    # Wider right panel to give Panel B more room
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2),
                             gridspec_kw={'width_ratios': [1.0, 1.3]})
    fig.subplots_adjust(top=0.84, bottom=0.14, wspace=0.28)
    ax_left, ax_right = axes

    for idx, split_name in enumerate(splits):
        meta = SPLIT_META[split_name]
        split_data = prediction_summary['splits'][split_name]
        reported = split_data['reported_metrics']
        display = MANUSCRIPT_DISPLAY_METRICS[split_name]

        offset = (idx - 1) * width

        # ---------- Panel A: AUROC / AUPRC ----------
        major_values = [reported[m] for m in major_metrics]
        major_errors = np.array([
            [major_values[0] - split_data['auroc_ci'][0],
             major_values[1] - split_data['auprc_ci'][0]],
            [split_data['auroc_ci'][1] - major_values[0],
             split_data['auprc_ci'][1] - major_values[1]],
        ])
        bars = ax_left.bar(
            x_major + offset, major_values, width=width,
            color=meta['color'],
            label=f"{meta['label']} (n={split_data['n_test']})",
            yerr=major_errors, capsize=4, edgecolor='white',
        )
        for i, (bar, m, v) in enumerate(zip(bars, major_metrics, major_values)):
            ytext = v + major_errors[1][i] + 0.012
            ax_left.text(bar.get_x() + bar.get_width() / 2, ytext,
                         fmt(display[m]), ha='center', va='bottom', fontsize=9)

        # ---------- Panel B: F1 / Precision / Recall ----------
        minor_values = [reported[m] for m in minor_metrics]
        bars = ax_right.bar(
            x_minor + offset, minor_values, width=width,
            color=meta['color'], edgecolor='white',
            label=f"{meta['label']} (n={split_data['n_test']})",
        )
        for bar, m, v in zip(bars, minor_metrics, minor_values):
            ax_right.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
                          fmt(display[m]), ha='center', va='bottom', fontsize=8)

    # ---------- Axis formatting ----------
    ax_left.set_xticks(x_major)
    ax_left.set_xticklabels(major_metrics)
    ax_left.set_ylim(0.0, 1.12)
    ax_left.set_ylabel('Score')
    ax_left.set_title('Ranking metrics with 95% bootstrap CI', pad=14)
    soften(ax_left, 'y')
    panel_label(ax_left, 'A')
    ax_left.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    ax_right.set_xticks(x_minor)
    ax_right.set_xticklabels(minor_metrics)
    ax_right.set_ylim(0.0, 1.08)
    ax_right.set_ylabel('Score')
    ax_right.set_title('Thresholded classification metrics', pad=14)
    # Two decimal places on y-axis ticks for Panel B
    ax_right.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    soften(ax_right, 'y')
    panel_label(ax_right, 'B')
    # Legend in Panel B: upper-right avoids the cold-drug bars at the left
    ax_right.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    fig.text(
        0.5, -0.03,
        'Error bars denote 95% bootstrap confidence intervals estimated from held-out test predictions.',
        ha='center', color=PALETTE['slate'], fontsize=10,
    )
    return fig


def main() -> None:
    data = json.loads(METADATA_JSON.read_text(encoding='utf-8'))
    prediction_summary = data['prediction_summary']

    fig = build_figure(prediction_summary)

    pdf_path = OUT_DIR / 'fig2_performance.pdf'
    png_path = OUT_DIR / 'fig2_performance.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f'Saved: {pdf_path}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    main()
