"""Standalone script to regenerate fig8_pharmacophore_comparison from saved JSON data.

Run from the BioInteract directory:
    python _regen_fig8.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent
METADATA_JSON = WORKSPACE / 'submission_acs' / 'manuscript' / 'figures' / 'metadata' / 'fig8_pharmacophore_comparison.json'
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


def short_case_name(target_name: str) -> str:
    return target_name.replace('ABL1', 'ABL1 ').replace('EGFR', 'EGFR ').replace('BRAF', 'BRAF ').strip()


def panel_label(ax, label: str, x: float = -0.075, y: float = 1.055) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            color=PALETTE['ink'], va='bottom', ha='left', clip_on=False)


def soften_axes(ax, grid_axis: str = 'y') -> None:
    ax.grid(axis=grid_axis, color='#d9dfdf', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def build_figure(groups: list[str], cases: list[dict], raw_matrix: list[list]) -> plt.Figure:
    # Reconstruct numpy matrix (NaN for None/null)
    matrix = np.array([
        [float('nan') if v is None or (isinstance(v, float) and math.isnan(v)) else v
         for v in row]
        for row in raw_matrix
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2),
                             gridspec_kw={'width_ratios': [1.42, 0.58]})
    fig.subplots_adjust(wspace=0.52, top=0.88, bottom=0.22)
    heat_ax, prob_ax = axes

    cmap = matplotlib.colormaps['YlOrBr'].copy()
    cmap.set_bad('#efefef')
    image = heat_ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            label = '--' if np.isnan(value) else f'{value:.2f}'
            color = PALETTE['ink'] if np.isnan(value) or value < 0.65 else 'white'
            heat_ax.text(col, row, label, ha='center', va='center', color=color, fontsize=8.5)

    heat_ax.set_xticks(np.arange(len(cases)))
    heat_ax.set_xticklabels([
        f"{short_case_name(case['target_name'])}\n{case['display_drug']}"
        for case in cases
    ], rotation=30, ha='right', fontsize=9)
    heat_ax.set_yticks(np.arange(len(groups)))
    heat_ax.set_yticklabels(groups)
    heat_ax.set_title('Target-dependent pharmacophore usage', pad=12)
    panel_label(heat_ax, 'A')
    fig.colorbar(image, ax=heat_ax, fraction=0.04, pad=0.03, shrink=0.82, anchor=(0.5, 0.3))

    probs = [case['prediction_prob'] for case in cases]
    prob_ax.barh(np.arange(len(cases)), probs,
                 color=[PALETTE['teal'], PALETTE['gold'], PALETTE['brick']],
                 edgecolor='white')
    for ypos, value in enumerate(probs):
        prob_ax.text(value + 0.02, ypos, f'{value:.3f}', va='center', ha='left', fontsize=9)
    prob_ax.set_xlim(0.0, 1.18)
    prob_ax.set_yticks(np.arange(len(cases)))
    prob_ax.set_yticklabels([case['display_drug'] for case in cases])
    prob_ax.invert_yaxis()
    prob_ax.set_xlabel('P(bind)')
    prob_ax.set_title('Prediction confidence', pad=12)
    soften_axes(prob_ax, 'x')
    panel_label(prob_ax, 'B', x=-0.12)

    return fig


def main() -> None:
    raw = json.loads(METADATA_JSON.read_text(encoding='utf-8'))
    groups = raw['groups']
    cases = raw['cases']
    matrix = raw['matrix']

    fig = build_figure(groups, cases, matrix)

    pdf_path = OUT_DIR / 'fig8_pharmacophore_comparison.pdf'
    png_path = OUT_DIR / 'fig8_pharmacophore_comparison.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f'Saved: {pdf_path}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    main()
