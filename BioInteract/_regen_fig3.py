"""Standalone script to regenerate fig3_sparsity from saved data files.

Run from the BioInteract directory:
    python _regen_fig3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path(__file__).resolve().parent.parent
FIGURE_DATA_DIR = WORKSPACE / 'BioInteract' / 'results' / 'figure_data'
NPZ_PATH = FIGURE_DATA_DIR / 'attention_distribution.npz'
SUMMARY_PATH = FIGURE_DATA_DIR / 'attention_distribution.json'
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


def soften_axes(ax, grid_axis: str = 'y') -> None:
    ax.grid(axis=grid_axis, color='#d9dfdf', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.13, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            color=PALETTE['ink'], va='bottom', ha='left', clip_on=False)


def build_figure(flat_scores: np.ndarray, percentiles: np.ndarray,
                 cumulative: np.ndarray, summary: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), gridspec_kw={'width_ratios': [1.05, 1.0]})
    fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.26)
    hist_ax, curve_ax = axes

    bins = np.logspace(-5, 0, 70)
    flat_clipped = np.clip(flat_scores, 1e-5, 1.0)
    hist_ax.hist(flat_clipped, bins=bins, color=PALETTE['teal'], alpha=0.9, edgecolor='white')
    hist_ax.set_xscale('log')
    hist_ax.set_xlabel('Normalised residue attention')
    hist_ax.set_ylabel('Count')
    hist_ax.set_title('Empirical distribution over all positive pairs', pad=14)
    hist_ax.axvline(summary['residue_attention_median'], color=PALETTE['slate'],
                    linestyle='--', linewidth=1.2, label='Median')
    hist_ax.axvline(summary['residue_attention_top5pct'], color=PALETTE['gold'],
                    linestyle='--', linewidth=1.2, label='95th percentile')
    hist_ax.axvline(summary['residue_attention_top1pct'], color=PALETTE['brick'],
                    linestyle='--', linewidth=1.2, label='99th percentile')
    hist_ax.legend(frameon=False, loc='upper right')
    soften_axes(hist_ax, 'y')
    panel_label(hist_ax, 'A')

    sample_idx = np.linspace(0, len(percentiles) - 1, 2500).astype(int)
    curve_ax.plot(percentiles[sample_idx], cumulative[sample_idx], color=PALETTE['brick'])
    curve_ax.set_xlim(0, 100)
    curve_ax.set_ylim(0, 100)
    curve_ax.set_xlabel('Top-ranked residue percentile')
    curve_ax.set_ylabel('Cumulative attention mass (%)')
    curve_ax.set_title('Concentration of attention mass', pad=14)
    for cutoff, label in [(1, 'Top 1%'), (5, 'Top 5%')]:
        idx = max(0, np.searchsorted(percentiles, cutoff, side='right') - 1)
        mass = cumulative[idx]
        curve_ax.scatter([cutoff], [mass], color=PALETTE['gold'], s=35, zorder=3)
        curve_ax.annotate(f'{label}: {mass:.1f}%', (cutoff, mass),
                          xytext=(cutoff + 6, mass - 12),
                          arrowprops={'arrowstyle': '->', 'color': PALETTE['gold'], 'lw': 1.0},
                          fontsize=9)
    curve_ax.text(0.96, 0.97,
                  f"Sparsity (<0.1): {summary['report_global_stats']['attention_sparsity'] * 100:.2f}%\n"
                  f"Positive pairs: {summary['report_global_stats']['n_samples']}",
                  transform=curve_ax.transAxes, ha='right', va='top', fontsize=9.5,
                  bbox={'boxstyle': 'round,pad=0.35', 'facecolor': PALETTE['cream'],
                        'edgecolor': '#d7d2c6'})
    soften_axes(curve_ax, 'both')
    panel_label(curve_ax, 'B')

    return fig


def main() -> None:
    arrays = np.load(NPZ_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))

    flat_scores = arrays['flat_scores']
    percentiles = arrays['percentiles']
    cumulative = arrays['cumulative_attention'] * 100.0

    fig = build_figure(flat_scores, percentiles, cumulative, summary)

    pdf_path = OUT_DIR / 'fig3_sparsity.pdf'
    png_path = OUT_DIR / 'fig3_sparsity.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f'Saved: {pdf_path}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    main()
