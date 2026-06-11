"""Generate baseline comparison and ablation figures for the manuscript."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
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
    'DeepDTA\n(2018)',
    'GraphDTA\n(2021)',
    'AttentionDTA\n(2019)',
    'MolTrans\n(2021)',
    'TransformerCPI\n(2020)',
    'DrugBAN\n(2023)',
    'BioInteract\n(Ours)',
]

RANDOM_BASELINE = {
    'AUROC': [0.878, 0.893, 0.900, 0.907, 0.910, 0.915, 0.921],
    'AUPRC': [0.352, 0.403, 0.425, 0.480, 0.492, 0.530, 0.608],
}

SPLIT_METHODS = ['DeepDTA', 'GraphDTA', 'MolTrans', 'DrugBAN', 'BioInteract']
SPLIT_HEATMAP = np.array([
    [0.878, 0.783, 0.592],
    [0.893, 0.815, 0.621],
    [0.907, 0.856, 0.668],
    [0.915, 0.874, 0.695],
    [0.921, 0.941, 0.739],
])

ABLATION_VARIANTS = [
    'Full\nBioInteract',
    'w/o\nCross-Attn',
    'w/o\nDomain Feat',
    'w/o\nESM-2',
    'w/o\nGraph Aug',
]

ABLATION_MATRIX = np.array([
    [0.921, 0.608, 0.941, 0.549],
    [0.864, 0.432, 0.851, 0.371],
    [0.906, 0.561, 0.912, 0.498],
    [0.845, 0.389, 0.793, 0.295],
    [0.908, 0.572, 0.926, 0.521],
])


def _bar_colors(n: int) -> list[str]:
    colors = [PALETTE['mist']] * n
    colors[-1] = PALETTE['gold']
    return colors


def fig_baseline_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    fig.subplots_adjust(top=0.82, bottom=0.15, wspace=0.20)
    y = np.arange(len(METHODS_ALL))

    for idx, metric in enumerate(['AUROC', 'AUPRC']):
        ax = axes[idx]
        values = RANDOM_BASELINE[metric]
        bars = ax.barh(y, values, color=_bar_colors(len(values)), edgecolor='white')
        bars[-1].set_edgecolor(PALETTE['ink'])
        bars[-1].set_linewidth(1.3)

        for bar, value in zip(bars, values):
            ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f'{value:.3f}',
                    va='center', ha='left', color=PALETTE['ink'], fontsize=9.5,
                    fontweight='bold' if value == values[-1] else 'normal')

        ax.set_xlabel(metric)
        ax.set_xlim((0.82, 0.95) if metric == 'AUROC' else (0.30, 0.66))
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels(METHODS_ALL)
        else:
            ax.tick_params(axis='y', length=0)
        ax.invert_yaxis()
        soften_axes(ax, 'x')
        panel_label(ax, 'A' if idx == 0 else 'B')
        ax.set_title(f'Random split {metric}', pad=12)

    fig.suptitle('Baseline comparison against published DTI models', x=0.52, y=0.98,
                 fontsize=14, fontweight='bold', color=PALETTE['ink'])
    fig.text(0.5, -0.02,
             'BioInteract improves both ranking metrics, with the largest gain on AUPRC.',
             ha='center', color=PALETTE['slate'], fontsize=10)

    save_figure(
        fig,
        'fig_comparison_random',
        metadata={
            'title': 'Baseline comparison on the random split',
            'sources': [
                'manuscript benchmark table values for baseline methods',
                'results/test_random.json for BioInteract metrics',
            ],
            'methods': METHODS_ALL,
            'metrics': RANDOM_BASELINE,
            'note': 'Baseline values are the manuscript comparison values used to reproduce Figure 1A.',
        },
    )


def fig_multisplit_comparison() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    fig.subplots_adjust(top=0.84, bottom=0.15)
    cmap = plt.cm.YlGnBu
    image = ax.imshow(SPLIT_HEATMAP, cmap=cmap, aspect='auto', vmin=0.55, vmax=0.95)

    for row in range(SPLIT_HEATMAP.shape[0]):
        for col in range(SPLIT_HEATMAP.shape[1]):
            value = SPLIT_HEATMAP[row, col]
            ax.text(col, row, f'{value:.3f}', ha='center', va='center',
                    color='white' if value > 0.82 else PALETTE['ink'],
                    fontsize=10, fontweight='bold' if row == len(SPLIT_METHODS) - 1 else 'normal')

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['Random', 'Cold-target', 'Cold-drug'])
    ax.set_yticks(np.arange(len(SPLIT_METHODS)))
    ax.set_yticklabels(SPLIT_METHODS)
    ax.set_title('AUROC across data splitting protocols', pad=14)
    panel_label(ax, 'A')

    highlight = Rectangle((-0.5, len(SPLIT_METHODS) - 1 - 0.5), 3, 1,
                          fill=False, edgecolor=PALETTE['gold'], linewidth=2.0)
    ax.add_patch(highlight)

    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label('AUROC')
    fig.text(0.5, -0.03,
             'BioInteract remains strongest in all three regimes, with a particularly large cold-target margin.',
             ha='center', color=PALETTE['slate'], fontsize=10)

    save_figure(
        fig,
        'fig_comparison_splits',
        metadata={
            'title': 'Multi-split AUROC comparison',
            'sources': ['manuscript comparison values across random, cold-target, and cold-drug protocols'],
            'methods': SPLIT_METHODS,
            'splits': ['random', 'cold_target', 'cold_drug'],
            'auroc_matrix': SPLIT_HEATMAP,
        },
    )


def fig_ablation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), gridspec_kw={'width_ratios': [1.25, 1.0]})
    fig.subplots_adjust(top=0.82, bottom=0.13, wspace=0.40)

    heatmap_ax, delta_ax = axes
    heatmap = heatmap_ax.imshow(ABLATION_MATRIX, cmap=plt.cm.YlOrBr, aspect='auto', vmin=0.25, vmax=0.95)
    for row in range(ABLATION_MATRIX.shape[0]):
        for col in range(ABLATION_MATRIX.shape[1]):
            value = ABLATION_MATRIX[row, col]
            heatmap_ax.text(col, row, f'{value:.3f}', ha='center', va='center',
                            color='white' if value > 0.72 else PALETTE['ink'],
                            fontsize=9.5, fontweight='bold' if row == 0 else 'normal')

    heatmap_ax.set_xticks(np.arange(4))
    heatmap_ax.set_xticklabels(['Random\nAUROC', 'Random\nAUPRC', 'Cold-target\nAUROC', 'Cold-target\nAUPRC'])
    heatmap_ax.set_yticks(np.arange(len(ABLATION_VARIANTS)))
    heatmap_ax.set_yticklabels(ABLATION_VARIANTS)
    heatmap_ax.set_title('Metric matrix', pad=12)
    panel_label(heatmap_ax, 'A', y=1.22)
    fig.colorbar(heatmap, ax=heatmap_ax, fraction=0.046, pad=0.03)

    full_random = ABLATION_MATRIX[0, 0]
    full_cold = ABLATION_MATRIX[0, 2]
    random_drop = ABLATION_MATRIX[1:, 0] - full_random
    cold_drop = ABLATION_MATRIX[1:, 2] - full_cold
    y = np.arange(len(ABLATION_VARIANTS) - 1)
    delta_ax.barh(y - 0.18, random_drop, height=0.32, color=PALETTE['sky'], label='Random AUROC')
    delta_ax.barh(y + 0.18, cold_drop, height=0.32, color=PALETTE['brick'], label='Cold-target AUROC')
    for values, offset in ((random_drop, -0.18), (cold_drop, 0.18)):
        for ypos, value in zip(y, values):
            delta_ax.text(value - 0.004, ypos + offset, f'{value:.3f}', va='center', ha='right',
                          color='white', fontsize=9, fontweight='bold')

    delta_ax.set_yticks(y)
    delta_ax.set_yticklabels(['Cross-attn', 'Domain feat', 'ESM-2', 'Graph aug'])
    delta_ax.set_xlim(-0.18, 0.01)
    delta_ax.axvline(0, color=PALETTE['ink'], linewidth=0.9)
    delta_ax.set_xlabel('Delta AUROC vs. full model')
    delta_ax.set_title('Performance drop after removing each component', pad=12)
    delta_ax.legend(frameon=False, loc='lower left')
    soften_axes(delta_ax, 'x')
    panel_label(delta_ax, 'B', x=-0.13, y=1.22)

    save_figure(
        fig,
        'fig_ablation',
        metadata={
            'title': 'Ablation study',
            'sources': ['Table 2 ablation values used in the manuscript'],
            'variants': ABLATION_VARIANTS,
            'metrics': ['random_auroc', 'random_auprc', 'cold_target_auroc', 'cold_target_auprc'],
            'matrix': ABLATION_MATRIX,
            'drops': {
                'random_auroc': random_drop,
                'cold_target_auroc': cold_drop,
            },
        },
    )


def main() -> None:
    print('Generating comparison and ablation figures...')
    fig_baseline_comparison()
    fig_multisplit_comparison()
    fig_ablation()
    write_manifest(['fig_comparison_random', 'fig_comparison_splits', 'fig_ablation'])
    print('Done.')


if __name__ == '__main__':
    main()