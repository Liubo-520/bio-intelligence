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
    'AUROC': [0.878, 0.893, 0.900, 0.907, 0.910, 0.915, 0.904],
    'AUPRC': [0.352, 0.403, 0.425, 0.480, 0.492, 0.530, 0.560],
}

SPLIT_METHODS = ['DeepDTA', 'GraphDTA', 'MolTrans', 'DrugBAN', 'BioInteract']
SPLIT_HEATMAP = np.array([
    [0.878, 0.783, 0.592],
    [0.893, 0.815, 0.621],
    [0.907, 0.856, 0.668],
    [0.915, 0.874, 0.695],
    [0.904, 0.930, 0.733],
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
        ax.set_title(f'Random-split {metric}', pad=12)

    fig.suptitle('Contextual comparison against published DTI models', x=0.52, y=0.98,
                 fontsize=14, fontweight='bold', color=PALETTE['ink'])
    fig.text(0.015, 0.955, 'A', fontsize=15, fontweight='bold', color=PALETTE['ink'])
    fig.text(0.5, -0.02,
             'Baseline values are historical; BioInteract values are canonical current-release checkpoint evaluations.',
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
    ax.set_xticklabels(['Random', 'Target-ID-held-out', 'Drug-ID-held-out'])
    ax.set_yticks(np.arange(len(SPLIT_METHODS)))
    ax.set_yticklabels(SPLIT_METHODS)
    ax.set_title('AUROC across identifier-based splitting protocols', pad=14)
    panel_label(ax, 'B')

    highlight = Rectangle((-0.5, len(SPLIT_METHODS) - 1 - 0.5), 3, 1,
                          fill=False, edgecolor=PALETTE['gold'], linewidth=2.0)
    ax.add_patch(highlight)

    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label('AUROC')
    fig.text(0.5, -0.03,
             'Compiled historical AUROCs; cross-study values are descriptive rather than a controlled benchmark.',
             ha='center', color=PALETTE['slate'], fontsize=10)

    save_figure(
        fig,
        'fig_comparison_splits',
        metadata={
            'title': 'Multi-split AUROC comparison',
            'sources': ['manuscript comparison values across random, Target-ID-held-out, and Drug-ID-held-out protocols'],
            'methods': SPLIT_METHODS,
            'splits': ['random', 'target_id_held_out', 'drug_id_held_out'],
            'auroc_matrix': SPLIT_HEATMAP,
        },
    )


def main() -> None:
    print('Generating canonical comparison figures...')
    fig_baseline_comparison()
    fig_multisplit_comparison()
    write_manifest(['fig_comparison_random', 'fig_comparison_splits'])
    print('Done.')


if __name__ == '__main__':
    main()
