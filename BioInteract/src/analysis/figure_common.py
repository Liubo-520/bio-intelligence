"""Common helpers for manuscript figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import MANUSCRIPT_FIGURES_DIR, RESULTS_DIR


FIGURE_DATA_DIR = RESULTS_DIR / 'figure_data'
FIGURE_METADATA_DIR = MANUSCRIPT_FIGURES_DIR / 'metadata'

for _path in (MANUSCRIPT_FIGURES_DIR, FIGURE_DATA_DIR, FIGURE_METADATA_DIR):
    _path.mkdir(parents=True, exist_ok=True)


PALETTE = {
    'ink': '#20303c',
    'teal': '#2f7f79',
    'gold': '#d89a34',
    'brick': '#b55a44',
    'sage': '#7b9a77',
    'sand': '#efe0c1',
    'sky': '#8fb8c9',
    'slate': '#5f6f7b',
    'mist': '#dbe6ea',
    'cream': '#fbf8f2',
}


def configure_matplotlib() -> None:
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


def panel_label(ax, label: str, x: float = -0.075, y: float = 1.055, fontsize: float = 14) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight='bold',
        color=PALETTE['ink'],
        va='bottom',
        ha='left',
        clip_on=False,
    )


def soften_axes(ax, grid_axis: str | None = 'y') -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color='#d9dfdf', linestyle='--', linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_figure(fig, stem: str, metadata: dict[str, Any] | None = None) -> None:
    pdf_path = MANUSCRIPT_FIGURES_DIR / f'{stem}.pdf'
    png_path = MANUSCRIPT_FIGURES_DIR / f'{stem}.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    if metadata is not None:
        payload = _json_ready(metadata)
        payload.setdefault('outputs', {})
        payload['outputs']['pdf'] = pdf_path.as_posix()
        payload['outputs']['png'] = png_path.as_posix()
        (FIGURE_METADATA_DIR / f'{stem}.json').write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding='utf-8',
        )
        (FIGURE_DATA_DIR / f'{stem}.json').write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding='utf-8',
        )


def write_manifest(stems: list[str]) -> None:
    manifest_path = FIGURE_METADATA_DIR / 'manifest.json'
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))

    combined = list(dict.fromkeys(existing.get('figures', []) + stems))
    manifest = {
        'figures': combined,
        'metadata_dir': FIGURE_METADATA_DIR.as_posix(),
        'data_dir': FIGURE_DATA_DIR.as_posix(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding='utf-8',
    )
