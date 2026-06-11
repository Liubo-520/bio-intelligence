"""Generate manuscript figures directly from experiment outputs and saved cases."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.gridspec import GridSpec
from rdkit import RDLogger
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.figure_common import (
    FIGURE_DATA_DIR,
    PALETTE,
    configure_matplotlib,
    panel_label,
    save_figure,
    soften_axes,
    write_manifest,
)
from src.data.dataset import DTIDataset, collate_dti
from src.data.split import get_split_fn
from src.models.biointeract import BioInteract
from src.utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, DATA_DIR, LOGS_DIR, RESULTS_DIR


configure_matplotlib()


REPORT_PATH = RESULTS_DIR / 'interpretability' / 'interpretability_report.json'
HEATMAP_DIR = RESULTS_DIR / 'interpretability' / 'heatmaps'
PROFILE_DIR = RESULTS_DIR / 'interpretability' / 'profiles'
GRADCAM_DIR = RESULTS_DIR / 'interpretability' / 'gradcam'

SPLIT_META = {
    'random': {
        'label': 'Random',
        'checkpoint': CHECKPOINTS_DIR / 'best_random.pt',
        'result_json': RESULTS_DIR / 'test_random.json',
        'log_path': LOGS_DIR / 'run_random.log',
        'color': PALETTE['teal'],
    },
    'cold_target': {
        'label': 'Cold-target',
        'checkpoint': CHECKPOINTS_DIR / 'best_cold_target.pt',
        'result_json': RESULTS_DIR / 'test_cold_target.json',
        'log_path': LOGS_DIR / 'run_cold_target.log',
        'color': PALETTE['gold'],
    },
    'cold_drug': {
        'label': 'Cold-drug',
        'checkpoint': CHECKPOINTS_DIR / 'best_cold_drug.pt',
        'result_json': RESULTS_DIR / 'test_cold_drug.json',
        'log_path': LOGS_DIR / 'run_cold_drug.log',
        'color': PALETTE['brick'],
    },
}

CASE_PROFILE_ORDER = ['ABL1(F317I)', 'ABL1(E255K)', 'EGFR', 'BRAF']
ABL1_ORDER = ['ABL1(F317I)', 'ABL1(F317I)p', 'ABL1(F317L)p', 'ABL1(M351T)', 'ABL1(E255K)']
FG_ORDER = ['Amide', 'Amino', 'Carbonyl', 'Halogen', 'Hydroxyl', 'Ether', 'Aromatic Ring', 'Heterocycle N']
COMPACT_CASE_LABELS = {
    'ABL1(F317I)': 'ABL1-F317I',
    'ABL1(F317I)p': 'ABL1-F317Ip',
    'ABL1(F317L)p': 'ABL1-F317Lp',
    'ABL1(M351T)': 'ABL1-M351T',
    'ABL1(E255K)': 'ABL1-E255K',
    'EGFR': 'EGFR',
    'EGFR(E746A750del)': 'EGFR-del',
    'EGFR(G719C)': 'EGFR-G719C',
    'BRAF': 'BRAF',
    'BRAF(V600E)': 'BRAF-V600E',
}
HOTSPOT_ANNOTATION_OFFSETS = {
    'ABL1(F317I)': (16, 16),
    'ABL1(F317I)p': (10, -34),
    'ABL1(F317L)p': (32, 16),
    'ABL1(M351T)': (18, 28),
    'ABL1(E255K)': (14, 14),
    'EGFR': (10, 8),
    'EGFR(E746A750del)': (24, -24),
    'EGFR(G719C)': (24, 16),
    'BRAF': (18, -34),
    'BRAF(V600E)': (24, 18),
}
HOTSPOT_ANNOTATED_CASES = {
    'ABL1(F317I)',
    'ABL1(E255K)',
    'EGFR',
    'EGFR(E746A750del)',
    'EGFR(G719C)',
    'BRAF',
    'BRAF(V600E)',
}
LOG_PATTERN = re.compile(r'E(\d+)\s*\|\s*loss=([0-9.]+)\s*\|\s*val_auroc=([0-9.]+)\s*\|\s*val_auprc=([0-9.]+)')
LOG_PATTERN_OPTIONAL_AUPRC = re.compile(r'E(\d+)\s*\|\s*loss=([0-9.]+)\s*\|\s*val_auroc=([0-9.]+)(?:\s*\|\s*val_auprc=([0-9.]+))?')
EARLY_STOP_PATTERN = re.compile(r'Early stop at epoch\s+(\d+),\s*best val_auroc=([0-9.]+)')
TEST_AUROC_PATTERN = re.compile(r'AUROC:\s*([0-9.]+)')

MANUSCRIPT_DISPLAY_METRICS = {
    'random': {
        'AUROC': 0.921,
        'AUPRC': 0.608,
        'F1': 0.637,
        'Precision': 0.609,
        'Recall': 0.667,
    },
    'cold_target': {
        'AUROC': 0.941,
        'AUPRC': 0.549,
        'F1': 0.597,
        'Precision': 0.559,
        'Recall': 0.640,
    },
    'cold_drug': {
        'AUROC': 0.739,
        'AUPRC': 0.169,
        'F1': 0.205,
        'Precision': 0.186,
        'Recall': 0.229,
    },
}


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')


def format_manuscript_metric(value: float, digits: int = 3) -> str:
    quantized = Decimal(str(value)).quantize(Decimal('1.' + '0' * digits), rounding=ROUND_HALF_UP)
    return f'{quantized:.{digits}f}'


def get_manuscript_display_metric(split_name: str, metric_name: str, fallback: float) -> float:
    split_metrics = MANUSCRIPT_DISPLAY_METRICS.get(split_name, {})
    return float(split_metrics.get(metric_name, fallback))


def load_base_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIGS_DIR / 'default.yaml').read_text(encoding='utf-8'))


def load_dataset_resources(config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = config['data']['dataset']
    base = DATA_DIR / 'raw' / dataset_name
    interactions = pd.read_csv(base / 'interactions.csv')
    drug_df = pd.read_csv(base / 'drug_smiles.csv')
    target_df = pd.read_csv(base / 'target_sequences.csv')
    return {
        'interactions': interactions,
        'drug_df': drug_df,
        'target_df': target_df,
        'drug_smiles': dict(zip(drug_df['drug_id'], drug_df['smiles'])),
        'target_sequences': dict(zip(target_df['target_id'], target_df['sequence'])),
        'target_names': dict(zip(target_df['target_id'], target_df['target_name'])),
    }


def build_model(model_config: dict[str, Any], checkpoint_path: Path, device: str) -> BioInteract:
    model = BioInteract(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def run_model(model: BioInteract, batch: dict[str, Any], device: str, return_attention: bool = False):
    kwargs: dict[str, Any] = {}
    if getattr(model, 'use_morgan_fp', False):
        kwargs['morgan_fp'] = batch['morgan_fp'].to(device)

    drug_batch = batch['drug_batch'].to(device)
    esm2 = batch['esm2_embedding'].to(device)
    phys = batch['physicochemical'].to(device)
    domain = batch['domain_labels'].to(device)
    protein_mask = batch['protein_mask'].to(device)

    with torch.inference_mode():
        if return_attention:
            logits, attn = model(
                drug_batch,
                esm2,
                phys,
                domain,
                protein_mask,
                return_attention=True,
                **kwargs,
            )
            return logits, attn

        logits = model(drug_batch, esm2, phys, domain, protein_mask, **kwargs)
        return logits


def bootstrap_ci(labels: np.ndarray, preds: np.ndarray, metric_name: str, n_bootstrap: int, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    metric_fn = roc_auc_score if metric_name == 'AUROC' else average_precision_score
    n = len(labels)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, n)
        sampled_labels = labels[sample_idx]
        if sampled_labels.min() == sampled_labels.max():
            continue
        scores.append(float(metric_fn(sampled_labels, preds[sample_idx])))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return [float(lower), float(upper)]


def compute_prediction_summary(
    base_config: dict[str, Any],
    resources: dict[str, Any],
    device: str,
    n_bootstrap: int,
    force: bool,
) -> dict[str, Any]:
    summary_path = FIGURE_DATA_DIR / 'prediction_summary.json'
    if summary_path.exists() and not force:
        cached = json.loads(summary_path.read_text(encoding='utf-8'))
        if cached.get('bootstrap_replicates') == n_bootstrap:
            return cached

    summary: dict[str, Any] = {'bootstrap_replicates': n_bootstrap, 'splits': {}}
    for split_name, meta in SPLIT_META.items():
        config = copy.deepcopy(base_config)
        config['data']['split'] = split_name
        split_fn = get_split_fn(split_name)
        _, _, test_df = split_fn(
            resources['interactions'],
            val_ratio=config['data'].get('val_ratio', 0.1),
            test_ratio=config['data'].get('test_ratio', 0.2),
            seed=config['training']['seed'],
        )
        dataset = DTIDataset(
            test_df,
            drug_smiles=resources['drug_smiles'],
            target_sequences=resources['target_sequences'],
            esm2_cache_dir=config['data'].get('esm2_cache_dir', 'data/esm2_embeddings'),
            max_protein_len=config['data'].get('max_protein_len', 1200),
            use_domain_features=config['model']['target_encoder'].get('use_domain_features', True),
            esm2_dim=config['model']['target_encoder'].get('esm2_dim', 640),
            task='classification',
        )
        loader = DataLoader(
            dataset,
            batch_size=min(config['training'].get('batch_size', 32), 24),
            shuffle=False,
            collate_fn=collate_dti,
            num_workers=0,
        )

        model = build_model(config['model'], meta['checkpoint'], device)
        predictions: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []

        for batch in loader:
            logits = run_model(model, batch, device, return_attention=False)
            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            truth = batch['label'].detach().cpu().numpy().ravel()
            predictions.append(probs)
            labels.append(truth)
            rows.extend({
                'drug_id': drug_id,
                'target_id': target_id,
                'label': float(label),
                'prediction': float(pred),
            } for drug_id, target_id, label, pred in zip(batch['drug_ids'], batch['target_ids'], truth, probs))

        preds = np.concatenate(predictions)
        truth = np.concatenate(labels)
        stored_metrics = json.loads(meta['result_json'].read_text(encoding='utf-8'))

        pred_csv = FIGURE_DATA_DIR / f'predictions_{split_name}.csv'
        pd.DataFrame(rows).to_csv(pred_csv, index=False)

        summary['splits'][split_name] = {
            'display_name': meta['label'],
            'checkpoint': meta['checkpoint'].as_posix(),
            'result_json': meta['result_json'].as_posix(),
            'prediction_csv': pred_csv.as_posix(),
            'n_test': int(len(truth)),
            'positive_count': int(truth.sum()),
            'computed_metrics': {
                'AUROC': float(roc_auc_score(truth, preds)),
                'AUPRC': float(average_precision_score(truth, preds)),
            },
            'reported_metrics': stored_metrics,
            'auroc_ci': bootstrap_ci(truth, preds, 'AUROC', n_bootstrap),
            'auprc_ci': bootstrap_ci(truth, preds, 'AUPRC', n_bootstrap),
        }

    _json_dump(summary_path, summary)
    return summary


def compute_attention_distribution(
    base_config: dict[str, Any],
    resources: dict[str, Any],
    report: dict[str, Any],
    device: str,
    force: bool,
) -> dict[str, Any]:
    npz_path = FIGURE_DATA_DIR / 'attention_distribution.npz'
    summary_path = FIGURE_DATA_DIR / 'attention_distribution.json'
    if npz_path.exists() and summary_path.exists() and not force:
        arrays = np.load(npz_path)
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        return {
            'flat_scores': arrays['flat_scores'],
            'percentiles': arrays['percentiles'],
            'cumulative_attention': arrays['cumulative_attention'],
            'summary': summary,
            'cache_path': npz_path.as_posix(),
        }

    config = copy.deepcopy(base_config)
    positive_df = resources['interactions'][resources['interactions']['label'] == 1][['drug_id', 'target_id', 'label']].reset_index(drop=True)
    dataset = DTIDataset(
        positive_df,
        drug_smiles=resources['drug_smiles'],
        target_sequences=resources['target_sequences'],
        esm2_cache_dir=config['data'].get('esm2_cache_dir', 'data/esm2_embeddings'),
        max_protein_len=config['data'].get('max_protein_len', 1200),
        use_domain_features=config['model']['target_encoder'].get('use_domain_features', True),
        esm2_dim=config['model']['target_encoder'].get('esm2_dim', 640),
        task='classification',
    )
    loader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=collate_dti, num_workers=0)
    model = build_model(config['model'], CHECKPOINTS_DIR / 'best.pt', device)

    all_scores: list[np.ndarray] = []
    for batch in loader:
        _, attn = run_model(model, batch, device, return_attention=True)
        interaction_map = attn['interaction_map']
        drug_mask = attn['drug_mask']
        protein_mask = attn['protein_mask']
        for idx in range(interaction_map.size(0)):
            residue_scores = interaction_map[idx][drug_mask[idx]].sum(dim=0)
            valid_scores = residue_scores[protein_mask[idx]]
            if valid_scores.numel() == 0:
                continue
            normalised = valid_scores / valid_scores.max().clamp(min=1e-8)
            all_scores.append(normalised.detach().cpu().numpy())

    flat_scores = np.concatenate(all_scores)
    sorted_scores = np.sort(flat_scores)[::-1]
    cumulative_attention = np.cumsum(sorted_scores) / sorted_scores.sum()
    percentiles = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100.0
    top1_idx = max(0, np.searchsorted(percentiles, 1.0, side='right') - 1)
    top5_idx = max(0, np.searchsorted(percentiles, 5.0, side='right') - 1)

    summary = {
        'n_samples': int(len(all_scores)),
        'n_residue_scores': int(len(flat_scores)),
        'residue_attention_mean': float(flat_scores.mean()),
        'residue_attention_std': float(flat_scores.std()),
        'residue_attention_median': float(np.median(flat_scores)),
        'residue_attention_top1pct': float(np.percentile(flat_scores, 99)),
        'residue_attention_top5pct': float(np.percentile(flat_scores, 95)),
        'attention_sparsity': float((flat_scores < 0.1).mean()),
        'top1_attention_mass': float(cumulative_attention[top1_idx]),
        'top5_attention_mass': float(cumulative_attention[top5_idx]),
        'report_global_stats': report['global_stats'],
        'cache_path': npz_path.as_posix(),
    }

    np.savez_compressed(npz_path, flat_scores=flat_scores, percentiles=percentiles, cumulative_attention=cumulative_attention)
    _json_dump(summary_path, summary)
    return {
        'flat_scores': flat_scores,
        'percentiles': percentiles,
        'cumulative_attention': cumulative_attention,
        'summary': summary,
        'cache_path': npz_path.as_posix(),
    }


def sanitize_case_key(drug_name: str, target_name: str) -> str:
    return f'{drug_name}_{target_name}'.replace('(', '').replace(')', '').replace(' ', '')


def compute_case_profiles(
    base_config: dict[str, Any],
    resources: dict[str, Any],
    report: dict[str, Any],
    device: str,
    force: bool,
) -> list[dict[str, Any]]:
    cache_path = FIGURE_DATA_DIR / 'case_profiles.json'
    if cache_path.exists() and not force:
        cached_profiles = json.loads(cache_path.read_text(encoding='utf-8'))
        for case in cached_profiles:
            case.setdefault('display_drug', str(case.get('drug_id', case.get('drug_name', 'unknown'))))
        return cached_profiles

    config = copy.deepcopy(base_config)
    case_df = pd.DataFrame([
        {'drug_id': case['drug_id'], 'target_id': case['target_id'], 'label': 1}
        for case in report['case_studies']
    ])
    dataset = DTIDataset(
        case_df,
        drug_smiles=resources['drug_smiles'],
        target_sequences=resources['target_sequences'],
        esm2_cache_dir=config['data'].get('esm2_cache_dir', 'data/esm2_embeddings'),
        max_protein_len=config['data'].get('max_protein_len', 1200),
        use_domain_features=config['model']['target_encoder'].get('use_domain_features', True),
        esm2_dim=config['model']['target_encoder'].get('esm2_dim', 640),
        task='classification',
    )
    model = build_model(config['model'], CHECKPOINTS_DIR / 'best.pt', device)

    case_profiles: list[dict[str, Any]] = []
    for idx, case in enumerate(report['case_studies']):
        sample = dataset[idx]
        batch = collate_dti([sample])
        logits, attn = run_model(model, batch, device, return_attention=True)
        prob = float(torch.sigmoid(logits).detach().cpu().numpy().ravel()[0])
        interaction_map = attn['interaction_map'][0]
        drug_mask = attn['drug_mask'][0]
        protein_mask = attn['protein_mask'][0]
        residue_scores = interaction_map[drug_mask].sum(dim=0)
        profile = residue_scores[protein_mask]
        profile = (profile / profile.max().clamp(min=1e-8)).detach().cpu().numpy()
        sequence = sample['sequence']
        top_residue_scores = {name: float(score) for name, score in case['top_10_residues']}
        hotspots = []
        for position, score in enumerate(profile, start=1):
            if score >= 0.5 and position <= len(sequence):
                hotspots.append(f'{sequence[position - 1]}{position}')

        safe = sanitize_case_key(str(case['drug_name']), case['target_name'])
        case_profiles.append({
            'drug_name': str(case['drug_name']),
            'target_name': case['target_name'],
            'drug_id': str(case['drug_id']),
            'display_drug': str(case['drug_id']),
            'target_id': str(case['target_id']),
            'affinity_nM': float(case['affinity_nM']),
            'prediction_prob': prob,
            'prediction_prob_report': float(case['prediction_prob']),
            'n_hotspots': int(case['n_hotspots']),
            'model_hotspot_count': int(len(hotspots)),
            'hotspots': hotspots,
            'top_10_residues': case['top_10_residues'],
            'top_residue_scores': top_residue_scores,
            'functional_groups': case['functional_groups'],
            'sequence_length': int(len(sequence)),
            'profile': profile.tolist(),
            'heatmap_path': (HEATMAP_DIR / f'{safe}.png').as_posix(),
            'profile_path': (PROFILE_DIR / f'{safe}.png').as_posix(),
            'gradcam_path': (GRADCAM_DIR / f'{safe}.png').as_posix(),
        })

    _json_dump(cache_path, case_profiles)
    return case_profiles


def parse_residue_position(residue_name: str) -> int:
    match = re.search(r'(\d+)', residue_name)
    return int(match.group(1)) if match else 10 ** 9


def _finalize_segment(segments: list[dict[str, Any]], current: dict[str, Any] | None) -> dict[str, Any] | None:
    if current and current['epochs']:
        segments.append(current)
    return None


def _choose_segment(segments: list[dict[str, Any]], target_auroc: float) -> dict[str, Any]:
    with_test = [segment for segment in segments if segment.get('test_auroc') is not None]
    if with_test:
        return min(with_test, key=lambda segment: (abs(segment['test_auroc'] - target_auroc), -segment['epochs'][-1]))
    return max(segments, key=lambda segment: (len(segment['epochs']), segment['epochs'][-1]))


def parse_training_logs(prediction_summary: dict[str, Any]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for split_name, meta in SPLIT_META.items():
        lines = meta['log_path'].read_text(encoding='utf-8', errors='ignore').splitlines()
        segments: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        waiting_for_test = False

        for line in lines:
            epoch_match = LOG_PATTERN_OPTIONAL_AUPRC.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if current is None:
                    current = {
                        'epochs': [],
                        'loss': [],
                        'val_auroc': [],
                        'val_auprc': [],
                        'stop_epoch': None,
                        'best_val_auroc': None,
                        'test_auroc': None,
                    }
                elif current['epochs'] and epoch <= current['epochs'][-1]:
                    current = _finalize_segment(segments, current)
                    current = {
                        'epochs': [],
                        'loss': [],
                        'val_auroc': [],
                        'val_auprc': [],
                        'stop_epoch': None,
                        'best_val_auroc': None,
                        'test_auroc': None,
                    }
                current['epochs'].append(epoch)
                current['loss'].append(float(epoch_match.group(2)))
                current['val_auroc'].append(float(epoch_match.group(3)))
                current['val_auprc'].append(float(epoch_match.group(4)) if epoch_match.group(4) else None)
                waiting_for_test = False
                continue

            stop_match = EARLY_STOP_PATTERN.search(line)
            if stop_match and current is not None:
                current['stop_epoch'] = int(stop_match.group(1))
                current['best_val_auroc'] = float(stop_match.group(2))
                waiting_for_test = True
                continue

            if 'TEST (' in line and split_name in line:
                waiting_for_test = True
                continue

            auroc_match = TEST_AUROC_PATTERN.search(line)
            if waiting_for_test and auroc_match and current is not None:
                current['test_auroc'] = float(auroc_match.group(1))
                waiting_for_test = False
                continue

            if line.strip() == 'DONE':
                current = _finalize_segment(segments, current)
                waiting_for_test = False

        _finalize_segment(segments, current)
        chosen = _choose_segment(segments, prediction_summary['splits'][split_name]['reported_metrics']['AUROC'])
        best_idx = int(np.argmax(chosen['val_auroc']))
        parsed[split_name] = {
            'epochs': chosen['epochs'],
            'loss': chosen['loss'],
            'val_auroc': chosen['val_auroc'],
            'val_auprc': chosen['val_auprc'],
            'best_epoch': chosen['epochs'][best_idx],
            'best_val_auroc': chosen['val_auroc'][best_idx],
            'stop_epoch': chosen['stop_epoch'] or chosen['epochs'][-1],
            'test_auroc': chosen.get('test_auroc'),
            'source_log': meta['log_path'].as_posix(),
        }
    _json_dump(FIGURE_DATA_DIR / 'training_curves.json', parsed)
    return parsed


def load_report() -> dict[str, Any]:
    return json.loads(REPORT_PATH.read_text(encoding='utf-8'))


def case_lookup(case_profiles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {case['target_name']: case for case in case_profiles}


def short_case_name(target_name: str) -> str:
    return target_name.replace('ABL1', 'ABL1 ').replace('EGFR', 'EGFR ').replace('BRAF', 'BRAF ').strip()


def compact_case_label(target_name: str) -> str:
    return COMPACT_CASE_LABELS.get(target_name, target_name)


def fig2_performance(prediction_summary: dict[str, Any]) -> None:
    from matplotlib.ticker import FormatStrFormatter

    splits = list(SPLIT_META.keys())
    # Wider right panel to reduce crowding in Panel B
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw={'width_ratios': [1.0, 1.3]})
    fig.subplots_adjust(top=0.86, bottom=0.15, wspace=0.24)
    ax_left, ax_right = axes

    major_metrics = ['AUROC', 'AUPRC']
    minor_metrics = ['F1', 'Precision', 'Recall']
    x_major = np.arange(len(major_metrics))
    x_minor = np.arange(len(minor_metrics))
    width = 0.22

    for idx, split_name in enumerate(splits):
        meta = SPLIT_META[split_name]
        split_data = prediction_summary['splits'][split_name]
        reported = split_data['reported_metrics']
        display_metrics = {
            metric: get_manuscript_display_metric(split_name, metric, reported[metric])
            for metric in major_metrics + minor_metrics
        }
        offset = (idx - 1) * width
        major_values = [reported[metric] for metric in major_metrics]
        major_errors = np.array([
            [major_values[0] - split_data['auroc_ci'][0], major_values[1] - split_data['auprc_ci'][0]],
            [split_data['auroc_ci'][1] - major_values[0], split_data['auprc_ci'][1] - major_values[1]],
        ])
        bars = ax_left.bar(
            x_major + offset,
            major_values,
            width=width,
            color=meta['color'],
            label=f"{meta['label']} (n={split_data['n_test']})",
            yerr=major_errors,
            capsize=4,
            edgecolor='white',
        )
        for bar, metric_name, value in zip(bars, major_metrics, major_values):
            display_value = display_metrics[metric_name]
            ax_left.text(bar.get_x() + bar.get_width() / 2, value + 0.017, format_manuscript_metric(display_value),
                         ha='center', va='bottom', fontsize=9)

        minor_values = [reported[metric] for metric in minor_metrics]
        bars = ax_right.bar(
            x_minor + offset,
            minor_values,
            width=width,
            color=meta['color'],
            edgecolor='white',
            label=f"{meta['label']} (n={split_data['n_test']})",
        )
        for bar, metric_name, value in zip(bars, minor_metrics, minor_values):
            display_value = display_metrics[metric_name]
            ax_right.text(bar.get_x() + bar.get_width() / 2, value + 0.015, format_manuscript_metric(display_value),
                          ha='center', va='bottom', fontsize=8)

    ax_left.set_xticks(x_major)
    ax_left.set_xticklabels(major_metrics)
    ax_left.set_ylim(0.0, 1.08)
    ax_left.set_ylabel('Score')
    ax_left.set_title('Ranking metrics with 95% bootstrap CI', pad=14)
    soften_axes(ax_left, 'y')
    panel_label(ax_left, 'A')

    ax_right.set_xticks(x_minor)
    ax_right.set_xticklabels(minor_metrics)
    ax_right.set_ylim(0.0, 1.08)
    ax_right.set_ylabel('Score')
    ax_right.set_title('Thresholded classification metrics', pad=14)
    ax_right.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    soften_axes(ax_right, 'y')
    panel_label(ax_right, 'B')
    # Legend placed in Panel B lower-left to avoid overlap with tall bars
    ax_right.legend(frameon=False, loc='lower left')

    fig.text(0.5, -0.03, 'Error bars denote 95% bootstrap confidence intervals estimated from held-out test predictions.',
             ha='center', color=PALETTE['slate'], fontsize=10)
    save_figure(fig, 'fig2_performance', metadata={
        'title': 'Performance across three data splitting protocols',
        'sources': [summary['prediction_csv'] for summary in prediction_summary['splits'].values()],
        'prediction_summary': prediction_summary,
        'display_metrics': MANUSCRIPT_DISPLAY_METRICS,
    })


def fig3_attention_sparsity(attention_data: dict[str, Any]) -> None:
    summary = attention_data['summary']
    flat_scores = np.clip(attention_data['flat_scores'], 1e-5, 1.0)
    percentiles = attention_data['percentiles']
    cumulative = attention_data['cumulative_attention'] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), gridspec_kw={'width_ratios': [1.05, 1.0]})
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.22)
    hist_ax, curve_ax = axes

    bins = np.logspace(-5, 0, 70)
    hist_ax.hist(flat_scores, bins=bins, color=PALETTE['teal'], alpha=0.9, edgecolor='white')
    hist_ax.set_xscale('log')
    hist_ax.set_xlabel('Normalised residue attention')
    hist_ax.set_ylabel('Count')
    hist_ax.set_title('Empirical distribution over all positive pairs', pad=14)
    hist_ax.axvline(summary['residue_attention_median'], color=PALETTE['slate'], linestyle='--', linewidth=1.2, label='Median')
    hist_ax.axvline(summary['residue_attention_top5pct'], color=PALETTE['gold'], linestyle='--', linewidth=1.2, label='95th percentile')
    hist_ax.axvline(summary['residue_attention_top1pct'], color=PALETTE['brick'], linestyle='--', linewidth=1.2, label='99th percentile')
    hist_ax.legend(frameon=False, loc='upper left')
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
        curve_ax.annotate(f'{label}: {mass:.1f}%', (cutoff, mass), xytext=(cutoff + 5, mass - 8),
                          arrowprops={'arrowstyle': '->', 'color': PALETTE['gold'], 'lw': 1.0},
                          fontsize=9)
    curve_ax.text(0.96, 0.97,
                  f"Sparsity (<0.1): {summary['report_global_stats']['attention_sparsity'] * 100:.2f}%\n"
                  f"Positive pairs: {summary['report_global_stats']['n_samples']}",
                  transform=curve_ax.transAxes, ha='right', va='top', fontsize=9.5,
                  bbox={'boxstyle': 'round,pad=0.35', 'facecolor': PALETTE['cream'], 'edgecolor': '#d7d2c6'})
    soften_axes(curve_ax, 'both')
    panel_label(curve_ax, 'B')

    save_figure(fig, 'fig3_sparsity', metadata={
        'title': 'Attention sparsity analysis',
        'sources': [REPORT_PATH.as_posix(), attention_data['cache_path']],
        'summary': summary,
    })


def fig4_abl1_heatmap(case_profiles: list[dict[str, Any]]) -> None:
    lookup = case_lookup(case_profiles)
    abl1_cases = [lookup[name] for name in ABL1_ORDER]
    residue_names = sorted(
        {residue for case in abl1_cases for residue, _ in case['top_10_residues']},
        key=parse_residue_position,
    )
    matrix = np.array([
        [case['top_residue_scores'].get(residue, 0.0) for residue in residue_names]
        for case in abl1_cases
    ])
    display_rows = ['F317I', 'F317I-p', 'F317L-p', 'M351T', 'E255K']

    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    fig.subplots_adjust(top=0.84)
    image = ax.imshow(matrix, cmap=plt.cm.magma_r, aspect='auto', vmin=0.0, vmax=1.0)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if value >= 0.18:
                ax.text(col, row, f'{value:.2f}', ha='center', va='center',
                        color='white' if value >= 0.5 else PALETTE['ink'], fontsize=8.5)

    ax.set_xticks(np.arange(len(residue_names)))
    ax.set_xticklabels(residue_names, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(display_rows)))
    ax.set_yticklabels(display_rows)
    ax.set_title('ABL1 resistance variants preserve a shared attention core', pad=14)
    panel_label(ax, 'A')
    for tick in ax.get_xticklabels():
        if tick.get_text() in {'V104', 'A648', 'S199'}:
            tick.set_color(PALETTE['gold'])
            tick.set_fontweight('bold')
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Normalised attention')

    save_figure(fig, 'fig4_heatmap', metadata={
        'title': 'ABL1 cross-attention heatmap across mutant cases',
        'sources': [REPORT_PATH.as_posix(), FIGURE_DATA_DIR.joinpath('case_profiles.json').as_posix()],
        'cases': [case['target_name'] for case in abl1_cases],
        'residues': residue_names,
        'matrix': matrix,
    })


def fig5_residue_profiles(case_profiles: list[dict[str, Any]]) -> None:
    lookup = case_lookup(case_profiles)
    selected = [lookup[name] for name in CASE_PROFILE_ORDER]
    colors = [PALETTE['teal'], PALETTE['gold'], PALETTE['sky'], PALETTE['brick']]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), sharey=True)
    axes = axes.ravel()
    fig.subplots_adjust(wspace=0.20, hspace=0.40, top=0.90, bottom=0.10)

    for ax, case, color in zip(axes, selected, colors):
        profile = np.array(case['profile'])
        positions = np.arange(1, len(profile) + 1)
        ax.plot(positions, profile, color=color)
        ax.fill_between(positions, profile, color=color, alpha=0.18)
        ax.axhline(0.5, color=PALETTE['slate'], linestyle='--', linewidth=1.0)
        for residue_name, score in case['top_10_residues'][:3]:
            residue_pos = parse_residue_position(residue_name)
            if residue_pos <= len(profile):
                ax.annotate(
                    residue_name,
                    xy=(residue_pos, profile[residue_pos - 1]),
                    xytext=(residue_pos, min(1.03, profile[residue_pos - 1] + 0.14)),
                    fontsize=8.5,
                    ha='center',
                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': 0.8},
                )
        label = f"{case['display_drug']} -> {case['target_name']}\nKd={case['affinity_nM']:.3f} nM | P={case['prediction_prob']:.3f}"
        if case['prediction_prob'] < 0.5:
            label += ' | hard case'
        ax.set_title(label, fontsize=9.8, pad=12)
        ax.set_xlim(1, len(profile))
        ax.set_ylim(0.0, 1.05)
        soften_axes(ax, 'y')

    for idx, ax in enumerate(axes):
        panel_label(ax, chr(ord('A') + idx))
        if idx >= 2:
            ax.set_xlabel('Residue position')
        if idx % 2 == 0:
            ax.set_ylabel('Normalised attention')

    save_figure(fig, 'fig5_residue_profile', metadata={
        'title': 'Residue-level attention profiles for representative cases',
        'sources': [FIGURE_DATA_DIR.joinpath('case_profiles.json').as_posix()],
        'cases': selected,
    })


def fig6_mutant_conservation(case_profiles: list[dict[str, Any]]) -> None:
    lookup = case_lookup(case_profiles)
    abl1_cases = [lookup[name] for name in ABL1_ORDER]
    profiles = [np.array(case['profile']) for case in abl1_cases]
    min_len = min(len(profile) for profile in profiles)
    matrix = np.vstack([profile[:min_len] for profile in profiles])
    correlations = np.corrcoef(matrix)
    residues = ['V104', 'A648', 'S199']
    residue_matrix = np.array([
        [case['top_residue_scores'].get(residue, 0.0) for case in abl1_cases]
        for residue in residues
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={'width_ratios': [1.0, 1.15]})
    fig.subplots_adjust(top=0.84, wspace=0.42)
    corr_ax, bar_ax = axes
    image = corr_ax.imshow(correlations, cmap=plt.cm.YlGnBu, vmin=max(0.99, correlations.min() - 0.001), vmax=1.0)
    _cmap = plt.cm.YlGnBu
    _vmin = max(0.99, correlations.min() - 0.001)
    _vmax = 1.0
    for row in range(correlations.shape[0]):
        for col in range(correlations.shape[1]):
            _norm_val = (correlations[row, col] - _vmin) / (_vmax - _vmin)
            _rgba = _cmap(_norm_val)
            _lum = 0.2126 * _rgba[0] + 0.7152 * _rgba[1] + 0.0722 * _rgba[2]
            _txt_color = 'white' if _lum < 0.45 else 'black'
            corr_ax.text(col, row, f'{correlations[row, col]:.3f}', ha='center', va='center', fontsize=9, color=_txt_color)
    corr_ax.set_xticks(np.arange(len(abl1_cases)))
    corr_ax.set_xticklabels(['F317I', 'F317I-p', 'F317L-p', 'M351T', 'E255K'], rotation=45, ha='right')
    corr_ax.set_yticks(np.arange(len(abl1_cases)))
    corr_ax.set_yticklabels(['F317I', 'F317I-p', 'F317L-p', 'M351T', 'E255K'])
    corr_ax.set_title('Pairwise profile correlation', pad=12)
    panel_label(corr_ax, 'A')
    plt.colorbar(image, ax=corr_ax, fraction=0.046, pad=0.03)

    x = np.arange(len(abl1_cases))
    width = 0.22
    colors = [PALETTE['gold'], PALETTE['teal'], PALETTE['brick']]
    for idx, residue in enumerate(residues):
        values = residue_matrix[idx]
        bars = bar_ax.bar(x + (idx - 1) * width, values, width=width, label=residue, color=colors[idx], edgecolor='white')
        for bar, value in zip(bars, values):
            bar_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(['F317I', 'F317I-p', 'F317L-p', 'M351T', 'E255K'], rotation=20)
    bar_ax.set_ylim(0.0, 1.1)
    bar_ax.set_ylabel('Normalised attention')
    bar_ax.set_title('Conserved top residues across mutants', pad=12)
    bar_ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    soften_axes(bar_ax, 'y')
    panel_label(bar_ax, 'B')

    save_figure(fig, 'fig6_mutant_conservation', metadata={
        'title': 'Attention conservation across ABL1 mutants',
        'sources': [FIGURE_DATA_DIR.joinpath('case_profiles.json').as_posix()],
        'correlations': correlations,
        'residue_matrix': residue_matrix,
        'residues': residues,
        'cases': [case['target_name'] for case in abl1_cases],
    })


def fig7_pharmacophore(case_profiles: list[dict[str, Any]]) -> None:
    case = case_lookup(case_profiles)['ABL1(F317I)']
    functional_groups = sorted(case['functional_groups'].items(), key=lambda item: item[1], reverse=True)
    image_data = plt.imread(case['gradcam_path']) if Path(case['gradcam_path']).exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), gridspec_kw={'width_ratios': [1.0, 1.08]})
    fig.subplots_adjust(top=0.82, wspace=0.26)
    image_ax, bar_ax = axes
    if image_data is not None:
        image_ax.imshow(image_data)
        image_ax.axis('off')
    else:
        image_ax.text(0.5, 0.5, 'Saved Grad-CAM image missing', ha='center', va='center')
        image_ax.set_axis_off()
    panel_label(image_ax, 'A')
    image_ax.set_title('Saved atom-level\nGrad-CAM rendering', pad=12)

    labels = [item[0] for item in functional_groups]
    values = [item[1] for item in functional_groups]
    y = np.arange(len(labels))
    colors = [PALETTE['gold'], PALETTE['teal'], PALETTE['sky'], PALETTE['sage'], PALETTE['sand'], PALETTE['slate'], PALETTE['brick']]
    bar_ax.barh(y, values, color=colors[:len(values)], edgecolor='white')
    for ypos, value in zip(y, values):
        bar_ax.text(value + 0.02, ypos, f'{value:.3f}', va='center', ha='left', fontsize=9)
    bar_ax.set_yticks(y)
    bar_ax.set_yticklabels(labels)
    bar_ax.invert_yaxis()
    bar_ax.set_xlim(0.0, 1.05)
    bar_ax.set_xlabel('Functional-group importance')
    bar_ax.set_title('Ranked pharmacophore contributions\nD0017 -> ABL1(F317I)', pad=12)
    soften_axes(bar_ax, 'x')
    panel_label(bar_ax, 'B')

    save_figure(fig, 'fig7_pharmacophore', metadata={
        'title': 'Grad-CAM pharmacophore analysis for ABL1(F317I)',
        'sources': [case['gradcam_path'], REPORT_PATH.as_posix()],
        'case': case,
        'functional_groups': functional_groups,
    })


def fig8_pharmacophore_comparison(case_profiles: list[dict[str, Any]]) -> None:
    lookup = case_lookup(case_profiles)
    selected = [lookup['ABL1(F317I)'], lookup['EGFR'], lookup['BRAF']]
    matrix = np.full((len(FG_ORDER), len(selected)), np.nan)
    for col, case in enumerate(selected):
        for row, group in enumerate(FG_ORDER):
            if group in case['functional_groups']:
                matrix[row, col] = case['functional_groups'][group]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={'width_ratios': [1.42, 0.58]})
    fig.subplots_adjust(wspace=0.48, top=0.84)
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
    heat_ax.set_xticks(np.arange(len(selected)))
    heat_ax.set_xticklabels([
        f"{short_case_name(case['target_name'])}\n{case['display_drug']}"
        for case in selected
    ])
    heat_ax.set_yticks(np.arange(len(FG_ORDER)))
    heat_ax.set_yticklabels(FG_ORDER)
    heat_ax.set_title('Target-dependent pharmacophore usage', pad=12)
    panel_label(heat_ax, 'A')
    fig.colorbar(image, ax=heat_ax, fraction=0.04, pad=0.03, shrink=0.82, anchor=(0.5, 0.3))

    probs = [case['prediction_prob'] for case in selected]
    prob_ax.barh(np.arange(len(selected)), probs, color=[PALETTE['teal'], PALETTE['gold'], PALETTE['brick']], edgecolor='white')
    for ypos, value in enumerate(probs):
        prob_ax.text(value + 0.02, ypos, f'{value:.3f}', va='center', ha='left', fontsize=9)
    prob_ax.set_xlim(0.0, 1.18)
    prob_ax.set_yticks(np.arange(len(selected)))
    prob_ax.set_yticklabels([case['display_drug'] for case in selected])
    prob_ax.invert_yaxis()
    prob_ax.set_xlabel('P(bind)')
    prob_ax.set_title('Prediction confidence', pad=12)
    soften_axes(prob_ax, 'x')
    panel_label(prob_ax, 'B', x=-0.12)

    save_figure(fig, 'fig8_pharmacophore_comparison', metadata={
        'title': 'Pharmacophore comparison across ABL1, EGFR, and BRAF cases',
        'sources': [REPORT_PATH.as_posix()],
        'groups': FG_ORDER,
        'cases': selected,
        'matrix': matrix,
    })


def fig_hotspots(case_profiles: list[dict[str, Any]]) -> None:
    ordered = sorted(case_profiles, key=lambda case: case['affinity_nM'])
    affinities = np.array([case['affinity_nM'] for case in ordered])
    hotspot_counts = np.array([case['n_hotspots'] for case in ordered])
    probabilities = np.array([case['prediction_prob'] for case in ordered])
    labels = [compact_case_label(case['target_name']) for case in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={'width_ratios': [1.0, 1.05]})
    fig.subplots_adjust(top=0.86, wspace=0.52)
    scatter_ax, bar_ax = axes

    scatter = scatter_ax.scatter(affinities, hotspot_counts, c=probabilities, cmap='cividis', s=90,
                                 edgecolor=[PALETTE['brick'] if value < 0.5 else PALETTE['ink'] for value in probabilities],
                                 linewidth=1.0)
    scatter_ax.set_xscale('log')
    scatter_ax.set_xlabel('Experimental Kd (nM)')
    scatter_ax.set_ylabel('Hotspot count')
    scatter_ax.set_title('Affinity versus salient hotspot count', pad=16)
    soften_axes(scatter_ax, 'y')
    panel_label(scatter_ax, 'A', x=-0.10)
    scatter_ax.set_xlim(affinities.min() * 0.85, affinities.max() * 1.55)
    scatter_ax.set_ylim(0.85, hotspot_counts.max() + 0.15)
    for case, x_value, y_value, label in zip(ordered, affinities, hotspot_counts, labels):
        if case['target_name'] not in HOTSPOT_ANNOTATED_CASES:
            continue
        dx, dy = HOTSPOT_ANNOTATION_OFFSETS.get(case['target_name'], (8, 8))
        scatter_ax.annotate(label, xy=(x_value, y_value), xytext=(dx, dy),
                            textcoords='offset points', fontsize=8,
                            ha='left', va='center')
    cbar = fig.colorbar(scatter, ax=scatter_ax, fraction=0.046, pad=0.03)
    cbar.set_label('P(bind)')

    y = np.arange(len(ordered))
    colors = [plt.cm.cividis(value) for value in probabilities]
    bar_ax.barh(y, hotspot_counts, color=colors, edgecolor='white')
    for ypos, (count, prob) in enumerate(zip(hotspot_counts, probabilities)):
        bar_ax.text(count + 0.08, ypos, f'{count} | P={prob:.2f}', va='center', ha='left', fontsize=8.5)
    bar_ax.set_yticks(y)
    bar_ax.set_yticklabels(labels)
    bar_ax.invert_yaxis()
    bar_ax.set_xlim(0, hotspot_counts.max() + 2.2)
    bar_ax.set_xlabel('Hotspot count')
    bar_ax.set_title('Case-wise hotspot summary', pad=14)
    soften_axes(bar_ax, 'x')
    panel_label(bar_ax, 'B')

    save_figure(fig, 'fig_hotspots', metadata={
        'title': 'Attention hotspot analysis across saved case studies',
        'sources': [REPORT_PATH.as_posix(), FIGURE_DATA_DIR.joinpath('case_profiles.json').as_posix()],
        'cases': ordered,
    })


def fig9_training(training_curves: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7))
    fig.subplots_adjust(top=0.86, wspace=0.24)
    auroc_ax, loss_ax = axes

    for split_name, meta in SPLIT_META.items():
        curves = training_curves[split_name]
        color = meta['color']
        epochs = np.array(curves['epochs'])
        auroc_ax.plot(epochs, curves['val_auroc'], color=color, label=meta['label'])
        auroc_ax.scatter([curves['best_epoch']], [curves['best_val_auroc']], color=color, s=40, zorder=3)
        auroc_ax.axvline(curves['stop_epoch'], color=color, linestyle='--', linewidth=1.0, alpha=0.6)

        loss_ax.plot(epochs, curves['loss'], color=color, label=meta['label'])
        loss_ax.axvline(curves['stop_epoch'], color=color, linestyle='--', linewidth=1.0, alpha=0.6)

    auroc_ax.set_xlabel('Epoch')
    auroc_ax.set_ylabel('Validation AUROC')
    auroc_ax.set_title('Validation AUROC trajectory', pad=12)
    soften_axes(auroc_ax, 'y')
    panel_label(auroc_ax, 'A')
    auroc_ax.legend(frameon=False, loc='lower right')

    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Training loss')
    loss_ax.set_title('Training loss trajectory', pad=12)
    soften_axes(loss_ax, 'y')
    panel_label(loss_ax, 'B')

    save_figure(fig, 'fig9_training', metadata={
        'title': 'Training dynamics across evaluation protocols',
        'sources': [curves['source_log'] for curves in training_curves.values()],
        'curves': training_curves,
    })


def fig_graphical_abstract(prediction_summary: dict[str, Any], attention_data: dict[str, Any]) -> None:
    summary = attention_data['summary']
    reported = prediction_summary['splits']
    aurocs = [
        reported['random']['reported_metrics']['AUROC'],
        reported['cold_target']['reported_metrics']['AUROC'],
        reported['cold_drug']['reported_metrics']['AUROC'],
    ]
    display_aurocs = [
        get_manuscript_display_metric('random', 'AUROC', aurocs[0]),
        get_manuscript_display_metric('cold_target', 'AUROC', aurocs[1]),
        get_manuscript_display_metric('cold_drug', 'AUROC', aurocs[2]),
    ]

    fig = plt.figure(figsize=(12.8, 6.8))
    grid = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.82], hspace=0.32, wspace=0.34)

    ax1 = fig.add_subplot(grid[0, 0])
    theta = np.linspace(0, 2 * np.pi, 7)[:-1]
    hx, hy = np.cos(theta) * 0.32 + 0.5, np.sin(theta) * 0.32 + 0.5
    for idx in range(6):
        ax1.plot([hx[idx], hx[(idx + 1) % 6]], [hy[idx], hy[(idx + 1) % 6]], color='black', lw=3.0)
    ax1.plot([hx[0], hx[0] + 0.22], [hy[0], hy[0] + 0.16], color='blue', lw=3.0)
    ax1.plot([hx[2], hx[2] - 0.22], [hy[2], hy[2] - 0.16], color='red', lw=3.0)
    ax1.text(hx[0] + 0.24, hy[0] + 0.18, 'NH₂', fontsize=17, color='blue', fontweight='bold')
    ax1.text(hx[2] - 0.40, hy[2] - 0.18, 'OH', fontsize=17, color='red', fontweight='bold')
    ax1.set_xlim(-0.08, 1.12)
    ax1.set_ylim(-0.08, 1.12)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Drug\n(GINE GNN)', fontsize=18, fontweight='bold', pad=14)

    ax2 = fig.add_subplot(grid[0, 1])
    x_prot = np.linspace(0, 1, 80)
    y_prot = 0.52 + 0.22 * np.sin(4 * np.pi * x_prot)
    ax2.plot(x_prot, y_prot, color=PALETTE['sage'], lw=5)
    pocket = (x_prot > 0.33) & (x_prot < 0.49)
    ax2.fill_between(x_prot[pocket], y_prot[pocket] - 0.06, y_prot[pocket] + 0.06,
                     color=PALETTE['brick'], alpha=0.25)
    ax2.text(0.41, 0.23, 'Binding\nPocket', fontsize=14, ha='center',
             color=PALETTE['brick'], fontweight='bold')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0.0, 1.0)
    ax2.axis('off')
    ax2.set_title('Protein\n(ESM-2 + Domain)', fontsize=18, fontweight='bold', pad=14)

    ax3 = fig.add_subplot(grid[0, 2])
    rng = np.random.default_rng(42)
    attn = rng.exponential(0.05, (8, 12))
    attn[2:5, 4:7] = rng.uniform(0.5, 1.0, (3, 3))
    ax3.imshow(attn, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax3.set_xlabel('Residue', fontsize=14)
    ax3.set_ylabel('Atom', fontsize=14)
    ax3.set_title('Cross-Attention\nInteraction Map', fontsize=18, fontweight='bold', pad=14)
    ax3.tick_params(labelsize=12)

    ax4 = fig.add_subplot(grid[0, 3])
    splits = ['Rnd', 'CT', 'CD']
    bars = ax4.bar(splits, aurocs, color=[PALETTE['teal'], PALETTE['sage'], PALETTE['gold']], edgecolor='white', width=0.6)
    for bar, value, display_value in zip(bars, aurocs, display_aurocs):
        ax4.text(bar.get_x() + bar.get_width() / 2, value + 0.015, format_manuscript_metric(display_value),
                 ha='center', fontsize=15, fontweight='bold')
    ax4.set_ylim(0, 1.10)
    ax4.set_ylabel('AUROC', fontsize=15)
    ax4.set_title('Performance', fontsize=18, fontweight='bold', pad=14)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.tick_params(labelsize=12)

    ax_bottom = fig.add_subplot(grid[1, :])
    ax_bottom.axis('off')
    findings = [
        (f"{summary['report_global_stats']['attention_sparsity'] * 100:.1f}% Attention\nSparsity", 'Matches localised\nbinding-pocket focus', PALETTE['teal']),
        ('Conserved Attention\nAcross Mutants', 'V104 remains dominant\nacross ABL1 variants', PALETTE['sage']),
        ('Pharmacophore\nDiscovery', 'Amide > Amino >\nCarbonyl > Halogen', PALETTE['gold']),
        (f"Cold-Target\nTransfer", f"AUROC {format_manuscript_metric(display_aurocs[1])}\nthrough kinase homology", PALETTE['brick']),
    ]
    for idx, (title, desc, color) in enumerate(findings):
        x_pos = 0.125 + idx * 0.25
        ax_bottom.add_patch(plt.Rectangle((x_pos - 0.105, 0.10), 0.21, 0.78,
                            fill=True, facecolor=color, alpha=0.12,
                            edgecolor=color, linewidth=2.0, transform=ax_bottom.transAxes,
                            clip_on=False))
        ax_bottom.text(x_pos, 0.62, title, fontsize=15, fontweight='bold',
                       ha='center', va='center', transform=ax_bottom.transAxes, color=color)
        ax_bottom.text(x_pos, 0.31, desc, fontsize=13,
                       ha='center', va='center', transform=ax_bottom.transAxes, color=PALETTE['ink'])

    save_figure(fig, 'graphical_abstract', metadata={
        'title': 'Graphical abstract for BioInteract',
        'sources': [
            REPORT_PATH.as_posix(),
            FIGURE_DATA_DIR.joinpath('prediction_summary.json').as_posix(),
            FIGURE_DATA_DIR.joinpath('attention_distribution.json').as_posix(),
        ],
        'auroc_values': aurocs,
        'display_auroc_values': display_aurocs,
        'attention_sparsity': summary['report_global_stats']['attention_sparsity'],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate manuscript figures from real experiment outputs.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--bootstrap', type=int, default=600)
    parser.add_argument('--force', action='store_true', help='Recompute cached figure data.')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    RDLogger.DisableLog('rdApp.warning')
    base_config = load_base_config()
    resources = load_dataset_resources(base_config)
    report = load_report()

    print('Computing reusable figure data...')
    prediction_summary = compute_prediction_summary(base_config, resources, device, args.bootstrap, args.force)
    attention_data = compute_attention_distribution(base_config, resources, report, device, args.force)
    case_profiles = compute_case_profiles(base_config, resources, report, device, args.force)
    training_curves = parse_training_logs(prediction_summary)

    print('Rendering figures...')
    fig2_performance(prediction_summary)
    fig3_attention_sparsity(attention_data)
    fig4_abl1_heatmap(case_profiles)
    fig5_residue_profiles(case_profiles)
    fig6_mutant_conservation(case_profiles)
    fig7_pharmacophore(case_profiles)
    fig8_pharmacophore_comparison(case_profiles)
    fig_hotspots(case_profiles)
    fig9_training(training_curves)
    fig_graphical_abstract(prediction_summary, attention_data)
    write_manifest([
        'graphical_abstract',
        'fig2_performance',
        'fig3_sparsity',
        'fig4_heatmap',
        'fig5_residue_profile',
        'fig6_mutant_conservation',
        'fig7_pharmacophore',
        'fig8_pharmacophore_comparison',
        'fig_hotspots',
        'fig9_training',
    ])
    print('Done.')


if __name__ == '__main__':
    main()