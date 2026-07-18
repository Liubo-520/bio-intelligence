"""Reproducible Davis split diagnostics used for the Reviewer 3 revision."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rapidfuzz import fuzz


RDLogger.DisableLog('rdApp.warning')


def _partition(values: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.RandomState(seed)
    values = values.copy()
    rng.shuffle(values)
    n_test = int(len(values) * test_ratio)
    n_val = int(len(values) * val_ratio)
    test = set(values[:n_test])
    val = set(values[n_test:n_test + n_val])
    train = set(values[n_test + n_val:])
    return train, val, test


def _summarise(values: list[float]) -> dict[str, float | int]:
    return {
        'n': len(values),
        'mean': round(float(mean(values)), 4),
        'median': round(float(median(values)), 4),
        'minimum': round(float(min(values)), 4),
        'maximum': round(float(max(values)), 4),
    }


def _fingerprints(drug_table: pd.DataFrame) -> dict[str, object]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints: dict[str, object] = {}
    for row in drug_table.itertuples(index=False):
        molecule = Chem.MolFromSmiles(row.smiles)
        if molecule is None:
            raise ValueError(f'Invalid SMILES for {row.drug_id}')
        fingerprints[row.drug_id] = generator.GetFingerprint(molecule)
    return fingerprints


def _max_fingerprint_similarity(test_ids: set[str], train_ids: set[str], fingerprints: dict[str, object]) -> tuple[list[float], list[dict[str, object]]]:
    train_order = sorted(train_ids)
    values: list[float] = []
    nearest: list[dict[str, object]] = []
    for test_id in sorted(test_ids):
        scores = [float(DataStructs.TanimotoSimilarity(fingerprints[test_id], fingerprints[train_id])) for train_id in train_order]
        index = int(np.argmax(scores))
        score = scores[index]
        values.append(score)
        nearest.append({'test_drug_id': test_id, 'nearest_train_drug_id': train_order[index], 'tanimoto': round(score, 4)})
    nearest.sort(key=lambda item: item['tanimoto'], reverse=True)
    return values, nearest[:10]


def _max_sequence_similarity(test_ids: set[str], train_ids: set[str], sequences: dict[str, str]) -> tuple[list[float], list[dict[str, object]]]:
    train_order = sorted(train_ids)
    values: list[float] = []
    nearest: list[dict[str, object]] = []
    for test_id in sorted(test_ids):
        scores = [float(fuzz.ratio(sequences[test_id], sequences[train_id]) / 100.0) for train_id in train_order]
        index = int(np.argmax(scores))
        score = scores[index]
        values.append(score)
        nearest.append({'test_target_id': test_id, 'nearest_train_target_id': train_order[index], 'normalised_indel_similarity': round(score, 4)})
    nearest.sort(key=lambda item: item['normalised_indel_similarity'], reverse=True)
    return values, nearest[:10]


def _exact_sequence_groups(sequences: dict[str, str]) -> list[list[str]]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for target_id, sequence in sequences.items():
        groups[sequence].append(target_id)
    duplicates = [sorted(group) for group in groups.values() if len(group) > 1]
    return sorted(duplicates, key=lambda group: (-len(group), group))


def _sequence_group_partition(targets: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> tuple[set[str], set[str], set[str]]:
    grouped_ids: defaultdict[str, list[str]] = defaultdict(list)
    for row in targets.itertuples(index=False):
        grouped_ids[row.sequence].append(row.target_id)
    groups = list(grouped_ids.values())
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    n_test = int(len(targets) * test_ratio)
    n_val = int(len(targets) * val_ratio)

    def take_until(target_size: int, remaining: list[list[str]]) -> set[str]:
        selected: set[str] = set()
        while remaining and len(selected) < target_size:
            selected.update(remaining.pop(0))
        return selected

    test_groups = groups.copy()
    test_ids = take_until(n_test, test_groups)
    val_ids = take_until(n_val, test_groups)
    train_ids = {target_id for group in test_groups for target_id in group}
    return train_ids, val_ids, test_ids


def sequence_group_cold_target_split(interactions: pd.DataFrame, targets: pd.DataFrame, seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a target-held-out split that keeps identical sequences together."""
    train_targets, val_targets, test_targets = _sequence_group_partition(targets, val_ratio, test_ratio, seed)
    train_df = interactions[interactions['target_id'].isin(train_targets)].copy()
    val_df = interactions[interactions['target_id'].isin(val_targets)].copy()
    test_df = interactions[interactions['target_id'].isin(test_targets)].copy()
    return train_df, val_df, test_df


def sequence_group_cold_both_split(interactions: pd.DataFrame, targets: pd.DataFrame, seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out both drug IDs and exact target-sequence groups."""
    drug_ids = interactions['drug_id'].drop_duplicates().to_numpy()
    train_drugs, val_drugs, test_drugs = _partition(drug_ids, val_ratio, test_ratio, seed)
    train_targets, val_targets, test_targets = _sequence_group_partition(targets, val_ratio, test_ratio, seed)
    train_df = interactions[interactions['drug_id'].isin(train_drugs) & interactions['target_id'].isin(train_targets)].copy()
    val_df = interactions[interactions['drug_id'].isin(val_drugs) & interactions['target_id'].isin(val_targets)].copy()
    test_df = interactions[interactions['drug_id'].isin(test_drugs) & interactions['target_id'].isin(test_targets)].copy()
    return train_df, val_df, test_df


def build_split_diagnostics(data_root: Path, seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.2) -> dict[str, object]:
    """Return deterministic diagnostics for the submitted Davis split protocol."""
    interactions = pd.read_csv(data_root / 'interactions.csv')
    drugs = pd.read_csv(data_root / 'drug_smiles.csv')
    targets = pd.read_csv(data_root / 'target_sequences.csv')
    drug_ids = interactions['drug_id'].drop_duplicates().to_numpy()
    target_ids = interactions['target_id'].drop_duplicates().to_numpy()

    train_drugs, val_drugs, test_drugs = _partition(drug_ids, val_ratio, test_ratio, seed)
    train_targets, val_targets, test_targets = _partition(target_ids, val_ratio, test_ratio, seed)
    fingerprints = _fingerprints(drugs)
    sequences = dict(zip(targets['target_id'], targets['sequence']))
    tanimoto, nearest_drugs = _max_fingerprint_similarity(test_drugs, train_drugs, fingerprints)
    sequence_similarity, nearest_targets = _max_sequence_similarity(test_targets, train_targets, sequences)

    cold_drug_train = interactions[interactions['drug_id'].isin(train_drugs)]
    cold_drug_val = interactions[interactions['drug_id'].isin(val_drugs)]
    cold_drug_test = interactions[interactions['drug_id'].isin(test_drugs)]
    cold_target_train = interactions[interactions['target_id'].isin(train_targets)]
    cold_target_val = interactions[interactions['target_id'].isin(val_targets)]
    cold_target_test = interactions[interactions['target_id'].isin(test_targets)]
    cold_both_train = interactions[interactions['drug_id'].isin(train_drugs) & interactions['target_id'].isin(train_targets)]
    cold_both_val = interactions[interactions['drug_id'].isin(val_drugs) & interactions['target_id'].isin(val_targets)]
    cold_both_test = interactions[interactions['drug_id'].isin(test_drugs) & interactions['target_id'].isin(test_targets)]
    duplicate_groups = _exact_sequence_groups(sequences)
    exact_train_test_duplicates = [
        group for group in duplicate_groups
        if set(group).intersection(train_targets) and set(group).intersection(test_targets)
    ]
    sequence_grouped_train, sequence_grouped_val, sequence_grouped_test = sequence_group_cold_target_split(
        interactions, targets, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio
    )
    sequence_grouped_both_train, sequence_grouped_both_val, sequence_grouped_both_test = sequence_group_cold_both_split(
        interactions, targets, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio
    )
    sequence_grouped_train_ids = set(sequence_grouped_train['target_id'].unique())
    sequence_grouped_test_ids = set(sequence_grouped_test['target_id'].unique())
    sequence_grouped_train_sequences = {sequences[target_id] for target_id in sequence_grouped_train_ids}
    sequence_grouped_test_sequences = {sequences[target_id] for target_id in sequence_grouped_test_ids}

    return {
        'dataset': 'Davis',
        'seed': seed,
        'ratios': {'train': round(1 - val_ratio - test_ratio, 3), 'validation': val_ratio, 'test': test_ratio},
        'similarity_definitions': {
            'drug': 'Morgan fingerprint (radius 2, 1024 bits) Tanimoto similarity.',
            'protein': 'Normalised Indel similarity calculated for each test sequence against every training sequence; this is a sequence-similarity diagnostic, not a clustering constraint.',
        },
        'cold_drug': {
            'split_sizes': {'train_pairs': len(cold_drug_train), 'validation_pairs': len(cold_drug_val), 'test_pairs': len(cold_drug_test), 'train_drugs': len(train_drugs), 'validation_drugs': len(val_drugs), 'test_drugs': len(test_drugs)},
            'entity_overlap': len(train_drugs.intersection(test_drugs)),
            'similarity': {'test_to_train_max': _summarise(tanimoto), 'nearest_pairs': nearest_drugs},
        },
        'cold_target': {
            'split_sizes': {'train_pairs': len(cold_target_train), 'validation_pairs': len(cold_target_val), 'test_pairs': len(cold_target_test), 'train_targets': len(train_targets), 'validation_targets': len(val_targets), 'test_targets': len(test_targets)},
            'entity_overlap': len(train_targets.intersection(test_targets)),
            'sequence_identity': {'test_to_train_max': _summarise(sequence_similarity), 'nearest_pairs': nearest_targets},
            'exact_sequence_redundancy': {'duplicate_groups_in_full_dataset': len(duplicate_groups), 'target_ids_in_duplicate_groups': int(sum(len(group) for group in duplicate_groups)), 'groups_crossing_train_test': len(exact_train_test_duplicates), 'crossing_groups': exact_train_test_duplicates},
        },
        'cold_both': {
            'split_sizes': {'train_pairs': len(cold_both_train), 'validation_pairs': len(cold_both_val), 'test_pairs': len(cold_both_test), 'train_drugs': len(train_drugs), 'validation_drugs': len(val_drugs), 'test_drugs': len(test_drugs), 'train_targets': len(train_targets), 'validation_targets': len(val_targets), 'test_targets': len(test_targets)},
            'drug_overlap': len(train_drugs.intersection(test_drugs)),
            'target_overlap': len(train_targets.intersection(test_targets)),
        },
        'sequence_grouped_cold_target': {
            'split_sizes': {'train_pairs': len(sequence_grouped_train), 'validation_pairs': len(sequence_grouped_val), 'test_pairs': len(sequence_grouped_test), 'train_targets': len(sequence_grouped_train_ids), 'validation_targets': len(sequence_grouped_val['target_id'].unique()), 'test_targets': len(sequence_grouped_test_ids)},
            'exact_sequence_train_test_overlap': len(sequence_grouped_train_sequences.intersection(sequence_grouped_test_sequences)),
        },
        'sequence_grouped_cold_both': {
            'split_sizes': {'train_pairs': len(sequence_grouped_both_train), 'validation_pairs': len(sequence_grouped_both_val), 'test_pairs': len(sequence_grouped_both_test), 'train_drugs': len(train_drugs), 'validation_drugs': len(val_drugs), 'test_drugs': len(test_drugs), 'train_targets': len(sequence_grouped_both_train['target_id'].unique()), 'validation_targets': len(sequence_grouped_both_val['target_id'].unique()), 'test_targets': len(sequence_grouped_both_test['target_id'].unique())},
            'drug_train_test_overlap': len(train_drugs.intersection(test_drugs)),
            'exact_sequence_train_test_overlap': len({sequences[target_id] for target_id in sequence_grouped_both_train['target_id'].unique()}.intersection({sequences[target_id] for target_id in sequence_grouped_both_test['target_id'].unique()})),
        },
    }


def render_diagnostics(report: dict[str, object], output_path: Path) -> None:
    tanimoto = [item['tanimoto'] for item in report['cold_drug']['similarity']['nearest_pairs']]
    protein_similarity = [item['normalised_indel_similarity'] for item in report['cold_target']['sequence_identity']['nearest_pairs']]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    summaries = [report['cold_drug']['similarity']['test_to_train_max'], report['cold_target']['sequence_identity']['test_to_train_max']]
    labels = ['Cold-drug\nTanimoto', 'Cold-target\nsequence similarity']
    means = [summary['mean'] for summary in summaries]
    medians = [summary['median'] for summary in summaries]
    bars = ax.bar(labels, means, color=['#4c78a8', '#59a14f'], width=0.58)
    for bar, summary, median_value in zip(bars, summaries, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"mean {summary['mean']:.3f}\nmedian {median_value:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Maximum test-to-train similarity')
    ax.set_title('Nearest-neighbour similarity diagnostic')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-0.15, 1.04, 'A', transform=ax.transAxes, fontweight='bold', fontsize=14)

    ax = axes[1]
    split = report['cold_both']['split_sizes']
    values = [split['train_pairs'], split['validation_pairs'], split['test_pairs']]
    bars = ax.bar(['Train', 'Validation', 'Test'], values, color=['#4c78a8', '#f28e2b', '#e15759'])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02, str(value), ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Drug--target pairs')
    ax.set_title('Cold-both evaluation partition')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(-0.15, 1.04, 'B', transform=ax.transAxes, fontweight='bold', fontsize=14)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate deterministic Davis split diagnostics for the revision.')
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'BioInteract' / 'data' / 'raw' / 'davis')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'analysis')
    args = parser.parse_args()
    report = build_split_diagnostics(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'split_diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    render_diagnostics(report, Path(__file__).resolve().parent / 'figures' / 'fig_split_diagnostics.pdf')


if __name__ == '__main__':
    main()
