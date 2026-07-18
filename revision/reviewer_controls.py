"""Leakage-aware Davis diagnostics requested by Reviewers 1 and 2.

The controls intentionally use only information available in the training
partition.  They are not replacements for external validation or structural
benchmarking; they quantify how much signal remains in a target-independent
drug representation under the same exact-sequence-grouped cold-target split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Align
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from revision_analysis import sequence_group_cold_target_split


plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


_ALIGNER = Align.PairwiseAligner()
_ALIGNER.mode = 'global'
_ALIGNER.match_score = 1
_ALIGNER.mismatch_score = 0
_ALIGNER.open_gap_score = -1
_ALIGNER.extend_gap_score = -0.5


def global_sequence_identity(first: str, second: str) -> float:
    """Return exact-residue identity divided by global alignment length."""
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    alignment = _ALIGNER.align(first, second)[0]
    first_indices, second_indices = alignment.indices
    aligned_positions = len(first_indices)
    matches = sum(
        first[first_index] == second[second_index]
        for first_index, second_index in zip(first_indices, second_indices)
        if first_index >= 0 and second_index >= 0
    )
    return matches / aligned_positions if aligned_positions else 0.0


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        'n': len(values),
        'mean': round(float(mean(values)), 6),
        'median': round(float(median(values)), 6),
        'minimum': round(float(min(values)), 6),
        'maximum': round(float(max(values)), 6),
    }


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        'AUROC': round(float(roc_auc_score(labels, scores)), 6),
        'AUPRC': round(float(average_precision_score(labels, scores)), 6),
        'positive_prevalence': round(float(np.mean(labels)), 6),
    }


def _morgan_features(drugs: pd.DataFrame) -> dict[str, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    features: dict[str, np.ndarray] = {}
    for row in drugs.itertuples(index=False):
        molecule = Chem.MolFromSmiles(row.smiles)
        if molecule is None:
            raise ValueError(f'Invalid SMILES for {row.drug_id}')
        vector = np.zeros(1024, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), vector)
        features[row.drug_id] = vector
    return features


def _logistic_scores(train: pd.DataFrame, test: pd.DataFrame, features: dict[str, np.ndarray], seed: int, shuffled_labels: bool = False) -> np.ndarray:
    train_labels = train['label'].to_numpy(dtype=int, copy=True)
    if shuffled_labels:
        np.random.RandomState(seed).shuffle(train_labels)
    classifier = LogisticRegression(
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000,
        random_state=seed,
    )
    train_matrix = np.vstack([features[drug_id] for drug_id in train['drug_id']])
    test_matrix = np.vstack([features[drug_id] for drug_id in test['drug_id']])
    classifier.fit(train_matrix, train_labels)
    return classifier.predict_proba(test_matrix)[:, 1]


def _max_train_identity(test_targets: set[str], train_targets: set[str], sequences: dict[str, str]) -> tuple[list[float], list[dict[str, object]]]:
    train_order = sorted(train_targets)
    values: list[float] = []
    per_target: list[dict[str, object]] = []
    for test_target in sorted(test_targets):
        scores = [global_sequence_identity(sequences[test_target], sequences[train_target]) for train_target in train_order]
        best_index = int(np.argmax(scores))
        best = float(scores[best_index])
        values.append(best)
        per_target.append({
            'test_target_id': test_target,
            'nearest_train_target_id': train_order[best_index],
            'maximum_global_sequence_identity': round(best, 6),
        })
    return values, per_target


def _identity_bins(per_target: list[dict[str, object]]) -> dict[str, int]:
    edges = [0.0, 0.4, 0.6, 0.8, 1.000001]
    labels = ['<0.40', '0.40--<0.60', '0.60--<0.80', '>=0.80']
    values = np.array([item['maximum_global_sequence_identity'] for item in per_target], dtype=float)
    return {
        label: int(np.sum((values >= lower) & (values < upper)))
        for label, lower, upper in zip(labels, edges[:-1], edges[1:])
    }


def identity_stratified_metrics(labels: np.ndarray, scores: np.ndarray, target_ids: list[str], identity_by_target: dict[str, float]) -> dict[str, dict[str, object]]:
    """Stratify pair-level scores by each target's maximum train identity."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    target_ids = np.asarray(target_ids, dtype=object)
    edges = [0.0, 0.4, 0.6, 0.8, 1.000001]
    bin_labels = ['<0.40', '0.40--<0.60', '0.60--<0.80', '>=0.80']
    target_identity = np.array([identity_by_target[target_id] for target_id in target_ids], dtype=float)
    report: dict[str, dict[str, object]] = {}
    for label, lower, upper in zip(bin_labels, edges[:-1], edges[1:]):
        mask = (target_identity >= lower) & (target_identity < upper)
        bin_labels_array = labels[mask]
        bin_scores = scores[mask]
        entry: dict[str, object] = {
            'test_pairs': int(np.sum(mask)),
            'positive_pairs': int(np.sum(bin_labels_array)),
        }
        if len(np.unique(bin_labels_array)) == 2:
            entry['metrics'] = _metrics(bin_labels_array, bin_scores)
        else:
            entry['metrics'] = None
        report[label] = entry
    return report


def strict_checkpoint_identity_stratification(data_root: Path, checkpoint_path: Path, identity_by_target: dict[str, float]) -> dict[str, dict[str, object]]:
    """Apply the saved strict cold-target model to identity-stratified test pairs."""
    import sys

    import torch
    import yaml
    from torch.utils.data import DataLoader

    workspace = Path(__file__).resolve().parents[1]
    project = workspace / 'BioInteract'
    for path in (workspace, project):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from src.data.dataset import DTIDataset, collate_dti
    from src.models.biointeract import BioInteract

    with (project / 'configs' / 'default.yaml').open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    interactions = pd.read_csv(data_root / 'interactions.csv')
    drugs = pd.read_csv(data_root / 'drug_smiles.csv')
    targets = pd.read_csv(data_root / 'target_sequences.csv')
    _, _, test = sequence_group_cold_target_split(interactions, targets, seed=42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    common = {
        'drug_smiles': dict(zip(drugs['drug_id'], drugs['smiles'])),
        'target_sequences': dict(zip(targets['target_id'], targets['sequence'])),
        'esm2_cache_dir': str(project / config['data']['esm2_cache_dir']),
        'max_protein_len': int(config['data']['max_protein_len']),
        'use_domain_features': bool(config['model']['target_encoder']['use_domain_features']),
        'esm2_dim': int(config['model']['target_encoder']['esm2_dim']),
        'task': 'classification',
    }
    loader = DataLoader(
        DTIDataset(test, **common),
        batch_size=int(config['training']['batch_size']),
        shuffle=False,
        collate_fn=collate_dti,
        num_workers=0,
    )
    model = BioInteract(checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    labels, scores, target_ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch['drug_batch'].to(device),
                batch['esm2_embedding'].to(device),
                batch['physicochemical'].to(device),
                batch['domain_labels'].to(device),
                batch['protein_mask'].to(device),
            )
            labels.extend(batch['label'].numpy().ravel().astype(int).tolist())
            scores.extend(torch.sigmoid(logits).cpu().numpy().ravel().tolist())
            target_ids.extend(batch['target_ids'])
    return identity_stratified_metrics(
        labels=np.asarray(labels),
        scores=np.asarray(scores),
        target_ids=target_ids,
        identity_by_target=identity_by_target,
    )


def build_reviewer_controls(data_root: Path, seed: int = 42, checkpoint_path: Path | None = None) -> dict[str, object]:
    """Build all controls from the Davis records without using held-out labels."""
    interactions = pd.read_csv(data_root / 'interactions.csv')
    drugs = pd.read_csv(data_root / 'drug_smiles.csv')
    targets = pd.read_csv(data_root / 'target_sequences.csv')
    train, validation, test = sequence_group_cold_target_split(interactions, targets, seed=seed)
    del validation

    train_target_ids = set(train['target_id'].unique())
    test_target_ids = set(test['target_id'].unique())
    sequences = dict(zip(targets['target_id'], targets['sequence']))
    train_sequences = {sequences[target_id] for target_id in train_target_ids}
    test_sequences = {sequences[target_id] for target_id in test_target_ids}
    identity_values, per_target_identity = _max_train_identity(test_target_ids, train_target_ids, sequences)

    features = _morgan_features(drugs)
    labels = test['label'].to_numpy(dtype=int)
    drug_only_scores = _logistic_scores(train, test, features, seed=seed)
    shuffled_scores = _logistic_scores(train, test, features, seed=seed, shuffled_labels=True)
    training_prevalence = float(train['label'].mean())
    drug_prevalence = train.groupby('drug_id')['label'].mean().to_dict()
    promiscuity_scores = np.array([drug_prevalence.get(drug_id, training_prevalence) for drug_id in test['drug_id']], dtype=float)
    constant_scores = np.full(len(test), training_prevalence, dtype=float)

    report: dict[str, object] = {
        'dataset': 'Davis',
        'seed': seed,
        'split_definition': 'Exact-sequence-grouped cold-target split; target IDs with identical full amino-acid sequences remain in one partition.',
        'sequence_grouped_cold_target': {
            'split': {
                'train_pairs': int(len(train)),
                'test_pairs': int(len(test)),
                'train_targets': int(len(train_target_ids)),
                'test_targets': int(len(test_target_ids)),
                'train_test_target_overlap': int(len(train_target_ids.intersection(test_target_ids))),
                'exact_sequence_train_test_overlap': int(len(train_sequences.intersection(test_sequences))),
            },
            'full_length_identity': {
                'definition': 'Exact-residue matches divided by global pairwise-alignment length; maximum score over all training targets for each held-out target.',
                'test_to_train_max': _summary(identity_values),
                'per_target': per_target_identity,
                'bins': _identity_bins(per_target_identity),
            },
        },
        'controls': {
            'drug_only_morgan_logistic': {
                'definition': 'Morgan fingerprint (radius 2, 1024 bits) logistic regression; no target representation is supplied.',
                'test_pairs': int(len(test)),
                'metrics': _metrics(labels, drug_only_scores),
            },
            'drug_promiscuity_prior': {
                'definition': 'Per-drug positive prevalence estimated only from training targets; no target representation is supplied.',
                'test_pairs': int(len(test)),
                'metrics': _metrics(labels, promiscuity_scores),
            },
            'target_only_constant_prior': {
                'definition': 'For held-out targets, any target-specific training label degree would leak test-target labels. The leakage-free target-only control is therefore the training positive prevalence applied uniformly.',
                'test_pairs': int(len(test)),
                'metrics': _metrics(labels, constant_scores),
            },
            'label_shuffled_drug_only': {
                'definition': 'The drug-only logistic procedure after a deterministic permutation of training labels; test labels remain unaltered.',
                'test_pairs': int(len(test)),
                'metrics': _metrics(labels, shuffled_scores),
            },
        },
    }
    if checkpoint_path is not None:
        identity_by_target = {
            item['test_target_id']: item['maximum_global_sequence_identity']
            for item in per_target_identity
        }
        report['sequence_grouped_cold_target']['strict_model_identity_stratification'] = (
            strict_checkpoint_identity_stratification(data_root, checkpoint_path, identity_by_target)
        )
    return report


def render_reviewer_controls(report: dict[str, object], output_path: Path) -> None:
    """Render compact evidence suitable for the response letter."""
    controls = report['controls']
    names = ['Drug-only\nMorgan', 'Drug\npromiscuity', 'Target-only\nconstant', 'Shuffled\ndrug-only']
    keys = ['drug_only_morgan_logistic', 'drug_promiscuity_prior', 'target_only_constant_prior', 'label_shuffled_drug_only']
    auroc = [controls[key]['metrics']['AUROC'] for key in keys]
    auprc = [controls[key]['metrics']['AUPRC'] for key in keys]
    identity = report['sequence_grouped_cold_target']['full_length_identity']

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), gridspec_kw={'width_ratios': [1.35, 1]})
    positions = np.arange(len(keys))
    width = 0.36
    axes[0].bar(positions - width / 2, auroc, width, label='AUROC', color='#2f7f78')
    axes[0].bar(positions + width / 2, auprc, width, label='AUPRC', color='#dd9c2b')
    axes[0].set_xticks(positions, names)
    axes[0].set_ylabel('Test metric')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Leakage-aware diagnostic controls')
    axes[0].legend(frameon=False, ncols=2, loc='upper center')
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].text(-0.12, 1.03, 'A', transform=axes[0].transAxes, fontweight='bold', fontsize=14)

    bins = identity['bins']
    labels = list(bins)
    counts = list(bins.values())
    bars = axes[1].bar(labels, counts, color='#6a82b5')
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, count + 0.8, str(count), ha='center', va='bottom', fontsize=9)
    axes[1].set_ylabel('Held-out targets')
    axes[1].set_title('Maximum full-length identity to train')
    axes[1].tick_params(axis='x', rotation=18)
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[1].text(-0.16, 1.03, 'B', transform=axes[1].transAxes, fontweight='bold', fontsize=14)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches='tight')
    figure.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Reviewer 1/2 leakage-aware Davis controls.')
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1] / 'BioInteract' / 'data' / 'raw' / 'davis')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'analysis')
    args = parser.parse_args()
    default_checkpoint = Path(__file__).resolve().parent / 'analysis' / 'sequence_grouped_cold_target.pt'
    report = build_reviewer_controls(
        args.data_root,
        checkpoint_path=default_checkpoint if default_checkpoint.exists() else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'reviewer_controls.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    render_reviewer_controls(report, Path(__file__).resolve().parent / 'figures' / 'fig_reviewer_controls.pdf')


if __name__ == '__main__':
    main()
