from pathlib import Path
import sys
from functools import lru_cache

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'revision'))

from reviewer_controls import (
    build_reviewer_controls,
    global_sequence_identity,
    identity_stratified_metrics,
    strict_checkpoint_identity_stratification,
)


DATA_ROOT = ROOT / 'BioInteract' / 'data' / 'raw' / 'davis'
CHECKPOINT = ROOT / 'revision' / 'analysis' / 'sequence_grouped_cold_target.pt'


@lru_cache(maxsize=1)
def control_report():
    return build_reviewer_controls(DATA_ROOT, seed=42)


def test_global_sequence_identity_reports_aligned_residue_fraction():
    assert global_sequence_identity('MKT', 'MKT') == 1.0
    assert global_sequence_identity('MKT', 'MRT') == 2 / 3
    assert global_sequence_identity('MKT', 'MK') == 2 / 3


def test_identity_stratification_keeps_target_level_similarity_with_each_pair():
    identity_by_target = {'T1': 0.25, 'T2': 0.75}
    report = identity_stratified_metrics(
        labels=np.array([0, 1, 0, 1]),
        scores=np.array([0.1, 0.9, 0.2, 0.8]),
        target_ids=['T1', 'T1', 'T2', 'T2'],
        identity_by_target=identity_by_target,
    )

    assert report['<0.40']['test_pairs'] == 2
    assert report['0.60--<0.80']['test_pairs'] == 2
    assert report['<0.40']['metrics']['AUROC'] == 1.0
    assert report['0.60--<0.80']['metrics']['AUPRC'] == 1.0


def test_reviewer_controls_are_leakage_aware_and_emit_valid_metrics():
    report = control_report()

    split = report['sequence_grouped_cold_target']['split']
    assert split['train_test_target_overlap'] == 0
    assert split['exact_sequence_train_test_overlap'] == 0

    identity = report['sequence_grouped_cold_target']['full_length_identity']
    assert identity['test_to_train_max']['n'] == split['test_targets']
    assert 0.0 <= identity['test_to_train_max']['median'] <= 1.0

    controls = report['controls']
    expected_controls = {
        'drug_only_morgan_logistic',
        'drug_promiscuity_prior',
        'target_only_constant_prior',
        'label_shuffled_drug_only',
    }
    assert expected_controls.issubset(controls)
    for name in expected_controls:
        metrics = controls[name]['metrics']
        assert controls[name]['test_pairs'] == split['test_pairs']
        assert 0.0 <= metrics['AUROC'] <= 1.0
        assert 0.0 <= metrics['AUPRC'] <= 1.0
        assert np.isfinite(metrics['positive_prevalence'])


def test_strict_checkpoint_metrics_can_be_stratified_by_target_identity():
    report = control_report()
    per_target = report['sequence_grouped_cold_target']['full_length_identity']['per_target']
    identity_by_target = {
        item['test_target_id']: item['maximum_global_sequence_identity']
        for item in per_target
    }
    stratified = strict_checkpoint_identity_stratification(DATA_ROOT, CHECKPOINT, identity_by_target)

    assert sum(entry['test_pairs'] for entry in stratified.values()) == 5984
    assert any(entry['metrics'] is not None for entry in stratified.values())
