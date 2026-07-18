from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_split_diagnostics_are_entity_disjoint():
    from revision_analysis import build_split_diagnostics

    data_root = Path(__file__).resolve().parents[2] / 'BioInteract' / 'data' / 'raw' / 'davis'
    report = build_split_diagnostics(data_root, seed=42)

    assert report['cold_drug']['entity_overlap'] == 0
    assert report['cold_target']['entity_overlap'] == 0
    assert report['cold_both']['drug_overlap'] == 0
    assert report['cold_both']['target_overlap'] == 0
    assert report['cold_drug']['similarity']['test_to_train_max']['n'] > 0
    assert report['cold_target']['sequence_identity']['test_to_train_max']['n'] > 0


def test_sequence_grouped_cold_target_prevents_exact_sequence_leakage():
    import pandas as pd
    from revision_analysis import sequence_group_cold_target_split

    data_root = Path(__file__).resolve().parents[2] / 'BioInteract' / 'data' / 'raw' / 'davis'
    interactions = pd.read_csv(data_root / 'interactions.csv')
    targets = pd.read_csv(data_root / 'target_sequences.csv')
    train_df, val_df, test_df = sequence_group_cold_target_split(interactions, targets, seed=42)
    sequences = dict(zip(targets['target_id'], targets['sequence']))
    train_sequences = {sequences[target] for target in train_df['target_id'].unique()}
    test_sequences = {sequences[target] for target in test_df['target_id'].unique()}

    assert not train_sequences.intersection(test_sequences)
    assert len(val_df) > 0
