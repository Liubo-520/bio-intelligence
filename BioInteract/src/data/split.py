"""Identifier-held-out data splitting strategies for DTI evaluation.

Historical function names are retained for compatibility. Public labels are
Drug-ID-held-out and Target-ID-held-out; identifier-held-out evaluation is not
an exact-sequence-grouped protocol.
"""
from typing import Tuple

import numpy as np
import pandas as pd


def random_split(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the archived random baseline partition."""
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    n_test = int(len(df) * test_ratio)
    n_val = int(len(df) * val_ratio)
    return df.iloc[indices[n_test + n_val:]], df.iloc[indices[n_test:n_test + n_val]], df.iloc[indices[:n_test]]


def cold_drug_split(
    df: pd.DataFrame,
    drug_col: str = "drug_id",
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a Drug-ID-held-out partition with unseen drug identifiers."""
    np.random.seed(seed)
    drugs = df[drug_col].unique()
    np.random.shuffle(drugs)
    n_test = int(len(drugs) * test_ratio)
    n_val = int(len(drugs) * val_ratio)
    test_drugs = set(drugs[:n_test])
    val_drugs = set(drugs[n_test:n_test + n_val])
    train_drugs = set(drugs[n_test + n_val:])
    return (
        df[df[drug_col].isin(train_drugs)],
        df[df[drug_col].isin(val_drugs)],
        df[df[drug_col].isin(test_drugs)],
    )


def cold_target_split(
    df: pd.DataFrame,
    target_col: str = "target_id",
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return a Target-ID-held-out partition with unseen target identifiers.

    Duplicate sequences can span identifiers, so this is not a strict
    exact-sequence-grouped evaluation.
    """
    np.random.seed(seed)
    targets = df[target_col].unique()
    np.random.shuffle(targets)
    n_test = int(len(targets) * test_ratio)
    n_val = int(len(targets) * val_ratio)
    test_targets = set(targets[:n_test])
    val_targets = set(targets[n_test:n_test + n_val])
    train_targets = set(targets[n_test + n_val:])
    return (
        df[df[target_col].isin(train_targets)],
        df[df[target_col].isin(val_targets)],
        df[df[target_col].isin(test_targets)],
    )


def cold_both_split(
    df: pd.DataFrame,
    drug_col: str = "drug_id",
    target_col: str = "target_id",
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return an identifier-held-out partition for both drug and target IDs."""
    np.random.seed(seed)
    drugs = df[drug_col].unique()
    targets = df[target_col].unique()
    np.random.shuffle(drugs)
    np.random.shuffle(targets)
    n_test_drugs = int(len(drugs) * test_ratio)
    n_val_drugs = int(len(drugs) * val_ratio)
    n_test_targets = int(len(targets) * test_ratio)
    n_val_targets = int(len(targets) * val_ratio)
    test_drugs = set(drugs[:n_test_drugs])
    val_drugs = set(drugs[n_test_drugs:n_test_drugs + n_val_drugs])
    test_targets = set(targets[:n_test_targets])
    val_targets = set(targets[n_test_targets:n_test_targets + n_val_targets])
    test_df = df[df[drug_col].isin(test_drugs) & df[target_col].isin(test_targets)]
    val_df = df[df[drug_col].isin(val_drugs) & df[target_col].isin(val_targets)]
    train_mask = ~(df[drug_col].isin(test_drugs | val_drugs)) & ~(df[target_col].isin(test_targets | val_targets))
    return df[train_mask], val_df, test_df


def get_split_fn(split_type: str):
    """Return a historical split function from its internal configuration key."""
    split_fns = {
        "random": random_split,
        "cold_drug": cold_drug_split,
        "cold_target": cold_target_split,
        "cold_both": cold_both_split,
    }
    if split_type not in split_fns:
        raise ValueError(f"Unknown split type: {split_type}. Choose from {list(split_fns.keys())}")
    return split_fns[split_type]
