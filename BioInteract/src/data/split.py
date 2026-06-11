"""
split.py — Cold-start data splitting strategies for DTI prediction.

In real drug discovery, we need to predict interactions for NEW drugs or
NEW targets. Random splits leak information and drastically overestimate
performance. Cold-start splits simulate the actual use case.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, List
from sklearn.model_selection import KFold


def random_split(df: pd.DataFrame,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.2,
                 seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simple random split. Baseline only — not realistic for DTI."""
    np.random.seed(seed)
    n = len(df)
    indices = np.random.permutation(n)
    
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


def cold_drug_split(df: pd.DataFrame,
                    drug_col: str = 'drug_id',
                    val_ratio: float = 0.1,
                    test_ratio: float = 0.2,
                    seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cold-drug split: test set contains drugs never seen during training.
    
    This simulates the scenario: "Given a novel compound, which known 
    targets might it bind to?"
    """
    np.random.seed(seed)
    drugs = df[drug_col].unique()
    np.random.shuffle(drugs)
    
    n_drugs = len(drugs)
    n_test = int(n_drugs * test_ratio)
    n_val = int(n_drugs * val_ratio)
    
    test_drugs = set(drugs[:n_test])
    val_drugs = set(drugs[n_test:n_test + n_val])
    train_drugs = set(drugs[n_test + n_val:])
    
    train_df = df[df[drug_col].isin(train_drugs)]
    val_df = df[df[drug_col].isin(val_drugs)]
    test_df = df[df[drug_col].isin(test_drugs)]
    
    return train_df, val_df, test_df


def cold_target_split(df: pd.DataFrame,
                      target_col: str = 'target_id',
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.2,
                      seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cold-target split: test set contains targets never seen during training.
    
    This simulates the scenario: "A new viral protein has been identified.
    Which existing drugs might bind to it?"
    
    This is the hardest and most biologically relevant setting.
    """
    np.random.seed(seed)
    targets = df[target_col].unique()
    np.random.shuffle(targets)
    
    n_targets = len(targets)
    n_test = int(n_targets * test_ratio)
    n_val = int(n_targets * val_ratio)
    
    test_targets = set(targets[:n_test])
    val_targets = set(targets[n_test:n_test + n_val])
    train_targets = set(targets[n_test + n_val:])
    
    train_df = df[df[target_col].isin(train_targets)]
    val_df = df[df[target_col].isin(val_targets)]
    test_df = df[df[target_col].isin(test_targets)]
    
    return train_df, val_df, test_df


def cold_both_split(df: pd.DataFrame,
                    drug_col: str = 'drug_id',
                    target_col: str = 'target_id',
                    val_ratio: float = 0.1,
                    test_ratio: float = 0.2,
                    seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cold-both split: test set contains BOTH unseen drugs AND unseen targets.
    
    Most stringent evaluation — neither the compound nor the protein has
    been observed during training. This is the ultimate test of
    generalisation, relevant for emerging diseases with novel targets.
    """
    np.random.seed(seed)
    
    drugs = df[drug_col].unique()
    targets = df[target_col].unique()
    np.random.shuffle(drugs)
    np.random.shuffle(targets)
    
    n_test_d = int(len(drugs) * test_ratio)
    n_val_d = int(len(drugs) * val_ratio)
    n_test_t = int(len(targets) * test_ratio)
    n_val_t = int(len(targets) * val_ratio)
    
    test_drugs = set(drugs[:n_test_d])
    val_drugs = set(drugs[n_test_d:n_test_d + n_val_d])
    test_targets = set(targets[:n_test_t])
    val_targets = set(targets[n_test_t:n_test_t + n_val_t])
    
    # test: both drug and target are new
    test_df = df[df[drug_col].isin(test_drugs) & df[target_col].isin(test_targets)]
    # val: both new
    val_df = df[df[drug_col].isin(val_drugs) & df[target_col].isin(val_targets)]
    # train: everything else
    train_mask = ~(df[drug_col].isin(test_drugs | val_drugs)) & \
                 ~(df[target_col].isin(test_targets | val_targets))
    train_df = df[train_mask]
    
    return train_df, val_df, test_df


def get_split_fn(split_type: str):
    """Get the splitting function by name."""
    split_fns = {
        'random': random_split,
        'cold_drug': cold_drug_split,
        'cold_target': cold_target_split,
        'cold_both': cold_both_split,
    }
    if split_type not in split_fns:
        raise ValueError(f"Unknown split type: {split_type}. "
                         f"Choose from {list(split_fns.keys())}")
    return split_fns[split_type]
