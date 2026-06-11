"""
protein_feat.py — Protein feature processing for BioInteract Space demo.

Adapted from the training version: removed dependency on internal
src.utils.paths module so this file runs as a standalone module.
ESM-2 embeddings are computed on-the-fly via transformers in app.py.
"""
import os
import json
import torch
import numpy as np
from typing import Optional


# ============================================================
# Standard amino acid vocabulary
# ============================================================

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_AA_IDX = len(AMINO_ACIDS)


# ============================================================
# Residue-level physicochemical properties (domain knowledge)
# ============================================================

HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

MOL_WEIGHT = {
    'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121,
    'E': 147, 'Q': 146, 'G': 75, 'H': 155, 'I': 131,
    'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
    'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117,
}

CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'E': -1, 'Q': 0, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

POLARITY = {
    'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0,
    'E': 1, 'Q': 1, 'G': 0, 'H': 1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0,
}


def residue_physicochemical_features(sequence: str) -> torch.Tensor:
    """
    Compute per-residue physicochemical feature vectors.
    Returns tensor of shape (L, 4).
    """
    features = []
    for aa in sequence:
        h = HYDROPHOBICITY.get(aa, 0.0) / 4.5
        w = MOL_WEIGHT.get(aa, 130.0) / 204.0
        c = CHARGE.get(aa, 0.0)
        p = POLARITY.get(aa, 0.0)
        features.append([h, w, c, p])
    return torch.tensor(features, dtype=torch.float)


# ============================================================
# Domain annotation helpers
# ============================================================

DOMAIN_TYPES = [
    'Pkinase', 'Pkinase_Tyr', '7tm_1', '7tm_2', '7tm_3',
    'Trypsin', 'Peptidase_C1', 'Metallopep', 'zf-CCHH', 'SH2',
    'SH3_1', 'PH', 'RRM_1', 'Ion_trans', 'Ank', 'WD40',
    'LRR_1', 'EGF', 'Ig', 'ABC_tran', 'Hormone_recep',
    'HATPase_c', 'Helicase_C', 'RVT_1', 'Topoisom_bac', 'NONE',
]

DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAIN_TYPES)}


def residue_domain_labels(
    sequence_length: int,
    domain_info: Optional[dict] = None
) -> torch.LongTensor:
    """
    Assign each residue a domain type index.
    Returns LongTensor of shape (L,).
    Residues without annotation get the 'NONE' label.
    """
    labels = torch.full(
        (sequence_length,), DOMAIN_TO_IDX['NONE'], dtype=torch.long
    )
    if domain_info is not None and 'domains' in domain_info:
        for dom in domain_info['domains']:
            dom_type = dom.get('type', 'NONE')
            if dom_type not in DOMAIN_TO_IDX:
                dom_type = 'NONE'
            idx = DOMAIN_TO_IDX[dom_type]
            start = dom.get('start', 0)
            end = min(dom.get('end', sequence_length), sequence_length)
            labels[start:end] = idx
    return labels
