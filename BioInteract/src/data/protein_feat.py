"""
protein_feat.py — Protein feature processing with functional domain annotations.

Integrates ESM-2 embeddings with InterPro/Pfam domain information to provide
biologically-informed residue-level representations.
"""
import os
import json
import torch
import numpy as np
from typing import Optional

from src.utils.paths import resolve_project_path


# ============================================================
# Standard amino acid vocabulary
# ============================================================

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_AA_IDX = len(AMINO_ACIDS)


# ============================================================
# Residue-level physicochemical properties (domain knowledge)
# ============================================================

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Molecular weight
MOL_WEIGHT = {
    'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121,
    'E': 147, 'Q': 146, 'G': 75, 'H': 155, 'I': 131,
    'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
    'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117,
}

# Charge at pH 7
CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'E': -1, 'Q': 0, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}

# Polarity
POLARITY = {
    'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0,
    'E': 1, 'Q': 1, 'G': 0, 'H': 1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0,
}


def residue_physicochemical_features(sequence: str) -> torch.Tensor:
    """
    Compute per-residue physicochemical feature vectors.
    
    Returns tensor of shape (L, 4) with columns:
      [hydrophobicity, mol_weight_normalised, charge, polarity]
    
    These features encode biological priors about amino acid properties
    relevant to binding: charged residues form salt bridges, hydrophobic
    residues cluster in binding pockets, etc.
    """
    features = []
    for aa in sequence:
        h = HYDROPHOBICITY.get(aa, 0.0) / 4.5     # normalise to [-1, 1]
        w = MOL_WEIGHT.get(aa, 130.0) / 204.0      # normalise to [0, 1]
        c = CHARGE.get(aa, 0.0)
        p = POLARITY.get(aa, 0.0)
        features.append([h, w, c, p])
    return torch.tensor(features, dtype=torch.float)


# ============================================================
# Domain annotation helpers
# ============================================================

# Commonly encountered Pfam domain types in drug targets
DOMAIN_TYPES = [
    'Pkinase',           # Protein kinase domain
    'Pkinase_Tyr',       # Tyrosine kinase domain
    '7tm_1',             # GPCR rhodopsin family
    '7tm_2',             # GPCR secretin family
    '7tm_3',             # GPCR metabotropic glutamate
    'Trypsin',           # Serine protease (trypsin-like)
    'Peptidase_C1',      # Cysteine protease
    'Metallopep',        # Metalloprotease
    'zf-CCHH',           # Zinc finger
    'SH2',               # SH2 domain
    'SH3_1',             # SH3 domain
    'PH',                # Pleckstrin homology
    'RRM_1',             # RNA recognition motif
    'Ion_trans',          # Ion transport
    'Ank',               # Ankyrin repeat
    'WD40',              # WD40 repeat
    'LRR_1',             # Leucine-rich repeat
    'EGF',               # EGF-like domain
    'Ig',                # Immunoglobulin domain
    'ABC_tran',          # ABC transporter
    'Hormone_recep',     # Nuclear hormone receptor
    'HATPase_c',         # ATPase
    'Helicase_C',        # Helicase
    'RVT_1',             # Reverse transcriptase
    'Topoisom_bac',      # Topoisomerase
    'NONE',              # No annotated domain (linker/disordered)
]

DOMAIN_TO_IDX = {d: i for i, d in enumerate(DOMAIN_TYPES)}


def load_domain_annotations(protein_id: str,
                            annotation_dir: str = 'data/domain_annotations'
                            ) -> Optional[dict]:
    """
    Load precomputed domain annotations for a protein.
    
    Expected format: JSON with structure
    {
        "protein_id": "P00533",
        "length": 1210,
        "domains": [
            {"start": 712, "end": 979, "type": "Pkinase_Tyr", "pfam": "PF07714"},
            ...
        ]
    }
    """
    path = resolve_project_path(annotation_dir) / f'{protein_id}.json'
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def residue_domain_labels(sequence_length: int,
                          domain_info: Optional[dict] = None) -> torch.LongTensor:
    """
    Assign each residue a domain type index.
    
    Returns a LongTensor of shape (L,) where each value indexes into DOMAIN_TYPES.
    Residues not covered by any domain annotation get the 'NONE' label.
    """
    labels = torch.full((sequence_length,),
                        DOMAIN_TO_IDX['NONE'],
                        dtype=torch.long)
    
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


# ============================================================
# ESM-2 embedding loader
# ============================================================

def load_esm2_embedding(protein_id: str,
                        cache_dir: str = 'data/esm2_embeddings'
                        ) -> Optional[torch.Tensor]:
    """
    Load precomputed ESM-2 residue-level embeddings.
    
    Returns tensor of shape (L, 1280) or None if not found.
    """
    path = resolve_project_path(cache_dir) / f'{protein_id}.pt'
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu', weights_only=True)


# ============================================================
# Combined protein feature builder
# ============================================================

class ProteinFeatureBuilder:
    """
    Builds the complete residue-level feature representation for a protein,
    combining:
      1. ESM-2 embeddings (1280-d, pretrained language model knowledge)
      2. Physicochemical properties (4-d, domain knowledge)
      3. Functional domain labels (for learnable domain embeddings)
    """
    
    def __init__(self,
                 esm2_cache_dir: str = 'data/esm2_embeddings',
                 domain_annotation_dir: str = 'data/domain_annotations',
                 max_protein_len: int = 1200,
                 use_domain_features: bool = True,
                 esm2_dim: int = 640):
        self.esm2_cache_dir = esm2_cache_dir
        self.domain_annotation_dir = domain_annotation_dir
        self.max_len = max_protein_len
        self.use_domain = use_domain_features
        self.esm2_dim = esm2_dim
    
    def build(self, protein_id: str, sequence: str) -> dict:
        """
        Build feature dict for a protein.
        
        Returns:
            {
                'esm2_embedding': Tensor (L, 1280),
                'physicochemical': Tensor (L, 4),
                'domain_labels': LongTensor (L,),
                'sequence_length': int,
                'protein_id': str,
            }
        """
        # truncate if needed
        seq = sequence[:self.max_len]
        L = len(seq)
        
        # 1. ESM-2 embeddings
        esm2_emb = load_esm2_embedding(protein_id, self.esm2_cache_dir)
        if esm2_emb is not None:
            esm2_emb = esm2_emb[:L]  # align with truncation
        else:
            # fallback: zero embedding (will be replaced after ESM-2 extraction)
            esm2_emb = torch.zeros(L, self.esm2_dim)
        
        # 2. physicochemical features
        physchem = residue_physicochemical_features(seq)
        
        # 3. domain annotations
        domain_info = None
        if self.use_domain:
            domain_info = load_domain_annotations(
                protein_id, self.domain_annotation_dir
            )
        domain_labels = residue_domain_labels(L, domain_info)
        
        return {
            'esm2_embedding': esm2_emb,
            'physicochemical': physchem,
            'domain_labels': domain_labels,
            'sequence_length': L,
            'protein_id': protein_id,
        }
