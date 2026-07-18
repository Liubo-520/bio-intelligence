"""Protein feature processing for the released BioInteract workflows.

The domain-label channel is a configured model input. Released workflows use
only the ``NONE`` label because no curated annotation resource is distributed.
"""
from typing import Optional

import torch

from src.utils.paths import resolve_project_path


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
UNK_AA_IDX = len(AMINO_ACIDS)

HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
MOL_WEIGHT = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121,
    "E": 147, "Q": 146, "G": 75, "H": 155, "I": 131,
    "L": 131, "K": 146, "M": 149, "F": 165, "P": 115,
    "S": 105, "T": 105, "W": 204, "Y": 181, "V": 117,
}
CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "E": -1, "Q": 0, "G": 0, "H": 0.1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}
POLARITY = {
    "A": 0, "R": 1, "N": 1, "D": 1, "C": 0,
    "E": 1, "Q": 1, "G": 0, "H": 1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 1, "T": 1, "W": 0, "Y": 1, "V": 0,
}


def residue_physicochemical_features(sequence: str) -> torch.Tensor:
    """Return four numeric residue descriptors for a sequence."""
    features = []
    for amino_acid in sequence:
        features.append([
            HYDROPHOBICITY.get(amino_acid, 0.0) / 4.5,
            MOL_WEIGHT.get(amino_acid, 130.0) / 204.0,
            CHARGE.get(amino_acid, 0.0),
            POLARITY.get(amino_acid, 0.0),
        ])
    return torch.tensor(features, dtype=torch.float)


# The checkpoint contains a learnable embedding table. Index zero is the only
# released input value; additional table capacity is preserved for compatibility.
DOMAIN_TYPES = ["NONE"]
DOMAIN_TO_IDX = {"NONE": 0}


def load_domain_annotations(
    protein_id: str,
    annotation_dir: str = "data/domain_annotations",
) -> None:
    """Return no annotation: released code ships no domain-label resource."""
    del protein_id, annotation_dir
    return None


def residue_domain_labels(
    sequence_length: int,
    domain_info: Optional[dict] = None,
) -> torch.LongTensor:
    """Return the configured ``NONE`` domain-label channel for every residue."""
    del domain_info
    return torch.full((sequence_length,), DOMAIN_TO_IDX["NONE"], dtype=torch.long)


def load_esm2_embedding(
    protein_id: str,
    cache_dir: str = "data/esm2_embeddings",
) -> Optional[torch.Tensor]:
    """Load a cached ESM-2 residue embedding when it is available locally."""
    path = resolve_project_path(cache_dir) / f"{protein_id}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


class ProteinFeatureBuilder:
    """Build released residue features with the configured ``NONE`` label channel."""

    def __init__(
        self,
        esm2_cache_dir: str = "data/esm2_embeddings",
        domain_annotation_dir: str = "data/domain_annotations",
        max_protein_len: int = 1200,
        use_domain_features: bool = True,
        esm2_dim: int = 640,
    ):
        self.esm2_cache_dir = esm2_cache_dir
        self.domain_annotation_dir = domain_annotation_dir
        self.max_len = max_protein_len
        self.use_domain = use_domain_features
        self.esm2_dim = esm2_dim

    def build(self, protein_id: str, sequence: str) -> dict:
        """Build cached ESM-2, numeric residue, and ``NONE`` label features."""
        sequence = sequence[:self.max_len]
        length = len(sequence)
        esm2_embedding = load_esm2_embedding(protein_id, self.esm2_cache_dir)
        if esm2_embedding is not None:
            esm2_embedding = esm2_embedding[:length]
        else:
            esm2_embedding = torch.zeros(length, self.esm2_dim)
        return {
            "esm2_embedding": esm2_embedding,
            "physicochemical": residue_physicochemical_features(sequence),
            "domain_labels": residue_domain_labels(length),
            "sequence_length": length,
            "protein_id": protein_id,
        }
