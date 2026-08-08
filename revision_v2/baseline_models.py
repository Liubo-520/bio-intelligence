"""Baseline architectures re-implemented for the matched Davis protocol.

Every model here is trained by ``run_baseline_suite.py`` under one shared
optimiser, schedule, split, seed, and threshold rule, so that differences
between rows reflect the encoders and the interaction module rather than
differences in training recipe or reporting convention.

Two families are provided:

* Sequence/graph baselines without a protein language model (``DeepDTA``,
  ``GraphDTA``), reproduced from their published architecture descriptions.
* ESM-2-matched baselines (``ESM2MLP``, ``ESM2GraphConcat``, ``ESM2Bilinear``)
  that consume exactly the frozen ``esm2_t30_150M_UR50D`` representation used
  by BioInteract, so the effect of ESM-2 pretraining is held constant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool, global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "BioInteract"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.models.drug_encoder import DrugEncoder  # noqa: E402


SMILES_VOCAB = (
    "#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy"
)
PROTEIN_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SMILES_STOI = {token: index + 1 for index, token in enumerate(SMILES_VOCAB)}
PROTEIN_STOI = {token: index + 1 for index, token in enumerate(PROTEIN_VOCAB)}
MAX_SMILES_LEN = 100
MAX_SEQUENCE_LEN = 1200


def encode_smiles(smiles: str, max_len: int = MAX_SMILES_LEN) -> torch.Tensor:
    """Return the fixed-width integer encoding used by the CNN baselines."""
    encoded = torch.zeros(max_len, dtype=torch.long)
    for position, token in enumerate(smiles[:max_len]):
        encoded[position] = SMILES_STOI.get(token, 0)
    return encoded


def encode_sequence(sequence: str, max_len: int = MAX_SEQUENCE_LEN) -> torch.Tensor:
    """Return the fixed-width integer encoding used by the CNN baselines."""
    encoded = torch.zeros(max_len, dtype=torch.long)
    for position, token in enumerate(sequence[:max_len]):
        encoded[position] = PROTEIN_STOI.get(token, 0)
    return encoded


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average ``values`` over valid positions only."""
    weight = mask.unsqueeze(-1).to(values.dtype)
    return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)


class CharCNN(nn.Module):
    """One-dimensional character CNN with global max pooling (DeepDTA style)."""

    def __init__(self, vocab_size: int, embed_dim: int, kernels: tuple[int, int, int],
                 channels: tuple[int, int, int] = (32, 64, 96)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        first, second, third = channels
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, first, kernels[0]), nn.ReLU(),
            nn.Conv1d(first, second, kernels[1]), nn.ReLU(),
            nn.Conv1d(second, third, kernels[2]), nn.ReLU(),
        )
        self.output_dim = third

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        return self.conv(embedded).max(dim=2).values


def prediction_head(input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> nn.Sequential:
    """Return the shared MLP classification head used by every baseline."""
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)])
        current = hidden
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


class DeepDTABaseline(nn.Module):
    """DeepDTA: parallel character CNNs over SMILES and amino-acid strings."""

    uses_esm2 = False

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.drug_cnn = CharCNN(len(SMILES_VOCAB), 128, (4, 6, 8))
        self.protein_cnn = CharCNN(len(PROTEIN_VOCAB), 128, (4, 8, 12))
        self.head = prediction_head(
            self.drug_cnn.output_dim + self.protein_cnn.output_dim, (1024, 512), dropout
        )

    def forward(self, batch: dict) -> torch.Tensor:
        drug = self.drug_cnn(batch["smiles_tokens"])
        protein = self.protein_cnn(batch["sequence_tokens"])
        return self.head(torch.cat([drug, protein], dim=1)).squeeze(-1)


class GraphDTABaseline(nn.Module):
    """GraphDTA (GIN variant): molecular graph encoder with a protein CNN."""

    uses_esm2 = False

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.drug_encoder = DrugEncoder(hidden_dim=hidden_dim, dropout=0.2)
        self.protein_cnn = CharCNN(len(PROTEIN_VOCAB), 128, (4, 8, 12))
        self.head = prediction_head(
            2 * hidden_dim + self.protein_cnn.output_dim, (1024, 512), dropout
        )

    def forward(self, batch: dict) -> torch.Tensor:
        atoms, index = self.drug_encoder(batch["drug_batch"])
        pooled = torch.cat(
            [global_mean_pool(atoms, index), global_max_pool(atoms, index)], dim=1
        )
        protein = self.protein_cnn(batch["sequence_tokens"])
        return self.head(torch.cat([pooled, protein], dim=1)).squeeze(-1)


class ESM2MLPBaseline(nn.Module):
    """ESM-2-matched control: mean-pooled ESM-2 with a Morgan-fingerprint MLP."""

    uses_esm2 = True

    def __init__(self, esm2_dim: int = 640, morgan_bits: int = 1024,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.protein_proj = nn.Sequential(
            nn.Linear(esm2_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.drug_proj = nn.Sequential(
            nn.Linear(morgan_bits, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.head = prediction_head(2 * hidden_dim, (256, 128), dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        protein = self.protein_proj(masked_mean(batch["esm2_embedding"], batch["protein_mask"]))
        drug = self.drug_proj(batch["morgan_fp"])
        return self.head(torch.cat([drug, protein], dim=1)).squeeze(-1)


class ESM2GraphConcatBaseline(nn.Module):
    """ESM-2-matched control: same encoders as BioInteract, concatenation fusion.

    This variant shares BioInteract's frozen ESM-2 representation and its GINE
    molecular graph encoder but replaces bidirectional cross-attention with a
    late concatenation of independently pooled representations.
    """

    uses_esm2 = True

    def __init__(self, esm2_dim: int = 640, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.drug_encoder = DrugEncoder(hidden_dim=hidden_dim, dropout=0.2)
        self.protein_proj = nn.Sequential(
            nn.Linear(esm2_dim + 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.head = prediction_head(2 * hidden_dim, (256, 128), dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        atoms, index = self.drug_encoder(batch["drug_batch"])
        drug = global_mean_pool(atoms, index)
        residues = torch.cat([batch["esm2_embedding"], batch["physicochemical"]], dim=-1)
        protein = masked_mean(self.protein_proj(residues), batch["protein_mask"])
        return self.head(torch.cat([drug, protein], dim=1)).squeeze(-1)


class ESM2BilinearBaseline(nn.Module):
    """ESM-2-matched bilinear interaction baseline in the DrugBAN family.

    Atom and residue representations are combined by a low-rank bilinear
    attention map followed by bilinear pooling, which is the interaction
    mechanism DrugBAN introduced. Only the interaction module differs from
    ``ESM2GraphConcatBaseline``.
    """

    uses_esm2 = True

    def __init__(self, esm2_dim: int = 640, hidden_dim: int = 256,
                 embedding_dim: int = 256, heads: int = 2, dropout: float = 0.3):
        super().__init__()
        self.drug_encoder = DrugEncoder(hidden_dim=hidden_dim, dropout=0.2)
        self.protein_proj = nn.Sequential(
            nn.Linear(esm2_dim + 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.drug_bilinear = nn.Linear(hidden_dim, embedding_dim * heads)
        self.protein_bilinear = nn.Linear(hidden_dim, embedding_dim * heads)
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = prediction_head(embedding_dim, (256, 128), dropout)

    def forward(self, batch: dict) -> torch.Tensor:
        atoms, index = self.drug_encoder(batch["drug_batch"])
        drug, drug_mask = to_dense(atoms, index)
        residues = torch.cat([batch["esm2_embedding"], batch["physicochemical"]], dim=-1)
        protein = self.protein_proj(residues)
        protein_mask = batch["protein_mask"]

        batch_size = drug.size(0)
        drug_map = self.drug_bilinear(drug).view(batch_size, drug.size(1), self.heads, -1)
        protein_map = self.protein_bilinear(protein).view(
            batch_size, protein.size(1), self.heads, -1
        )
        # Low-rank bilinear attention over the atom-residue grid.
        attention = torch.einsum("bnhd,bmhd->bhnm", drug_map, protein_map)
        pair_mask = drug_mask.unsqueeze(1).unsqueeze(-1) & protein_mask.unsqueeze(1).unsqueeze(2)
        attention = attention.masked_fill(~pair_mask, float("-inf"))
        attention = torch.softmax(attention.flatten(start_dim=2), dim=-1).view_as(attention)
        attention = torch.nan_to_num(attention)

        # Bilinear pooling: sum the attended atom-residue outer products, then
        # average the bilinear heads into a single joint representation.
        joint = torch.einsum("bhnm,bnhd,bmhd->bhd", attention, drug_map, protein_map)
        return self.head(self.norm(joint.mean(dim=1))).squeeze(-1)


def to_dense(atoms: torch.Tensor, batch_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert packed atom features into a padded ``(B, N, D)`` tensor and mask."""
    from torch_geometric.utils import to_dense_batch

    dense, mask = to_dense_batch(atoms, batch_index)
    return dense, mask


class BioInteractWrapper(nn.Module):
    """Adapter so BioInteract shares the baseline suite's batch interface."""

    uses_esm2 = True

    def __init__(self, model_config: dict):
        super().__init__()
        from src.models.biointeract import BioInteract

        self.model = BioInteract(model_config)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(
            batch["drug_batch"],
            batch["esm2_embedding"],
            batch["physicochemical"],
            batch["domain_labels"],
            batch["protein_mask"],
        ).squeeze(-1)


MODEL_REGISTRY = {
    "DeepDTA": DeepDTABaseline,
    "GraphDTA": GraphDTABaseline,
    "ESM2-MLP": ESM2MLPBaseline,
    "ESM2-GraphConcat": ESM2GraphConcatBaseline,
    "ESM2-Bilinear": ESM2BilinearBaseline,
}
