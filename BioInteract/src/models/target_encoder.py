"""
target_encoder.py — Protein target encoder combining ESM-2 with domain knowledge.

Takes pre-extracted ESM-2 embeddings and enriches them with:
  1. Physicochemical amino acid properties (hydrophobicity, charge, etc.)
  2. Learnable functional domain embeddings (Pfam/InterPro)

This design lets us inject biological knowledge about protein structure
and function without any additional computational cost at training time.
"""
import torch
import torch.nn as nn


class TargetEncoder(nn.Module):
    """
    Protein residue-level encoder.
    
    Input:
        - ESM-2 embeddings: (B, L, 1280) — pretrained, frozen
        - Physicochemical features: (B, L, 4) — amino acid properties
        - Domain labels: (B, L) — functional domain type indices
    
    Output:
        - Residue representations: (B, L, projection_dim)
    
    The domain embedding adds awareness of which functional region each
    residue belongs to (e.g., kinase domain, SH2 domain). This helps the
    cross-attention module learn family-specific interaction patterns:
    a kinase inhibitor should attend differently to residues in the
    catalytic domain vs. a regulatory domain.
    """
    
    def __init__(self,
                 esm2_dim: int = 1280,
                 projection_dim: int = 256,
                 physchem_dim: int = 4,
                 domain_embed_dim: int = 32,
                 num_domain_types: int = 50,
                 use_domain_features: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.use_domain = use_domain_features
        self.projection_dim = projection_dim
        
        # total input dimension
        input_dim = esm2_dim + physchem_dim
        if use_domain_features:
            input_dim += domain_embed_dim
            self.domain_embedding = nn.Embedding(
                num_domain_types + 1,  # +1 for padding
                domain_embed_dim,
                padding_idx=num_domain_types,
            )
        
        # projection: compress concatenated features to target dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
        )
    
    def forward(self,
                esm2_embedding: torch.Tensor,
                physicochemical: torch.Tensor,
                domain_labels: torch.Tensor = None,
                protein_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            esm2_embedding: (B, L, 1280) pretrained embeddings
            physicochemical: (B, L, 4) amino acid properties
            domain_labels: (B, L) LongTensor of domain type indices
            protein_mask: (B, L) boolean mask (True = valid residue)
        
        Returns:
            residue_repr: (B, L, projection_dim) residue representations
        """
        parts = [esm2_embedding, physicochemical]
        
        if self.use_domain and domain_labels is not None:
            domain_emb = self.domain_embedding(domain_labels)
            parts.append(domain_emb)
        
        # concatenate all feature sources
        combined = torch.cat(parts, dim=-1)
        
        # project
        residue_repr = self.projection(combined)
        
        # zero out padded positions
        if protein_mask is not None:
            residue_repr = residue_repr * protein_mask.unsqueeze(-1).float()
        
        return residue_repr
