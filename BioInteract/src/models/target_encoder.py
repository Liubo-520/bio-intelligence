"""Protein target encoder combining ESM-2 with residue descriptors.

It accepts frozen ESM-2 embeddings, four physicochemical descriptors, and an
optional categorical label channel. The public Davis release supplies the
``NONE`` label when curated annotations are absent; that channel must not be
interpreted as Pfam/InterPro annotation or evidence of protein function.
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
    
    The optional label embedding is a model input channel. In the public Davis
    workflow it is the shared ``NONE`` label because no curated annotations are
    distributed. It is not a domain, binding-site, or functional assignment.
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
