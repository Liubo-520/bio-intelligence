"""
interaction.py — Cross-Attention interaction module.

This is the core of BioInteract: computing fine-grained atom-residue
interaction maps that are both predictive and interpretable.

Design philosophy:
    Traditional DTI models concatenate drug and protein vectors and
    lose spatial information. Our cross-attention computes an explicit
    interaction matrix M ∈ R^(n_atoms × n_residues), where M[i,j]
    represents the "interaction strength" between drug atom i and
    protein residue j.
    
    This matrix directly corresponds to the biological concept of
    molecular contacts — and can be validated against crystal structures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionInteraction(nn.Module):
    """
    Multi-head cross-attention between drug atoms and protein residues.
    
    Returns:
        - Fused representation for prediction
        - Attention weights for interpretability (atom × residue interaction map)
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # drug atoms attend to protein residues
        self.W_q_drug = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_prot = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_prot = nn.Linear(hidden_dim, hidden_dim)
        
        # protein residues attend to drug atoms (bidirectional)
        self.W_q_prot = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_drug = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_drug = nn.Linear(hidden_dim, hidden_dim)
        
        # output projections
        self.out_proj_drug = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_prot = nn.Linear(hidden_dim, hidden_dim)
        
        # layer norms
        self.ln_drug = nn.LayerNorm(hidden_dim)
        self.ln_prot = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def _attention(self, Q, K, V, mask=None):
        """
        Standard scaled dot-product attention.
        
        Returns:
            output: attended values
            attn_weights: softmax attention weights (for interpretability)
        """
        # Q: (B, H, Lq, d), K: (B, H, Lk, d), V: (B, H, Lk, d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B,H,Lq,Lk)
        
        if mask is not None:
            # mask shape: (B, 1, 1, Lk) — broadcast over heads and queries
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)  # (B, H, Lq, d)
        return output, attn_weights
    
    def _reshape_to_heads(self, x, batch_size):
        """(B, L, D) → (B, H, L, d)"""
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(self,
                drug_repr: torch.Tensor,
                drug_mask: torch.Tensor,
                prot_repr: torch.Tensor,
                prot_mask: torch.Tensor):
        """
        Bidirectional cross-attention between drug atoms and protein residues.
        
        Args:
            drug_repr: (B, N_atoms, D) — per-atom drug representations
            drug_mask: (B, N_atoms) — boolean mask for drug atoms
            prot_repr: (B, L_residues, D) — per-residue protein representations
            prot_mask: (B, L_residues) — boolean mask for protein residues
        
        Returns:
            drug_updated: (B, N_atoms, D) — drug repr enriched by protein context
            prot_updated: (B, L_residues, D) — protein repr enriched by drug context
            interaction_map: (B, N_atoms, L_residues) — attention-based interaction matrix
                this is the key output for interpretability analysis
        """
        B = drug_repr.size(0)
        
        # --- Drug → Protein attention ---
        Q_d = self._reshape_to_heads(self.W_q_drug(drug_repr), B)
        K_p = self._reshape_to_heads(self.W_k_prot(prot_repr), B)
        V_p = self._reshape_to_heads(self.W_v_prot(prot_repr), B)
        
        # mask: (B, L_residues) → (B, 1, 1, L_residues)
        prot_attn_mask = prot_mask.unsqueeze(1).unsqueeze(2) if prot_mask is not None else None
        
        drug_attended, drug_to_prot_attn = self._attention(Q_d, K_p, V_p, prot_attn_mask)
        # drug_to_prot_attn: (B, H, N_atoms, L_residues)
        
        drug_attended = drug_attended.transpose(1, 2).contiguous().view(B, -1, self.hidden_dim)
        drug_attended = self.out_proj_drug(drug_attended)
        drug_updated = self.ln_drug(drug_repr + drug_attended)
        
        # --- Protein → Drug attention ---
        Q_p = self._reshape_to_heads(self.W_q_prot(prot_repr), B)
        K_d = self._reshape_to_heads(self.W_k_drug(drug_repr), B)
        V_d = self._reshape_to_heads(self.W_v_drug(drug_repr), B)
        
        drug_attn_mask = drug_mask.unsqueeze(1).unsqueeze(2) if drug_mask is not None else None
        
        prot_attended, _ = self._attention(Q_p, K_d, V_d, drug_attn_mask)
        
        prot_attended = prot_attended.transpose(1, 2).contiguous().view(B, -1, self.hidden_dim)
        prot_attended = self.out_proj_prot(prot_attended)
        prot_updated = self.ln_prot(prot_repr + prot_attended)
        
        # --- Interaction map (averaged over heads) ---
        # This is what we visualise and validate against binding sites
        interaction_map = drug_to_prot_attn.mean(dim=1)  # (B, N_atoms, L_residues)
        
        return drug_updated, prot_updated, interaction_map


class GatedPooling(nn.Module):
    """
    Gated pooling for aggregating atom/residue-level features into
    a fixed-size vector for prediction.
    
    Instead of mean/max pooling, we learn which atoms and residues
    are most important for the final prediction. The gate weights
    are themselves interpretable signals.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) — sequence of feature vectors
            mask: (B, L) — boolean mask
        
        Returns:
            pooled: (B, D) — single feature vector per sample
        """
        gate_scores = self.gate(x).squeeze(-1)  # (B, L)
        
        if mask is not None:
            gate_scores = gate_scores.masked_fill(~mask, float('-inf'))
        
        gate_weights = F.softmax(gate_scores, dim=-1)  # (B, L)
        
        transformed = self.transform(x)  # (B, L, D)
        pooled = torch.bmm(gate_weights.unsqueeze(1), transformed).squeeze(1)  # (B, D)
        
        return pooled
