"""
biointeract.py — Full BioInteract model (v2: dual-channel + focal loss ready).

Connects drug encoder (GINE + Morgan FP), target encoder (ESM-2),
cross-attention interaction module, and prediction head.

v2 changes:
    - Dual-channel drug encoding: GNN atoms + Morgan fingerprint global vector
    - Graph augmentation (DropNode/DropEdge) for cold-start robustness
    - Morgan fingerprint branch fused at prediction head
"""
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .drug_encoder import DrugEncoder, MorganFPEncoder, GraphAugmentation
from .target_encoder import TargetEncoder
from .interaction import CrossAttentionInteraction, GatedPooling


class BioInteract(nn.Module):
    """
    BioInteract: Interpretable Drug-Target Interaction Prediction
    via Residue-Level Cross-Attention with Biological Prior Knowledge.
    
    Forward pass:
        1. Encode drug molecular graph → per-atom representations
        2. Encode protein (ESM-2 + domain features) → per-residue representations
        3. Cross-attention interaction → atom-residue interaction map
        4. Gated pooling → fixed-size vector
        5. Prediction head → binding probability / affinity
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        drug_cfg = config.get('drug_encoder', {})
        target_cfg = config.get('target_encoder', {})
        inter_cfg = config.get('interaction', {})
        pred_cfg = config.get('predictor', {})
        
        hidden_dim = drug_cfg.get('hidden_dim', 256)
        
        # --- configurable Morgan FP ---
        self.use_morgan_fp = drug_cfg.get('use_morgan_fp', False)
        
        # --- graph augmentation (training only) ---
        self.graph_aug = GraphAugmentation(
            p_node=drug_cfg.get('drop_node', 0.0),
            p_edge=drug_cfg.get('drop_edge', 0.0),
        )
        
        # --- encoders ---
        self.drug_encoder = DrugEncoder(
            num_atom_features=drug_cfg.get('num_atom_features', 52),
            edge_dim=drug_cfg.get('edge_dim', 16),
            hidden_dim=hidden_dim,
            num_layers=drug_cfg.get('num_layers', 3),
            dropout=drug_cfg.get('dropout', 0.2),
            jk=drug_cfg.get('jk', 'last'),
        )
        
        # Morgan fingerprint branch (optional)
        if self.use_morgan_fp:
            morgan_nbits = drug_cfg.get('morgan_nbits', 1024)
            self.morgan_encoder = MorganFPEncoder(
                input_dim=morgan_nbits,
                hidden_dim=hidden_dim,
                dropout=drug_cfg.get('dropout', 0.2),
            )
        
        self.target_encoder = TargetEncoder(
            esm2_dim=target_cfg.get('esm2_dim', 1280),
            projection_dim=hidden_dim,
            physchem_dim=4,
            domain_embed_dim=target_cfg.get('domain_embed_dim', 32),
            num_domain_types=target_cfg.get('num_domain_types', 50),
            use_domain_features=target_cfg.get('use_domain_features', True),
        )
        
        # --- interaction ---
        self.interaction = CrossAttentionInteraction(
            hidden_dim=hidden_dim,
            num_heads=inter_cfg.get('num_heads', 8),
            dropout=inter_cfg.get('dropout', 0.1),
        )
        
        # --- pooling ---
        self.drug_pooling = GatedPooling(hidden_dim)
        self.prot_pooling = GatedPooling(hidden_dim)
        
        # --- prediction head ---
        # fusion dimension depends on whether Morgan FP is used
        if self.use_morgan_fp:
            pred_input_dim = hidden_dim * 3  # drug_gnn + protein + morgan
        else:
            pred_input_dim = hidden_dim * 2  # drug_gnn + protein
        
        pred_hidden = pred_cfg.get('hidden_dims', [256, 128])
        pred_dropout = pred_cfg.get('dropout', 0.3)
        self.task = pred_cfg.get('task', 'classification')
        
        layers = []
        in_dim = pred_input_dim
        for h_dim in pred_hidden:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(pred_dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.predictor = nn.Sequential(*layers)
    
    def forward(self,
                drug_batch: Batch,
                esm2_embedding: torch.Tensor,
                physicochemical: torch.Tensor,
                domain_labels: torch.Tensor,
                protein_mask: torch.Tensor,
                morgan_fp: torch.Tensor = None,
                return_attention: bool = False):
        """
        Args:
            drug_batch: PyG Batch of molecular graphs
            esm2_embedding: (B, L, esm2_dim)
            physicochemical: (B, L, 4)
            domain_labels: (B, L)
            protein_mask: (B, L) boolean
            morgan_fp: (B, 1024) Morgan fingerprint vectors
            return_attention: if True, also return attention maps
        
        Returns:
            prediction: (B, 1) — binding score
            attention_data: dict (only if return_attention=True)
        """
        batch_size = esm2_embedding.size(0)
        
        # 0. graph augmentation (training only)
        drug_batch = self.graph_aug(drug_batch)
        
        # 1. encode drug → per-atom representations
        atom_repr, batch_index = self.drug_encoder(drug_batch)
        
        # 2. encode protein → per-residue representations
        residue_repr = self.target_encoder(
            esm2_embedding, physicochemical, domain_labels, protein_mask
        )
        
        # 3. reshape drug atoms into (B, max_atoms, D) for cross-attention
        drug_repr_padded, drug_mask = self._pad_drug_atoms(
            atom_repr, batch_index, batch_size
        )
        
        # 4. cross-attention interaction
        drug_updated, prot_updated, interaction_map = self.interaction(
            drug_repr_padded, drug_mask, residue_repr, protein_mask
        )
        
        # 5. gated pooling to fixed-size vectors
        drug_pooled = self.drug_pooling(drug_updated, drug_mask)  # (B, D)
        prot_pooled = self.prot_pooling(prot_updated, protein_mask)  # (B, D)
        
        # 6. Fusion — conditionally include Morgan FP
        if self.use_morgan_fp and morgan_fp is not None:
            morgan_repr = self.morgan_encoder(morgan_fp)  # (B, D)
            fused = torch.cat([drug_pooled, prot_pooled, morgan_repr], dim=-1)  # (B, 3D)
        else:
            fused = torch.cat([drug_pooled, prot_pooled], dim=-1)  # (B, 2D)
        
        # 7. predict
        prediction = self.predictor(fused)  # (B, 1)
        
        # NOTE: for classification, we return raw logits here.
        # Apply sigmoid only during inference (not training with AMP).
        # Use BCEWithLogitsLoss for training.
        
        if return_attention:
            return prediction, {
                'interaction_map': interaction_map,
                'drug_mask': drug_mask,
                'protein_mask': protein_mask,
            }
        
        return prediction
    
    def predict_proba(self, *args, **kwargs):
        """Return probabilities (sigmoid applied) for inference."""
        logits = self.forward(*args, **kwargs)
        if isinstance(logits, tuple):
            return torch.sigmoid(logits[0]), logits[1]
        return torch.sigmoid(logits)
    
    def _pad_drug_atoms(self, atom_repr, batch_index, batch_size):
        """
        Convert scattered atom representations to padded batch tensor.
        
        Args:
            atom_repr: (total_atoms, D) — all atoms from all graphs
            batch_index: (total_atoms,) — which graph each atom belongs to
            batch_size: int
        
        Returns:
            padded: (B, max_atoms, D)
            mask: (B, max_atoms) boolean
        """
        device = atom_repr.device
        D = atom_repr.size(-1)
        
        # count atoms per graph
        counts = torch.bincount(batch_index, minlength=batch_size)
        max_atoms = counts.max().item()
        
        padded = torch.zeros(batch_size, max_atoms, D, device=device)
        mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            atom_indices = (batch_index == i).nonzero(as_tuple=True)[0]
            n = atom_indices.size(0)
            padded[i, :n] = atom_repr[atom_indices]
            mask[i, :n] = True
        
        return padded, mask
