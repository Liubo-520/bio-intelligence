"""
drug_encoder.py — GINE-based molecular graph encoder + Morgan fingerprint fusion.

Uses Graph Isomorphism Network with Edge features (GINE) for learned
molecular representations, combined with Morgan fingerprints (ECFP4)
for knowledge-driven chemical similarity encoding.

v2: adds DropNode/DropEdge graph augmentation for cold-start robustness,
    and a MorganFPEncoder for dual-channel drug representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, BatchNorm
from torch_geometric.data import Batch


class GINELayer(nn.Module):
    """Single GINE layer with batch norm and residual connection."""
    
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.2):
        super().__init__()
        # GINE uses an MLP as the update function
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.conv = GINEConv(nn=mlp, edge_dim=edge_dim)
        self.bn = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr):
        h = self.conv(x, edge_index, edge_attr)
        h = self.bn(h)
        h = F.relu(h)
        h = self.dropout(h)
        # residual connection
        return h + x


class DrugEncoder(nn.Module):
    """
    GINE-based drug molecular graph encoder.
    
    Produces per-atom representations suitable for cross-attention
    with protein residue representations.
    
    Architecture:
        Input projection → N × GINE layers → atom-level output
    
    We intentionally do NOT apply global readout (mean/sum pooling)
    here — the cross-attention module needs individual atom vectors
    to compute residue-level interaction maps.
    """
    
    def __init__(self,
                 num_atom_features: int = 52,
                 edge_dim: int = 16,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 jk: str = 'last'):
        """
        Args:
            num_atom_features: input atom feature dimension
            edge_dim: bond feature dimension
            hidden_dim: GINE hidden dimension
            num_layers: number of GINE layers
            dropout: dropout rate
            jk: jumping knowledge mode ('last' or 'cat')
        """
        super().__init__()
        self.num_layers = num_layers
        self.jk = jk
        
        # input projection
        self.input_proj = nn.Sequential(
            nn.Linear(num_atom_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # edge feature projection (match edge_attr dim to hidden_dim for GINE)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # GINE layers
        self.layers = nn.ModuleList([
            GINELayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # jumping knowledge output
        if jk == 'cat':
            self.jk_proj = nn.Linear(hidden_dim * num_layers, hidden_dim)
        
        self.output_dim = hidden_dim
    
    def forward(self, drug_batch: Batch) -> tuple:
        """
        Args:
            drug_batch: PyG Batch of molecular graphs
        
        Returns:
            atom_repr: (total_atoms, hidden_dim) — per-atom representations
            batch_index: (total_atoms,) — which graph each atom belongs to
        """
        x = drug_batch.x
        edge_index = drug_batch.edge_index
        edge_attr = drug_batch.edge_attr
        batch_index = drug_batch.batch
        
        # project inputs
        h = self.input_proj(x)
        edge_features = self.edge_proj(edge_attr) if edge_attr.size(0) > 0 \
            else edge_attr
        
        # message passing
        layer_outputs = []
        for layer in self.layers:
            h = layer(h, edge_index, edge_features)
            layer_outputs.append(h)
        
        # jumping knowledge
        if self.jk == 'cat':
            h = torch.cat(layer_outputs, dim=-1)
            h = self.jk_proj(h)
        else:
            h = layer_outputs[-1]
        
        return h, batch_index


class GraphAugmentation(nn.Module):
    """
    Stochastic graph augmentation for regularisation during training.

    DropNode: randomly masks node features (zero-out) with probability p_node.
    DropEdge: randomly removes edges with probability p_edge.

    These augmentations prevent the GNN from memorising specific molecular
    graphs, which is critical for cold-start generalisation where test
    molecules are unseen during training.

    Reference: Rong et al., "DropEdge: Towards Deep Graph ConvNets", ICLR 2020
    """

    def __init__(self, p_node: float = 0.1, p_edge: float = 0.15):
        super().__init__()
        self.p_node = p_node
        self.p_edge = p_edge

    def forward(self, drug_batch: Batch) -> Batch:
        if not self.training:
            return drug_batch

        # DropNode: zero-out whole node feature vectors
        if self.p_node > 0:
            mask = torch.rand(drug_batch.x.size(0), 1,
                              device=drug_batch.x.device) > self.p_node
            drug_batch.x = drug_batch.x * mask.float()

        # DropEdge: remove random edges
        if self.p_edge > 0 and drug_batch.edge_index.size(1) > 0:
            n_edges = drug_batch.edge_index.size(1)
            keep = torch.rand(n_edges, device=drug_batch.edge_index.device) > self.p_edge
            drug_batch.edge_index = drug_batch.edge_index[:, keep]
            if drug_batch.edge_attr is not None and drug_batch.edge_attr.size(0) == n_edges:
                drug_batch.edge_attr = drug_batch.edge_attr[keep]

        return drug_batch


class MorganFPEncoder(nn.Module):
    """
    Encoder for Morgan (ECFP4) molecular fingerprints.

    Morgan fingerprints capture circular substructure patterns and provide
    a chemistry-prior-based global drug representation that inherently
    generalises to unseen molecules — unlike GNN features that depend on
    message passing over specific graph topologies.

    This encoder projects the binary fingerprint to a dense vector and
    is fused with the GNN representation for dual-channel drug encoding.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fp: (B, input_dim) binary Morgan fingerprint
        Returns:
            (B, hidden_dim) dense drug representation
        """
        return self.encoder(fp)
