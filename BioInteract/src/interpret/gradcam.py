"""
gradcam.py — Gradient-weighted Class Activation Mapping for GNN drug encoder.

Identifies which atoms in the drug molecule contribute most to the
binding prediction. The biological interpretation: these are the
pharmacophore-relevant atoms — the functional groups responsible
for the drug's binding activity.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization", ICCV 2017.
Adapted for graph neural networks.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GNNGradCAM:
    """
    Grad-CAM adapted for graph neural networks.
    
    Computes atom-level importance scores by:
      1. Doing a forward pass and extracting activations from target GNN layer
      2. Computing gradients of the prediction w.r.t. these activations
      3. Weighting activations by their gradients (= importance)
      4. ReLU to keep only positive contributions
    
    The resulting heatmap can be mapped onto the 2D molecular structure
    to highlight pharmacophore regions.
    """
    
    def __init__(self, model, target_layer_name: str = 'drug_encoder.layers.2'):
        """
        Args:
            model: BioInteract model
            target_layer_name: which GNN layer to compute Grad-CAM on
                               (typically the last GINE layer)
        """
        self.model = model
        self.activations = None
        self.gradients = None
        
        # register hooks on the target layer
        target_layer = self._get_layer(model, target_layer_name)
        target_layer.register_forward_hook(self._activation_hook)
        target_layer.register_full_backward_hook(self._gradient_hook)
    
    def _get_layer(self, model, layer_name: str):
        """Access nested module by dot-separated name."""
        parts = layer_name.split('.')
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _activation_hook(self, module, input, output):
        """Store activations during forward pass."""
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
    
    def _gradient_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        self.gradients = grad_output[0].detach()
    
    def compute(self,
                batch: dict,
                device: str = 'cuda',
                target_class: int = 1) -> np.ndarray:
        """
        Compute Grad-CAM atom importance scores.
        
        Args:
            batch: collated DTI batch
            device: cuda or cpu
            target_class: which class to explain (1 = binding)
        
        Returns:
            atom_importance: (total_atoms,) array of importance scores
        """
        self.model.eval()
        # we need gradients for Grad-CAM, but only w.r.t. activations
        self.model.zero_grad()
        
        drug_batch = batch['drug_batch'].to(device)
        drug_batch.x.requires_grad_(True)
        
        esm2 = batch['esm2_embedding'].to(device)
        phys = batch['physicochemical'].to(device)
        domain = batch['domain_labels'].to(device)
        prot_mask = batch['protein_mask'].to(device)
        
        # forward
        prediction = self.model(
            drug_batch, esm2, phys, domain, prot_mask,
            return_attention=False
        )
        
        # backward from prediction
        self.model.zero_grad()
        prediction.sum().backward(retain_graph=True)
        
        # Grad-CAM computation
        if self.gradients is None or self.activations is None:
            return np.zeros(drug_batch.x.size(0))
        
        # global average pooling of gradients → weights per feature channel
        weights = self.gradients.mean(dim=0)  # (D,) or per-node
        
        if weights.dim() == 1:
            # weights: (D,), activations: (N, D)
            cam = (self.activations * weights.unsqueeze(0)).sum(dim=-1)
        else:
            cam = (self.activations * weights).sum(dim=-1)
        
        # ReLU: only keep positive influence
        cam = F.relu(cam)
        
        # normalise per graph
        batch_index = drug_batch.batch
        cam_np = cam.cpu().numpy()
        batch_np = batch_index.cpu().numpy()
        
        # normalise within each graph
        for graph_idx in np.unique(batch_np):
            mask = batch_np == graph_idx
            graph_cam = cam_np[mask]
            max_val = graph_cam.max()
            if max_val > 0:
                cam_np[mask] = graph_cam / max_val
        
        return cam_np
    
    def get_atom_importance_per_graph(self,
                                      cam_scores: np.ndarray,
                                      batch_index: np.ndarray
                                      ) -> list:
        """
        Split per-graph atom importance scores.
        
        Returns:
            list of arrays, one per graph in the batch
        """
        result = []
        for graph_idx in range(batch_index.max() + 1):
            mask = batch_index == graph_idx
            result.append(cam_scores[mask])
        return result


def map_atom_importance_to_substructures(atom_scores: np.ndarray,
                                          smiles: str) -> dict:
    """
    Aggregate atom-level Grad-CAM scores to functional group level.
    
    This bridges the gap between raw atom scores and pharmacophore
    interpretation: instead of "atom 7 is important", we can say
    "the phenyl ring at position 3-8 is important for binding."
    
    Args:
        atom_scores: per-atom importance scores
        smiles: drug SMILES string
    
    Returns:
        dict mapping functional group names to average importance
    """
    from rdkit import Chem
    from rdkit.Chem import Fragments
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    # define common pharmacophore-relevant substructures
    substructures = {
        'Aromatic Ring': Chem.MolFromSmarts('c1ccccc1'),
        'Hydroxyl': Chem.MolFromSmarts('[OX2H]'),
        'Amino': Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),
        'Amide': Chem.MolFromSmarts('[NX3][CX3](=[OX1])'),
        'Carboxyl': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
        'Sulfonyl': Chem.MolFromSmarts('[SX4](=O)(=O)'),
        'Halogen': Chem.MolFromSmarts('[F,Cl,Br,I]'),
        'Ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
        'Heterocycle N': Chem.MolFromSmarts('[nR]'),
        'Carbonyl': Chem.MolFromSmarts('[CX3]=[OX1]'),
    }
    
    results = {}
    for name, pattern in substructures.items():
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            # average importance across all atoms in all matches
            all_atoms = set()
            for match in matches:
                all_atoms.update(match)
            
            valid_atoms = [a for a in all_atoms if a < len(atom_scores)]
            if valid_atoms:
                avg_score = np.mean([atom_scores[a] for a in valid_atoms])
                results[name] = float(avg_score)
    
    return results
