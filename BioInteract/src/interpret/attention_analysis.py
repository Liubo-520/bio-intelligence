"""
attention_analysis.py — Extract and analyse cross-attention interaction maps.

The core idea: the attention weight M[i,j] between drug atom i and protein
residue j is a learned proxy for "interaction strength". If the model has
truly captured the binding mechanism, the high-attention residues should
correspond to the experimentally determined binding pocket.
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


def extract_interaction_map(model, batch: dict, device: str = 'cuda') -> dict:
    """
    Run a forward pass with attention output and collect interaction maps.
    
    Returns:
        {
            'interaction_map': np.array (B, N_atoms, L_residues),
            'drug_ids': list,
            'target_ids': list,
            'drug_mask': np.array (B, N_atoms),
            'protein_mask': np.array (B, L_residues),
        }
    """
    model.eval()
    with torch.no_grad():
        drug_batch = batch['drug_batch'].to(device)
        esm2 = batch['esm2_embedding'].to(device)
        phys = batch['physicochemical'].to(device)
        domain = batch['domain_labels'].to(device)
        prot_mask = batch['protein_mask'].to(device)
        
        _, attn_data = model(
            drug_batch, esm2, phys, domain, prot_mask,
            return_attention=True
        )
    
    return {
        'interaction_map': attn_data['interaction_map'].cpu().numpy(),
        'drug_mask': attn_data['drug_mask'].cpu().numpy(),
        'protein_mask': attn_data['protein_mask'].cpu().numpy(),
        'drug_ids': batch['drug_ids'],
        'target_ids': batch['target_ids'],
    }


def get_top_k_residues(interaction_map: np.ndarray,
                       drug_mask: np.ndarray,
                       k: int = 20) -> np.ndarray:
    """
    For each sample, aggregate atom-level attention over all drug atoms
    and return the indices of the top-K most attended protein residues.
    
    The aggregation (sum over drug atoms) represents the total "interaction
    pressure" on each residue — biologically, this identifies which
    residues are most involved in binding across the entire drug molecule.
    
    Args:
        interaction_map: (B, N_atoms, L_residues)
        drug_mask: (B, N_atoms) boolean
        k: number of top residues to return
    
    Returns:
        top_k_indices: (B, K) — indices of top-K residues per sample
    """
    B = interaction_map.shape[0]
    top_k_indices = np.zeros((B, k), dtype=np.int64)
    
    for b in range(B):
        # mask out padding atoms, sum attention across valid atoms
        valid_atoms = drug_mask[b]
        attn = interaction_map[b]  # (N_atoms, L_residues)
        attn[~valid_atoms] = 0
        
        residue_scores = attn.sum(axis=0)  # (L_residues,)
        
        # get top-K indices
        top_indices = np.argsort(residue_scores)[::-1][:k]
        top_k_indices[b, :len(top_indices)] = top_indices
    
    return top_k_indices


def binding_site_recall(predicted_residues: np.ndarray,
                        true_binding_residues: List[set],
                        k_values: List[int] = [10, 15, 20, 30]
                        ) -> Dict[str, float]:
    """
    Evaluate how well the attention-predicted binding residues match
    experimentally known binding site residues.
    
    This is the key quantitative validation for interpretability:
    "Does the model's attention focus on the real binding pocket?"
    
    Args:
        predicted_residues: (B, max_K) — predicted residue indices
        true_binding_residues: list of sets, each containing ground-truth
                               binding residue indices for that sample
        k_values: evaluate at these K values
    
    Returns:
        dict with Recall@K, Precision@K, F1@K for each K value
    """
    results = {}
    B = len(true_binding_residues)
    
    for k in k_values:
        recalls = []
        precisions = []
        
        for b in range(B):
            pred_set = set(predicted_residues[b, :k].tolist())
            true_set = true_binding_residues[b]
            
            if len(true_set) == 0:
                continue
            
            hits = len(pred_set & true_set)
            recall = hits / len(true_set)
            precision = hits / k
            
            recalls.append(recall)
            precisions.append(precision)
        
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_precision = np.mean(precisions) if precisions else 0.0
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)
        
        results[f'Recall@{k}'] = avg_recall
        results[f'Precision@{k}'] = avg_precision
        results[f'F1@{k}'] = f1
    
    return results


def residue_attention_profile(interaction_map: np.ndarray,
                              drug_mask: np.ndarray,
                              protein_mask: np.ndarray,
                              sequence: str) -> Dict[str, float]:
    """
    Compute per-residue attention profile for a single drug-target pair.
    
    Returns a dict mapping residue position (1-indexed, with amino acid)
    to its normalised attention score above the background.
    
    Useful for generating detailed case study analysis.
    """
    # sum over drug atoms
    valid_atoms = drug_mask
    attn = interaction_map.copy()
    attn[~valid_atoms] = 0
    
    residue_scores = attn.sum(axis=0)  # (L,)
    valid_residues = protein_mask
    residue_scores[~valid_residues] = 0
    
    # normalise to [0, 1]
    max_score = residue_scores.max()
    if max_score > 0:
        residue_scores = residue_scores / max_score
    
    profile = {}
    for i, score in enumerate(residue_scores):
        if valid_residues[i]:
            aa = sequence[i] if i < len(sequence) else '?'
            profile[f'{aa}{i+1}'] = float(score)
    
    return profile


def identify_interaction_hotspots(profile: Dict[str, float],
                                  threshold: float = 0.5) -> List[str]:
    """
    Identify residues with attention scores above the threshold.
    
    These are the predicted binding hotspot residues — the model
    believes these are critical for drug-target recognition.
    """
    return [res for res, score in profile.items() if score >= threshold]


def cross_family_analysis(interaction_maps: Dict[str, np.ndarray],
                          family_labels: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    Cluster interaction patterns by protein family.
    
    For each protein family (e.g., Kinase, GPCR), compute the average
    interaction pattern. This reveals family-specific binding preferences:
    - Kinases: attention on hinge region and DFG motif
    - GPCRs: attention on transmembrane helices
    - Proteases: attention on catalytic triad residues
    
    Args:
        interaction_maps: {target_id: residue_score_array}
        family_labels: {target_id: family_name}
    
    Returns:
        {family_name: average_attention_distribution}
    """
    family_patterns = defaultdict(list)
    
    for target_id, attn_scores in interaction_maps.items():
        family = family_labels.get(target_id, 'Unknown')
        family_patterns[family].append(attn_scores)
    
    averaged = {}
    for family, patterns in family_patterns.items():
        # normalise each pattern to same length (use interpolation)
        max_len = max(len(p) for p in patterns)
        interpolated = []
        for p in patterns:
            x_old = np.linspace(0, 1, len(p))
            x_new = np.linspace(0, 1, max_len)
            interp = np.interp(x_new, x_old, p)
            interpolated.append(interp)
        averaged[family] = np.mean(interpolated, axis=0)
    
    return averaged
