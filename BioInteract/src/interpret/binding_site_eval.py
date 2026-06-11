"""
binding_site_eval.py — Evaluate model interpretability against known binding sites.

Uses PDB crystal structures as ground truth to quantitatively assess
whether the model's attention patterns correspond to real molecular contacts.

This is the "proof" that the model has learned genuine biology:
if attention hotspots match experimentally determined binding pockets,
the model isn't just fitting statistics — it's capturing the physics
of molecular recognition.
"""
import os
import json
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BindingSiteInfo:
    """Ground-truth binding site information from PDB."""
    pdb_id: str
    drug_name: str
    target_name: str
    contact_residues: Set[int]         # residue indices within 4Å of ligand
    hbond_residues: Set[int]           # residues forming hydrogen bonds
    hydrophobic_residues: Set[int]     # residues in hydrophobic contacts
    key_residues: Set[int]             # literature-reported key residues


def load_binding_site_ground_truth(case_study_dir: str = 'data/case_studies'
                                    ) -> Dict[str, BindingSiteInfo]:
    """
    Load precomputed binding site annotations.
    
    These should be prepared manually from PDB structures using tools like:
      - PLIP (Protein-Ligand Interaction Profiler)
      - PyMOL distance measurements
      - Published literature
    
    Expected JSON format per case:
    {
        "pdb_id": "1IEP",
        "drug_name": "Imatinib",
        "target_name": "ABL1",
        "contact_residues_4A": [271, 286, 290, 315, 317, ...],
        "hbond_residues": [315, 381, 382],
        "hydrophobic_residues": [271, 286, 290, 299, 313],
        "key_residues_literature": [315, 317, 381, 382, 400]
    }
    """
    cases = {}
    
    if not os.path.exists(case_study_dir):
        return cases
    
    for fname in os.listdir(case_study_dir):
        if not fname.endswith('.json'):
            continue
        
        with open(os.path.join(case_study_dir, fname), 'r') as f:
            data = json.load(f)
        
        pdb_id = data['pdb_id']
        cases[pdb_id] = BindingSiteInfo(
            pdb_id=pdb_id,
            drug_name=data.get('drug_name', ''),
            target_name=data.get('target_name', ''),
            contact_residues=set(data.get('contact_residues_4A', [])),
            hbond_residues=set(data.get('hbond_residues', [])),
            hydrophobic_residues=set(data.get('hydrophobic_residues', [])),
            key_residues=set(data.get('key_residues_literature', [])),
        )
    
    return cases


def evaluate_binding_site_prediction(predicted_top_k: np.ndarray,
                                      ground_truth: BindingSiteInfo,
                                      k_values: List[int] = [10, 15, 20, 30]
                                      ) -> Dict[str, dict]:
    """
    Comprehensive binding site prediction evaluation.
    
    Evaluates attention predictions against multiple levels of ground truth:
      1. All contact residues (4Å cutoff) — general binding pocket
      2. H-bond residues — specific polar interactions
      3. Key residues from literature — expert-curated critical contacts
    
    This multi-level evaluation shows the model captures both broad
    binding pocket geometry and specific interaction chemistry.
    """
    results = {}
    
    evaluations = {
        'contact_4A': ground_truth.contact_residues,
        'hbond': ground_truth.hbond_residues,
        'key_literature': ground_truth.key_residues,
    }
    
    for eval_name, true_residues in evaluations.items():
        if not true_residues:
            continue
        
        eval_results = {}
        for k in k_values:
            pred_set = set(predicted_top_k[:k].tolist())
            
            hits = len(pred_set & true_residues)
            recall = hits / len(true_residues)
            precision = hits / k
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            eval_results[f'Recall@{k}'] = recall
            eval_results[f'Precision@{k}'] = precision
            eval_results[f'F1@{k}'] = f1
            eval_results[f'Hits@{k}'] = hits
        
        results[eval_name] = eval_results
    
    return results


def interaction_type_analysis(interaction_map: np.ndarray,
                               drug_mask: np.ndarray,
                               ground_truth: BindingSiteInfo,
                               protein_mask: np.ndarray) -> dict:
    """
    Analyse what types of interactions the model attends to most.
    
    Compare average attention scores on:
      - H-bond residues vs non-H-bond contact residues
      - Hydrophobic contact residues vs polar residues
      - Key residues vs other contact residues
    
    If the model gives significantly higher attention to H-bond and
    key residues, it indicates the model has learned the hierarchy
    of interaction importance — not just proximity.
    """
    # compute residue-level attention scores
    valid_atoms = drug_mask
    attn = interaction_map.copy()
    attn[~valid_atoms] = 0
    residue_scores = attn.sum(axis=0)
    
    valid = protein_mask
    
    # categorise residues
    all_contact = ground_truth.contact_residues
    hbond = ground_truth.hbond_residues
    hydrophobic = ground_truth.hydrophobic_residues
    key = ground_truth.key_residues
    non_contact = set(range(len(residue_scores))) - all_contact
    
    def safe_mean(indices):
        valid_idx = [i for i in indices if i < len(residue_scores) and valid[i]]
        if not valid_idx:
            return 0.0
        return float(np.mean([residue_scores[i] for i in valid_idx]))
    
    return {
        'avg_attention_contact': safe_mean(all_contact),
        'avg_attention_non_contact': safe_mean(non_contact),
        'avg_attention_hbond': safe_mean(hbond),
        'avg_attention_hydrophobic': safe_mean(hydrophobic),
        'avg_attention_key': safe_mean(key),
        'contact_vs_noncontact_ratio': (
            safe_mean(all_contact) / (safe_mean(non_contact) + 1e-8)
        ),
        'key_vs_contact_ratio': (
            safe_mean(key) / (safe_mean(all_contact) + 1e-8)
        ),
    }


def generate_case_study_report(drug_name: str,
                                target_name: str,
                                pdb_id: str,
                                residue_profile: Dict[str, float],
                                binding_eval: Dict[str, dict],
                                substructure_importance: Dict[str, float],
                                interaction_types: dict) -> str:
    """
    Generate a human-readable case study report for the paper.
    
    This produces the text that goes into "Section 5.5: Case Studies"
    of the manuscript.
    """
    report = []
    report.append(f"=== Case Study: {drug_name} — {target_name} (PDB: {pdb_id}) ===\n")
    
    # binding site prediction quality
    report.append("1. Binding Site Localisation:")
    if 'contact_4A' in binding_eval:
        for metric, value in binding_eval['contact_4A'].items():
            report.append(f"   {metric}: {value:.3f}")
    
    if 'key_literature' in binding_eval:
        report.append("\n   Key Residue Recovery:")
        for metric, value in binding_eval['key_literature'].items():
            report.append(f"   {metric}: {value:.3f}")
    
    # interaction type preference
    report.append("\n2. Interaction Type Analysis:")
    report.append(f"   Contact vs Non-contact attention ratio: "
                  f"{interaction_types.get('contact_vs_noncontact_ratio', 0):.2f}x")
    report.append(f"   H-bond residue avg attention: "
                  f"{interaction_types.get('avg_attention_hbond', 0):.4f}")
    report.append(f"   Hydrophobic residue avg attention: "
                  f"{interaction_types.get('avg_attention_hydrophobic', 0):.4f}")
    
    # top attended residues
    report.append("\n3. Top-10 Attended Residues:")
    sorted_residues = sorted(residue_profile.items(),
                              key=lambda x: x[1], reverse=True)[:10]
    for res, score in sorted_residues:
        report.append(f"   {res}: {score:.4f}")
    
    # drug substructure importance
    if substructure_importance:
        report.append("\n4. Drug Substructure Importance (Grad-CAM):")
        sorted_sub = sorted(substructure_importance.items(),
                             key=lambda x: x[1], reverse=True)
        for sub, score in sorted_sub:
            report.append(f"   {sub}: {score:.4f}")
    
    return '\n'.join(report)
