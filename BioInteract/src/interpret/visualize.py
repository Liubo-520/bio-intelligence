"""
visualize.py — Visualisation utilities for BioInteract interpretability.

Generates publication-quality figures for:
  1. Interaction heatmaps (atom × residue)
  2. Residue attention profiles (along protein sequence)
  3. Molecular structure annotation (2D drug with atom importance)
  4. PyMOL script generation (3D protein with attention colouring)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os


def plot_interaction_heatmap(interaction_map: np.ndarray,
                              drug_atoms: List[str],
                              residue_labels: List[str],
                              title: str = 'Drug-Target Interaction Map',
                              save_path: str = None,
                              figsize: Tuple[int, int] = (14, 8)):
    """
    Plot the atom × residue interaction attention heatmap.
    
    This is the key figure for the paper: it shows which drug atoms
    interact with which protein residues, according to the model.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # subsample if too many residues for readability
    max_residues = 80
    if len(residue_labels) > max_residues:
        # show only top-attended region
        residue_scores = interaction_map.sum(axis=0)
        center = np.argmax(residue_scores)
        start = max(0, center - max_residues // 2)
        end = min(len(residue_labels), start + max_residues)
        
        interaction_map = interaction_map[:, start:end]
        residue_labels = residue_labels[start:end]
    
    sns.heatmap(
        interaction_map,
        xticklabels=residue_labels,
        yticklabels=drug_atoms,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Attention Score'},
    )
    
    ax.set_xlabel('Protein Residue', fontsize=12)
    ax.set_ylabel('Drug Atom', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residue_attention_profile(residue_scores: np.ndarray,
                                    sequence: str,
                                    binding_residues: set = None,
                                    title: str = 'Residue Attention Profile',
                                    save_path: str = None,
                                    figsize: Tuple[int, int] = (16, 5)):
    """
    Plot attention scores along the protein sequence.
    
    Optionally overlay experimental binding site residues to visually
    assess the correspondence between model attention and reality.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    positions = np.arange(len(residue_scores))
    
    # attention profile
    ax.fill_between(positions, residue_scores, alpha=0.3, color='steelblue')
    ax.plot(positions, residue_scores, color='steelblue', linewidth=0.8,
            label='Model Attention')
    
    # mark binding site residues
    if binding_residues:
        for res_idx in binding_residues:
            if res_idx < len(residue_scores):
                ax.axvline(x=res_idx, color='red', alpha=0.3,
                           linewidth=0.5, linestyle='--')
        
        # highlight region
        binding_list = sorted(binding_residues)
        ax.scatter(binding_list,
                   [residue_scores[i] if i < len(residue_scores) else 0
                    for i in binding_list],
                   color='red', s=15, zorder=5,
                   label='Known Binding Residues')
    
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Normalised Attention Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, len(residue_scores))
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_drug_atom_importance(smiles: str,
                               atom_scores: np.ndarray,
                               title: str = 'Drug Atom Importance',
                               save_path: str = None):
    """
    Render 2D molecular structure with atoms coloured by importance.
    
    Uses RDKit drawing utilities to produce a publication-quality
    molecular diagram with Grad-CAM heatmap overlay.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("RDKit required for molecular visualisation")
        return
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    
    AllChem.Compute2DCoords(mol)
    
    # create atom colour map
    norm = Normalize(vmin=0, vmax=max(atom_scores.max(), 1e-6))
    cmap = plt.cm.YlOrRd
    
    atom_colours = {}
    atom_radii = {}
    for idx in range(mol.GetNumAtoms()):
        if idx < len(atom_scores):
            score = atom_scores[idx]
            rgba = cmap(norm(score))
            atom_colours[idx] = rgba[:3]  # RGB only
            atom_radii[idx] = 0.3 + 0.4 * score
    
    # draw
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
    drawer.drawOptions().useBWAtomPalette()
    
    # Use compatible API: DrawMolecule with highlight lists
    highlight_atoms = list(atom_colours.keys())
    highlight_atom_colors = {k: tuple(v) for k, v in atom_colours.items()}
    highlight_atom_radii = atom_radii
    
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightAtomRadii=highlight_atom_radii,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(drawer.GetDrawingText())


def generate_pymol_script(protein_id: str,
                           pdb_file: str,
                           residue_scores: Dict[int, float],
                           save_path: str = None,
                           colour_scheme: str = 'blue_white_red') -> str:
    """
    Generate a PyMOL .pml script that colours the protein structure
    by attention scores.
    
    This produces the 3D figure for the paper: a protein surface
    coloured by how much the model attends to each residue,
    with the drug molecule shown in the binding pocket.
    
    Usage: Open in PyMOL → File → Run Script → save as PNG
    """
    lines = []
    lines.append(f'# PyMOL script for {protein_id} attention visualisation')
    lines.append(f'# Generated by BioInteract')
    lines.append('')
    lines.append(f'load {pdb_file}, {protein_id}')
    lines.append(f'hide everything')
    lines.append(f'show cartoon, {protein_id}')
    lines.append(f'color grey80, {protein_id}')
    lines.append('')
    lines.append('# Colour residues by attention score')
    
    # bin scores into colour levels
    if residue_scores:
        max_score = max(residue_scores.values())
        min_score = min(residue_scores.values())
        
        for res_idx, score in sorted(residue_scores.items()):
            if max_score > min_score:
                normalised = (score - min_score) / (max_score - min_score)
            else:
                normalised = 0.5
            
            # map to blue-white-red colour
            if normalised < 0.5:
                r = normalised * 2
                g = normalised * 2
                b = 1.0
            else:
                r = 1.0
                g = (1 - normalised) * 2
                b = (1 - normalised) * 2
            
            lines.append(
                f'set_color attn_{res_idx}, [{r:.3f}, {g:.3f}, {b:.3f}]'
            )
            lines.append(
                f'color attn_{res_idx}, resi {res_idx + 1}'
            )
    
    lines.append('')
    lines.append('# Show binding pocket residues as sticks')
    
    # top-20 residues shown as sticks
    sorted_residues = sorted(residue_scores.items(),
                              key=lambda x: x[1], reverse=True)[:20]
    top_resi = '+'.join(str(r + 1) for r, _ in sorted_residues)
    lines.append(f'show sticks, resi {top_resi}')
    lines.append('')
    lines.append('# Camera settings')
    lines.append('set ray_shadow, 0')
    lines.append('set surface_quality, 2')
    lines.append('bg_color white')
    lines.append('set antialias, 2')
    lines.append(f'orient {protein_id}')
    
    script = '\n'.join(lines)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(script)
    
    return script
