"""
interpret.py — Full interpretability analysis for BioInteract.

Produces publication-quality analyses:
  1. Attention-based binding site analysis on high-confidence predictions
  2. Grad-CAM drug atom importance
  3. Per-residue attention profiles for case studies
  4. Interaction heatmaps (drug atom x protein residue)
  5. Functional group importance ranking
  6. Summary statistics and figures

Usage:
    python -m src.cli.interpret --config configs/default.yaml --checkpoint checkpoints/best.pt
"""
import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict

from src.data.dataset import DTIDataset, collate_dti
from src.models.biointeract import BioInteract
from src.interpret.attention_analysis import (
    extract_interaction_map,
    get_top_k_residues,
    residue_attention_profile,
    identify_interaction_hotspots,
)
from src.interpret.gradcam import GNNGradCAM, map_atom_importance_to_substructures
from src.interpret.visualize import (
    plot_interaction_heatmap,
    plot_residue_attention_profile,
    plot_drug_atom_importance,
    generate_pymol_script,
)
from src.utils.logger import setup_logger
from src.utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, DATA_DIR, RESULTS_DIR, resolve_project_path


def load_config(path):
    with open(resolve_project_path(path), 'r') as f:
        return yaml.safe_load(f)


def find_case_study_pairs(interactions_df, drug_df, target_df, case_names):
    """Find drug-target pair indices matching case study target name patterns."""
    drug_lookup = dict(zip(drug_df['drug_id'],
                           drug_df.get('drug_name', drug_df['drug_id'])))
    target_lookup = dict(zip(target_df['target_id'], target_df['target_name']))

    matches = []
    for case in case_names:
        target_pattern = case['target']
        matched_targets = [(tid, tname) for tid, tname in target_lookup.items()
                           if target_pattern.upper() in tname.upper()]
        if not matched_targets:
            continue
        for tid, tname in matched_targets[:3]:
            binding = interactions_df[
                (interactions_df['target_id'] == tid) & (interactions_df['label'] == 1)
            ]
            if len(binding) > 0:
                best = binding.sort_values('affinity').iloc[0]
                did = best['drug_id']
                matches.append({
                    'drug_id': did, 'target_id': tid,
                    'drug_name': str(drug_lookup.get(did, did)),
                    'target_name': tname,
                    'affinity': best['affinity'],
                    'pdb': case.get('pdb', 'N/A'),
                })
    return matches


def analyse_single_pair(model, dataset, idx, device, top_k=20):
    """Run full interpretability analysis for one drug-target pair."""
    sample = dataset[idx]
    if sample is None:
        return None
    batch = collate_dti([sample])
    attn_data = extract_interaction_map(model, batch, device)

    interaction_map = attn_data['interaction_map'][0]
    drug_mask = attn_data['drug_mask'][0]
    protein_mask = attn_data['protein_mask'][0]

    sequence = sample.get('sequence', '')
    profile = residue_attention_profile(interaction_map, drug_mask, protein_mask, sequence)
    top_k_indices = get_top_k_residues(
        attn_data['interaction_map'], attn_data['drug_mask'], k=top_k
    )[0]
    hotspots = identify_interaction_hotspots(profile, threshold=0.5)

    with torch.no_grad():
        drug_batch = batch['drug_batch'].to(device)
        esm2 = batch['esm2_embedding'].to(device)
        phys = batch['physicochemical'].to(device)
        domain = batch['domain_labels'].to(device)
        prot_mask = batch['protein_mask'].to(device)
        logit = model(drug_batch, esm2, phys, domain, prot_mask)
        prob = torch.sigmoid(logit).item()

    return {
        'interaction_map': interaction_map,
        'drug_mask': drug_mask,
        'protein_mask': protein_mask,
        'residue_profile': profile,
        'top_k_residues': top_k_indices,
        'hotspots': hotspots,
        'prediction_prob': prob,
    }


def run_gradcam_analysis(gradcam, dataset, idx, device):
    """Run Grad-CAM for a single sample."""
    sample = dataset[idx]
    if sample is None:
        return None
    batch = collate_dti([sample])
    try:
        cam_scores = gradcam.compute(batch, device)
        per_graph = gradcam.get_atom_importance_per_graph(
            cam_scores, batch['drug_batch'].batch.cpu().numpy()
        )
        return per_graph[0] if per_graph else None
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        return None


def global_attention_statistics(model, dataloader, device, n_batches=50):
    """Compute global attention statistics across the dataset."""
    all_residue_scores = []
    model.eval()
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        attn_data = extract_interaction_map(model, batch, device)
        for b in range(attn_data['interaction_map'].shape[0]):
            imap = attn_data['interaction_map'][b]
            dmask = attn_data['drug_mask'][b]
            pmask = attn_data['protein_mask'][b]
            attn = imap.copy()
            attn[~dmask] = 0
            rscore = attn.sum(axis=0)
            rscore[~pmask] = 0
            valid = rscore[pmask]
            if len(valid) > 0:
                mx = valid.max()
                if mx > 0:
                    valid = valid / mx
                all_residue_scores.append(valid)
    if not all_residue_scores:
        return {}
    flat = np.concatenate(all_residue_scores)
    return {
        'residue_attention_mean': float(flat.mean()),
        'residue_attention_std': float(flat.std()),
        'residue_attention_median': float(np.median(flat)),
        'residue_attention_top1pct': float(np.percentile(flat, 99)),
        'residue_attention_top5pct': float(np.percentile(flat, 95)),
        'n_samples': len(all_residue_scores),
        'attention_sparsity': float((flat < 0.1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(CONFIGS_DIR / 'default.yaml'))
    parser.add_argument('--checkpoint', type=str, default=str(CHECKPOINTS_DIR / 'best.pt'))
    parser.add_argument('--output_dir', type=str, default=str(RESULTS_DIR / 'interpretability'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_cases', type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = resolve_project_path(args.output_dir)
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger = setup_logger('interpret')

    for sub in ['heatmaps', 'profiles', 'gradcam']:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- load model ----
    logger.info("Loading trained model...")
    model = BioInteract(config['model']).to(device)
    ckpt = torch.load(resolve_project_path(args.checkpoint), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f"Checkpoint epoch {ckpt.get('epoch','?')}, "
                f"best={ckpt.get('best_metric',0):.4f}")

    # ---- load data ----
    base = DATA_DIR / 'raw' / config['data']['dataset']
    interactions = pd.read_csv(base / 'interactions.csv')
    drug_df = pd.read_csv(base / 'drug_smiles.csv')
    target_df = pd.read_csv(base / 'target_sequences.csv')
    drug_smiles = dict(zip(drug_df['drug_id'], drug_df['smiles']))
    target_sequences = dict(zip(target_df['target_id'], target_df['sequence']))
    target_names = dict(zip(target_df['target_id'], target_df['target_name']))

    # ---- find case study pairs ----
    interpret_cfg = config.get('interpret', {})
    case_defs = interpret_cfg.get('case_study_pairs', [])
    top_k = interpret_cfg.get('top_k_residues', 20)

    named_cases = find_case_study_pairs(interactions, drug_df, target_df, case_defs)
    logger.info(f"Found {len(named_cases)} named case study pairs")

    # pick additional high-confidence binders
    positive_pairs = interactions[interactions['label'] == 1].sort_values('affinity')
    extra = []
    for _, row in positive_pairs.head(args.n_cases * 2).iterrows():
        did, tid = row['drug_id'], row['target_id']
        tname = target_names.get(tid, tid)
        if not any(c['drug_id'] == did and c['target_id'] == tid for c in named_cases):
            dname = drug_df[drug_df['drug_id'] == did]['drug_name'].values
            extra.append({
                'drug_id': did, 'target_id': tid,
                'drug_name': str(dname[0]) if len(dname) > 0 else did,
                'target_name': tname,
                'affinity': row['affinity'],
                'pdb': 'N/A',
            })
        if len(extra) >= args.n_cases - len(named_cases):
            break

    all_cases = named_cases + extra
    logger.info(f"Total case studies: {len(all_cases)}")

    # ---- build dataset ----
    case_df = pd.DataFrame([
        {'drug_id': c['drug_id'], 'target_id': c['target_id'], 'label': 1}
        for c in all_cases
    ])
    esm2_dim = config['model']['target_encoder'].get('esm2_dim', 640)
    case_dataset = DTIDataset(
        case_df, drug_smiles=drug_smiles, target_sequences=target_sequences,
        esm2_cache_dir=config['data'].get('esm2_cache_dir', 'data/esm2_embeddings'),
        max_protein_len=config['data'].get('max_protein_len', 1200),
        use_domain_features=config['model']['target_encoder'].get('use_domain_features', True),
        esm2_dim=esm2_dim, task='classification',
    )

    # ---- Grad-CAM ----
    gradcam = GNNGradCAM(model, target_layer_name='drug_encoder.layers.2')

    # ---- analyse each case ----
    reports = []
    for idx, case in enumerate(all_cases):
        logger.info(f"\n{'='*60}")
        logger.info(f"Case {idx+1}/{len(all_cases)}: "
                     f"{case['drug_name']} - {case['target_name']} "
                     f"(Kd={case['affinity']:.1f} nM)")
        logger.info(f"{'='*60}")

        result = analyse_single_pair(model, case_dataset, idx, device, top_k)
        if result is None:
            logger.warning("  Skipped")
            continue

        logger.info(f"  Pred prob: {result['prediction_prob']:.4f}")
        logger.info(f"  Hotspots (>0.5): {len(result['hotspots'])}")

        sorted_profile = sorted(result['residue_profile'].items(),
                                key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"  Top-10 attended residues:")
        for rn, sc in sorted_profile:
            logger.info(f"    {rn}: {sc:.4f}")

        # Grad-CAM
        atom_imp = run_gradcam_analysis(gradcam, case_dataset, idx, device)
        substruct = {}
        smiles = drug_smiles.get(case['drug_id'], '')
        if atom_imp is not None and smiles:
            substruct = map_atom_importance_to_substructures(atom_imp, smiles)
            if substruct:
                logger.info(f"  Functional groups:")
                for g, s in sorted(substruct.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"    {g}: {s:.4f}")

        # ---- figures ----
        safe = f"{case['drug_name']}_{case['target_name']}".replace('(','').replace(')','')
        seq = target_sequences.get(case['target_id'], '')[:config['data'].get('max_protein_len', 1200)]
        nA = int(result['drug_mask'].sum())
        nR = int(result['protein_mask'].sum())

        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        atom_labels = ([f"{a.GetSymbol()}{a.GetIdx()}" for a in mol.GetAtoms()]
                       if mol else [f"A{i}" for i in range(nA)])
        res_labels = [f"{seq[i]}{i+1}" if i < len(seq) else f"?{i+1}" for i in range(nR)]

        imap = result['interaction_map'][:nA, :nR]

        try:
            plot_interaction_heatmap(
                imap, atom_labels[:nA], np.array(res_labels),
                title=f"{case['drug_name']} - {case['target_name']}",
                save_path=str(output_dir / 'heatmaps' / f'{safe}.png'),
            )
            logger.info("  Saved heatmap")
        except Exception as e:
            logger.warning(f"  Heatmap error: {e}")

        rscore = imap.sum(axis=0)
        mx = rscore.max()
        if mx > 0:
            rscore = rscore / mx
        try:
            plot_residue_attention_profile(
                rscore, seq[:nR],
                title=f"Attention: {case['drug_name']} - {case['target_name']}",
                save_path=str(output_dir / 'profiles' / f'{safe}.png'),
            )
            logger.info("  Saved attention profile")
        except Exception as e:
            logger.warning(f"  Profile error: {e}")

        if atom_imp is not None and smiles:
            try:
                plot_drug_atom_importance(
                    smiles, atom_imp,
                    title=f"Atom Importance: {case['drug_name']}",
                    save_path=str(output_dir / 'gradcam' / f'{safe}.png'),
                )
                logger.info("  Saved Grad-CAM figure")
            except Exception as e:
                logger.warning(f"  Grad-CAM fig error: {e}")

        reports.append({
            'drug_name': case['drug_name'], 'target_name': case['target_name'],
            'drug_id': case['drug_id'], 'target_id': case['target_id'],
            'affinity_nM': case['affinity'],
            'prediction_prob': result['prediction_prob'],
            'n_hotspots': len(result['hotspots']),
            'top_10_residues': sorted_profile,
            'functional_groups': substruct,
        })

    # ---- global stats ----
    logger.info(f"\n{'='*60}")
    logger.info("Computing global attention statistics...")
    all_pos = interactions[interactions['label'] == 1]
    pos_ds = DTIDataset(
        all_pos, drug_smiles=drug_smiles, target_sequences=target_sequences,
        esm2_cache_dir=config['data'].get('esm2_cache_dir', 'data/esm2_embeddings'),
        max_protein_len=config['data'].get('max_protein_len', 1200),
        use_domain_features=config['model']['target_encoder'].get('use_domain_features', True),
        esm2_dim=esm2_dim, task='classification',
    )
    nw = 0 if sys.platform == 'win32' else 4
    pos_loader = DataLoader(pos_ds, batch_size=32, shuffle=False,
                            collate_fn=collate_dti, num_workers=nw)
    gstats = global_attention_statistics(model, pos_loader, device, n_batches=50)
    for k, v in gstats.items():
        logger.info(f"  {k}: {v:.4f}")

    # ---- save JSON ----
    report_out = {
        'model_info': {
            'epoch': ckpt.get('epoch', '?'),
            'best_metric': float(ckpt.get('best_metric', 0)),
            'esm2_dim': esm2_dim,
            'params': sum(p.numel() for p in model.parameters()),
        },
        'global_stats': gstats,
        'case_studies': reports,
    }
    rpath = output_dir / 'interpretability_report.json'
    with open(rpath, 'w', encoding='utf-8') as f:
        json.dump(report_out, f, indent=2, default=str, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_dir}/")
    logger.info("Interpretability analysis complete.")


if __name__ == '__main__':
    main()
