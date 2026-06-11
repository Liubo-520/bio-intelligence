"""
evaluate_splits.py — Evaluate BioInteract across multiple data split strategies.

Runs training + test evaluation for: random, cold_drug, cold_target
Saves results to results/split_comparison.json

Usage:
    python -m src.cli.evaluate_splits --config configs/default.yaml
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os, sys, yaml, json, torch, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

from src.data.dataset import DTIDataset, collate_dti
from src.data.split import get_split_fn
from src.models.biointeract import BioInteract
from src.utils.metrics import classification_metrics
from src.utils.logger import setup_logger
from src.utils.paths import CONFIGS_DIR, DATA_DIR, RESULTS_DIR, resolve_project_path


def load_config(path):
    with open(resolve_project_path(path), 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(dataset_name):
    base = DATA_DIR / 'raw' / dataset_name
    interactions = pd.read_csv(base / 'interactions.csv')
    drug_df = pd.read_csv(base / 'drug_smiles.csv')
    target_df = pd.read_csv(base / 'target_sequences.csv')
    drug_smiles = dict(zip(drug_df['drug_id'], drug_df['smiles']))
    target_sequences = dict(zip(target_df['target_id'], target_df['sequence']))
    return interactions, drug_smiles, target_sequences


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=True):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0, 0
    for batch in dataloader:
        drug_batch = batch['drug_batch'].to(device)
        esm2 = batch['esm2_embedding'].to(device)
        phys = batch['physicochemical'].to(device)
        domain = batch['domain_labels'].to(device)
        prot_mask = batch['protein_mask'].to(device)
        labels = batch['label'].to(device)
        with autocast(enabled=use_amp):
            preds = model(drug_batch, esm2, phys, domain, prot_mask)
            loss = criterion(preds, labels)
        total_loss += loss.item()
        n += 1
        all_preds.append(torch.sigmoid(preds).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    metrics = classification_metrics(labels, preds)
    metrics['loss'] = total_loss / max(n, 1)
    return metrics


def train_and_evaluate(config, split_type, device, logger):
    """Train from scratch with given split, return test metrics."""
    set_seed(config['training']['seed'])
    
    interactions, drug_smiles, target_sequences = load_data(config['data']['dataset'])
    
    split_fn = get_split_fn(split_type)
    train_df, val_df, test_df = split_fn(
        interactions,
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.2),
        seed=config['training']['seed'],
    )
    
    logger.info(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    esm2_dim = config['model']['target_encoder'].get('esm2_dim', 640)
    common = dict(
        drug_smiles=drug_smiles, target_sequences=target_sequences,
        esm2_cache_dir=config['data'].get('esm2_cache_dir', 'data/esm2_embeddings'),
        max_protein_len=config['data'].get('max_protein_len', 1200),
        use_domain_features=config['model']['target_encoder'].get('use_domain_features', True),
        esm2_dim=esm2_dim, task='classification',
    )
    
    train_ds = DTIDataset(train_df, **common)
    val_ds = DTIDataset(val_df, **common)
    test_ds = DTIDataset(test_df, **common)
    
    bs = config['training']['batch_size']
    nw = 0 if sys.platform == 'win32' else 4
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_dti, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=collate_dti, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             collate_fn=collate_dti, num_workers=nw)
    
    model = BioInteract(config['model']).to(device)
    
    # pos_weight
    n_pos = train_df['label'].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device) if n_pos > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['training']['lr'],
                                 weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs'], eta_min=1e-6)
    scaler = GradScaler(enabled=config['training'].get('amp', True))
    
    best_metric = -float('inf')
    patience_counter = 0
    patience = config['training'].get('patience', 15)
    best_state = None
    
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss, n_batch = 0, 0
        for batch in train_loader:
            drug_batch = batch['drug_batch'].to(device)
            esm2 = batch['esm2_embedding'].to(device)
            phys = batch['physicochemical'].to(device)
            domain = batch['domain_labels'].to(device)
            prot_mask = batch['protein_mask'].to(device)
            labels = batch['label'].to(device)
            
            with autocast(enabled=config['training'].get('amp', True)):
                preds = model(drug_batch, esm2, phys, domain, prot_mask)
                loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_batch += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batch, 1)
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        auroc = val_metrics['AUROC']
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val AUROC: {auroc:.4f}")
        
        if auroc > best_metric:
            best_metric = auroc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stop at epoch {epoch}")
                break
    
    # load best and evaluate test
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    return test_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(CONFIGS_DIR / 'default.yaml'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger = setup_logger('eval_splits')
    
    splits = ['random', 'cold_drug', 'cold_target']
    results = {}
    
    for split in splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating split: {split}")
        logger.info(f"{'='*60}")
        
        metrics = train_and_evaluate(config, split, device, logger)
        results[split] = {k: round(float(v), 4) for k, v in metrics.items()}
        
        logger.info(f"  TEST RESULTS ({split}):")
        for k, v in results[split].items():
            logger.info(f"    {k}: {v}")
    
    # save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'split_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # summary table
    logger.info(f"\n{'='*60}")
    logger.info("SPLIT COMPARISON SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Split':<15} {'AUROC':<10} {'AUPRC':<10} {'F1':<10}")
    logger.info("-" * 45)
    for split, m in results.items():
        logger.info(f"{split:<15} {m['AUROC']:<10} {m['AUPRC']:<10} {m['F1']:<10}")
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
