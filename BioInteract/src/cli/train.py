"""
train.py — Main training script for BioInteract.

Handles:
  - Config loading
  - Dataset preparation with cold-start splits
  - Model training with AMP + gradient accumulation
  - Validation and early stopping
  - Checkpoint saving
  - TensorBoard logging

Usage:
    python -m src.cli.train --config configs/default.yaml
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from src.data.dataset import DTIDataset, collate_dti
from src.data.split import get_split_fn
from src.models.biointeract import BioInteract
from src.utils.metrics import classification_metrics, regression_metrics
from src.utils.logger import setup_logger, ExperimentTracker
from src.utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, RUNS_DIR, resolve_project_path


def load_config(config_path: str) -> dict:
    with open(resolve_project_path(config_path), 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_dataset_raw(dataset_name: str, data_dir: str = 'data/raw'):
    """
    Load raw dataset.
    
    Expected files in data/raw/{dataset_name}/:
      - interactions.csv: columns [drug_id, target_id, label]
      - drug_smiles.csv: columns [drug_id, smiles]
      - target_sequences.csv: columns [target_id, sequence]
    """
    import pandas as pd
    
    base = resolve_project_path(data_dir) / dataset_name
    
    interactions = pd.read_csv(base / 'interactions.csv')
    
    drug_df = pd.read_csv(base / 'drug_smiles.csv')
    drug_smiles = dict(zip(drug_df['drug_id'], drug_df['smiles']))
    
    target_df = pd.read_csv(base / 'target_sequences.csv')
    target_sequences = dict(zip(target_df['target_id'], target_df['sequence']))
    
    return interactions, drug_smiles, target_sequences


def train_epoch(model, dataloader, criterion, optimizer, scaler,
                config, device, epoch, logger):
    """Single training epoch with AMP and gradient accumulation."""
    model.train()
    total_loss = 0
    n_batches = 0
    accum_steps = config['training'].get('gradient_accumulation', 1)
    
    optimizer.zero_grad()
    
    total_batches = len(dataloader)
    for step, batch in enumerate(dataloader):
        drug_batch = batch['drug_batch'].to(device)
        esm2 = batch['esm2_embedding'].to(device)
        phys = batch['physicochemical'].to(device)
        domain = batch['domain_labels'].to(device)
        prot_mask = batch['protein_mask'].to(device)
        labels = batch['label'].to(device)
        
        with autocast(enabled=config['training'].get('amp', True)):
            predictions = model(drug_batch, esm2, phys, domain, prot_mask)
            loss = criterion(predictions, labels)
            loss = loss / accum_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
        n_batches += 1
        
        if (step + 1) % 100 == 0 or step == total_batches - 1:
            logger.info(f"  Epoch {epoch} [{step+1}/{total_batches}] loss={total_loss/n_batches:.4f}")
    
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, config, device):
    """Evaluate model on validation/test set."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        drug_batch = batch['drug_batch'].to(device)
        esm2 = batch['esm2_embedding'].to(device)
        phys = batch['physicochemical'].to(device)
        domain = batch['domain_labels'].to(device)
        prot_mask = batch['protein_mask'].to(device)
        labels = batch['label'].to(device)
        
        with autocast(enabled=config['training'].get('amp', True)):
            predictions = model(drug_batch, esm2, phys, domain, prot_mask)
            loss = criterion(predictions, labels)
        
        total_loss += loss.item()
        n_batches += 1
        
        # apply sigmoid for classification metrics
        if config['model']['predictor'].get('task', 'classification') == 'classification':
            predictions = torch.sigmoid(predictions)
        all_preds.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()
    
    avg_loss = total_loss / max(n_batches, 1)
    
    task = config['model']['predictor'].get('task', 'classification')
    if task == 'classification':
        metrics = classification_metrics(all_labels, all_preds)
    else:
        metrics = regression_metrics(all_labels, all_preds)
    
    metrics['loss'] = avg_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train BioInteract model')
    parser.add_argument('--config', type=str, default=str(CONFIGS_DIR / 'default.yaml'))
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    set_seed(config['training']['seed'])
    
    logger = setup_logger('train')
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")
    
    # ---- data ----
    dataset_name = config['data']['dataset']
    logger.info(f"Loading dataset: {dataset_name}")
    
    interactions, drug_smiles, target_sequences = load_dataset_raw(dataset_name)
    
    # cold-start split
    split_fn = get_split_fn(config['data']['split'])
    train_df, val_df, test_df = split_fn(
        interactions,
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.2),
        seed=config['training']['seed'],
    )
    
    logger.info(f"Split ({config['data']['split']}): "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # datasets
    common_kwargs = dict(
        drug_smiles=drug_smiles,
        target_sequences=target_sequences,
        esm2_cache_dir=config['data'].get('esm2_cache_dir', 'data/esm2_embeddings'),
        max_protein_len=config['data'].get('max_protein_len', 1200),
        use_domain_features=config['model']['target_encoder'].get('use_domain_features', True),
        esm2_dim=config['model']['target_encoder'].get('esm2_dim', 640),
        task=config['model']['predictor'].get('task', 'classification'),
    )
    
    train_dataset = DTIDataset(train_df, **common_kwargs)
    val_dataset = DTIDataset(val_df, **common_kwargs)
    test_dataset = DTIDataset(test_df, **common_kwargs)
    
    bs = config['training']['batch_size']
    # num_workers=0 on Windows to avoid multiprocessing deadlocks
    num_workers = 0 if sys.platform == 'win32' else 4
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              collate_fn=collate_dti, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                            collate_fn=collate_dti, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                             collate_fn=collate_dti, num_workers=num_workers)
    
    # ---- model ----
    model = BioInteract(config['model']).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")
    
    # ---- training setup ----
    task = config['model']['predictor'].get('task', 'classification')
    if task == 'classification':
        # Handle class imbalance with pos_weight
        n_pos = train_df['label'].sum()
        n_neg = len(train_df) - n_pos
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], device=device)
            logger.info(f"Class balance: pos={n_pos}, neg={n_neg}, "
                        f"pos_weight={pos_weight.item():.2f}")
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # cosine annealing scheduler
    scheduler = None
    sched_type = config['training'].get('scheduler', 'cosine')
    if sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=1e-6
        )
    
    scaler = GradScaler(enabled=config['training'].get('amp', True))
    
    # tensorboard
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(RUNS_DIR / 'biointeract'))
    
    # ---- training loop ----
    best_metric = -float('inf')
    patience_counter = 0
    patience = config['training'].get('patience', 15)
    
    # primary metric for early stopping
    primary_metric = 'AUROC' if task == 'classification' else 'CI'
    
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            config, device, epoch, logger
        )
        
        val_metrics = evaluate(model, val_loader, criterion, config, device)
        
        if scheduler is not None:
            scheduler.step()
        
        # logging
        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val {primary_metric}: {val_metrics[primary_metric]:.4f}"
        )
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        for k, v in val_metrics.items():
            if k != 'loss':
                writer.add_scalar(f'Val/{k}', v, epoch)
        
        # early stopping
        current_metric = val_metrics[primary_metric]
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'config': config,
            }, CHECKPOINTS_DIR / 'best.pt')
            logger.info(f"  → New best {primary_metric}: {best_metric:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # ---- test evaluation ----
    logger.info("Loading best model for test evaluation...")
    checkpoint = torch.load(CHECKPOINTS_DIR / 'best.pt', map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, config, device)
    
    logger.info("=" * 60)
    logger.info(f"TEST RESULTS ({config['data']['split']} split):")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 60)
    
    # save results
    tracker = ExperimentTracker()
    tracker.log_experiment(
        config=config,
        metrics=test_metrics,
        split_type=config['data']['split'],
        dataset=dataset_name,
    )
    
    writer.close()
    logger.info("Training complete.")


if __name__ == '__main__':
    main()
