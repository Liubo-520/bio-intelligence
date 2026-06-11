"""
run_split_final.py — Train + evaluate BioInteract (final) on a single split.

Final configuration for SCI paper:
    - Original v1 architecture (GINE GNN + ESM-2 + Cross-Attention)
    - No Morgan FP, No GraphAugmentation
    - BCE with pos_weight for class imbalance
    - AdamW optimizer with cosine warmup
    - Optimal F1 threshold from PR curve
    - Label smoothing 0.05

Usage:
    python -m src.experiments.run_split_final --split random
    python -m src.experiments.run_split_final --split cold_drug
    python -m src.experiments.run_split_final --split cold_target
"""
import warnings
warnings.filterwarnings('ignore')

import os, sys, yaml, json, time, torch, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.nn as nn

from src.data.dataset import DTIDataset, collate_dti
from src.data.split import get_split_fn
from src.models.biointeract import BioInteract
from src.utils.metrics import classification_metrics
from src.utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, DATA_DIR, LOGS_DIR, RESULTS_DIR, resolve_project_path


_LOG_FILE = None

def P(msg):
    print(msg, flush=True)
    if _LOG_FILE:
        _LOG_FILE.write(msg + '\n')
        _LOG_FILE.flush()
        import os as _os
        _os.fsync(_LOG_FILE.fileno())


def load_config(path='configs/default.yaml'):
    with open(resolve_project_path(path), 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def smooth_labels(labels, smoothing=0.05):
    """Label smoothing: pos→0.975, neg→0.025"""
    return labels * (1.0 - smoothing) + smoothing * 0.5


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        drug_batch = batch['drug_batch'].to(device)
        esm2 = batch['esm2_embedding'].to(device)
        phys = batch['physicochemical'].to(device)
        domain = batch['domain_labels'].to(device)
        prot_mask = batch['protein_mask'].to(device)
        labels = batch['label'].to(device)
        with autocast('cuda'):
            logits = model(drug_batch, esm2, phys, domain, prot_mask)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        n += 1
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()
    m = classification_metrics(labels, preds)  # auto optimal threshold
    m['loss'] = total_loss / max(n, 1)
    return m


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True,
                        choices=['random', 'cold_drug', 'cold_target'])
    parser.add_argument('--config', default=str(CONFIGS_DIR / 'default.yaml'))
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_type = args.split
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(cfg['training']['seed'])

    # Final: disable Morgan FP and GraphAugmentation
    cfg['model']['drug_encoder']['use_morgan_fp'] = False
    cfg['model']['drug_encoder']['drop_node'] = 0.0
    cfg['model']['drug_encoder']['drop_edge'] = 0.0

    global _LOG_FILE
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(LOGS_DIR / f'run_{split_type}.log', 'w')

    P(f"=== Split: {split_type} (FINAL: v1-arch + improved training) ===")

    # Data
    base = DATA_DIR / 'raw' / cfg['data']['dataset']
    interactions = pd.read_csv(base / 'interactions.csv')
    drug_df = pd.read_csv(base / 'drug_smiles.csv')
    target_df = pd.read_csv(base / 'target_sequences.csv')
    drug_smiles = dict(zip(drug_df['drug_id'], drug_df['smiles']))
    target_sequences = dict(zip(target_df['target_id'], target_df['sequence']))

    split_fn = get_split_fn(split_type)
    train_df, val_df, test_df = split_fn(
        interactions,
        val_ratio=cfg['data'].get('val_ratio', 0.1),
        test_ratio=cfg['data'].get('test_ratio', 0.2),
        seed=cfg['training']['seed'],
    )
    P(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    esm2_dim = cfg['model']['target_encoder'].get('esm2_dim', 640)
    common = dict(
        drug_smiles=drug_smiles, target_sequences=target_sequences,
        esm2_cache_dir=cfg['data'].get('esm2_cache_dir', 'data/esm2_embeddings'),
        max_protein_len=cfg['data'].get('max_protein_len', 1200),
        use_domain_features=cfg['model']['target_encoder'].get('use_domain_features', True),
        esm2_dim=esm2_dim, task='classification',
    )

    train_ds = DTIDataset(train_df, **common)
    val_ds = DTIDataset(val_df, **common)
    test_ds = DTIDataset(test_df, **common)

    bs = cfg['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_dti, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=collate_dti, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             collate_fn=collate_dti, num_workers=0)

    model = BioInteract(cfg['model']).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    P(f"  params={n_params:,}")
    P(f"  use_morgan_fp=False")

    # BCE with pos_weight (proven effective in v1)
    n_pos = train_df['label'].sum()
    n_neg = len(train_df) - n_pos
    pw = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    P(f"  loss=BCE(pos_weight={pw.item():.2f}) + label_smooth=0.05")

    lr = cfg['training']['lr']
    wd = cfg['training']['weight_decay']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    P(f"  optimizer=AdamW(lr={lr}, wd={wd})")

    # Cosine warmup scheduler
    warmup_epochs = cfg['training'].get('warmup_epochs', 5)
    total_epochs = cfg['training']['epochs']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    best_auroc = -1.0
    patience_counter = 0
    patience = cfg['training'].get('patience', 15)
    best_state = None
    label_smooth = 0.05

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batch = 0.0, 0
        for batch in train_loader:
            drug_batch = batch['drug_batch'].to(device)
            esm2 = batch['esm2_embedding'].to(device)
            phys = batch['physicochemical'].to(device)
            domain = batch['domain_labels'].to(device)
            prot_mask = batch['protein_mask'].to(device)
            labels = batch['label'].to(device)

            # label smoothing
            smooth_lab = smooth_labels(labels, label_smooth)

            with autocast('cuda'):
                logits = model(drug_batch, esm2, phys, domain, prot_mask)
                loss = criterion(logits, smooth_lab)

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

        val_m = evaluate(model, val_loader, criterion, device)
        auroc = val_m['AUROC']
        dt = time.time() - t0

        P(f"  E{epoch:03d} | loss={avg_loss:.4f} | val_auroc={auroc:.4f} | val_auprc={val_m['AUPRC']:.4f} | {dt:.1f}s")

        if auroc > best_auroc:
            best_auroc = auroc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                P(f"  Early stop at epoch {epoch}, best val_auroc={best_auroc:.4f}")
                break

    # Test
    model.load_state_dict(best_state)
    model.to(device)
    test_m = evaluate(model, test_loader, criterion, device)

    P(f"\n  TEST ({split_type}):")
    for k, v in test_m.items():
        if isinstance(v, float):
            P(f"    {k}: {v:.4f}")
        else:
            P(f"    {k}: {v}")

    # Save result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f'test_{split_type}.json'
    with open(out_file, 'w') as f:
        json.dump({k: round(float(v), 4) for k, v in test_m.items()}, f, indent=2)
    P(f"  Saved to {out_file}")

    # Save checkpoint
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_file = CHECKPOINTS_DIR / f'best_{split_type}.pt'
    torch.save({
        'model_state_dict': best_state,
        'config': cfg['model'],
        'split': split_type,
        'best_val_auroc': best_auroc,
        'version': 'final',
    }, ckpt_file)
    P(f"  Checkpoint: {ckpt_file}")
    P("DONE")
    if _LOG_FILE:
        _LOG_FILE.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        P(f"FATAL ERROR:\n{err_msg}")
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOGS_DIR / 'run_error.log', 'w') as ef:
            ef.write(err_msg)
        raise
