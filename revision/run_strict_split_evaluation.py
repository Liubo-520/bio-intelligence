"""Train BioInteract under exact-sequence-grouped cold-start protocols."""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'BioInteract'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

from revision_analysis import sequence_group_cold_both_split, sequence_group_cold_target_split
from src.data.dataset import DTIDataset, collate_dti
from src.models.biointeract import BioInteract


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collect_predictions(model: BioInteract, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels, probabilities, losses = [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch['drug_batch'].to(device),
                batch['esm2_embedding'].to(device),
                batch['physicochemical'].to(device),
                batch['domain_labels'].to(device),
                batch['protein_mask'].to(device),
            )
            probability = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            truth = batch['label'].detach().cpu().numpy().ravel()
            labels.append(truth)
            probabilities.append(probability)
    return np.concatenate(labels), np.concatenate(probabilities), float(np.mean(losses)) if losses else float('nan')


def metrics_from_validation_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probabilities >= threshold).astype(int)
    tp = int(np.sum((prediction == 1) & (labels == 1)))
    fp = int(np.sum((prediction == 1) & (labels == 0)))
    fn = int(np.sum((prediction == 0) & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'AUROC': float(roc_auc_score(labels, probabilities)),
        'AUPRC': float(average_precision_score(labels, probabilities)),
        'F1': float(f1),
        'Precision': float(precision),
        'Recall': float(recall),
        'threshold_from_validation': float(threshold),
    }


def training_loss(logits: torch.Tensor, labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    smoothed = labels * 0.95 + 0.025
    return criterion(logits, smoothed)


def train_one_protocol(protocol: str, interactions: pd.DataFrame, targets: pd.DataFrame, drug_smiles: dict[str, str], sequences: dict[str, str], config: dict, output_dir: Path) -> dict[str, object]:
    seed = int(config['training']['seed'])
    set_seed(seed)
    if protocol == 'sequence_grouped_cold_target':
        train_df, val_df, test_df = sequence_group_cold_target_split(interactions, targets, seed=seed)
    elif protocol == 'sequence_grouped_cold_both':
        train_df, val_df, test_df = sequence_group_cold_both_split(interactions, targets, seed=seed)
    else:
        raise ValueError(protocol)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    common = {
        'drug_smiles': drug_smiles,
        'target_sequences': sequences,
        'esm2_cache_dir': str(PROJECT / config['data']['esm2_cache_dir']),
        'max_protein_len': int(config['data']['max_protein_len']),
        'use_domain_features': bool(config['model']['target_encoder']['use_domain_features']),
        'esm2_dim': int(config['model']['target_encoder']['esm2_dim']),
        'task': 'classification',
    }
    train_loader = DataLoader(DTIDataset(train_df, **common), batch_size=int(config['training']['batch_size']), shuffle=True, collate_fn=collate_dti, num_workers=0, pin_memory=device == 'cuda')
    val_loader = DataLoader(DTIDataset(val_df, **common), batch_size=int(config['training']['batch_size']), shuffle=False, collate_fn=collate_dti, num_workers=0)
    test_loader = DataLoader(DTIDataset(test_df, **common), batch_size=int(config['training']['batch_size']), shuffle=False, collate_fn=collate_dti, num_workers=0)

    model_config = copy.deepcopy(config['model'])
    model_config['drug_encoder']['use_morgan_fp'] = False
    model_config['drug_encoder']['drop_node'] = 0.0
    model_config['drug_encoder']['drop_edge'] = 0.0
    model = BioInteract(model_config).to(device)
    positive_weight = torch.tensor([(len(train_df) - float(train_df['label'].sum())) / float(train_df['label'].sum())], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['lr']), weight_decay=float(config['training']['weight_decay']))
    warmup_epochs = int(config['training']['warmup_epochs'])
    total_epochs = int(config['training']['epochs'])

    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        fraction = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = GradScaler('cuda', enabled=device == 'cuda')
    best_state, best_validation_auroc, waiting = None, -np.inf, 0
    patience = int(config['training']['patience'])
    history: list[dict[str, float]] = []

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            labels = batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=device == 'cuda'):
                logits = model(batch['drug_batch'].to(device), batch['esm2_embedding'].to(device), batch['physicochemical'].to(device), batch['domain_labels'].to(device), batch['protein_mask'].to(device))
                loss = training_loss(logits, labels, criterion)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation_labels, validation_probabilities, _ = collect_predictions(model, val_loader, device)
        validation_auroc = float(roc_auc_score(validation_labels, validation_probabilities))
        history.append({'epoch': epoch, 'training_loss': float(np.mean(epoch_losses)), 'validation_auroc': validation_auroc, 'validation_auprc': float(average_precision_score(validation_labels, validation_probabilities))})
        print(f'{protocol} epoch {epoch:03d}: loss={history[-1]["training_loss"]:.4f}; val_AUROC={validation_auroc:.4f}; val_AUPRC={history[-1]["validation_auprc"]:.4f}', flush=True)
        if validation_auroc > best_validation_auroc:
            best_validation_auroc = validation_auroc
            waiting = 0
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    validation_labels, validation_probabilities, _ = collect_predictions(model, val_loader, device)
    precision, recall, thresholds = precision_recall_curve(validation_labels, validation_probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    threshold = float(thresholds[int(np.argmax(f1))])
    test_labels, test_probabilities, _ = collect_predictions(model, test_loader, device)
    metrics = metrics_from_validation_threshold(test_labels, test_probabilities, threshold)
    torch.save({'model_state_dict': best_state, 'model_config': model_config, 'protocol': protocol, 'seed': seed, 'best_validation_auroc': best_validation_auroc}, output_dir / f'{protocol}.pt')
    return {
        'protocol': protocol,
        'seed': seed,
        'device': device,
        'split_sizes': {'train_pairs': len(train_df), 'validation_pairs': len(val_df), 'test_pairs': len(test_df), 'train_positives': int(train_df['label'].sum()), 'validation_positives': int(val_df['label'].sum()), 'test_positives': int(test_df['label'].sum())},
        'best_validation_auroc': float(best_validation_auroc),
        'epochs_completed': len(history),
        'metrics': metrics,
        'history': history,
    }


def main() -> None:
    output_dir = ROOT / 'revision' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    with (PROJECT / 'configs' / 'default.yaml').open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    data_root = PROJECT / 'data' / 'raw' / 'davis'
    interactions = pd.read_csv(data_root / 'interactions.csv')
    drugs = pd.read_csv(data_root / 'drug_smiles.csv')
    targets = pd.read_csv(data_root / 'target_sequences.csv')
    drug_smiles = dict(zip(drugs['drug_id'], drugs['smiles']))
    sequences = dict(zip(targets['target_id'], targets['sequence']))
    results = {
        'sequence_grouped_cold_target': train_one_protocol('sequence_grouped_cold_target', interactions, targets, drug_smiles, sequences, config, output_dir),
        'sequence_grouped_cold_both': train_one_protocol('sequence_grouped_cold_both', interactions, targets, drug_smiles, sequences, config, output_dir),
    }
    (output_dir / 'strict_split_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
