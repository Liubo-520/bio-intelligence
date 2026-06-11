"""
evaluate.py — Standalone evaluation script.

Loads a trained checkpoint and evaluates on test set with all metrics.

Usage:
    python -m src.cli.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
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
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from src.data.dataset import DTIDataset, collate_dti
from src.data.split import get_split_fn
from src.models.biointeract import BioInteract
from src.utils.metrics import classification_metrics, regression_metrics
from src.utils.logger import setup_logger
from src.utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, RESULTS_DIR, resolve_project_path


def load_config(path):
    with open(resolve_project_path(path), 'r') as f:
        return yaml.safe_load(f)


def load_dataset_raw(dataset_name, data_dir='data/raw'):
    import pandas as pd
    base = resolve_project_path(data_dir) / dataset_name
    interactions = pd.read_csv(base / 'interactions.csv')
    drug_df = pd.read_csv(base / 'drug_smiles.csv')
    drug_smiles = dict(zip(drug_df['drug_id'], drug_df['smiles']))
    target_df = pd.read_csv(base / 'target_sequences.csv')
    target_sequences = dict(zip(target_df['target_id'], target_df['sequence']))
    return interactions, drug_smiles, target_sequences


@torch.no_grad()
def evaluate_model(model, dataloader, config, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_drug_ids = []
    all_target_ids = []

    for batch in dataloader:
        drug_batch = batch['drug_batch'].to(device)
        esm2 = batch['esm2_embedding'].to(device)
        phys = batch['physicochemical'].to(device)
        domain = batch['domain_labels'].to(device)
        prot_mask = batch['protein_mask'].to(device)
        labels = batch['label'].to(device)

        with autocast(enabled=config['training'].get('amp', True)):
            predictions = model(drug_batch, esm2, phys, domain, prot_mask)

        task = config['model']['predictor'].get('task', 'classification')
        if task == 'classification':
            predictions = torch.sigmoid(predictions)

        all_preds.append(predictions.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_drug_ids.extend(batch['drug_ids'])
        all_target_ids.extend(batch['target_ids'])

    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    task = config['model']['predictor'].get('task', 'classification')
    if task == 'classification':
        metrics = classification_metrics(all_labels, all_preds)
    else:
        metrics = regression_metrics(all_labels, all_preds)

    return metrics, all_preds, all_labels, all_drug_ids, all_target_ids


def main():
    parser = argparse.ArgumentParser(description='Evaluate BioInteract')
    parser.add_argument('--config', type=str, default=str(CONFIGS_DIR / 'default.yaml'))
    parser.add_argument('--checkpoint', type=str, default=str(CHECKPOINTS_DIR / 'best.pt'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save per-sample predictions to CSV')
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger = setup_logger('evaluate')

    # data
    dataset_name = config['data']['dataset']
    interactions, drug_smiles, target_sequences = load_dataset_raw(dataset_name)

    split_fn = get_split_fn(config['data']['split'])
    _, _, test_df = split_fn(
        interactions,
        val_ratio=config['data'].get('val_ratio', 0.1),
        test_ratio=config['data'].get('test_ratio', 0.2),
        seed=config['training']['seed'],
    )

    test_dataset = DTIDataset(
        test_df,
        drug_smiles=drug_smiles,
        target_sequences=target_sequences,
        esm2_cache_dir=config['data'].get('esm2_cache_dir', 'data/esm2_embeddings'),
        max_protein_len=config['data'].get('max_protein_len', 1200),
        use_domain_features=config['model']['target_encoder'].get('use_domain_features', True),
        task=config['model']['predictor'].get('task', 'classification'),
    )

    num_workers = 0 if sys.platform == 'win32' else 4
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                             shuffle=False, collate_fn=collate_dti, num_workers=num_workers)

    # model
    model = BioInteract(config['model']).to(device)
    checkpoint = torch.load(resolve_project_path(args.checkpoint), map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # evaluate
    metrics, preds, labels, drug_ids, target_ids = evaluate_model(
        model, test_loader, config, device
    )

    logger.info("=" * 60)
    logger.info(f"TEST RESULTS — {dataset_name} / {config['data']['split']}")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # save predictions
    if args.save_predictions:
        import pandas as pd
        pred_df = pd.DataFrame({
            'drug_id': drug_ids,
            'target_id': target_ids,
            'label': labels,
            'prediction': preds,
        })
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pred_path = RESULTS_DIR / 'test_predictions.csv'
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to {pred_path}")


if __name__ == '__main__':
    main()
