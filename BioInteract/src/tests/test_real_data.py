"""Quick test: load one batch of real Davis data and do a forward pass."""
import sys
import yaml
import torch
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DTIDataset, collate_dti
from src.data.split import get_split_fn
from src.models.biointeract import BioInteract
from src.utils.paths import CONFIGS_DIR, DATA_DIR

print("Loading config...", flush=True)
with open(CONFIGS_DIR / 'default.yaml') as f:
    config = yaml.safe_load(f)

print("Loading data...", flush=True)
base = DATA_DIR / 'raw' / 'davis'
interactions = pd.read_csv(base / 'interactions.csv')
drug_df = pd.read_csv(base / 'drug_smiles.csv')
drug_smiles = dict(zip(drug_df['drug_id'], drug_df['smiles']))
target_df = pd.read_csv(base / 'target_sequences.csv')
target_sequences = dict(zip(target_df['target_id'], target_df['sequence']))

print(f"Interactions: {len(interactions)}, Drugs: {len(drug_smiles)}, Targets: {len(target_sequences)}", flush=True)

# Just take first 200 samples for quick test
small_df = interactions.head(200)

print("Building dataset...", flush=True)
dataset = DTIDataset(
    small_df,
    drug_smiles=drug_smiles,
    target_sequences=target_sequences,
    esm2_cache_dir=str(DATA_DIR / 'esm2_embeddings'),
    max_protein_len=1200,
    use_domain_features=True,
    task='classification',
)
print(f"Dataset size: {len(dataset)}", flush=True)

print("Getting single sample...", flush=True)
sample = dataset[0]
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}", flush=True)
    else:
        print(f"  {k}: {v}", flush=True)

print("\nCollating batch of 4...", flush=True)
batch = collate_dti([dataset[i] for i in range(4)])
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}", flush=True)
    elif hasattr(v, 'num_graphs'):
        print(f"  {k}: PyG Batch with {v.num_graphs} graphs", flush=True)
    else:
        print(f"  {k}: {type(v).__name__}", flush=True)

print("\nBuilding model...", flush=True)
model = BioInteract(config['model']).cuda()
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

print("\nForward pass...", flush=True)
device = 'cuda'
drug_batch = batch['drug_batch'].to(device)
esm2 = batch['esm2_embedding'].to(device)
phys = batch['physicochemical'].to(device)
domain = batch['domain_labels'].to(device)
prot_mask = batch['protein_mask'].to(device)
labels = batch['label'].to(device)

with torch.no_grad():
    pred = model(drug_batch, esm2, phys, domain, prot_mask)
    print(f"Predictions: {pred.flatten().tolist()}", flush=True)
    print(f"Labels: {labels.flatten().tolist()}", flush=True)

# Test loss
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(pred, labels)
print(f"Loss: {loss.item():.4f}", flush=True)

print("\nALL PASSED!", flush=True)
