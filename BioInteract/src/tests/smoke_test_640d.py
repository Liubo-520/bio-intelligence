"""Quick smoke test for 640d features."""
import torch, yaml, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DTIDataset, collate_dti
from src.models.biointeract import BioInteract
from torch.utils.data import DataLoader
import pandas as pd
from src.utils.paths import CONFIGS_DIR, DATA_DIR

with open(CONFIGS_DIR / 'default.yaml') as f:
    config = yaml.safe_load(f)

base = DATA_DIR / 'raw' / 'davis'
interactions = pd.read_csv(base / 'interactions.csv')
drug_df = pd.read_csv(base / 'drug_smiles.csv')
target_df = pd.read_csv(base / 'target_sequences.csv')
drug_smiles = dict(zip(drug_df['drug_id'], drug_df['smiles']))
target_seqs = dict(zip(target_df['target_id'], target_df['sequence']))

test_df = interactions.head(16)
ds = DTIDataset(test_df, drug_smiles=drug_smiles, target_sequences=target_seqs,
                esm2_cache_dir=str(DATA_DIR / 'esm2_embeddings'), esm2_dim=640, max_protein_len=1200,
                use_domain_features=True, task='classification')
print(f'Dataset created, len={len(ds)}')

loader = DataLoader(ds, batch_size=4, collate_fn=collate_dti, shuffle=False)
batch = next(iter(loader))
print(f'Batch loaded:')
print(f'  esm2_embedding: {batch["esm2_embedding"].shape}')
print(f'  physicochemical: {batch["physicochemical"].shape}')
print(f'  domain_labels: {batch["domain_labels"].shape}')

model = BioInteract(config['model']).cuda()
drug_batch = batch['drug_batch'].to('cuda')
esm2 = batch['esm2_embedding'].to('cuda')
phys = batch['physicochemical'].to('cuda')
dom = batch['domain_labels'].to('cuda')
mask = batch['protein_mask'].to('cuda')

with torch.no_grad():
    out = model(drug_batch, esm2, phys, dom, mask)
print(f'Output shape: {out.shape}')
print('Smoke test PASSED!')
