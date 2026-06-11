"""
Smoke test: verify the full BioInteract pipeline runs end-to-end
with synthetic data on GPU.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch_geometric.data import Batch

from src.data.mol_graph import smiles_to_graph, get_atom_feature_dim
from src.data.protein_feat import residue_physicochemical_features
from src.models.biointeract import BioInteract


def test_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. test molecule graph construction
    print("\n[1] Molecular graph construction...")
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',     # Aspirin
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F',  # Celecoxib
        'C1=CC=C(C=C1)C(=O)O',            # Benzoic acid
    ]
    
    graphs = []
    for smi in test_smiles:
        g = smiles_to_graph(smi)
        assert g is not None, f"Failed to parse: {smi}"
        graphs.append(g)
        print(f"  {smi[:30]:30s} -> {g.num_atoms} atoms, {g.edge_index.size(1)} edges, feat={g.x.shape}")
    
    atom_feat_dim = get_atom_feature_dim()
    print(f"  Atom feature dim: {atom_feat_dim}")
    
    # 2. test protein features
    print("\n[2] Protein feature construction...")
    test_sequences = [
        'MTEYKLVVVGAVGVGKSALTIQLIQNH' * 5,  # ~135 residues
        'MKWVTFISLLFLFSSAYS' * 8,             # ~144 residues
        'GIVEQCCTSICSLYQLEN' * 6,             # ~108 residues
    ]
    
    batch_size = len(test_sequences)
    max_len = max(len(s) for s in test_sequences)
    esm2_dim = 1280
    
    # simulate ESM-2 embeddings (normally pre-extracted)
    esm2_emb = torch.randn(batch_size, max_len, esm2_dim)
    physchem = torch.zeros(batch_size, max_len, 4)
    domain_labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    prot_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, seq in enumerate(test_sequences):
        L = len(seq)
        physchem[i, :L] = residue_physicochemical_features(seq)
        prot_mask[i, :L] = True
    
    print(f"  Batch: {batch_size} proteins, max_len={max_len}")
    print(f"  ESM-2 shape: {esm2_emb.shape}")
    print(f"  Physicochemical shape: {physchem.shape}")
    
    # 3. test model forward pass
    print("\n[3] Model forward pass...")
    config = {
        'drug_encoder': {
            'num_atom_features': atom_feat_dim,
            'edge_dim': 16,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.2,
            'jk': 'last',
        },
        'target_encoder': {
            'esm2_dim': 1280,
            'projection_dim': 256,
            'domain_embed_dim': 32,
            'num_domain_types': 50,
            'use_domain_features': True,
        },
        'interaction': {
            'num_heads': 8,
            'attn_dim': 256,
            'dropout': 0.1,
        },
        'predictor': {
            'hidden_dims': [256, 128],
            'dropout': 0.3,
            'task': 'classification',
        },
    }
    
    model = BioInteract(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    
    # prepare batch
    drug_batch = Batch.from_data_list(graphs).to(device)
    esm2_emb = esm2_emb.to(device)
    physchem = physchem.to(device)
    domain_labels = domain_labels.to(device)
    prot_mask = prot_mask.to(device)
    
    # forward with attention
    with torch.no_grad():
        prediction, attn_data = model(
            drug_batch, esm2_emb, physchem, domain_labels, prot_mask,
            return_attention=True
        )
    
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Predictions: {prediction.squeeze().cpu().numpy()}")
    print(f"  Interaction map shape: {attn_data['interaction_map'].shape}")
    print(f"  Drug mask shape: {attn_data['drug_mask'].shape}")
    
    # 4. test AMP training
    print("\n[4] Mixed precision training step...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()  # AMP-safe
    scaler = torch.amp.GradScaler('cuda')
    
    labels = torch.tensor([[1.0], [0.0], [1.0]], device=device)
    
    with torch.amp.autocast('cuda'):
        pred = model(drug_batch, esm2_emb, physchem, domain_labels, prot_mask)
        loss = criterion(pred, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    print(f"  Loss: {loss.item():.4f}")
    
    # 5. GPU memory usage
    if device == 'cuda':
        mem_allocated = torch.cuda.max_memory_allocated() / 1e6
        mem_reserved = torch.cuda.max_memory_reserved() / 1e6
        print(f"\n[5] GPU Memory:")
        print(f"  Allocated: {mem_allocated:.1f} MB")
        print(f"  Reserved:  {mem_reserved:.1f} MB")
        print(f"  Available: ~{17200 - mem_reserved:.0f} MB remaining")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! Pipeline is ready.")
    print("=" * 50)


if __name__ == '__main__':
    test_pipeline()
