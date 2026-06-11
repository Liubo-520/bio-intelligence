"""
dataset.py — PyTorch dataset for drug-target interaction prediction.
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .mol_graph import smiles_to_graph, smiles_to_morgan
from .protein_feat import ProteinFeatureBuilder


class DTIDataset(Dataset):
    """
    Drug-Target Interaction dataset.
    
    Each sample consists of:
      - A molecular graph (PyG Data) for the drug
      - Residue-level features for the target protein
      - A label (binary or continuous affinity)
    """
    
    def __init__(self,
                 data_df: pd.DataFrame,
                 drug_smiles: dict,         # drug_id -> SMILES
                 target_sequences: dict,     # target_id -> amino acid sequence
                 esm2_cache_dir: str = 'data/esm2_embeddings',
                 domain_annotation_dir: str = 'data/domain_annotations',
                 max_protein_len: int = 1200,
                 use_domain_features: bool = True,
                 esm2_dim: int = 640,
                 task: str = 'classification',
                 label_col: str = 'label',
                 drug_col: str = 'drug_id',
                 target_col: str = 'target_id'):
        """
        Args:
            data_df: DataFrame with columns [drug_col, target_col, label_col]
            drug_smiles: mapping from drug ID to SMILES string
            target_sequences: mapping from target ID to protein sequence
            task: 'classification' or 'regression'
        """
        self.data = data_df.reset_index(drop=True)
        self.drug_smiles = drug_smiles
        self.target_sequences = target_sequences
        self.task = task
        self.label_col = label_col
        self.drug_col = drug_col
        self.target_col = target_col
        
        self.protein_builder = ProteinFeatureBuilder(
            esm2_cache_dir=esm2_cache_dir,
            domain_annotation_dir=domain_annotation_dir,
            max_protein_len=max_protein_len,
            use_domain_features=use_domain_features,
            esm2_dim=esm2_dim,
        )
        
        self.morgan_nbits = 1024

        # pre-compute molecular graphs (they're small, fits in RAM)
        self._drug_graph_cache = {}
        self._morgan_cache = {}
        self._protein_feat_cache = {}
        
        self._precompute_graphs()
    
    def _precompute_graphs(self):
        """Cache all molecular graphs and Morgan fingerprints."""
        unique_drugs = self.data[self.drug_col].unique()
        failed = 0
        for drug_id in unique_drugs:
            smiles = self.drug_smiles.get(drug_id, '')
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self._drug_graph_cache[drug_id] = graph
            else:
                failed += 1
            # Morgan fingerprint (separate from graph, always try)
            fp = smiles_to_morgan(smiles, n_bits=self.morgan_nbits)
            if fp is not None:
                self._morgan_cache[drug_id] = fp
        
        if failed > 0:
            print(f"Warning: {failed}/{len(unique_drugs)} drugs failed "
                  f"SMILES parsing and will be skipped.")
    
    def _get_protein_features(self, target_id: str) -> dict:
        """Get or cache protein features."""
        if target_id not in self._protein_feat_cache:
            sequence = self.target_sequences.get(target_id, '')
            self._protein_feat_cache[target_id] = \
                self.protein_builder.build(target_id, sequence)
        return self._protein_feat_cache[target_id]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_id = row[self.drug_col]
        target_id = row[self.target_col]
        
        # drug graph
        drug_graph = self._drug_graph_cache.get(drug_id, None)
        if drug_graph is None:
            # return a dummy minimal graph if SMILES parsing failed
            drug_graph = Data(
                x=torch.zeros(1, 52),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 16),
                smiles='',
                num_atoms=1,
            )
        
        # Morgan fingerprint
        morgan_fp = self._morgan_cache.get(drug_id,
                        torch.zeros(self.morgan_nbits))

        # protein features
        prot_feat = self._get_protein_features(target_id)
        
        # label
        label_val = row[self.label_col]
        if self.task == 'classification':
            label = torch.tensor([float(label_val)], dtype=torch.float)
        else:
            label = torch.tensor([float(label_val)], dtype=torch.float)
        
        return {
            'drug_graph': drug_graph,
            'morgan_fp': morgan_fp,
            'esm2_embedding': prot_feat['esm2_embedding'],
            'physicochemical': prot_feat['physicochemical'],
            'domain_labels': prot_feat['domain_labels'],
            'protein_length': prot_feat['sequence_length'],
            'label': label,
            'drug_id': drug_id,
            'target_id': target_id,
            'sequence': self.target_sequences.get(target_id, '')[:self.protein_builder.max_len],
        }


def collate_dti(batch: list) -> dict:
    """
    Custom collate function for DTI batches.
    
    Drug graphs are batched via PyG's Batch.from_data_list.
    Protein features are padded to the max length in the batch.
    """
    from torch_geometric.data import Batch
    
    drug_graphs = [item['drug_graph'] for item in batch]
    drug_batch = Batch.from_data_list(drug_graphs)
    
    # protein padding
    max_prot_len = max(item['protein_length'] for item in batch)
    batch_size = len(batch)
    esm2_dim = batch[0]['esm2_embedding'].size(-1)
    phys_dim = batch[0]['physicochemical'].size(-1)
    
    esm2_padded = torch.zeros(batch_size, max_prot_len, esm2_dim)
    phys_padded = torch.zeros(batch_size, max_prot_len, phys_dim)
    domain_padded = torch.zeros(batch_size, max_prot_len, dtype=torch.long)
    prot_mask = torch.zeros(batch_size, max_prot_len, dtype=torch.bool)
    
    labels = []
    drug_ids = []
    target_ids = []
    
    for i, item in enumerate(batch):
        L = item['protein_length']
        esm2_padded[i, :L] = item['esm2_embedding']
        phys_padded[i, :L] = item['physicochemical']
        domain_padded[i, :L] = item['domain_labels']
        prot_mask[i, :L] = True
        labels.append(item['label'])
        drug_ids.append(item['drug_id'])
        target_ids.append(item['target_id'])
    
    # Morgan fingerprints
    morgan_fps = torch.stack([item['morgan_fp'] for item in batch])  # (B, 1024)

    return {
        'drug_batch': drug_batch,
        'morgan_fp': morgan_fps,
        'esm2_embedding': esm2_padded,
        'physicochemical': phys_padded,
        'domain_labels': domain_padded,
        'protein_mask': prot_mask,
        'label': torch.stack(labels),
        'drug_ids': drug_ids,
        'target_ids': target_ids,
    }
