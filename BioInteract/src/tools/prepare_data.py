"""
prepare_data.py — Download and preprocess DTI datasets.

Handles Davis and KIBA datasets:
  1. Download raw data files via TDC when available
  2. Parse into standardised CSV format
  3. Generate FASTA files for ESM-2 extraction

Usage:
    python -m src.tools.prepare_data --dataset davis
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.utils.paths import DATA_DIR, resolve_project_path


def _resolve_raw_dir(raw_dir: str | None, dataset_name: str) -> Path:
    if raw_dir:
        return resolve_project_path(raw_dir)
    return DATA_DIR / 'raw' / dataset_name


def prepare_davis(raw_dir: str | None = None):
    """
    Prepare the Davis kinase binding affinity dataset.

    Davis dataset contains 442 drugs × 379 kinase targets with Kd values.
    Source: Davis et al., Nature Biotechnology 2011.
    """
    raw_path = _resolve_raw_dir(raw_dir, 'davis')
    raw_path.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Davis Dataset Preparation')
    print('=' * 60)
    print()
    print('Please download the following files and place them in:')
    print(f'  {raw_path}/')
    print()
    print('Required files:')
    print('  1. Y (affinity matrix): from DeepDTA GitHub or TDC')
    print('  2. ligands_can.txt: canonical SMILES for each drug')
    print('  3. proteins.txt: amino acid sequences for each target')
    print()
    print('Recommended source: Therapeutics Data Commons (TDC)')
    print('  pip install PyTDC')
    print("  from tdc.multi_pred import DTI")
    print("  data = DTI(name='DAVIS')")
    print('  df = data.get_data()')
    print()

    tdc_path = raw_path / 'tdc_davis.csv'
    if tdc_path.exists():
        print(f'Found TDC data at {tdc_path}')
        df = pd.read_csv(tdc_path)
        _process_tdc_format(df, raw_path)
        return

    try:
        from tdc.multi_pred import DTI

        print('Downloading Davis dataset via TDC...')
        data = DTI(name='DAVIS')
        df = data.get_data()
        df.to_csv(tdc_path, index=False)
        _process_tdc_format(df, raw_path)
        print('Davis dataset prepared successfully!')
    except ImportError:
        print('\nTDC not installed. Install with: pip install PyTDC')
        print('Or manually download and run this script again.')
        _create_placeholder(raw_path)


def _process_tdc_format(df: pd.DataFrame, output_dir: Path):
    """Process TDC-format DTI data into the project standard format."""
    if 'Drug_ID' in df.columns:
        df = df.rename(columns={
            'Drug_ID': 'drug_id',
            'Drug': 'smiles',
            'Target_ID': 'target_id',
            'Target': 'sequence',
            'Y': 'label',
        })

    interactions = df[['drug_id', 'target_id', 'label']].copy()
    interactions['affinity'] = interactions['label']
    interactions['label_binary'] = (interactions['label'] < 30).astype(int)
    interactions['pKd'] = -np.log10(interactions['label'] / 1e9 + 1e-10)
    interactions.to_csv(output_dir / 'interactions.csv', index=False)

    drug_smiles = df[['drug_id', 'smiles']].drop_duplicates('drug_id')
    drug_smiles.to_csv(output_dir / 'drug_smiles.csv', index=False)

    target_seqs = df[['target_id', 'sequence']].drop_duplicates('target_id')
    target_seqs.to_csv(output_dir / 'target_sequences.csv', index=False)

    fasta_path = output_dir / 'targets.fasta'
    with open(fasta_path, 'w', encoding='utf-8') as handle:
        for _, row in target_seqs.iterrows():
            handle.write(f">{row['target_id']}\n")
            seq = row['sequence']
            for index in range(0, len(seq), 80):
                handle.write(seq[index:index + 80] + '\n')

    print('\nDataset Summary:')
    print(f'  Interactions: {len(interactions):,}')
    print(f"  Unique drugs: {drug_smiles['drug_id'].nunique()}")
    print(f"  Unique targets: {target_seqs['target_id'].nunique()}")
    print(f"  Positive pairs (Kd < 30nM): {interactions['label_binary'].sum():,}")
    print(f'  FASTA file: {fasta_path}')
    print(f'\nFiles saved to: {output_dir}/')


def _create_placeholder(output_dir: Path):
    """Create placeholder files with expected schema."""
    pd.DataFrame({
        'drug_id': ['D001'],
        'target_id': ['T001'],
        'label': [0],
    }).to_csv(output_dir / 'interactions.csv', index=False)

    pd.DataFrame({
        'drug_id': ['D001'],
        'smiles': ['CC(=O)OC1=CC=CC=C1C(O)=O'],
    }).to_csv(output_dir / 'drug_smiles.csv', index=False)

    pd.DataFrame({
        'target_id': ['T001'],
        'sequence': ['MTEYKLVVVGAVGVGKSAL'],
    }).to_csv(output_dir / 'target_sequences.csv', index=False)

    print(f'\nPlaceholder files created in {output_dir}/')
    print('Replace with actual data before training.')


def prepare_kiba(raw_dir: str | None = None):
    """Prepare the KIBA dataset via TDC if available."""
    raw_path = _resolve_raw_dir(raw_dir, 'kiba')
    raw_path.mkdir(parents=True, exist_ok=True)

    try:
        from tdc.multi_pred import DTI

        print('Downloading KIBA dataset via TDC...')
        data = DTI(name='KIBA')
        df = data.get_data()
        tdc_path = raw_path / 'tdc_kiba.csv'
        df.to_csv(tdc_path, index=False)
        _process_tdc_format(df, raw_path)
        print('KIBA dataset prepared successfully!')
    except ImportError:
        print('TDC not installed. Install with: pip install PyTDC')
        _create_placeholder(raw_path)


def main():
    parser = argparse.ArgumentParser(description='Prepare DTI datasets')
    parser.add_argument('--dataset', type=str, default='davis',
                        choices=['davis', 'kiba', 'all'])
    parser.add_argument('--raw_dir', type=str, default=None,
                        help='Optional raw dataset directory override')
    args = parser.parse_args()

    if args.dataset in {'davis', 'all'}:
        prepare_davis(args.raw_dir if args.dataset == 'davis' else None)

    if args.dataset in {'kiba', 'all'}:
        prepare_kiba(args.raw_dir if args.dataset == 'kiba' else None)


if __name__ == '__main__':
    main()