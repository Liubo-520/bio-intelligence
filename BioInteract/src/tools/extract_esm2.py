"""
extract_esm2.py — Extract residue-level ESM-2 embeddings.

Saves one .pt tensor per target protein under data/esm2_embeddings.

Usage:
    python -m src.tools.extract_esm2 --dataset davis --model esm2_t30_150M_UR50D
"""
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from src.utils.paths import DATA_DIR, resolve_project_path


def parse_fasta(fasta_path: str | Path) -> list[tuple[str, str]]:
    path = resolve_project_path(fasta_path)
    records: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []

    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(chunks)))
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        records.append((header, ''.join(chunks)))

    return records


def load_target_sequences(dataset_name: str, fasta_path: str | None = None) -> list[tuple[str, str]]:
    if fasta_path:
        return parse_fasta(fasta_path)

    csv_path = DATA_DIR / 'raw' / dataset_name / 'target_sequences.csv'
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        return [(str(row.target_id), str(row.sequence)) for row in frame.itertuples(index=False)]

    fallback_fasta = DATA_DIR / 'raw' / dataset_name / 'targets.fasta'
    if fallback_fasta.exists():
        return parse_fasta(fallback_fasta)

    raise FileNotFoundError(
        f'No target_sequences.csv or targets.fasta found for dataset {dataset_name!r}'
    )


def infer_repr_layer(model_name: str, model) -> int:
    if hasattr(model, 'num_layers'):
        return int(model.num_layers)
    match = re.search(r'_t(\d+)_', model_name)
    return int(match.group(1)) if match else 33


def choose_device(device_arg: str) -> str:
    if device_arg != 'auto':
        return device_arg
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def build_parser(default_model: str = 'esm2_t30_150M_UR50D') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extract ESM-2 embeddings')
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--model', type=str, default=default_model)
    parser.add_argument('--output_dir', type=str, default=str(DATA_DIR / 'esm2_embeddings'))
    parser.add_argument('--fasta', type=str, default=None,
                        help='Optional FASTA override; otherwise reads target_sequences.csv')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--skip_existing', action='store_true')
    return parser


def run_extraction(args: argparse.Namespace):
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError('fair-esm is required. Install with: pip install fair-esm') from exc

    device = choose_device(args.device)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_target_sequences(args.dataset, args.fasta)
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = infer_repr_layer(args.model, model)

    print(f'Loaded {args.model} on {device}; repr_layer={repr_layer}')
    print(f'Extracting {len(sequences)} protein embeddings to {output_dir}')

    with torch.inference_mode():
        for protein_id, sequence in tqdm(sequences, desc='ESM-2 extraction'):
            save_path = output_dir / f'{protein_id}.pt'
            if args.skip_existing and save_path.exists():
                continue

            seq = sequence[:args.max_length]
            _, _, tokens = batch_converter([(protein_id, seq)])
            tokens = tokens.to(device)
            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            residue_repr = results['representations'][repr_layer][0, 1:len(seq) + 1].cpu()
            torch.save(residue_repr, save_path)

    print('Extraction complete.')


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_extraction(args)


if __name__ == '__main__':
    main()