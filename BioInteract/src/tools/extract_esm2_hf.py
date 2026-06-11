"""
extract_esm2_hf.py — Extract residue embeddings with Hugging Face ESM models.

Usage:
    python -m src.tools.extract_esm2_hf --dataset davis --model facebook/esm2_t30_150M_UR50D
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from src.tools.extract_esm2 import choose_device, load_target_sequences
from src.utils.paths import DATA_DIR, resolve_project_path


def main():
    parser = argparse.ArgumentParser(description='Extract ESM-2 embeddings with Transformers')
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--model', type=str, default='facebook/esm2_t30_150M_UR50D')
    parser.add_argument('--output_dir', type=str, default=str(DATA_DIR / 'esm2_embeddings'))
    parser.add_argument('--fasta', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError as exc:
        raise RuntimeError('transformers is required. Install with: pip install transformers') from exc

    device = choose_device(args.device)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_target_sequences(args.dataset, args.fasta)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model).to(device)
    model.eval()

    print(f'Loaded {args.model} on {device}')
    print(f'Extracting {len(sequences)} protein embeddings to {output_dir}')

    with torch.inference_mode():
        for protein_id, sequence in tqdm(sequences, desc='HF ESM-2 extraction'):
            save_path = output_dir / f'{protein_id}.pt'
            if args.skip_existing and save_path.exists():
                continue

            seq = sequence[:args.max_length]
            encoded = tokenizer(
                seq,
                return_tensors='pt',
                truncation=True,
                max_length=args.max_length + 2,
                add_special_tokens=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            residue_repr = outputs.last_hidden_state[0, 1:len(seq) + 1].cpu()
            torch.save(residue_repr, save_path)

    print('Extraction complete.')


if __name__ == '__main__':
    main()