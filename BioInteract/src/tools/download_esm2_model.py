"""Pre-download an ESM-2 model into the local cache."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description='Download an ESM-2 model into cache')
    parser.add_argument('--backend', choices=['fair-esm', 'huggingface'], default='fair-esm')
    parser.add_argument('--model', type=str, default='esm2_t30_150M_UR50D')
    args = parser.parse_args()

    if args.backend == 'fair-esm':
        try:
            import esm
        except ImportError as exc:
            raise RuntimeError('fair-esm is required. Install with: pip install fair-esm') from exc

        model, _ = esm.pretrained.load_model_and_alphabet(args.model)
        model.eval()
        print(f'Cached fair-esm model: {args.model}')
        return

    try:
        from transformers import AutoTokenizer, EsmModel
    except ImportError as exc:
        raise RuntimeError('transformers is required. Install with: pip install transformers') from exc

    model_name = args.model if '/' in args.model else f'facebook/{args.model}'
    AutoTokenizer.from_pretrained(model_name)
    EsmModel.from_pretrained(model_name)
    print(f'Cached Hugging Face model: {model_name}')


if __name__ == '__main__':
    main()