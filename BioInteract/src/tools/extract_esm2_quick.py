"""
extract_esm2_quick.py — Quick wrapper around ESM-2 extraction.

Uses the same 640d default model as the main pipeline, but skips existing
files by default and is intended for incremental extraction.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.extract_esm2 import build_parser, run_extraction


def main():
    parser = build_parser(default_model='esm2_t30_150M_UR50D')
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()
    print('Quick mode: only missing embeddings will be generated.')
    run_extraction(args)


if __name__ == '__main__':
    main()