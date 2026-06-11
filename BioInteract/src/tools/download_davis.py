"""Compatibility wrapper for preparing the Davis dataset."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.prepare_data import prepare_davis
from src.utils.paths import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description='Download or prepare Davis dataset')
    parser.add_argument('--output_dir', type=str,
                        default=str(DATA_DIR / 'raw' / 'davis'))
    args = parser.parse_args()
    prepare_davis(args.output_dir)


if __name__ == '__main__':
    main()