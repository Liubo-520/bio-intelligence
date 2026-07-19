"""Create the prespecified strict BindingDB Kd-only external cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.bindingdb_external import curate_archive
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line contract for a fixed, traceable BindingDB snapshot."""
    parser = argparse.ArgumentParser(
        description="Curate a strict Kd-only, double-novel BindingDB external cohort"
    )
    parser.add_argument("--archive", required=True, help="BindingDB TSV ZIP snapshot")
    parser.add_argument("--out-dir", required=True, help="Local output directory")
    parser.add_argument("--source-url", required=True, help="Exact archive download URL")
    parser.add_argument("--source-version", required=True, help="Fixed source version/date")
    parser.add_argument(
        "--davis-dir",
        default="data/raw/davis",
        help="Released Davis raw-data directory used for identity exclusion",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Run strict curation and print the persisted manifest for audit logs."""
    args = build_parser().parse_args(argv)
    manifest = curate_archive(
        archive=resolve_project_path(args.archive),
        out_dir=resolve_project_path(args.out_dir),
        source_url=args.source_url,
        source_version=args.source_version,
        davis_dir=resolve_project_path(args.davis_dir),
        chunksize=args.chunksize,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
