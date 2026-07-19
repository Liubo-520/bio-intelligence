"""Release-archive contract for the strict BindingDB revision."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision"))

from build_release_archive import release_files  # noqa: E402


def test_lean_release_archive_never_requires_unversioned_runtime_logs():
    """A portable archive must build from tracked/reviewed release artifacts only."""
    files = release_files(include_esm_cache=False)
    relative_paths = {path.relative_to(ROOT).as_posix() for path in files}

    assert "BioInteract/src/analysis/bindingdb_external.py" in relative_paths
    assert "BioInteract/results/external_validation/bindingdb_external_manifest.json" in relative_paths
    assert not any(path.startswith("BioInteract/logs/") for path in relative_paths)
