# Zenodo archive note

This v1.0.1 DOI package contains the executable source, raw Davis inputs,
released checkpoints, configurations, deterministic canonical prediction CSVs,
provenance hashes, strict-split artifacts, tests, and final revision documents.

The 852 MiB derived ESM cache is intentionally omitted from this permanent ZIP
to avoid duplicating it. Retrieve the cache from the matching GitHub LFS tag:

```bash
git clone https://github.com/Liubo-520/bio-intelligence.git
cd bio-intelligence
git checkout v1.0.1
git lfs pull
```

`revision/analysis/original_entity_split_metrics.json` records the required
cache digest and file count. The release archive can be rebuilt with:

```bash
python revision/build_release_archive.py --omit-esm-cache \
  --output biointeract-v1.0.1.zip
```
