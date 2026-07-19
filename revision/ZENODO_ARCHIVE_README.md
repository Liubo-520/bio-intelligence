# Zenodo archive note

This v1.0.4 DOI package contains the executable source, raw Davis inputs,
released checkpoints, configurations, deterministic canonical prediction CSVs,
provenance hashes, strict-split artifacts, tests, and final revision documents.
It includes the figure-generation code and the restored Davis baseline-comparison
and component-removal figures used in the resubmission.
It also contains strict BindingDB external-validation code, aggregate manifests,
and an audit report; it deliberately excludes the raw third-party BindingDB
archive, derived strict pairs, external ESM cache, and pair-level predictions.
The Space-local checkpoint copy is also omitted because it is byte-identical to
the included canonical `BioInteract/checkpoints/best.pt` checkpoint
(SHA-256 `9680d73224ee90aa464b92df30b2b6297405825d3e3b58864d1c38b9726df065`).

The 852 MiB derived ESM cache is intentionally omitted from this permanent ZIP
to avoid duplicating it. Retrieve the cache from the matching GitHub LFS tag:

```bash
git clone https://github.com/Liubo-520/bio-intelligence.git
cd bio-intelligence
git checkout v1.0.4
git lfs pull
```

`revision/analysis/original_entity_split_metrics.json` records the required
cache digest and file count. The release archive can be rebuilt with:

```bash
python revision/build_release_archive.py --omit-esm-cache \
  --output biointeract-v1.0.4.zip
```
