# Strict BindingDB external-validation report

## Protocol

This report records the pre-specified external test in
`docs/superpowers/specs/2026-07-19-bindingdb-external-validation-design.md`.
The source was the official BindingDB curated-articles TSV snapshot
`BindingDB_BindingDB_Articles_202607_tsv.zip`, downloaded from
`https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202607_tsv.zip`
on 2026-07-19 UTC. The raw ZIP is 18,114,757 bytes and has SHA-256
`d2584d1519318d00ab5f46289da5ab3549affe732d598a5072f8777b6b3b5262`.

Only a finite, exact numerical Kd in nM was retained. Ki, IC50, EC50,
inequality/censored Kd values, multi-chain targets, invalid structures, invalid
or >1,200-residue sequences, and duplicate pairs whose valid replicate Kd values
fell on opposite sides of 30 nM were excluded. High affinity is Kd <30 nM.
No artificial negatives were sampled.

External ligand identity was canonical isomeric SMILES; target identity was the
full normalized amino-acid sequence. Every external ligand whose canonical SMILES
occurred in the released Davis data and every external target whose full sequence
occurred in Davis was removed before pair reconciliation. Thus the resulting
cohort is strictly double-novel by exact ligand and sequence identity relative to
the entire released Davis collection.

## Curation accounting

| Item | Count |
|---|---:|
| BindingDB source rows | 93,712 |
| Censored/non-exact Kd excluded | 82,988 |
| Multi-chain targets excluded | 8,261 |
| Invalid sequences excluded | 139 |
| Invalid SMILES excluded | 8 |
| Davis ligand overlaps excluded | 64 |
| Davis sequence overlaps excluded | 94 |
| Eligible measurements before reconciliation | 2,158 |
| Threshold-conflicting replicate pairs excluded | 15 pairs / 44 measurement rows |
| Threshold-consistent measurement rows | 2,114 |
| Retained pairs with two or more consistent measurements | 143 |
| Repeated measurement rows collapsed by within-pair median $K_d$ | 182 |
| Final external pairs | 1,932 |
| High-affinity pairs (Kd <30 nM) | 516 |
| Low-affinity pairs (Kd >=30 nM) | 1,416 |
| Unique ligands | 1,195 |
| Unique targets | 396 |

The 15 threshold-conflicting entries are a emph{pair-level} count, not a row
count. Thus the correct closure is $2{,}158-44=2{,}114$ consistent measurement
rows, followed by $2{,}114-182=1{,}932$ final ligand--target pairs. For each
threshold-consistent pair, the exact $K_d$ used for the final label is the median
of its replicate measurements. The predeclared minimum of 100 pairs in each
class was met.

## Frozen inference and reproducibility

The primary model was `BioInteract/checkpoints/best_random.pt` (SHA-256
`a91f5e75e123af4c0a66ff9302842b0bcacdce2780abb9f7f5f4170b69f3e347`).
The checkpoint was not retrained, fine-tuned, recalibrated, or reselected. The
decision threshold was the existing Davis random-validation threshold
0.5959881544113159; it was not optimized on BindingDB.

All 396 external targets received new 1,200-residue-capped
`fair-esm` 2.0.0 `esm2_t30_150M_UR50D` residue embeddings on CUDA. The ESM model
file SHA-256 was `881c7176cf198ef8dec26a3c375d40eb58d0c33df95c22562ca6cc6d3f812c62`.
The external cache contains 396 tensors (402,950,480 bytes; deterministic
name-and-content SHA-256
`693bc3f8f9cb541a55f46e061a69791e0d57965deedb0b3389e14f34d7dff2d6`). Every
tensor passed the required 640-dimensional and sequence-length checks; the
zero-feature fallback was never permitted.

The historical Davis cache could not be regenerated bit-for-bit under the
current documented `fair-esm` 2.0.0 runtime, although the model ID and tensor
shape matched. On a fixed Davis target (T0000), the embedding RMSE was 0.003316
and the maximum absolute coordinate difference was 0.092427. Running the frozen
classifier over its 68 observed T0000 pairs gave mean and maximum probability
differences of 0.000565 and 0.013464, respectively, with no frozen-threshold
crossings. This runtime-level limitation is recorded here; it does not justify
using a different ESM model, dimension, or zero padding.

Two independent full-precision CUDA runs produced the same per-pair probability
SHA-256: `9e50dfadbdcb9974d6bc2dcefbdc4edbf4bd089b14598d76724da48fa40bba1c`.
`CUBLAS_WORKSPACE_CONFIG=:4096:8` was set before importing PyTorch so cuBLAS
matrix multiplication was deterministic. The final gate therefore passed.

## Results

Pair-stratified bootstrap intervals use 600 replicates, NumPy seed 42, and the
2.5th/97.5th percentiles.

| Metric | Value |
|---|---:|
| AUROC | 0.5597 [0.5313, 0.5885] |
| AUPRC | 0.3404 [0.3138, 0.3724] |
| Frozen threshold | 0.5959881544 |
| F1 | 0.0701 |
| Precision | 0.7308 |
| Recall | 0.0368 |

## Interpretation and manuscript boundary

This is an independent exact-entity external test, but it does not support a
broad DTI-transfer claim. The modest ranking performance and very low
frozen-threshold recall show that a Davis-trained, kinase-focused binary
classifier does not transfer reliably to this chemically and proteomically
broader strict cohort. The result neither changes nor validates the reported
within-Davis checkpoint reconstructions, and it provides no evidence about
attention maps, residue contacts, binding pockets, or mechanisms.

The manuscript and responses should therefore report the result transparently as
a limitation of cross-domain transfer. It must not be described as a successful
external validation or used to strengthen claims beyond the Davis benchmark.
