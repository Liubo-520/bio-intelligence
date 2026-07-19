# BioInteract v1.0.2

This immutable reproducibility release accompanies the revised BioInteract manuscript.

- Adds strict BindingDB $K_d$-only exact-double-novel external-validation code,
  contract tests, aggregate manifests, and audit report. The frozen Davis
  `best_random` checkpoint achieves AUROC 0.5597 [0.5313, 0.5885] and AUPRC
  0.3404 [0.3138, 0.3724] on 1,932 pairs (516 positives); this is reported as
  limited transfer, not broad DTI generalisation.
- Keeps the raw BindingDB archive, derived pairs, ESM cache, and pair-level
  predictions out of the release archive. Their source URL, source SHA-256,
  curation accounting, and deterministic regeneration procedure are included.

- Canonical entity-split metrics are regenerated from the released checkpoints and inputs under deterministic full-precision inference.
- Figure 4 now uses only checkpoint-matched contiguous training traces; Figure 5 uses the archived 1,301,380 raw attention scores.
- Classification thresholds are selected once on validation predictions and frozen for test reporting.
- Legacy entity-split prediction artifacts that could not be recreated from the current released checkpoints and inputs are retired.
- The archive includes checkpoint/configuration hashes, input hashes, prediction CSVs, bootstrap intervals, and strict within-Davis split artifacts.
- Numerical ablation claims are intentionally not included because matching ablated-model checkpoint/prediction artifacts are unavailable.

Use the repository and Zenodo archive together: the `v1.0.2` tag identifies the executable source, while [DOI: 10.5281/zenodo.21436351](https://doi.org/10.5281/zenodo.21436351) identifies the immutable deposited record.
