# BioInteract v1.0.4

This immutable resubmission release supersedes v1.0.3.

- Restores the first-submission Davis numerical comparison against DeepDTA,
  GraphDTA, AttentionDTA, MolTrans, TransformerCPI, and DrugBAN. The BioInteract
  row uses the canonical current-release entity-split values.
- Restores the component-removal ablation table and redrawn figure. The full-model
  row uses canonical current-release metrics, while the component-removed values
  retain the original submitted numerical results; the plotted differences are
  therefore descriptive rather than controlled matched-retraining effects.
- Preserves the v1.0.3 strict-split training provenance, closed BindingDB curation
  audit, and claim-safe Q9UQB9 hosted-Space workflow.

Use the repository and Zenodo archive together: the `v1.0.4` tag identifies the
executable source, while [DOI: 10.5281/zenodo.21437753](https://doi.org/10.5281/zenodo.21437753)
identifies the immutable deposited record.

# BioInteract v1.0.3

This immutable final-audit release supersedes v1.0.2 for the revised manuscript.

- Documents that the exact-sequence-grouped cold-target and cold-both models
  were trained from initialization, with their optimization settings,
  early-stopping selections, checkpoint paths, and SHA-256 digests recorded in
  the Supporting Information and strict-split metrics artifact.
- Removes unsupported cross-model numerical baselines, their figure, and their
  duplicated SI table. The release retains only a non-numerical representational
  contrast and BioInteract results that map to released artifacts.
- Closes the BindingDB audit trail with sequential row accounting, pair-level
  threshold-conflict removal, and within-pair median aggregation of consistent
  replicate measurements.
- Ships the claim-safe custom-prediction Space source, including the labelled
  Aurora kinase C (Q9UQB9) demonstration, rather than retired fixed case-study
  views.

Use the repository and Zenodo archive together: the `v1.0.3` tag identifies the
executable source, while [DOI: 10.5281/zenodo.21436919](https://doi.org/10.5281/zenodo.21436919)
identifies the immutable deposited record.

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
