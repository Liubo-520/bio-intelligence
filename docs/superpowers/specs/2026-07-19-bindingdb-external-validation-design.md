# BindingDB Kd-only External Validation Design

## Objective

Add one genuinely independent, reproducible external validation to the BioInteract
revision in direct response to Reviewer 2, Comment 3, without changing the Davis
training data, model parameters, checkpoint selection, threshold selection, or any
existing manuscript conclusion.

## Scientific boundary

This is a narrow transferability test of the existing Davis binary classifier. It
does **not** establish broad drug--target-interaction generalisation, validate the
model's structural interpretation, or replace the existing Davis entity-held-out
analyses.

The trained model is frozen. The primary inference artifact is
`BioInteract/checkpoints/best_random.pt`; the decision threshold is the canonical
random-split validation threshold `0.5959881544`. Neither may be reselected from
external data. No training, fine-tuning, calibration, feature selection, or threshold
optimization may use the external cohort.

## External source and snapshot

The source is BindingDB. The implementation must record the exact download URL,
source-provided version or archive date, UTC download timestamp, local filename,
file size, and SHA-256 hash in a manifest committed with the results. The raw archive
must remain excluded from Git and from the Zenodo source archive if redistribution is
not permitted. The manuscript must cite BindingDB and state the fixed snapshot used.

Only measurements that satisfy every condition below are eligible:

1. A finite, exact numerical Kd in nM is reported. Ki, IC50, EC50, composite scores,
   and inequality/censored values (for example `<`, `>`, `~`) are excluded.
2. The ligand has a parseable RDKit structure and a canonical isomeric SMILES.
3. The target has a valid amino-acid sequence with length from 1 through 1,200.
4. The target and measurement fields can be traced to the downloaded record.

The binary outcome is defined exactly as in the Davis analysis: high affinity is
`Kd < 30 nM`; low affinity is `Kd >= 30 nM`. Artificially sampled negatives are
prohibited.

## Replicates and identity controls

Canonical ligand identity is the canonical isomeric SMILES plus InChIKey where
available. Protein identity is the full normalized amino-acid sequence. For each
external ligand--protein pair, retain it only when all valid replicate Kd values lie
on the same side of 30 nM. Assign the pair the median Kd and its corresponding binary
label. Exclude threshold-discordant pairs; record all counts.

The primary cohort is strict double-novel external data:

1. Exclude an external ligand if its canonical ligand identity occurs anywhere in the
   released Davis interaction data.
2. Exclude an external protein if its full normalized sequence occurs anywhere in the
   released Davis interaction data.
3. Report the number of records and pairs removed at each filtering stage and the
   remaining number of unique ligands, proteins, and pairs.

No weakening to a pair-unseen or source-only set is allowed merely because this
strict cohort is small.

## Feature and inference protocol

Each retained external protein must receive a fair-ESM
`esm2_t30_150M_UR50D` representation generated with the repository extractor and a
dedicated external cache. Missing ESM representations are a hard error; zero-filled
fallback tensors must never be used for the external evaluation. Domain labels use
the model-compatible `NONE` representation only and are described as such.

The evaluator loads the model configuration embedded in the frozen checkpoint,
reconstructs the model without changing its configuration, runs deterministic
full-precision inference, and emits one row per external pair with source IDs,
canonical identities, Kd, true label, predicted probability, and frozen-threshold
prediction. Every evaluator input and output must be traceable through the manifest.

## Acceptance gates and reporting

The primary external result may be included in the manuscript only if the strict
cohort contains at least 100 positive and 100 negative pairs after all eligibility,
replicate, and identity filters. It must also have a complete raw-data manifest,
valid ESM representations for every retained protein, and a deterministic rerun that
reproduces all reported metrics.

If any gate fails, stop before changing manuscript claims. Produce an auditable
feasibility report containing counts, exclusions, and the reason the cohort was not
used; do not relax the protocol.

For a passing cohort, report its sample counts and class prevalence; AUROC and AUPRC
with stratified pair-bootstrap 95% confidence intervals; and F1, precision, and
recall at the frozen Davis validation threshold. Do not compare the external result
with prior publications evaluated under different cohorts or label definitions.

## Revision and release deliverables

For a passing cohort, create:

- a source and curation manifest, deterministic evaluator, and automated tests;
- a compact CSV/JSON result artifact and a supplementary external-validation table;
- manuscript and Supporting Information text that state the source, Kd-only label
  definition, strict double-novel filtering, frozen inference protocol, exact sample
  counts, metrics, and scope limitation;
- a point-by-point response to Reviewer 2, Comment 3 mapping every claim to an
  actual location and artifact; and
- an updated immutable `v1.0.2` GitHub/Zenodo release with an updated DOI where
  available. Version `v1.0.1` remains unchanged.

For a failing cohort, create only the feasibility artifact and a response that
truthfully explains that the prespecified strict cohort was insufficient; do not claim
that external validation was completed.

## Non-negotiable integrity constraints

- Never inspect external labels to select checkpoints, thresholds, preprocessing, or
  post hoc inclusion rules.
- Never write a numerical result, sample count, confidence interval, figure, or
  reviewer-response claim before it is generated and independently rechecked.
- Preserve pre-existing user changes and stage only files created or intentionally
  modified for this task.
