# Final pre-submission audit — BioInteract revision

Date: 18 July 2026
Release target: GitHub tag `v1.0.0` and Zenodo DOI
[`10.5281/zenodo.21429673`](https://doi.org/10.5281/zenodo.21429673)

## Decision

The revised submission is suitable for final production only with the
canonical, current-release entity-split results reported below. Historical
entity-split CSV/JSON scores were not reproducible from the released inputs and
released checkpoints, so they have been retired rather than retained as a
second, conflicting result set. Historical numerical ablations are withdrawn
because matching ablated checkpoints and prediction files are unavailable.

| Protocol | AUROC | AUPRC | F1 | Test pairs (positive) |
| --- | ---: | ---: | ---: | ---: |
| Random entity split | 0.9039 | 0.5601 | 0.5654 | 6,011 (276) |
| Target-ID-held-out | 0.9300 | 0.5245 | 0.5338 | 5,984 (250) |
| Drug-ID-held-out | 0.7334 | 0.1663 | 0.1000 | 5,746 (245) |

Every test threshold in this table was selected once on the corresponding
validation predictions and frozen before test evaluation. The full-precision
`torch.inference_mode()` procedure, input and checkpoint hashes, prediction
CSVs, and bootstrap intervals are recorded in
`revision/analysis/original_entity_split_metrics.json`.

## Scope and interpretation controls

- The Davis evaluation is binary high-affinity interaction classification, not
  binding-affinity regression or clinical prediction.
- The `Target-ID-held-out` partition is identifier-held-out; it is not a strict
  exact-sequence-held-out partition because duplicate sequences occur in Davis.
- Exact-sequence-grouped cold-target and cold-both retraining results are
  reported separately in `revision/analysis/strict_split_metrics.json` and are
  not conflated with checkpoint reconstruction.
- Attention and Grad-CAM are model-native, hypothesis-generating attributions;
  they are not contact, pocket, binding-residue, or structural-mechanism
  evidence.
- No released curated per-protein domain annotation files are available. The
  configured domain-label channel therefore uses the unknown label and does not
  support a biological-domain claim.
- The public web demonstration has a 512-residue Transformers input limit and
  uncalibrated classifier score. It is not numerically equivalent to the
  reported 1,200-residue cached-ESM evaluation pipeline.

## Title recommendation requiring author approval

The current title includes “with Biological Prior Knowledge,” but the released
input audit finds no curated domain annotations. To eliminate a possible
reviewer concern, the authors may wish to adopt the narrower title:

> BioInteract: Interpretable Drug–Target Interaction Prediction via
> Residue-Level Cross-Attention and Pretrained Protein Representations

This audit does not alter the title automatically, because title changes are an
authorial decision.

## Files audited

- `manuscript_clean.tex` and `manuscript_marked.tex`: canonical metrics,
  threshold protocol, withheld-ablation disclosure, public-scope wording, and
  repository/DOI availability statements.
- `supporting_information.tex`: input audit, computation environment, canonical
  metric table, confidence intervals, and reproducibility procedure.
- `response_to_reviewers.tex`: editor DOI request, result reconciliation,
  withdrawal of irreproducible ablations, and scope limits.
- `figures/fig2_performance.*`: original identifier-split checkpoint
  reconstruction distinguished from strict split stress tests.
- `BioInteract/results/`: canonical public prediction CSVs and result JSON
  generated only by `revision/promote_canonical_entity_metrics.py`.

## Remaining production gate

Compile all four TeX deliverables without unresolved references, render-check
the PDFs, and verify the public GitHub release and published Zenodo record.
