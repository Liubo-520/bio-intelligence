# BioInteract final consistency audit

Audit date: 2026-07-18
Release: GitHub tag `v1.0.1`; Zenodo DOI
[`10.5281/zenodo.21431147`](https://doi.org/10.5281/zenodo.21431147)

## Decision

The revised submission is internally consistent with the released inputs,
archived checkpoints, raw attribution artifact, and public code. No model was
retrained, no data, labels, seed, or split was changed, and no structural or
experimental validation was invented.

## Corrected issues and evidence

| Issue | Before | Final correction | Evidence | Metric/conclusion impact |
| --- | --- | --- | --- | --- |
| Entity-split reconstruction | CUDA outputs varied at the last floating-point bits between independent processes. | Deterministic full-precision inference is set before CUDA initialisation; two independent runs produced identical hashes for all six prediction CSVs. | `revision/recompute_entity_split_metrics.py`; `revision/analysis/original_entity_split_metrics.json`; reproducibility manifest. | Canonical rounded values are Random 0.904/0.560, Target-ID-held-out 0.930/0.525, and Drug-ID-held-out 0.733/0.167 (AUROC/AUPRC). |
| Threshold protocol | Historical values could be confused with a test-selected decision threshold. | The F1 threshold is selected once on each validation partition then frozen for its test partition. | Canonical JSON records validation predictions and thresholds. | Threshold-dependent test metrics are no longer described as threshold-selected on the test set. |
| Training traces (Fig. 4) | Repeated/restarted log fragments could be concatenated into a visually continuous trace. | One contiguous trace per protocol is selected only when its peak validation AUROC agrees with the archived checkpoint; restart fragments are excluded. No fitted line, slope, interpolation, or early-stop marker is drawn. | `BioInteract/logs/run_*.log`; checkpoint `best_val_auroc`; `fig4_training.json`. | Figure is descriptive single-seed provenance, not evidence of no overfitting. |
| Attention distribution (Fig. 5) | Summary/curve could be regenerated from a non-identical flattened array. | Both panels use the archived `flat_scores` vector: distribution plus descending-score cumulative mass. | `BioInteract/results/figure_data/attention_distribution.npz`; 1,301,380 scores. | Mean 0.00435138; median 0.000272190; 99th percentile 0.0571928; 99.2319% below 0.1; top 1%/5% mass 0.788528/0.904465. Attribution remains model-native only. |
| GINE specification | Edge handling and epsilon status were under-specified. | The manuscript now states that 16-dimensional edge features are projected to 256 dimensions before `GINEConv`; epsilon is fixed at zero rather than learnable. | `BioInteract/src/models/drug_encoder.py`; checkpoint configuration. | Method description now matches executable architecture. |
| Historical optimizer claims | A uniform AdamW/warmup/label-smoothing recipe was not recoverable for all checkpoints. | The text distinguishes Random/Drug-ID final-run log headers from the checkpoint-matched Target-ID log, for which optimizer, scheduler, warmup, and smoothing are not recoverable. | Checkpoint configs; archived runner logs; SI Table S3. | No unsupported uniform historical-training claim remains. |
| Ablations | Submitted numeric ablation claims lacked matching ablated checkpoints and predictions. | Numerical ablation figure, table, and component-effect claims are withdrawn. | Response R1.5/R3.11; manuscript limitation. | No irreproducible ablation conclusion is retained. |
| Public interpretation scope | Domain-label and attention wording risked implying curated annotations or contacts. | Public code/docs state that Davis uses the unknown-domain representation and attention/Grad-CAM are not contacts, pockets, residue validation, or mechanisms. | `README.md`; `hf_space`; model/interpretation source. | No structural-contact or Pfam/InterPro annotation claim is made. |

## Canonical entity-split outputs

| Protocol | AUROC | AUPRC | Validation threshold | F1 | Test pairs (positive) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.9039055886 | 0.5601694008 | 0.5959881544 | 0.5653846154 | 6,011 (276) |
| Target-ID-held-out | 0.9299874433 | 0.5245233707 | 0.6022372842 | 0.5338491296 | 5,984 (250) |
| Drug-ID-held-out | 0.7334477219 | 0.1672303538 | 0.8456150889 | 0.1000000000 | 5,746 (245) |

## Required limitations retained

- The identifier-held-out target split is not an exact-sequence-held-out claim;
  strict within-Davis stress tests are reported separately.
- The target training log cannot prove a complete historical optimizer/scheduler
  recipe.
- No ablated checkpoint, external assay, PDB/PLIP analysis, docking result,
  or experimental validation is added or implied.
- The title retains the authors' phrase “biological prior knowledge”; its
  definition is narrowed in the manuscript to frozen ESM-2 and residue
  physicochemical features, not curated domain annotation. A shorter title is a
  stylistic author decision rather than an evidentiary prerequisite.

## Verification record

- `python -m pytest revision/analysis/test_entity_split_metrics.py -q`
- `python -m pytest BioInteract/src/tests/test_public_figure_provenance.py -q`
- `python BioInteract/hf_space/test_public_copy.py`
- Figure generation from archived artifacts and four successful PDF builds.
- Source scans found no obsolete figure-renumbering language, no assertion that
  attention is a physical contact, and no uniform historical
  AdamW/warmup/label-smoothing claim.
