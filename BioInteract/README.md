# BioInteract

BioInteract is a Davis-benchmark model for **binary high-affinity interaction classification**. Its public outputs are a sigmoid **classifier score** and **model-native atom-residue attention attribution**. Attribution is hypothesis-generating and **not physical contacts** or evidence of binding residues, a binding pocket, or a structural mechanism.

## Evaluation scope

The reported pipeline uses cached fair-ESM embeddings with a **1,200-residue** sequence limit. The Hugging Face browser demonstration instead uses Transformers with a **512-residue** limit and an **unknown-domain representation** for arbitrary user inputs. It is **not numerically equivalent** to the 1,200-residue reported pipeline; its classifier score is not calibrated.

The archived **Target-ID-held-out** result is identifier-held-out, not a strict exact-sequence-held-out result because Davis contains duplicate sequence groups.

## Canonical current-release results

The following test-set values are generated from the released checkpoints and
released Davis inputs under `torch.inference_mode()` in full precision.  For
each split, the F1 decision threshold is selected once from that split's
validation predictions and then frozen before evaluating its test predictions.
They supersede earlier metric JSON/CSV artifacts that could not be reproduced
from the current released checkpoints and inputs.

| Protocol | Test pairs (positive) | AUROC | AUPRC | F1 | Validation-selected threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random entity split | 6,011 (276) | 0.904 | 0.560 | 0.565 | 0.596 |
| Target-ID-held-out | 5,984 (250) | 0.930 | 0.524 | 0.534 | 0.602 |
| Drug-ID-held-out | 5,746 (245) | 0.733 | 0.166 | 0.100 | 0.846 |

The stricter exact-sequence-grouped cold-target and cold-both retraining
experiments are separate stress tests, not replacements for these archived
identifier-split checkpoint reconstructions. Their full artifacts are in
`revision/analysis/strict_split_metrics.json`.

## Setup

```bash
pip install -r requirements.txt
```

## Core commands

```bash
python -m src.tools.prepare_data --dataset davis
python -m src.tools.extract_esm2 --dataset davis --model esm2_t30_150M_UR50D
python -m src.cli.train --config configs/default.yaml
python -m src.cli.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
python -m src.cli.interpret --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## Reproduce the reported release artifacts

Run these commands from the repository root. The canonical script instantiates
each released model from `checkpoint['config']`, writes validation and test
prediction CSVs with hashes, and never chooses a test-set threshold. The second
script promotes only a canonical artifact to the public result JSON/CSV files
used by the figures.

```bash
python revision/recompute_entity_split_metrics.py --output-dir revision/analysis --device cuda
python revision/promote_canonical_entity_metrics.py \
  --artifact revision/analysis/original_entity_split_metrics.json \
  --results-dir BioInteract/results
cd BioInteract
python -m src.analysis.generate_comparison_figures
python -m src.analysis.generate_figures --device cuda
```

The exact audit environment is pinned in `requirements-audit.txt`; the
machine-readable provenance, input/checkpoint hashes, prediction CSVs, and
bootstrap intervals are in `revision/analysis/`.

## Evaluation runners

`src.experiments.run_split_final` is the original entity-held-out runner. `revision/run_strict_split_evaluation.py` is the strict exact-sequence-grouped runner. They serve different protocols and their outputs should not be conflated.

## Project layout

- `src/`: source code, entrypoints, experiments, analysis, tools, and tests
- `configs/`: model and workflow configuration
- `data/`: input data and cached embeddings
- `checkpoints/`, `results/`, `logs/`, `runs/`: experiment artifacts
- `../revision/`: release reconciliation, canonical recomputation, strict-split
  analyses, provenance records, and verification tests
