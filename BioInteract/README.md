# BioInteract

BioInteract is a Davis-benchmark model for **binary high-affinity interaction classification**. Its public outputs are a sigmoid **classifier score** and **model-native atom-residue attention attribution**. Attribution is hypothesis-generating and **not physical contacts** or evidence of binding residues, a binding pocket, or a structural mechanism.

## Evaluation scope

The reported pipeline uses cached fair-ESM embeddings with a **1,200-residue** sequence limit. The Hugging Face browser demonstration instead uses Transformers with a **512-residue** limit and an **unknown-domain representation** for arbitrary user inputs. It is **not numerically equivalent** to the 1,200-residue reported pipeline; its classifier score is not calibrated.

The archived **Target-ID-held-out** result is identifier-held-out, not a strict exact-sequence-held-out result because Davis contains duplicate sequence groups.

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

## Evaluation runners

`src.experiments.run_split_final` is the original entity-held-out runner. `revision/run_strict_split_evaluation.py` is the strict exact-sequence-grouped runner. They serve different protocols and their outputs should not be conflated.

## Project layout

- `src/`: source code, entrypoints, experiments, analysis, tools, and tests
- `configs/`: model and workflow configuration
- `data/`: input data and cached embeddings
- `checkpoints/`, `results/`, `logs/`, `runs/`: experiment artifacts
