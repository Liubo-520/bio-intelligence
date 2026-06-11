# BioInteract

Interpretable Drug-Target Interaction Prediction via Residue-Level Cross-Attention with Biological Prior Knowledge.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
```bash
python -m src.tools.prepare_data --dataset davis
```

### 2. Extract ESM-2 Features
```bash
python -m src.tools.extract_esm2 --dataset davis --model esm2_t30_150M_UR50D
```

### 3. Train
```bash
python -m src.cli.train --config configs/default.yaml
```

### 4. Evaluate
```bash
python -m src.cli.evaluate --config configs/default.yaml --checkpoint checkpoints/best.pt
```

### 5. Interpretability Analysis
```bash
python -m src.cli.interpret --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## Project Layout

- src/: all Python source code, entrypoints, experiments, analysis, tools, and tests
- configs/: training and model configuration
- data/: raw data and precomputed embeddings
- checkpoints/, results/, logs/, runs/: experiment artifacts
- ../submission/manuscript/: manuscript, figures, and journal submission assets
- ../docs/: planning notes and result summaries
