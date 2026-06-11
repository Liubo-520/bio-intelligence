---
title: BioInteract Drug-Target Interaction
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.20.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Interpretable DTI prediction with cross-attention
---

# BioInteract

**Interpretable Drug–Target Interaction Prediction via Residue-Level Cross-Attention with Biological Prior Knowledge**

BioInteract is a deep learning framework that predicts whether a drug molecule will bind to a protein target, while simultaneously generating an interpretable atom–residue interaction map that shows *which* drug atoms interact with *which* protein residues.

## How to Use

### Tab 1 — Case Studies
Explore three pre-computed, clinically validated drug–target pairs:
- **ABL1(E255K) + Drug 5328940** (resistance mutant, Kd = 0.047 nM)
- **EGFR + Drug 156414** (kinase inhibitor)
- **BRAF + Drug 11717001** (RAF inhibitor)

Each case shows the full atom–residue interaction heatmap, top binding residues, and model prediction probability.

### Tab 2 — Custom Prediction
Enter any drug SMILES string and protein amino acid sequence.  
The model will:
1. Encode the drug via pharmacophore-aware GINE molecular graph
2. Encode the protein using ESM-2 (150M) residue embeddings + physicochemical features
3. Compute bidirectional cross-attention to generate an interaction map
4. Predict binding probability

> **Note:** First prediction on CPU may take 1–3 minutes as ESM-2 initialises.  
> Sequences are truncated to 512 residues for demo speed.

## Model Architecture

```
Drug (SMILES) → GINE encoder → atom representations
Protein (AA seq) → ESM-2 + physicochemical → residue representations
        ↓ bidirectional cross-attention ↓
  atom × residue interaction map  →  gated pooling  →  binding score
```

## Performance (Davis Kinase Dataset)

| Split | AUROC | AUPRC |
|-------|-------|-------|
| Random | 0.921 | 0.608 |
| Cold-Drug | 0.739 | 0.169 |
| Cold-Target | **0.941** | 0.549 |

## Citation

> BioInteract: Interpretable Drug–Target Interaction Prediction via Residue-Level Cross-Attention with Biological Prior Knowledge. *PLOS Computational Biology*, 2026.
