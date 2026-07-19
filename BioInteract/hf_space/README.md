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
short_description: Binary high-affinity DTI with model-native attribution
---

# BioInteract

BioInteract supports **binary high-affinity interaction classification** on the Davis kinase benchmark. The browser interface returns a **classifier score** and **model-native atom-residue attention attribution**. Attribution is hypothesis-generating; it is **not physical contacts**, binding-residue evidence, or a structural mechanism.

## Browser demonstration scope

The custom workflow accepts a drug SMILES string and a protein sequence. It uses Hugging Face Transformers with a **512-residue** input limit and initialises ESM-2 during application startup. For arbitrary user-supplied sequences, the configured domain-label channel uses an **unknown-domain representation** because curated annotations are not released with this demonstration.

The browser demonstration is **not numerically equivalent** to the reported **1,200-residue** evaluation pipeline, which uses cached fair-ESM embeddings. It must not be used to reproduce the reported metrics or to interpret its sigmoid classifier score as a calibrated measure.

## Reported Davis results

| Partition | AUROC | AUPRC |
|---|---:|---:|
| Random | 0.904 | 0.560 |
| Drug-ID-held-out | 0.733 | 0.167 |
| Target-ID-held-out | 0.930 | 0.525 |

Target-ID-held-out is an archived identifier-based split result. Duplicate Davis protein sequences mean it is not a strict exact-sequence-held-out estimate.

## Interpretation limit

The heatmap and residue chart rank model-native attribution. They do not identify binding residues, key contacts, a binding pocket, or a mechanistic interaction map.
