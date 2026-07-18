# BioInteract v1.0.0

This immutable reproducibility release accompanies the revised BioInteract manuscript.

- Canonical entity-split metrics are regenerated from the released checkpoints and inputs under full-precision inference.
- Classification thresholds are selected once on validation predictions and frozen for test reporting.
- Legacy entity-split prediction artifacts that could not be recreated from the current released checkpoints and inputs are retired.
- The archive includes checkpoint/configuration hashes, input hashes, prediction CSVs, bootstrap intervals, and strict within-Davis split artifacts.
- Numerical ablation claims are intentionally not included because matching ablated-model checkpoint/prediction artifacts are unavailable.

Use the repository and Zenodo archive together: the `v1.0.0` tag identifies the executable source, while [DOI: 10.5281/zenodo.21429673](https://doi.org/10.5281/zenodo.21429673) identifies the immutable deposited record.
