# Numerical consistency report

## Authoritative numerical source

`revision/analysis/original_entity_split_metrics.json` is the single
authoritative source for current released-checkpoint entity-split results. It is
machine-readable and lists full-precision runtime details, model/data hashes,
validation thresholds, test metrics, and the associated CSV file names.

| Split | AUROC | AUPRC | F1 | Precision | Recall | Frozen threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.903905 | 0.560125 | 0.565385 | 0.602459 | 0.532609 | 0.595988 |
| Target-ID-held-out | 0.929987 | 0.524523 | 0.533849 | 0.516854 | 0.552000 | 0.602237 |
| Drug-ID-held-out | 0.733448 | 0.167230 | 0.100000 | 0.272727 | 0.061224 | 0.845615 |

## Bootstrap confidence intervals (600 paired seed-42 resamples)

| Split | AUROC 95% CI | AUPRC 95% CI |
| --- | --- | --- |
| Random | [0.880747, 0.924368] | [0.497896, 0.624200] |
| Target-ID-held-out | [0.910988, 0.947690] | [0.454408, 0.583863] |
| Drug-ID-held-out | [0.703006, 0.763186] | [0.128860, 0.210610] |

The manuscript, marked revision, supporting information, figures, README,
public Space copy, result JSON/CSVs, and reviewer response use the rounded
values derived from this source. Baseline values are explicitly labelled as
contextual values reproduced from the original submitted comparison table; they
are not newly executed baseline replications.

## Separate strict split results

The stricter sequence-grouped cold-target and cold-both analyses use newly
trained models and are stored independently in
`revision/analysis/strict_split_metrics.json`. They must never be substituted
for the three archived-checkpoint reconstruction values above.
