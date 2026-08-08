# Second revision package — BioInteract (Scientific Reports, ID 56f53c41-a86e-4948-8cfd-a3c11b83fd1b)

Files to upload to the journal:

| File | Purpose |
|---|---|
| `manuscript_clean.pdf` / `.tex` | Revised manuscript, no markup |
| `manuscript_marked.pdf` / `.tex` | Same manuscript, new and revised material in red |
| `supporting_information.pdf` / `.tex` | Supporting Information, Tables S1–S17 |
| `response_to_reviewers.pdf` / `.tex` | Point-by-point response to Editor and Reviewers 1–3 |
| `cover_letter.pdf` / `.tex` | Cover letter |
| `figures/` | All six main-text figures as PDF (and PNG previews) |
| `references.bib` | Bibliography (45 entries) |

Support files required to compile: `wlscirep.cls`, `naturemag-doi.bst`,
`jabbrv.sty`, `jabbrv-ltwa-all.ldf`, `jabbrv-ltwa-en.ldf`.

## Building

`manuscript_clean.tex` and `manuscript_marked.tex` are **generated**; do not edit
them directly. Edit `manuscript_source.tex`, which carries `%%REV-START` /
`%%REV-END` markers around every block that is new or substantively revised in
this round, then run:

```
python build_manuscripts.py     # writes both manuscript variants
latexmk -pdf manuscript_clean.tex
latexmk -pdf manuscript_marked.tex
latexmk -pdf supporting_information.tex
latexmk -pdf response_to_reviewers.tex
latexmk -pdf cover_letter.tex
```

`build_manuscripts.py` also substitutes two table bodies from the analysis
artifacts in `../revision_v2/analysis/`, so no measured value is transcribed by
hand:

- `%%MATCHED-ROWS%%` ← `matched_table_body.tex` (main-text Table 1)
- `%%FULLDB-RESULT%%` ← `fulldb_result.tex` (main-text Table 5 and its paragraph)

## Reproducing the analyses added in this round

All scripts live in `../revision_v2/`:

| Script | Produces |
|---|---|
| `run_baseline_suite.py` | Matched-protocol training runs → `analysis/baseline_suite.json` |
| `baseline_models.py` | The re-implemented comparison architectures |
| `render_matched_table.py` | Table 1, Table S15 and Table S17 bodies |
| `bindingdb_waterfall.py` | Ordered filtering waterfall for a BindingDB snapshot |
| `applicability_domain.py` | Ligand/target applicability-domain stratification |
| `build_applicability_reference.py` | Davis reference bundle used by the web server |
| `render_fulldb_result.py` | Full-release external-evaluation fragment |
| `capture_webserver.py` | Figure 6 screenshot and attribution payload |
| `figures_v2.py` | Figures 1, 3 and 6 |

The web application updated for this revision is
`../BioInteract/hf_space/app.py`.

## Float inventory

8 tables and 6 figures, down from 9 tables and 7 figures in the previous
revision. The mapping from the previous version is tabulated at the end of
`response_to_reviewers.pdf`.
