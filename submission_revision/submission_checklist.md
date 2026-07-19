# BioInteract submission checklist

Release target: GitHub `v1.0.2` and a matching new Zenodo version. The
preceding v1.0.1 record ([`10.5281/zenodo.21431147`](https://doi.org/10.5281/zenodo.21431147))
remains immutable.

- [x] `manuscript_clean.pdf` compiled from editable `manuscript_clean.tex`.
- [x] `manuscript_marked.pdf` compiled from editable `manuscript_marked.tex`.
- [x] `supporting_information.pdf` compiled from editable
  `supporting_information.tex`.
- [x] `response_to_reviewers.pdf` compiled from editable
  `response_to_reviewers.tex`.
- [x] Figure map is final: Fig. 1 comparison, Fig. 2 original/strict split,
  Fig. 3 diagnostic controls/relatedness, Fig. 4 training, Fig. 5 attention,
  Fig. 6 web interface.
- [x] Fig. 4 uses checkpoint-matched contiguous training records only.
- [x] Fig. 5 is derived directly from the archived 1,301,380 raw scores.
- [x] All final entity-split metrics originate from the deterministic canonical
  checkpoint reconstruction and a validation-selected frozen threshold.
- [x] Strict BindingDB $K_d$-only double-novel evaluation uses the frozen
  `best_random` checkpoint and the existing Davis random-validation threshold;
  its limited transfer result is reported without external recalibration.
- [x] Numerical ablation claims and figures without recoverable artifacts are
  withdrawn.
- [x] Public scope text states that attribution is not structural-contact,
  pocket, residue-validation, or mechanism evidence.
- [ ] The new v1.0.2 Zenodo DOI is in manuscript, marked manuscript, response
  letter, citation metadata, and release notes after the version is minted.
- [x] Code, checkpoint/input provenance, figures, editable sources, and this
  audit are included in the release archive; the derived ESM cache is retrieved
  from the matching Git LFS tag as documented.
