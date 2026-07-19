"""Guard against overstating the strict BindingDB external result in submission copy."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission_revision"
SUMMARY = ROOT / "BioInteract" / "results" / "external_validation" / "bindingdb_external_summary.json"


def _section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[begin:finish]


def test_external_validation_claims_match_verified_summary_and_remain_limited():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["primary_gate_passed"] is True
    assert summary["total_pairs"] == 1932
    assert summary["positive_pairs"] == 516
    assert summary["negative_pairs"] == 1416
    assert summary["metrics"]["AUROC"] == 0.5596943021066001
    assert summary["metrics"]["AUPRC"] == 0.34042591899153285

    manuscript = (SUBMISSION / "manuscript_clean.tex").read_text(encoding="utf-8")
    marked = (SUBMISSION / "manuscript_marked.tex").read_text(encoding="utf-8")
    supporting = (SUBMISSION / "supporting_information.tex").read_text(encoding="utf-8")
    response = (SUBMISSION / "response_to_reviewers.tex").read_text(encoding="utf-8")

    for text in (manuscript, marked):
        assert "\\label{tab:bindingdb_external}" in text
        assert "1{,}932" in text
        assert "516" in text and "1{,}416" in text
        assert "0.560" in text and "0.340" in text
        assert "does not support a broad DTI-transfer claim" in text
        assert "gilson2016bindingdb" in text

    assert "\\label{tab:bindingdb_external_si}" in supporting
    assert "BindingDB_BindingDB_Articles_202607" in supporting
    assert "0.5959881544" in supporting
    assert "0.5597 [0.5313, 0.5885]" in supporting

    r23 = _section(response, "\\subsection*{R2.3.", "\\subsection*{R2.4.")
    r31 = _section(response, "\\subsection*{R3.1.", "\\subsection*{R3.2.")
    for section in (r23, r31):
        assert "BindingDB" in section
        assert "1{,}932" in section
        assert "0.560" in section and "0.340" in section
        assert "does not support a broad DTI-transfer claim" in section
