"""Regression guard for the Davis ABL1 nominal-variant input audit."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "revision" / "analysis"))

from abl1_variant_audit import audit_abl1_variants  # noqa: E402


def test_nominal_abl1_variants_have_one_input_sequence_digest():
    report = audit_abl1_variants(ROOT / "BioInteract" / "data" / "raw" / "davis")

    assert report["variant_count"] == 15
    assert report["unique_sequence_digest_count"] == 1
    assert {row["target_name"] for row in report["variants"]} >= {
        "ABL1",
        "ABL1(E255K)",
        "ABL1(F317I)",
        "ABL1(F317L)",
        "ABL1(M351T)",
    }
    assert {row["drug_id"] for row in report["compound_records"]} == {"D0010", "D0017"}
