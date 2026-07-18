"""Regression contracts for evidence-backed public figure generation."""

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.generate_figures import (  # noqa: E402
    prepare_attention_curve,
    select_checkpoint_matching_segment,
    split_training_log_segments,
)


def test_log_parser_keeps_a_single_complete_segment_matching_checkpoint():
    log_text = "\n".join(
        [
            "E001 | loss=1.0 | val_auroc=0.7000",
            "E002 | loss=0.8 | val_auroc=0.8000",
            "E001 | loss=1.1 | val_auroc=0.7100",
            "E002 | loss=0.9 | val_auroc=0.8100",
        ]
    )

    segments = split_training_log_segments(log_text)
    selected, selected_index = select_checkpoint_matching_segment(
        segments, checkpoint_best_val_auroc=0.80004
    )

    assert len(segments) == 2
    assert selected_index == 0
    assert [record["epoch"] for record in selected] == [1, 2]
    assert max(record["val_auroc"] for record in selected) == pytest.approx(0.8)


def test_log_parser_rejects_a_segment_without_checkpoint_match():
    segments = split_training_log_segments(
        "E001 | loss=1.0 | val_auroc=0.7000\nE002 | loss=0.8 | val_auroc=0.8000"
    )

    with pytest.raises(ValueError, match="does not match"):
        select_checkpoint_matching_segment(segments, checkpoint_best_val_auroc=0.95)


def test_attention_curve_is_recomputed_from_raw_scores_and_normalises_to_one():
    curve = prepare_attention_curve(np.array([0.8, 0.1, 0.1], dtype=np.float32))

    assert curve["percentiles"].tolist() == pytest.approx(
        [100 / 3, 200 / 3, 100.0]
    )
    assert curve["cumulative_attention"].tolist() == pytest.approx([0.8, 0.9, 1.0])
