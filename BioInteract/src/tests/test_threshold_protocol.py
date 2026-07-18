"""Behavioral regression tests for validation-only threshold selection."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import classification_metrics, select_f1_threshold  # noqa: E402
from src.cli import evaluate as evaluate_cli  # noqa: E402
from src.cli import evaluate_splits, train  # noqa: E402
from src.experiments import run_split, run_split_final, run_split_v2, run_split_v3  # noqa: E402


CLASSIFICATION_CONFIG = {
    "model": {"predictor": {"task": "classification"}},
    "training": {"amp": False},
}
EVALUATOR_NAMES = (
    "train",
    "evaluate",
    "evaluate_splits",
    "run_split",
    "run_split_v2",
    "run_split_v3",
    "run_split_final",
)
_MISSING = object()


class _FixedLogitModel(torch.nn.Module):
    """Return precomputed logits encoded in the dummy drug batch."""

    def forward(self, drug_batch, *args, **kwargs):
        return drug_batch


def _batch_from_scores(labels, scores):
    logits = torch.logit(torch.tensor(scores, dtype=torch.float32))
    size = len(labels)
    return {
        "drug_batch": logits,
        "morgan_fp": torch.zeros(size, 1),
        "esm2_embedding": torch.zeros(size, 1, 1),
        "physicochemical": torch.zeros(size, 1, 4),
        "domain_labels": torch.zeros(size, 1, dtype=torch.long),
        "protein_mask": torch.ones(size, 1, dtype=torch.bool),
        "label": torch.tensor(labels, dtype=torch.float32),
        "drug_ids": [f"drug-{index}" for index in range(size)],
        "target_ids": [f"target-{index}" for index in range(size)],
    }


def _evaluate(entry_point, batch, threshold=_MISSING):
    model = _FixedLogitModel()
    criterion = torch.nn.BCEWithLogitsLoss()
    loader = [batch]

    if entry_point == "train":
        args = (model, loader, criterion, CLASSIFICATION_CONFIG, "cpu")
        evaluator = train.evaluate
        kwargs = {}
    elif entry_point == "evaluate":
        args = (model, loader, CLASSIFICATION_CONFIG, "cpu")
        evaluator = evaluate_cli.evaluate_model
        kwargs = {}
    elif entry_point == "evaluate_splits":
        args = (model, loader, criterion, "cpu")
        evaluator = evaluate_splits.evaluate
        kwargs = {"use_amp": False}
    elif entry_point == "run_split":
        args = (model, loader, criterion, "cpu")
        evaluator = run_split.evaluate
        kwargs = {}
    elif entry_point == "run_split_v2":
        args = (model, loader, criterion, "cpu")
        evaluator = run_split_v2.evaluate
        kwargs = {}
    elif entry_point == "run_split_v3":
        args = (model, loader, criterion, "cpu")
        evaluator = run_split_v3.evaluate
        kwargs = {}
    else:
        args = (model, loader, criterion, "cpu")
        evaluator = run_split_final.evaluate
        kwargs = {}

    if threshold is not _MISSING:
        kwargs["threshold"] = threshold
    result = evaluator(*args, **kwargs)
    return result[0] if entry_point == "evaluate" else result


def test_threshold_selection_is_explicit_and_validation_only():
    validation_labels = np.array([0, 0, 1, 1])
    validation_scores = np.array([0.10, 0.35, 0.60, 0.90])
    test_labels = np.array([0, 1, 0, 1])
    test_scores = np.array([0.20, 0.40, 0.55, 0.95])

    threshold = select_f1_threshold(validation_labels, validation_scores)
    metrics = classification_metrics(test_labels, test_scores, threshold=threshold)

    assert metrics["threshold"] == threshold
    with pytest.raises(ValueError, match="validation-selected threshold"):
        classification_metrics(test_labels, test_scores)


def test_test_metrics_keep_validation_threshold_when_test_optimum_conflicts():
    validation_labels = np.array([0, 0, 1, 1])
    validation_scores = np.array([0.10, 0.35, 0.60, 0.90])
    test_labels = np.array([0, 0, 0, 0, 1])
    test_scores = np.array([0.20, 0.40, 0.55, 0.65, 0.75])

    validation_threshold = select_f1_threshold(
        validation_labels, validation_scores
    )
    test_optimum = select_f1_threshold(test_labels, test_scores)

    metrics = classification_metrics(
        test_labels, test_scores, threshold=validation_threshold
    )

    assert validation_threshold == pytest.approx(0.60)
    assert test_optimum == pytest.approx(0.75)
    assert test_optimum != validation_threshold
    assert metrics["threshold"] == validation_threshold
    assert metrics["F1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("entry_point", EVALUATOR_NAMES)
def test_evaluation_helpers_require_explicit_classification_threshold(entry_point):
    batch = _batch_from_scores([0, 1], [0.20, 0.80])

    with pytest.raises(TypeError):
        _evaluate(entry_point, batch)


@pytest.mark.parametrize("entry_point", EVALUATOR_NAMES)
def test_evaluation_helpers_apply_validation_threshold_without_reoptimising_test(
    entry_point,
):
    validation_threshold = select_f1_threshold(
        np.array([0, 0, 1, 1]), np.array([0.10, 0.35, 0.60, 0.90])
    )
    test_scores = np.array([0.20, 0.40, 0.55, 0.65, 0.75])
    test_labels = np.array([0, 0, 0, 0, 1])
    test_optimum = select_f1_threshold(test_labels, test_scores)

    metrics = _evaluate(
        entry_point,
        _batch_from_scores(test_labels, test_scores),
        threshold=validation_threshold,
    )

    assert test_optimum != validation_threshold
    assert metrics["threshold"] == pytest.approx(validation_threshold)
    assert metrics["F1"] == pytest.approx(2 / 3)


def test_select_f1_threshold_rejects_empty_inputs():
    with pytest.raises(ValueError, match="empty inputs"):
        select_f1_threshold(np.array([]), np.array([]))


def test_select_f1_threshold_rejects_missing_pr_thresholds(monkeypatch):
    monkeypatch.setattr(
        "src.utils.metrics.precision_recall_curve",
        lambda y_true, y_pred_prob: (
            np.array([1.0]), np.array([1.0]), np.array([])
        ),
    )

    with pytest.raises(ValueError, match="does not provide a threshold"):
        select_f1_threshold(np.array([1]), np.array([0.9]))
