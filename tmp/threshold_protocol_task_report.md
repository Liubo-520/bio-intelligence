# Validation-only classification-threshold protocol

## Scope

Changed only the protocol implementation and its focused regression test:

- `BioInteract/src/utils/metrics.py`
- `BioInteract/src/cli/train.py`
- `BioInteract/src/cli/evaluate.py`
- `BioInteract/src/cli/evaluate_splits.py`
- `BioInteract/src/experiments/run_split.py`
- `BioInteract/src/experiments/run_split_v2.py`
- `BioInteract/src/experiments/run_split_v3.py`
- `BioInteract/src/experiments/run_split_final.py`
- `BioInteract/src/tests/test_threshold_protocol.py`

No data, split implementation, random seed, model code, dependencies,
manuscript files, checkpoints, or result files were modified or regenerated.

## Implementation

`select_f1_threshold(y_true, y_pred_prob)` is now the public, explicit PR-curve
selector. It preserves the prior maximum-F1 convention, rejects empty input,
and raises rather than manufacturing a threshold if the PR curve supplies no
thresholds. `classification_metrics` now raises `ValueError` containing
`validation-selected threshold` when called without a threshold; it no longer
uses labels to choose one. AUROC and AUPRC remain computed from unthresholded
probabilities.

The focused regression test additionally parses every supported entry point to
ensure its `classification_metrics` calls have an explicit `threshold` keyword
and that validation-threshold selection is present.

## Concrete validation-to-test data flow

| Entry point | Validation selection | Frozen test use |
| --- | --- | --- |
| `src/cli/train.py` | Each validation evaluation selects F1; when validation AUROC improves, the matching threshold is saved with the checkpoint. | The loaded best checkpoint's `validation_threshold` is passed to test evaluation. Legacy checkpoints without it first derive one from the reconstructed validation split. |
| `src/cli/evaluate.py` | Reconstructs the same validation and test splits; predicts validation data and selects F1 once. | Passes that `validation_threshold` unchanged into test evaluation; the emitted test metrics contain it. |
| `src/cli/evaluate_splits.py` | Selects on every validation evaluation and retains the value paired with the best validation-AUROC state. | Supplies `best_validation_threshold` to the test call. |
| `src/experiments/run_split.py` | Same best-validation-AUROC pairing. | Supplies the paired value to test evaluation and saves it as checkpoint metadata. |
| `src/experiments/run_split_v2.py` | Same best-validation-AUROC pairing. | Supplies the paired value to test evaluation and saves it as checkpoint metadata. |
| `src/experiments/run_split_v3.py` | Same best-validation-AUROC pairing. | Supplies the paired value to test evaluation and saves it as checkpoint metadata. |
| `src/experiments/run_split_final.py` | Same best-validation-AUROC pairing. | Supplies the paired value to test evaluation and saves it as checkpoint metadata. |

## TDD and verification evidence

1. RED, before the helper existed:

   ```powershell
   python -m pytest BioInteract\src\tests\test_threshold_protocol.py -q
   ```

   Exit code 1. Collection failed as expected with:
   `ImportError: cannot import name 'select_f1_threshold' from 'src.utils.metrics'`.

2. RED for the added entry-point protocol test, after the metrics helper was
   implemented but before callers were changed:

   ```powershell
   python -m pytest BioInteract\src\tests\test_threshold_protocol.py -q
   ```

   Exit code 1. The two metric tests passed and all seven entry-point cases
   failed because `select_f1_threshold` and the frozen
   `validation_threshold` flow were absent.

3. GREEN after the minimal caller changes:

   ```powershell
   python -m pytest BioInteract\src\tests\test_threshold_protocol.py -q
   ```

   Exit code 0: `9 passed in 0.99s`.

4. Required focused integration command:

   ```powershell
   python -m pytest BioInteract\src\tests\test_pipeline.py BioInteract\src\tests\test_real_data.py -q
   ```

   Exit code 0: `1 passed in 6.53s`. `test_real_data.py` contains no pytest
   test function, so the command collected the pipeline test only.

5. Syntax and whitespace checks:

   ```powershell
   python -m py_compile BioInteract\src\utils\metrics.py BioInteract\src\cli\train.py BioInteract\src\cli\evaluate.py BioInteract\src\cli\evaluate_splits.py BioInteract\src\experiments\run_split.py BioInteract\src\experiments\run_split_v2.py BioInteract\src\experiments\run_split_v3.py BioInteract\src\experiments\run_split_final.py
   git diff --check
   ```

   Both completed with exit code 0.

6. Read-only protocol probe confirmed that empty arrays raise and that a
   validation-derived threshold (`0.60` for the probe inputs) is preserved in
   test metrics.

## Concerns

No implementation blockers remain. The requested real-data pytest path has no
collectable test function, so its current command validates collection without
running a real-data forward pass; this pre-existing test structure was not
changed.

## Review follow-up: explicit helper thresholds

Following review, the generic evaluation helpers no longer accept a default
threshold or select from the labels they are asked to score. Each now has a
required `threshold` parameter for classification metrics and is backed by a
raw `_collect_predictions` helper. F1 selection appears only in explicit
validation-only functions:

- `evaluate_validation` in `train.py`, `evaluate_splits.py`, and each
  `run_split*.py` entry point;
- `select_validation_threshold` in standalone `evaluate.py`.

The training loops call their explicit validation function before checking
validation AUROC, retain the threshold paired with the winning state, and pass
that value to the regular test helper. `train.py` retains its legacy-checkpoint
fallback, but it now invokes `evaluate_validation` rather than a generic test
or evaluation helper. Regression calls explicitly pass `threshold=None`; that
value is never used by `classification_metrics`.

The former AST/source-structure check was replaced with behavioral coverage.
For all seven helpers, the test suite now verifies that:

1. omitting a classification threshold raises `TypeError` at the helper API;
2. a supplied validation threshold is preserved when the test split's
   independently F1-optimal threshold conflicts with it (`0.60` validation
   versus `0.75` test); and
3. the resulting test F1 is evaluated at the validation threshold rather than
   at the better test-only optimum.

The selector boundary tests now also cover empty inputs and a PR-curve result
with no thresholds (simulated through the metrics module dependency).

### Follow-up RED/GREEN evidence

RED after adding the behavioral helper requirement, before changing the helper
implementations:

```powershell
python -m pytest BioInteract\src\tests\test_threshold_protocol.py -q
```

Exit code 1: seven cases failed with `Failed: DID NOT RAISE <class 'TypeError'>`,
one for every supported helper, demonstrating their prior `threshold=None`
fallback.

GREEN after separating raw prediction collection from explicit validation
selection:

```powershell
python -m pytest BioInteract\src\tests\test_threshold_protocol.py -q
```

Exit code 0: `18 passed in 6.53s`. The run emitted three pre-existing PyTorch
AMP deprecation warnings from the CLI modules; no new warning suppression was
added.

Focused integration and syntax checks:

```powershell
python -m pytest BioInteract\src\tests\test_pipeline.py BioInteract\src\tests\test_real_data.py -q
python -m py_compile BioInteract\src\utils\metrics.py BioInteract\src\cli\train.py BioInteract\src\cli\evaluate.py BioInteract\src\cli\evaluate_splits.py BioInteract\src\experiments\run_split.py BioInteract\src\experiments\run_split_v2.py BioInteract\src\experiments\run_split_v3.py BioInteract\src\experiments\run_split_final.py
git diff --check
```

All completed with exit code 0. The pytest command reported `1 passed in
6.59s`; as above, `test_real_data.py` has no collectable pytest test function.
