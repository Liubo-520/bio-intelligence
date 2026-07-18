"""Generate aggregate performance and attribution figures from saved outputs.

The public generator retains benchmark performance, aggregate model-native
attribution, and training-dynamics figures. It intentionally does not create
pair-specific structural or variant interpretation figures.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import RDLogger
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.figure_common import (
    FIGURE_DATA_DIR,
    PALETTE,
    configure_matplotlib,
    panel_label,
    save_figure,
    soften_axes,
    write_manifest,
)
from src.data.dataset import DTIDataset, collate_dti
from src.data.split import get_split_fn
from src.models.biointeract import BioInteract
from src.utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, DATA_DIR, LOGS_DIR, RESULTS_DIR


configure_matplotlib()

REPORT_PATH = RESULTS_DIR / "interpretability" / "interpretability_report.json"
STRICT_METRICS_PATH = PROJECT_ROOT.parent / "revision" / "analysis" / "strict_split_metrics.json"
SPLIT_META = {
    "random": {
        "label": "Random",
        "checkpoint": CHECKPOINTS_DIR / "best_random.pt",
        "result_json": RESULTS_DIR / "test_random.json",
        "log_path": LOGS_DIR / "run_random.log",
        "color": PALETTE["teal"],
    },
    "cold_target": {
        "label": "Target-ID-held-out",
        "checkpoint": CHECKPOINTS_DIR / "best_cold_target.pt",
        "result_json": RESULTS_DIR / "test_cold_target.json",
        "log_path": LOGS_DIR / "run_cold_target.log",
        "color": PALETTE["gold"],
    },
    "cold_drug": {
        "label": "Drug-ID-held-out",
        "checkpoint": CHECKPOINTS_DIR / "best_cold_drug.pt",
        "result_json": RESULTS_DIR / "test_cold_drug.json",
        "log_path": LOGS_DIR / "run_cold_drug.log",
        "color": PALETTE["brick"],
    },
}
LOG_PATTERN = re.compile(r"E(\d+)\s*\|\s*loss=([0-9.]+)\s*\|\s*val_auroc=([0-9.]+)(?:\s*\|\s*val_auprc=([0-9.]+))?")


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_base_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIGS_DIR / "default.yaml").read_text(encoding="utf-8"))


def load_dataset_resources(config: dict[str, Any]) -> dict[str, Any]:
    base = DATA_DIR / "raw" / config["data"]["dataset"]
    interactions = pd.read_csv(base / "interactions.csv")
    drug_df = pd.read_csv(base / "drug_smiles.csv")
    target_df = pd.read_csv(base / "target_sequences.csv")
    return {
        "interactions": interactions,
        "drug_smiles": dict(zip(drug_df["drug_id"], drug_df["smiles"])),
        "target_sequences": dict(zip(target_df["target_id"], target_df["sequence"])),
    }


def build_model(model_config: dict[str, Any], checkpoint_path: Path, device: str) -> BioInteract:
    model = BioInteract(model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_model(model: BioInteract, batch: dict[str, Any], device: str, return_attention: bool = False):
    drug_batch = batch["drug_batch"].to(device)
    esm2 = batch["esm2_embedding"].to(device)
    physchem = batch["physicochemical"].to(device)
    domain = batch["domain_labels"].to(device)
    protein_mask = batch["protein_mask"].to(device)
    with torch.inference_mode():
        return model(drug_batch, esm2, physchem, domain, protein_mask, return_attention=return_attention)


def load_canonical_prediction_summary() -> dict[str, Any]:
    """Load the sole approved source for original entity-split metrics.

    ``revision.promote_canonical_entity_metrics`` writes this file after checking
    the full-precision checkpoint artifact.  Figures must not silently rerun
    inference or recover the retired historical prediction CSVs.
    """
    summary_path = FIGURE_DATA_DIR / "prediction_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Missing canonical prediction summary. Run "
            "python revision/promote_canonical_entity_metrics.py first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("canonical"):
        raise ValueError("prediction_summary.json is not marked canonical")
    missing = set(SPLIT_META) - set(summary.get("splits", {}))
    if missing:
        raise ValueError(f"canonical prediction summary is missing splits: {sorted(missing)}")
    return summary


def prepare_attention_curve(flat_scores: np.ndarray) -> dict[str, np.ndarray]:
    """Build a complete cumulative curve directly from archived raw scores."""
    scores = np.asarray(flat_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("attention scores must be finite, non-negative, and non-empty")
    total_attention = float(scores.sum())
    if total_attention <= 0:
        raise ValueError("attention scores must have positive total mass")
    sorted_scores = np.sort(scores)[::-1]
    return {
        "sorted_scores": sorted_scores,
        "percentiles": np.arange(1, scores.size + 1, dtype=np.float64) / scores.size * 100,
        "cumulative_attention": np.cumsum(sorted_scores) / total_attention,
    }


def load_attention_distribution(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load the archived 1,301,380-score attribution distribution without re-inference."""
    raw_path = FIGURE_DATA_DIR / "attention_distribution.npz"
    if not raw_path.exists():
        raise FileNotFoundError(
            "Missing archived attention_distribution.npz; do not regenerate a public "
            "distribution from a different model state."
        )
    raw = np.load(raw_path)
    if "flat_scores" not in raw:
        raise ValueError("attention_distribution.npz is missing flat_scores")
    flat_scores = raw["flat_scores"]
    curve = prepare_attention_curve(flat_scores)
    global_stats = report.get("global_stats", {})
    summary = {
        "n_samples": int(global_stats.get("n_samples", 0)),
        "n_residue_scores": int(flat_scores.size),
        "residue_attention_mean": float(np.mean(flat_scores)),
        "residue_attention_std": float(np.std(flat_scores)),
        "residue_attention_median": float(np.median(flat_scores)),
        "residue_attention_top1pct": float(np.percentile(flat_scores, 99)),
        "residue_attention_top5pct": float(np.percentile(flat_scores, 95)),
        "attention_sparsity": float(np.mean(flat_scores < 0.1)),
        "top_1pct_attention_mass": float(curve["cumulative_attention"][int(np.ceil(flat_scores.size * 0.01)) - 1]),
        "top_5pct_attention_mass": float(curve["cumulative_attention"][int(np.ceil(flat_scores.size * 0.05)) - 1]),
        "raw_scores_artifact": "BioInteract/results/figure_data/attention_distribution.npz",
        "normalisation": "pair-normalised residue scores pooled over all positive Davis pairs",
    }
    if summary["n_samples"] <= 0:
        raise ValueError("interpretability report does not provide a positive sample count")
    for key in (
        "residue_attention_mean",
        "residue_attention_std",
        "residue_attention_median",
        "residue_attention_top1pct",
        "residue_attention_top5pct",
        "attention_sparsity",
    ):
        if key in global_stats and not np.isclose(summary[key], global_stats[key], rtol=0, atol=2e-8):
            raise ValueError(f"raw attention statistic for {key} disagrees with interpretability report")
    _json_dump(FIGURE_DATA_DIR / "attention_distribution.json", summary)
    return summary, curve


def split_training_log_segments(log_text: str) -> list[list[dict[str, float | int]]]:
    """Split a log whenever its epoch counter resets, preserving only real traces."""
    segments: list[list[dict[str, float | int]]] = []
    current: list[dict[str, float | int]] = []
    previous_epoch: int | None = None
    for match in LOG_PATTERN.finditer(log_text):
        epoch = int(match.group(1))
        if previous_epoch is not None and epoch <= previous_epoch:
            if current:
                segments.append(current)
            current = []
        current.append(
            {
                "epoch": epoch,
                "loss": float(match.group(2)),
                "val_auroc": float(match.group(3)),
            }
        )
        previous_epoch = epoch
    if current:
        segments.append(current)
    return segments


def select_checkpoint_matching_segment(
    segments: list[list[dict[str, float | int]]], checkpoint_best_val_auroc: float, tolerance: float = 5e-4
) -> tuple[list[dict[str, float | int]], int]:
    """Select the earliest complete trace whose peak agrees with the saved checkpoint."""
    if not segments:
        raise ValueError("training log has no parseable epoch records")
    candidates = [
        (abs(max(float(record["val_auroc"]) for record in segment) - checkpoint_best_val_auroc), -len(segment), index, segment)
        for index, segment in enumerate(segments)
    ]
    error, _, index, segment = min(candidates, key=lambda item: item[:3])
    if error > tolerance:
        raise ValueError(
            "no contiguous training-log segment does not match the checkpoint best validation AUROC"
        )
    return segment, index


def parse_training_logs(prediction_summary: dict[str, Any]) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for split_name, metadata in SPLIT_META.items():
        checkpoint = torch.load(metadata["checkpoint"], map_location="cpu", weights_only=False)
        checkpoint_best = checkpoint.get("best_val_auroc", checkpoint.get("best_metric"))
        if checkpoint_best is None:
            raise ValueError(f"{metadata['checkpoint']} lacks a saved validation AUROC")
        segments = split_training_log_segments(
            metadata["log_path"].read_text(encoding="utf-8", errors="ignore")
        )
        selected, selected_index = select_checkpoint_matching_segment(segments, float(checkpoint_best))
        epochs = [int(record["epoch"]) for record in selected]
        losses = [float(record["loss"]) for record in selected]
        aurocs = [float(record["val_auroc"]) for record in selected]
        best_index = int(np.argmax(aurocs))
        curves[split_name] = {
            "epochs": epochs,
            "loss": losses,
            "val_auroc": aurocs,
            "best_epoch": epochs[best_index],
            "best_val_auroc": aurocs[best_index],
            "checkpoint_best_val_auroc": float(checkpoint_best),
            "source_log": metadata["log_path"].relative_to(PROJECT_ROOT).as_posix(),
            "segments_detected": len(segments),
            "selected_segment": selected_index + 1,
            "discarded_segment_ranges": [
                {"start_epoch": int(segment[0]["epoch"]), "end_epoch": int(segment[-1]["epoch"])}
                for index, segment in enumerate(segments)
                if index != selected_index
            ],
            "reported_auroc": prediction_summary["splits"][split_name]["reported_metrics"]["AUROC"],
        }
    return curves


def load_strict_metrics() -> dict[str, Any]:
    """Load reviewer-requested strict metrics without conflating their protocol."""
    if not STRICT_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing strict-split artifact: {STRICT_METRICS_PATH}")
    strict = json.loads(STRICT_METRICS_PATH.read_text(encoding="utf-8"))
    expected = {"sequence_grouped_cold_target", "sequence_grouped_cold_both"}
    if missing := expected - set(strict):
        raise ValueError(f"strict metrics are missing protocols: {sorted(missing)}")
    return strict


def fig2_performance(prediction_summary: dict[str, Any], strict_metrics: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    figure.subplots_adjust(bottom=0.27, wspace=0.20)
    panels = (
        (
            axes[0],
            "Original identifier-based partitions",
            [
                (SPLIT_META[name]["label"], prediction_summary["splits"][name]["reported_metrics"])
                for name in SPLIT_META
            ],
            [SPLIT_META[name]["color"] for name in SPLIT_META],
        ),
        (
            axes[1],
            "Exact-sequence-grouped within-Davis tests",
            [
                ("Cold-target", strict_metrics["sequence_grouped_cold_target"]["metrics"]),
                ("Cold-both", strict_metrics["sequence_grouped_cold_both"]["metrics"]),
            ],
            [PALETTE["sage"], PALETTE["brick"]],
        ),
    )
    x = np.arange(2)
    for axis, title, records, colors in panels:
        width = 0.72 / len(records)
        for index, ((label, metrics), color) in enumerate(zip(records, colors)):
            offset = (index - (len(records) - 1) / 2) * width
            values = [metrics["AUROC"], metrics["AUPRC"]]
            bars = axis.bar(x + offset, values, width=width, color=color, label=label)
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.018,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.7,
                )
        axis.set_xticks(x)
        axis.set_xticklabels(["AUROC", "AUPRC"])
        axis.set_ylim(0, 1.05)
        axis.set_title(title, pad=12)
        axis.legend(frameon=False, fontsize=8.2, loc="upper right")
        soften_axes(axis, "y")
    axes[0].set_ylabel("Score")
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    save_figure(
        figure,
        "fig2_performance",
        metadata={
            "title": "Canonical entity-split and strict within-Davis performance",
            "canonical_prediction_summary": "BioInteract/results/figure_data/prediction_summary.json",
            "strict_metrics": "revision/analysis/strict_split_metrics.json",
        },
    )


def fig5_attention_sparsity(attention_data: dict[str, Any], curve: dict[str, np.ndarray]) -> None:
    """Plot the saved score distribution and its directly recomputed cumulative mass."""
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    distribution_axis, cumulative_axis = axes
    bins = np.geomspace(max(float(curve["sorted_scores"][-1]), 1e-7), 1.0, 55)
    distribution_axis.hist(
        curve["sorted_scores"], bins=bins, color=PALETTE["sky"], edgecolor="white", linewidth=0.25
    )
    distribution_axis.axvline(0.1, color=PALETTE["brick"], linestyle="--", linewidth=1.1)
    distribution_axis.set_xscale("log")
    distribution_axis.set_yscale("log")
    distribution_axis.set_xlabel("Pair-normalised residue attribution (log scale)")
    distribution_axis.set_ylabel("Residue observations (log scale)")
    distribution_axis.set_title("Attribution-score distribution")
    cumulative_axis.plot(
        curve["percentiles"], curve["cumulative_attention"], color=PALETTE["teal"]
    )
    cumulative_axis.set_xlabel("Top residue percentile included (%)")
    cumulative_axis.set_ylabel("Cumulative share of total attribution")
    cumulative_axis.set_ylim(0, 1.02)
    cumulative_axis.set_xlim(0, 100)
    cumulative_axis.set_title("Cumulative attribution concentration")
    for axis, label in zip(axes, ("A", "B")):
        soften_axes(axis, "y")
        panel_label(axis, label)
    save_figure(
        figure,
        "fig5_attention_sparsity",
        metadata={
            "title": "Archived aggregate model-native attribution distribution",
            "summary": attention_data,
            "curve": {
                "source": "flat_scores sorted descending and cumulatively normalised at figure generation",
                "n_points": int(curve["percentiles"].size),
                "terminal_cumulative_attention": float(curve["cumulative_attention"][-1]),
            },
        },
    )


def fig4_training(training_curves: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.7))
    auroc_axis, loss_axis = axes
    for split_name, curves in training_curves.items():
        metadata = SPLIT_META[split_name]
        auroc_axis.plot(curves["epochs"], curves["val_auroc"], color=metadata["color"], label=metadata["label"])
        loss_axis.plot(curves["epochs"], curves["loss"], color=metadata["color"], label=metadata["label"])
    auroc_axis.set_xlabel("Epoch")
    auroc_axis.set_ylabel("Validation AUROC")
    auroc_axis.set_title("Validation AUROC trajectory")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Training loss")
    loss_axis.set_title("Training loss trajectory")
    for axis in axes:
        soften_axes(axis, "y")
        axis.legend(frameon=False)
    metadata = {
        "title": "Archived checkpoint-matched training dynamics",
        "curves": training_curves,
        "note": (
            "Each line is one contiguous archived epoch trace selected because its "
            "peak validation AUROC matches the saved checkpoint. Restarted or duplicated "
            "log segments are excluded; no fitted or interpolated trends are plotted."
        ),
    }
    save_figure(figure, "fig4_training", metadata=metadata)
    _json_dump(FIGURE_DATA_DIR / "training_curves.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aggregate public BioInteract figures.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--force-attention-refresh",
        action="store_true",
        help="Recompute the separate aggregate-attention summary; metrics remain canonical.",
    )
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    RDLogger.DisableLog("rdApp.warning")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {"global_stats": {}}
    prediction_summary = load_canonical_prediction_summary()
    strict_metrics = load_strict_metrics()
    base_config = load_base_config()
    resources = load_dataset_resources(base_config)
    attention_data, attention_curve = load_attention_distribution(report)
    training_curves = parse_training_logs(prediction_summary)
    fig2_performance(prediction_summary, strict_metrics)
    fig5_attention_sparsity(attention_data, attention_curve)
    if training_curves:
        fig4_training(training_curves)
        manifest = ["fig2_performance", "fig4_training", "fig5_attention_sparsity"]
    else:
        manifest = ["fig2_performance", "fig5_attention_sparsity"]
    write_manifest(manifest, replace=True)


if __name__ == "__main__":
    main()
