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


def compute_attention_distribution(
    base_config: dict[str, Any], resources: dict[str, Any], report: dict[str, Any], device: str, force: bool
) -> dict[str, Any]:
    summary_path = FIGURE_DATA_DIR / "attention_distribution.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    config = copy.deepcopy(base_config)
    positive_df = resources["interactions"][resources["interactions"]["label"] == 1][["drug_id", "target_id", "label"]].reset_index(drop=True)
    dataset = DTIDataset(
        positive_df,
        drug_smiles=resources["drug_smiles"],
        target_sequences=resources["target_sequences"],
        esm2_cache_dir=config["data"].get("esm2_cache_dir", "data/esm2_embeddings"),
        max_protein_len=config["data"].get("max_protein_len", 1200),
        use_domain_features=config["model"]["target_encoder"].get("use_domain_features", True),
        esm2_dim=config["model"]["target_encoder"].get("esm2_dim", 640),
        task="classification",
    )
    loader = DataLoader(dataset, batch_size=12, shuffle=False, collate_fn=collate_dti, num_workers=0)
    model = build_model(config["model"], CHECKPOINTS_DIR / "best.pt", device)
    scores: list[np.ndarray] = []
    for batch in loader:
        _, attention = run_model(model, batch, device, return_attention=True)
        for index in range(attention["interaction_map"].size(0)):
            residue_scores = attention["interaction_map"][index][attention["drug_mask"][index]].sum(dim=0)
            valid_scores = residue_scores[attention["protein_mask"][index]]
            if valid_scores.numel():
                scores.append((valid_scores / valid_scores.max().clamp(min=1e-8)).detach().cpu().numpy())
    flattened = np.concatenate(scores)
    summary = {
        "n_samples": int(len(scores)),
        "n_residue_scores": int(len(flattened)),
        "residue_attention_mean": float(flattened.mean()),
        "residue_attention_std": float(flattened.std()),
        "residue_attention_median": float(np.median(flattened)),
        "residue_attention_top1pct": float(np.percentile(flattened, 99)),
        "residue_attention_top5pct": float(np.percentile(flattened, 95)),
        "attention_sparsity": float((flattened < 0.1).mean()),
        "report_global_stats": report.get("global_stats", {}),
    }
    _json_dump(summary_path, summary)
    return summary


def parse_training_logs(prediction_summary: dict[str, Any]) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for split_name, metadata in SPLIT_META.items():
        epochs, losses, aurocs = [], [], []
        for line in metadata["log_path"].read_text(encoding="utf-8", errors="ignore").splitlines():
            match = LOG_PATTERN.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                aurocs.append(float(match.group(3)))
        if not epochs:
            continue
        best_index = int(np.argmax(aurocs))
        curves[split_name] = {
            "epochs": epochs,
            "loss": losses,
            "val_auroc": aurocs,
            "best_epoch": epochs[best_index],
            "best_val_auroc": aurocs[best_index],
            "source_log": metadata["log_path"].relative_to(PROJECT_ROOT).as_posix(),
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


def fig3_attention_sparsity(attention_data: dict[str, Any]) -> None:
    values = [
        attention_data["residue_attention_mean"],
        attention_data["residue_attention_median"],
        attention_data["residue_attention_top5pct"],
        attention_data["residue_attention_top1pct"],
    ]
    labels = ["Mean", "Median", "Top 5%", "Top 1%"]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.bar(labels, values, color=[PALETTE["slate"], PALETTE["sky"], PALETTE["teal"], PALETTE["gold"]])
    axis.set_ylabel("Normalised model-native attention attribution")
    axis.set_title("Aggregate attribution distribution")
    soften_axes(axis, "y")
    save_figure(figure, "fig3_sparsity", metadata={"title": "Aggregate model-native attribution", "summary": attention_data})


def fig9_training(training_curves: dict[str, Any]) -> None:
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
        "title": "Archived training dynamics with canonical checkpoint metrics",
        "curves": training_curves,
        "note": (
            "Lines are archived training-log diagnostics. The reported AUROC for each "
            "split is the canonical full-precision checkpoint evaluation, not the "
            "legacy test metric recorded in the log."
        ),
    }
    save_figure(figure, "fig9_training", metadata=metadata)
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
    attention_data = compute_attention_distribution(
        base_config, resources, report, device, args.force_attention_refresh
    )
    training_curves = parse_training_logs(prediction_summary)
    fig2_performance(prediction_summary, strict_metrics)
    fig3_attention_sparsity(attention_data)
    if training_curves:
        fig9_training(training_curves)
        manifest = ["fig2_performance", "fig3_sparsity", "fig9_training"]
    else:
        manifest = ["fig2_performance", "fig3_sparsity"]
    write_manifest(manifest)


if __name__ == "__main__":
    main()
