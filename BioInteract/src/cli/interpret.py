"""Generate a global model-native attribution summary for BioInteract.

The released command reports aggregate attention statistics for binary
high-affinity interaction classification. It does not generate pair-specific
structural or contact outputs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DTIDataset, collate_dti
from src.interpret.attention_analysis import extract_interaction_map
from src.models.biointeract import BioInteract
from src.utils.logger import setup_logger
from src.utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, DATA_DIR, RESULTS_DIR, resolve_project_path


def load_config(path: str) -> dict:
    """Load a workflow configuration."""
    with open(resolve_project_path(path), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def global_attention_statistics(model, dataloader, device: str, n_batches: int = 50) -> dict:
    """Compute aggregate normalised model-native attention statistics."""
    all_residue_scores: list[np.ndarray] = []
    model.eval()
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= n_batches:
            break
        attention = extract_interaction_map(model, batch, device)
        for row in range(attention["interaction_map"].shape[0]):
            interaction_map = attention["interaction_map"][row]
            drug_mask = attention["drug_mask"][row]
            protein_mask = attention["protein_mask"][row]
            residue_scores = interaction_map.copy()
            residue_scores[~drug_mask] = 0
            residue_scores = residue_scores.sum(axis=0)
            residue_scores[~protein_mask] = 0
            valid_scores = residue_scores[protein_mask]
            if len(valid_scores) == 0:
                continue
            maximum = valid_scores.max()
            if maximum > 0:
                valid_scores = valid_scores / maximum
            all_residue_scores.append(valid_scores)
    if not all_residue_scores:
        return {}
    flattened = np.concatenate(all_residue_scores)
    return {
        "residue_attention_mean": float(flattened.mean()),
        "residue_attention_std": float(flattened.std()),
        "residue_attention_median": float(np.median(flattened)),
        "residue_attention_top1pct": float(np.percentile(flattened, 99)),
        "residue_attention_top5pct": float(np.percentile(flattened, 95)),
        "n_samples": len(all_residue_scores),
        "attention_sparsity": float((flattened < 0.1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise global model-native attention attribution.")
    parser.add_argument("--config", default=str(CONFIGS_DIR / "default.yaml"))
    parser.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "best.pt"))
    parser.add_argument("--output_dir", default=str(RESULTS_DIR / "interpretability"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    logger = setup_logger("interpret")
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading trained model for global attention summary")
    model = BioInteract(config["model"]).to(device)
    checkpoint = torch.load(resolve_project_path(args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset_root = DATA_DIR / "raw" / config["data"]["dataset"]
    interactions = pd.read_csv(dataset_root / "interactions.csv")
    drug_df = pd.read_csv(dataset_root / "drug_smiles.csv")
    target_df = pd.read_csv(dataset_root / "target_sequences.csv")
    positive_interactions = interactions[interactions["label"] == 1]
    dataset = DTIDataset(
        positive_interactions,
        drug_smiles=dict(zip(drug_df["drug_id"], drug_df["smiles"])),
        target_sequences=dict(zip(target_df["target_id"], target_df["sequence"])),
        esm2_cache_dir=config["data"].get("esm2_cache_dir", "data/esm2_embeddings"),
        max_protein_len=config["data"].get("max_protein_len", 1200),
        use_domain_features=config["model"]["target_encoder"].get("use_domain_features", True),
        esm2_dim=config["model"]["target_encoder"].get("esm2_dim", 640),
        task="classification",
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_dti, num_workers=0)
    n_batches = config.get("interpret", {}).get("attention_batches", 50)
    global_stats = global_attention_statistics(model, loader, device, n_batches=n_batches)

    report_path = output_dir / "interpretability_report.json"
    report_path.write_text(json.dumps({"global_stats": global_stats}, indent=2), encoding="utf-8")
    logger.info("Saved global model-native attribution summary to %s", report_path)


if __name__ == "__main__":
    main()
