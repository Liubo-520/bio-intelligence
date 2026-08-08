"""Train every comparison model under one identical Davis protocol.

Reviewer 2 asked for (i) a baseline that also consumes ESM-2 representations
and (ii) comparisons with additional models re-evaluated in our own setting
rather than quoted from other papers. This script trains BioInteract and five
re-implemented baselines with the same partitions, seed, optimiser, schedule,
loss, model-selection rule, and validation-selected decision threshold, and
writes one JSON record per (model, split) run.

Usage::

    python revision_v2/run_baseline_suite.py --out revision_v2/analysis/baseline_suite.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "BioInteract"
for path in (str(ROOT), str(PROJECT), str(ROOT / "revision"), str(ROOT / "revision_v2")):
    if path not in sys.path:
        sys.path.insert(0, path)

from baseline_models import (  # noqa: E402
    MODEL_REGISTRY,
    BioInteractWrapper,
    encode_sequence,
    encode_smiles,
)
from revision_analysis import (  # noqa: E402
    sequence_group_cold_both_split,
    sequence_group_cold_target_split,
)
from src.data.mol_graph import smiles_to_graph, smiles_to_morgan  # noqa: E402
from src.data.protein_feat import ProteinFeatureBuilder  # noqa: E402
from src.data.split import cold_drug_split, cold_target_split, random_split  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

MAX_PROTEIN_LEN = 1200
ESM2_DIM = 640

# Resource budget. The cross-attention models allocate an (atoms x residues)
# map per pair, so a large physical batch over 1,200-residue proteins is what
# drives accelerator memory. We therefore train on a small physical batch with
# gradient accumulation, which keeps the effective batch at the configured 64
# while cutting peak activation memory, run under automatic mixed precision as
# the released training scripts do, and cap the fraction of the device this
# process may reserve so that the suite cannot exhaust a shared GPU.
MICRO_BATCH = 16
EVAL_BATCH = 16
GPU_MEMORY_FRACTION = 0.5


def set_seed(seed: int) -> None:
    """Fix every random source used by a single training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FeatureStore:
    """Featurise every Davis drug and target once and share it across runs."""

    def __init__(self, data_root: Path, esm2_cache_dir: Path):
        drugs = pd.read_csv(data_root / "drug_smiles.csv")
        targets = pd.read_csv(data_root / "target_sequences.csv")
        builder = ProteinFeatureBuilder(
            esm2_cache_dir=str(esm2_cache_dir),
            max_protein_len=MAX_PROTEIN_LEN,
            use_domain_features=True,
            esm2_dim=ESM2_DIM,
        )
        self.graphs: dict[str, Data] = {}
        self.morgan: dict[str, torch.Tensor] = {}
        self.smiles_tokens: dict[str, torch.Tensor] = {}
        for row in drugs.itertuples(index=False):
            graph = smiles_to_graph(row.smiles)
            if graph is None:
                raise ValueError(f"RDKit could not parse Davis drug {row.drug_id}")
            self.graphs[row.drug_id] = graph
            self.morgan[row.drug_id] = smiles_to_morgan(row.smiles, n_bits=1024)
            self.smiles_tokens[row.drug_id] = encode_smiles(row.smiles)

        self.esm2: dict[str, torch.Tensor] = {}
        self.physchem: dict[str, torch.Tensor] = {}
        self.domains: dict[str, torch.Tensor] = {}
        self.lengths: dict[str, int] = {}
        self.sequence_tokens: dict[str, torch.Tensor] = {}
        for row in targets.itertuples(index=False):
            features = builder.build(row.target_id, row.sequence)
            # Half precision halves the resident feature store; the training
            # loop casts back to float32 before the forward pass.
            self.esm2[row.target_id] = features["esm2_embedding"].half()
            self.physchem[row.target_id] = features["physicochemical"]
            self.domains[row.target_id] = features["domain_labels"]
            self.lengths[row.target_id] = int(features["sequence_length"])
            self.sequence_tokens[row.target_id] = encode_sequence(
                row.sequence[:MAX_PROTEIN_LEN]
            )


class PairDataset(Dataset):
    """Index drug--target pairs into the shared :class:`FeatureStore`."""

    def __init__(self, pairs: pd.DataFrame, store: FeatureStore, need_esm: bool):
        self.pairs = pairs.reset_index(drop=True)
        self.store = store
        self.need_esm = need_esm

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        row = self.pairs.iloc[index]
        drug_id, target_id = row["drug_id"], row["target_id"]
        item = {
            "drug_graph": self.store.graphs[drug_id],
            "morgan_fp": self.store.morgan[drug_id],
            "smiles_tokens": self.store.smiles_tokens[drug_id],
            "sequence_tokens": self.store.sequence_tokens[target_id],
            "length": self.store.lengths[target_id],
            "label": float(row["label"]),
        }
        if self.need_esm:
            item["esm2"] = self.store.esm2[target_id]
            item["physchem"] = self.store.physchem[target_id]
            item["domains"] = self.store.domains[target_id]
        return item


def collate(batch: list[dict]) -> dict:
    """Pad proteins to the batch maximum and build the PyG drug batch."""
    max_len = max(item["length"] for item in batch)
    size = len(batch)
    mask = torch.zeros(size, max_len, dtype=torch.bool)
    for index, item in enumerate(batch):
        mask[index, : item["length"]] = True

    collated = {
        "drug_batch": Batch.from_data_list([item["drug_graph"] for item in batch]),
        "morgan_fp": torch.stack([item["morgan_fp"] for item in batch]),
        "smiles_tokens": torch.stack([item["smiles_tokens"] for item in batch]),
        "sequence_tokens": torch.stack([item["sequence_tokens"] for item in batch]),
        "protein_mask": mask,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float),
    }
    if "esm2" in batch[0]:
        esm2 = torch.zeros(size, max_len, ESM2_DIM)
        physchem = torch.zeros(size, max_len, 4)
        domains = torch.zeros(size, max_len, dtype=torch.long)
        for index, item in enumerate(batch):
            length = item["length"]
            esm2[index, :length] = item["esm2"].float()
            physchem[index, :length] = item["physchem"]
            domains[index, :length] = item["domains"]
        collated["esm2_embedding"] = esm2
        collated["physicochemical"] = physchem
        collated["domain_labels"] = domains
    return collated


def to_device(batch: dict, device: str) -> dict:
    """Move every tensor field of a collated batch onto ``device``."""
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and sigmoid probabilities for one loader.

    Evaluation stays in full precision so the reported metrics do not depend on
    the mixed-precision training path.
    """
    model.eval()
    labels, probabilities = [], []
    with torch.inference_mode():
        for batch in loader:
            moved = to_device(batch, device)
            logits = model(moved)
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy().ravel())
            labels.append(batch["label"].numpy().ravel())
            del moved, logits
    return np.concatenate(labels), np.concatenate(probabilities)


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Compute ranking metrics plus F1/precision/recall at a frozen threshold."""
    prediction = (probabilities >= threshold).astype(int)
    true_positive = int(np.sum((prediction == 1) & (labels == 1)))
    false_positive = int(np.sum((prediction == 1) & (labels == 0)))
    false_negative = int(np.sum((prediction == 0) & (labels == 1)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AUPRC": float(average_precision_score(labels, probabilities)),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "threshold_from_validation": float(threshold),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select the maximum-F1 threshold on validation predictions only."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[int(np.argmax(f1))])


def build_model(name: str, config: dict) -> nn.Module:
    """Instantiate one comparison model by registry name."""
    if name == "BioInteract":
        model_config = copy.deepcopy(config["model"])
        model_config["drug_encoder"]["use_morgan_fp"] = False
        model_config["drug_encoder"]["drop_node"] = 0.0
        model_config["drug_encoder"]["drop_edge"] = 0.0
        return BioInteractWrapper(model_config)
    return MODEL_REGISTRY[name]()


def make_splits(interactions: pd.DataFrame, targets: pd.DataFrame, protocol: str, seed: int):
    """Return the train/validation/test partition for one protocol name."""
    if protocol == "random":
        return random_split(interactions, seed=seed)
    if protocol == "target_id_held_out":
        return cold_target_split(interactions, seed=seed)
    if protocol == "drug_id_held_out":
        return cold_drug_split(interactions, seed=seed)
    if protocol == "sequence_grouped_cold_target":
        return sequence_group_cold_target_split(interactions, targets, seed=seed)
    if protocol == "sequence_grouped_cold_both":
        return sequence_group_cold_both_split(interactions, targets, seed=seed)
    raise ValueError(f"unknown protocol: {protocol}")


def train_run(model_name: str, protocol: str, store: FeatureStore,
              interactions: pd.DataFrame, targets: pd.DataFrame, config: dict,
              max_epochs: int, patience: int) -> dict:
    """Train and evaluate one (model, protocol) pair under the shared recipe."""
    seed = int(config["training"]["seed"])
    set_seed(seed)
    train_df, val_df, test_df = make_splits(interactions, targets, protocol, seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, config).to(device)
    need_esm = bool(getattr(model, "uses_esm2", True))
    effective_batch = int(config["training"]["batch_size"])
    accumulation = max(1, effective_batch // MICRO_BATCH)
    loaders = {
        name: DataLoader(
            PairDataset(frame, store, need_esm),
            batch_size=MICRO_BATCH if name == "train" else EVAL_BATCH,
            shuffle=name == "train",
            collate_fn=collate,
            num_workers=0,
        )
        for name, frame in (("train", train_df), ("val", val_df), ("test", test_df))
    }

    positives = float(train_df["label"].sum())
    pos_weight = torch.tensor([(len(train_df) - positives) / positives], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    warmup = int(config["training"]["warmup_epochs"])

    def schedule(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        fraction = (epoch - warmup) / max(max_epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = GradScaler("cuda", enabled=device == "cuda")
    best_state, best_auroc, best_epoch, waiting = None, -np.inf, None, 0
    history: list[dict] = []
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loaders["train"], start=1):
            moved = to_device(batch, device)
            with autocast("cuda", enabled=device == "cuda"):
                logits = model(moved)
                # 0.05 label smoothing, matching the strict-split training recipe.
                loss = criterion(logits, moved["label"] * 0.95 + 0.025)
            losses.append(float(loss.detach().cpu()))
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            del moved, logits, loss
        scheduler.step()
        val_labels, val_probabilities = predict(model, loaders["val"], device)
        val_auroc = float(roc_auc_score(val_labels, val_probabilities))
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_auroc": val_auroc,
                "validation_auprc": float(average_precision_score(val_labels, val_probabilities)),
            }
        )
        print(
            f"[{model_name}|{protocol}] epoch {epoch:03d} loss={history[-1]['training_loss']:.4f} "
            f"val_AUROC={val_auroc:.4f}",
            flush=True,
        )
        if val_auroc > best_auroc:
            best_auroc, best_epoch, waiting = val_auroc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    val_labels, val_probabilities = predict(model, loaders["val"], device)
    threshold = select_threshold(val_labels, val_probabilities)
    test_labels, test_probabilities = predict(model, loaders["test"], device)
    parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0.0
    )
    record = {
        "model": model_name,
        "protocol": protocol,
        "seed": seed,
        "device": device,
        "uses_esm2": need_esm,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": accumulation,
        "effective_batch": MICRO_BATCH * accumulation,
        "mixed_precision": device == "cuda",
        "peak_gpu_memory_gib": round(peak_memory, 2),
        "trainable_parameters": parameters,
        "split_sizes": {
            "train_pairs": int(len(train_df)),
            "validation_pairs": int(len(val_df)),
            "test_pairs": int(len(test_df)),
            "train_positives": int(train_df["label"].sum()),
            "validation_positives": int(val_df["label"].sum()),
            "test_positives": int(test_df["label"].sum()),
        },
        "best_validation_auroc": float(best_auroc),
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "wall_clock_seconds": round(time.time() - started, 1),
        "metrics": threshold_metrics(test_labels, test_probabilities, threshold),
        "history": history,
    }

    # Release the model, optimiser state and cached blocks before the next run,
    # so that peak memory is set by a single model rather than by the suite.
    del model, optimizer, scheduler, scaler, best_state, loaders
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return record


def main(argv: list[str] | None = None) -> None:
    """Run the requested subset of the matched baseline suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="revision_v2/analysis/baseline_suite.json")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["DeepDTA", "GraphDTA", "ESM2-MLP", "ESM2-GraphConcat", "ESM2-Bilinear", "BioInteract"],
    )
    parser.add_argument(
        "--protocols",
        nargs="*",
        default=["random", "target_id_held_out", "drug_id_held_out"],
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=GPU_MEMORY_FRACTION,
        help="Upper bound on the fraction of device memory this process may reserve",
    )
    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(
            f"GPU budget: {args.gpu_memory_fraction:.0%} of {total:.1f} GiB; "
            f"micro-batch {MICRO_BATCH} with gradient accumulation to "
            f"effective batch 64; mixed precision enabled",
            flush=True,
        )

    config = yaml.safe_load((PROJECT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    data_root = PROJECT / "data" / "raw" / "davis"
    interactions = pd.read_csv(data_root / "interactions.csv")
    targets = pd.read_csv(data_root / "target_sequences.csv")
    store = FeatureStore(data_root, PROJECT / config["data"]["esm2_cache_dir"])

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    done = {(run["model"], run["protocol"]) for run in runs}

    for protocol in args.protocols:
        for model_name in args.models:
            if (model_name, protocol) in done:
                print(f"skip {model_name}|{protocol} (already recorded)", flush=True)
                continue
            record = train_run(
                model_name, protocol, store, interactions, targets, config,
                args.max_epochs, args.patience,
            )
            runs.append(record)
            out_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
            print(
                f"done {model_name}|{protocol}: AUROC={record['metrics']['AUROC']:.4f} "
                f"AUPRC={record['metrics']['AUPRC']:.4f} ({record['wall_clock_seconds']}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
