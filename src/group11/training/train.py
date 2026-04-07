"""
train.py — Training Loop
==========================
Loads gnn_dataset.pt, splits topologies 400/50/50 (train/val/test),
trains a specified GNN model, and saves the best checkpoint.

Usage:
    python src/group11/training/train.py --model gcn
    python src/group11/training/train.py --model gat
    python src/group11/training/train.py --model mpnn

Checkpoints saved to:
    data/processed/checkpoints/<model>_best.pt

Run from repo root.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.group11.models import GCNLatencyPredictor, GATLatencyPredictor, MPNNLatencyPredictor, RouteNetFermiPredictor, EdgeWeightGNN, SpatialTemporalGNN
from src.group11.dataset_assembly.augment_dataset import RouteNetData
from src.group11.dataset_assembly.build_temporal_dataset import TemporalData  # noqa: F401 — required for torch.load to deserialise TemporalData objects

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH          = os.path.join("data", "processed", "dataset", "gnn_dataset.pt")
ROUTENET_DATASET_PATH = os.path.join("data", "processed", "dataset", "routenet_dataset.pt")
TEMPORAL_DATASET_PATH = os.path.join("data", "processed", "dataset", "temporal_dataset.pt")
CHECKPOINT_DIR        = os.path.join("data", "processed", "checkpoints")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
TRAIN_SIZE  = 400
VAL_SIZE    = 50
TEST_SIZE   = 50
SEED        = 42
BATCH_SIZE  = 16
EPOCHS      = 50
LR          = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIM  = 64
DROPOUT     = 0.1

LOAD_CONDITIONS = ["low", "medium", "high", "flash"]

# ---------------------------------------------------------------------------
# Per-model peak training configs
# Tuned based on model capacity and dataset characteristics.
# ---------------------------------------------------------------------------
PER_MODEL_CONFIGS = {
    "gcn": {
        "lr":                5e-4,
        "weight_decay":      1e-5,
        "epochs":            200,
        "sched_patience":    10,
        "early_stop_patience": 25,
    },
    "gat": {
        "lr":                5e-4,
        "weight_decay":      1e-5,
        "epochs":            200,
        "sched_patience":    10,
        "early_stop_patience": 25,
    },
    "mpnn": {
        # 308K params (18× GCN): needs lower LR, stronger regularisation, more epochs
        "lr":                3e-4,
        "weight_decay":      1e-3,
        "epochs":            250,
        "sched_patience":    15,
        "early_stop_patience": 35,
    },
    "routenet": {
        # batch_size=1 → many updates per epoch; fewer max epochs are enough
        "lr":                5e-4,
        "weight_decay":      1e-5,
        "epochs":            150,
        "sched_patience":    10,
        "early_stop_patience": 25,
    },
    "custom": {
        # EdgeWeightGNN (42K): moderate capacity, slightly lower LR than GCN
        "lr":                3e-4,
        "weight_decay":      1e-4,
        "epochs":            200,
        "sched_patience":    10,
        "early_stop_patience": 25,
    },
    "tgnn": {
        # Temporal forecasting, batch_size=1, Transformer inside
        "lr":                3e-4,
        "weight_decay":      1e-4,
        "epochs":            200,
        "sched_patience":    15,
        "early_stop_patience": 30,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fix_u(data: Data) -> Data:
    """Ensure u is 2-D: (1, 11). Handles both (11,) and (1, 11) shapes."""
    if data.u.dim() == 1:
        data.u = data.u.unsqueeze(0)
    return data


def load_dataset(path: str):
    dataset = torch.load(path, weights_only=False)
    dataset = [_fix_u(d) for d in dataset]
    return dataset


def split_dataset(dataset, seed: int = SEED):
    """Reproducible topology-level split: 400 train / 50 val / 50 test."""
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng).tolist()
    train_idx = indices[:TRAIN_SIZE]
    val_idx   = indices[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test_idx  = indices[TRAIN_SIZE + VAL_SIZE :]
    return (
        [dataset[i] for i in train_idx],
        [dataset[i] for i in val_idx],
        [dataset[i] for i in test_idx],
    )


def build_model(name: str, device: torch.device) -> nn.Module:
    name = name.lower()
    if name == "gcn":
        model = GCNLatencyPredictor(hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    elif name == "gat":
        model = GATLatencyPredictor(hidden_dim=32, heads=4, dropout=DROPOUT)
    elif name == "mpnn":
        model = MPNNLatencyPredictor(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, steps=3)
    elif name == "routenet":
        model = RouteNetFermiPredictor(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, steps=8)
    elif name == "custom":
        model = EdgeWeightGNN(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, steps=3)
    elif name == "tgnn":
        model = SpatialTemporalGNN(hidden_dim=HIDDEN_DIM, dropout=DROPOUT, steps=6, nhead=4)
    else:
        raise ValueError(f"Unknown model: {name}. Choose gcn, gat, mpnn, routenet, custom, or tgnn.")
    return model.to(device)


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimiser, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_edges = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)

            # u shape fix inside batched loader: (B, 11)
            if batch.u.dim() == 1:
                batch.u = batch.u.unsqueeze(0)

            # Dispatch: SpatialTemporalGNN and RouteNet-Fermi take the full
            # batch object; standard models use kwargs.
            if isinstance(model, SpatialTemporalGNN):
                pred = model(batch)
                loss = criterion(pred, batch.y_next)
            elif isinstance(model, RouteNetFermiPredictor):
                pred = model(batch)
                loss = criterion(pred, batch.y_edge)
            else:
                pred = model(
                    x          = batch.x,
                    edge_index = batch.edge_index,
                    edge_attr  = batch.edge_attr,
                    u          = batch.u,
                    batch      = batch.batch,
                )
                loss = criterion(pred, batch.y_edge)

            if train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimiser.step()

            y_ref        = batch.y_next if isinstance(model, SpatialTemporalGNN) else batch.y_edge
            num_edges    = y_ref.size(0)
            total_loss  += loss.item() * num_edges
            total_edges += num_edges

    return total_loss / total_edges if total_edges > 0 else float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train(model_name: str):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Pick dataset and batch size for each model family
    if model_name == "routenet":
        dataset_path = ROUTENET_DATASET_PATH
        batch_size   = 1
    elif model_name == "tgnn":
        dataset_path = TEMPORAL_DATASET_PATH
        batch_size   = 1
    else:
        dataset_path = DATASET_PATH
        batch_size   = BATCH_SIZE

    print(f"Loading dataset from {dataset_path} ...")
    dataset = load_dataset(dataset_path)
    print(f"  {len(dataset)} graphs loaded.")

    train_data, val_data, test_data = split_dataset(dataset)
    print(f"  Split: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    model     = build_model(model_name, device)
    criterion = nn.MSELoss()

    cfg          = PER_MODEL_CONFIGS[model_name]
    lr           = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    epochs       = cfg["epochs"]
    es_patience  = cfg["early_stop_patience"]

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=cfg["sched_patience"]
    )

    best_val_loss   = float("inf")
    no_improve      = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")

    print(f"\nTraining {model_name.upper()} — max {epochs} epochs, early-stop patience {es_patience}\n")
    header = f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'LR':>8}  {'Time':>6}  {'Best':>6}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr_now  = optimiser.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_name":  model_name,
                    "model_state": model.state_dict(),
                    "val_loss":    val_loss,
                },
                checkpoint_path,
            )
            marker = "*"
        else:
            no_improve += 1
            marker = ""

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {lr_now:>8.2e}  {elapsed:>5.1f}s  {marker}")

        if no_improve >= es_patience:
            print(f"\nEarly stop: no improvement for {es_patience} consecutive epochs.")
            break

    print(f"\nBest val MSE: {best_val_loss:.4f}")
    print(f"Checkpoint saved -> {checkpoint_path}")
    print(f"\nTo evaluate: python src/group11/training/evaluate.py --model {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN latency predictor.")
    parser.add_argument(
        "--model", type=str, default="mpnn",
        choices=["gcn", "gat", "mpnn", "routenet", "custom", "tgnn"],
        help="Model architecture to train (default: mpnn)",
    )
    args = parser.parse_args()
    train(args.model)
