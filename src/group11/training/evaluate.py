"""
evaluate.py — Evaluation Metrics
==================================
Loads a trained model checkpoint and evaluates it on the held-out test set
(50 topologies). Reports per-condition MAE, RMSE, and R² for all 4 load
conditions (low, medium, high, flash), plus aggregated averages.

Usage:
    python src/group11/training/evaluate.py --model gcn
    python src/group11/training/evaluate.py --model gat
    python src/group11/training/evaluate.py --model mpnn

Run from repo root.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.group11.models import GCNLatencyPredictor, GATLatencyPredictor, MPNNLatencyPredictor, RouteNetFermiPredictor, EdgeWeightGNN, SpatialTemporalGNN
from src.group11.dataset_assembly.augment_dataset import RouteNetData  # noqa: F401 — required for torch.load unpickling
from src.group11.dataset_assembly.build_temporal_dataset import TemporalData  # noqa: F401 — required for torch.load unpickling
from src.group11.training.train import (
    DATASET_PATH,
    ROUTENET_DATASET_PATH,
    TEMPORAL_DATASET_PATH,
    CHECKPOINT_DIR,
    HIDDEN_DIM,
    DROPOUT,
    BATCH_SIZE,
    LOAD_CONDITIONS,
    load_dataset,
    split_dataset,
    build_model,
    _fix_u,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor):
    """
    Compute MAE, RMSE, and R² per load condition and overall average.

    For standard models: preds/targets are (E, 4) — four load conditions.
    For tgnn (forecasting): preds/targets are (E, 1) — single next-step target.
    """
    results = {}

    if targets.shape[1] == 1:
        # Temporal GNN: single next-step latency prediction
        p = preds[:, 0]
        t = targets[:, 0]
        mae  = (p - t).abs().mean().item()
        rmse = ((p - t) ** 2).mean().sqrt().item()
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2     = (1.0 - ss_res / (ss_tot + 1e-8)).item()
        metric = {"mae_ms": mae, "rmse_ms": rmse, "r2": r2}
        results["per_condition"] = {"next_step": metric}
        results["average"]       = metric
        return results

    # Per-condition metrics (standard 4-condition case)
    per = {}
    for i, cond in enumerate(LOAD_CONDITIONS):
        p = preds[:, i]
        t = targets[:, i]

        mae  = (p - t).abs().mean().item()
        rmse = ((p - t) ** 2).mean().sqrt().item()

        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2     = (1.0 - ss_res / (ss_tot + 1e-8)).item()

        per[cond] = {"mae_ms": mae, "rmse_ms": rmse, "r2": r2}

    results["per_condition"] = per
    results["average"] = {
        "mae_ms":  sum(v["mae_ms"]  for v in per.values()) / len(per),
        "rmse_ms": sum(v["rmse_ms"] for v in per.values()) / len(per),
        "r2":      sum(v["r2"]      for v in per.values()) / len(per),
    }
    return results


# ---------------------------------------------------------------------------
# Inference over a DataLoader
# ---------------------------------------------------------------------------
def collect_predictions(model, loader, device):
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if batch.u.dim() == 1:
                batch.u = batch.u.unsqueeze(0)

            # SpatialTemporalGNN and RouteNet-Fermi take the full batch object;
            # standard models use kwargs.
            if isinstance(model, SpatialTemporalGNN):
                pred   = model(batch)
                target = batch.y_next
            elif isinstance(model, RouteNetFermiPredictor):
                pred   = model(batch)
                target = batch.y_edge
            else:
                pred = model(
                    x          = batch.x,
                    edge_index = batch.edge_index,
                    edge_attr  = batch.edge_attr,
                    u          = batch.u,
                    batch      = batch.batch,
                )
                target = batch.y_edge
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------
def print_results(model_name: str, metrics: dict):
    print(f"\n{'='*60}")
    print(f"  Evaluation Results — {model_name.upper()}")
    print(f"{'='*60}")
    print(f"{'Condition':<12} {'MAE (ms)':>10} {'RMSE (ms)':>10} {'R²':>8}")
    print(f"{'-'*42}")
    for cond, vals in metrics["per_condition"].items():
        print(f"{cond:<12} {vals['mae_ms']:>10.3f} {vals['rmse_ms']:>10.3f} {vals['r2']:>8.4f}")
    print(f"{'-'*42}")
    avg = metrics["average"]
    print(f"{'AVERAGE':<12} {avg['mae_ms']:>10.3f} {avg['rmse_ms']:>10.3f} {avg['r2']:>8.4f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            f"Run: python src/group11/training/train.py --model {model_name}"
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(model_name, device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Trained for {ckpt['epoch']} epochs | best val MSE: {ckpt['val_loss']:.4f}")

    print(f"\nLoading dataset ...")
    if model_name == "routenet":
        dataset_path = ROUTENET_DATASET_PATH
        batch_size   = 1
    elif model_name == "tgnn":
        dataset_path = TEMPORAL_DATASET_PATH
        batch_size   = 1
    else:
        dataset_path = DATASET_PATH
        batch_size   = BATCH_SIZE
    dataset = load_dataset(dataset_path)
    _, _, test_data = split_dataset(dataset)
    print(f"  Test set: {len(test_data)} graphs")

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    preds, targets = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(preds, targets)
    print_results(model_name, metrics)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN latency predictor.")
    parser.add_argument(
        "--model", type=str, default="mpnn",
        choices=["gcn", "gat", "mpnn", "routenet", "custom", "tgnn"],
        help="Model architecture to evaluate (default: mpnn)",
    )
    args = parser.parse_args()
    evaluate(args.model)
