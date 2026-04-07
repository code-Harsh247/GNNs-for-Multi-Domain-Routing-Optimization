"""
benchmark.py — Architecture Comparison
=========================================
Trains all three models (GCN, GAT, MPNN) sequentially and prints a
side-by-side comparison table of MAE, RMSE, and R² on the test set.

Usage:
    python src/group11/training/benchmark.py

Options:
    --epochs N        Override number of training epochs (default: uses train.py default)
    --skip-training   Skip training and just evaluate existing checkpoints

Run from repo root.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.group11.training.train import train, EPOCHS, LOAD_CONDITIONS
from src.group11.training.evaluate import evaluate
from src.group11.dataset_assembly.augment_dataset import RouteNetData  # noqa: F401 — required for torch.load unpickling
from src.group11.dataset_assembly.build_temporal_dataset import TemporalData  # noqa: F401 — required for torch.load unpickling

MODELS = ["gcn", "gat", "mpnn", "routenet", "custom", "tgnn"]


# ---------------------------------------------------------------------------
# Pretty summary table
# ---------------------------------------------------------------------------
def print_comparison(all_metrics: dict):
    conds = LOAD_CONDITIONS

    # Header
    col_w = 12
    cond_w = 10
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS — All Architectures vs. All Load Conditions")
    print("=" * 80)

    # --- MAE Table ---
    print(f"\n  MAE (ms)  — lower is better")
    print(f"  {'Model':<{col_w}}", end="")
    for c in conds:
        print(f"  {c.capitalize():<{cond_w}}", end="")
    print(f"  {'Average':<{cond_w}}")
    print("  " + "-" * (col_w + (cond_w + 2) * (len(conds) + 1)))
    for model_name, metrics in all_metrics.items():
        print(f"  {model_name.upper():<{col_w}}", end="")
        if "next_step" in metrics["per_condition"]:
            # tgnn: single condition — pad remaining columns with dashes
            v = metrics["per_condition"]["next_step"]["mae_ms"]
            print(f"  {v:<{cond_w}.3f}", end="")
            for _ in range(len(conds) - 1):
                print(f"  {'—':<{cond_w}}", end="")
        else:
            for c in conds:
                v = metrics["per_condition"][c]["mae_ms"]
                print(f"  {v:<{cond_w}.3f}", end="")
        print(f"  {metrics['average']['mae_ms']:<{cond_w}.3f}")

    # --- RMSE Table ---
    print(f"\n  RMSE (ms)  — lower is better")
    print(f"  {'Model':<{col_w}}", end="")
    for c in conds:
        print(f"  {c.capitalize():<{cond_w}}", end="")
    print(f"  {'Average':<{cond_w}}")
    print("  " + "-" * (col_w + (cond_w + 2) * (len(conds) + 1)))
    for model_name, metrics in all_metrics.items():
        print(f"  {model_name.upper():<{col_w}}", end="")
        if "next_step" in metrics["per_condition"]:
            v = metrics["per_condition"]["next_step"]["rmse_ms"]
            print(f"  {v:<{cond_w}.3f}", end="")
            for _ in range(len(conds) - 1):
                print(f"  {'—':<{cond_w}}", end="")
        else:
            for c in conds:
                v = metrics["per_condition"][c]["rmse_ms"]
                print(f"  {v:<{cond_w}.3f}", end="")
        print(f"  {metrics['average']['rmse_ms']:<{cond_w}.3f}")

    # --- R² Table ---
    print(f"\n  R²  — higher is better (max 1.0)")
    print(f"  {'Model':<{col_w}}", end="")
    for c in conds:
        print(f"  {c.capitalize():<{cond_w}}", end="")
    print(f"  {'Average':<{cond_w}}")
    print("  " + "-" * (col_w + (cond_w + 2) * (len(conds) + 1)))
    for model_name, metrics in all_metrics.items():
        print(f"  {model_name.upper():<{col_w}}", end="")
        if "next_step" in metrics["per_condition"]:
            v = metrics["per_condition"]["next_step"]["r2"]
            print(f"  {v:<{cond_w}.4f}", end="")
            for _ in range(len(conds) - 1):
                print(f"  {'—':<{cond_w}}", end="")
        else:
            for c in conds:
                v = metrics["per_condition"][c]["r2"]
                print(f"  {v:<{cond_w}.4f}", end="")
        print(f"  {metrics['average']['r2']:<{cond_w}.4f}")

    # --- Winner summary ---
    print("\n  " + "=" * 60)
    best_mae  = min(all_metrics, key=lambda m: all_metrics[m]["average"]["mae_ms"])
    best_rmse = min(all_metrics, key=lambda m: all_metrics[m]["average"]["rmse_ms"])
    best_r2   = max(all_metrics, key=lambda m: all_metrics[m]["average"]["r2"])
    print(f"  Best avg MAE : {best_mae.upper()}")
    print(f"  Best avg RMSE: {best_rmse.upper()}")
    print(f"  Best avg R²  : {best_r2.upper()}")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def benchmark(skip_training: bool = False):
    t_start = time.time()

    if not skip_training:
        for model_name in MODELS:
            print(f"\n{'#'*60}")
            print(f"#  Training: {model_name.upper()}")
            print(f"{'#'*60}")
            train(model_name)

    print(f"\n{'#'*60}")
    print(f"#  Evaluating all models on test set")
    print(f"{'#'*60}")

    all_metrics = {}
    for model_name in MODELS:
        print(f"\n--- {model_name.upper()} ---")
        all_metrics[model_name] = evaluate(model_name)

    print_comparison(all_metrics)

    total_time = time.time() - t_start
    print(f"Total benchmark time: {total_time/60:.1f} min\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all three GNN architectures.")
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training and evaluate existing checkpoints only.",
    )
    args = parser.parse_args()
    benchmark(skip_training=args.skip_training)
