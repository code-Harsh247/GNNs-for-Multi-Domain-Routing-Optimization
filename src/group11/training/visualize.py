"""
visualize.py — Visual Model Comparison
========================================
Loads all six trained checkpoints, runs inference on the test set, and
produces five publication-ready figures saved to docs/figures/.

Figures produced
----------------
1. average_summary.png      — Bar chart: average MAE / RMSE / R² for all 6 models
2. metric_comparison.png    — Grouped bars: MAE & R² per model per load condition
3. pred_vs_actual.png       — Scatter (2×3): predicted vs actual latency per model
4. error_distribution.png   — Box plots: |error| distribution per model
5. edge_weights.png         — EdgeWeightGNN α_e distribution (intra vs inter-domain)

Usage
-----
    python src/group11/training/visualize.py

Run from repo root.
"""

import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import matplotlib
matplotlib.use("Agg")          # headless / no display required
import matplotlib.pyplot as plt

from src.group11.models import (
    GCNLatencyPredictor, GATLatencyPredictor, MPNNLatencyPredictor,
    RouteNetFermiPredictor, EdgeWeightGNN, SpatialTemporalGNN,
)
from src.group11.dataset_assembly.augment_dataset import RouteNetData          # noqa: F401
from src.group11.dataset_assembly.build_temporal_dataset import TemporalData   # noqa: F401
from src.group11.training.train import (
    DATASET_PATH, ROUTENET_DATASET_PATH, TEMPORAL_DATASET_PATH,
    CHECKPOINT_DIR, BATCH_SIZE, LOAD_CONDITIONS,
    load_dataset, split_dataset, build_model,
)
from src.group11.training.evaluate import collect_predictions, compute_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURES_DIR   = os.path.join("docs", "figures")
MODEL_NAMES   = ["gcn", "gat", "mpnn", "routenet", "custom", "tgnn"]
MODEL_LABELS  = ["GCN",  "GAT", "MPNN", "RouteNet", "EdgeGNN", "ST-GNN"]
PALETTE       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 — Average summary (all 6 models)
# ---------------------------------------------------------------------------
def plot_average_summary(all_metrics: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Performance Summary — Test Set Average", fontsize=14, fontweight="bold")

    present = [
        (mn, ml, PALETTE[i])
        for i, (mn, ml) in enumerate(zip(MODEL_NAMES, MODEL_LABELS))
        if mn in all_metrics
    ]
    labels     = [x[1] for x in present]
    colors     = [x[2] for x in present]
    mae_vals   = [all_metrics[mn]["average"]["mae_ms"]  for mn, *_ in present]
    rmse_vals  = [all_metrics[mn]["average"]["rmse_ms"] for mn, *_ in present]
    r2_vals    = [all_metrics[mn]["average"]["r2"]      for mn, *_ in present]

    for ax, vals, title, ylabel in zip(
        axes,
        [mae_vals, rmse_vals, r2_vals],
        ["Average MAE (ms)", "Average RMSE (ms)", "Average R²"],
        ["MAE (ms)", "RMSE (ms)", "R²"],
    ):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        offset = 0.005 * max(vals) if max(vals) > 0 else 0.001
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    _save(fig, "average_summary.png")


# ---------------------------------------------------------------------------
# Figure 2 — Per-condition grouped bars (static models only)
# ---------------------------------------------------------------------------
def plot_metric_comparison(all_metrics: dict) -> None:
    static_names  = ["gcn", "gat", "mpnn", "routenet", "custom"]
    static_labels = ["GCN",  "GAT", "MPNN", "RouteNet", "EdgeGNN"]
    static_colors = PALETTE[:5]

    available = [(mn, ml, c) for mn, ml, c in zip(static_names, static_labels, static_colors)
                 if mn in all_metrics]
    if not available:
        print("  [SKIP] metric_comparison — no static model checkpoints found")
        return

    n      = len(available)
    x      = np.arange(len(LOAD_CONDITIONS))
    width  = 0.15
    offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Static-Load Models — Per-Condition Comparison", fontsize=14, fontweight="bold")

    for ax, metric_key, ylabel, title in zip(
        axes,
        ["mae_ms", "r2"],
        ["MAE (ms)", "R²"],
        ["Mean Absolute Error — lower is better", "R² Score — higher is better"],
    ):
        for i, (mn, ml, c) in enumerate(available):
            vals = [all_metrics[mn]["per_condition"][cond][metric_key]
                    for cond in LOAD_CONDITIONS]
            ax.bar(x + offsets[i], vals, width, label=ml, color=c, alpha=0.85,
                   edgecolor="white", linewidth=0.6)
        ax.set_xlabel("Load Condition")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in LOAD_CONDITIONS])
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "metric_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3 — Predicted vs Actual scatter (2×3 grid)
# ---------------------------------------------------------------------------
def plot_pred_vs_actual(all_preds: dict, all_targets: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Predicted vs Actual Latency — Test Set", fontsize=14, fontweight="bold")
    flat_axes = axes.flatten()

    rng = np.random.default_rng(0)

    for plot_i, (mn, ml) in enumerate(zip(MODEL_NAMES, MODEL_LABELS)):
        ax = flat_axes[plot_i]
        if mn not in all_preds:
            ax.set_visible(False)
            continue

        preds   = all_preds[mn].numpy()
        targets = all_targets[mn].numpy()

        # medium load (index 1) for 4-condition models; index 0 for tgnn
        col = 1 if preds.shape[1] == 4 else 0
        p   = preds[:, col]
        t   = targets[:, col]
        cond_label = "Medium Load" if preds.shape[1] == 4 else "Next-Step"

        # Subsample for readability
        if len(p) > 2_000:
            sample = rng.choice(len(p), 2_000, replace=False)
            p, t   = p[sample], t[sample]

        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())

        ax.scatter(t, p, alpha=0.25, s=5, color=PALETTE[plot_i])
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")

        # R² annotation
        ss_res = ((t - p) ** 2).sum()
        ss_tot = ((t - t.mean()) ** 2).sum()
        r2     = 1.0 - ss_res / (ss_tot + 1e-8)
        ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
                fontsize=9, va="top", color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        ax.set_xlabel("Actual (ms)", fontsize=9)
        ax.set_ylabel("Predicted (ms)", fontsize=9)
        ax.set_title(f"{ml} — {cond_label}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "pred_vs_actual.png")


# ---------------------------------------------------------------------------
# Figure 4 — Absolute-error box plots (all models)
# ---------------------------------------------------------------------------
def plot_error_distribution(all_preds: dict, all_targets: dict) -> None:
    present = [(mn, ml, PALETTE[i])
               for i, (mn, ml) in enumerate(zip(MODEL_NAMES, MODEL_LABELS))
               if mn in all_preds]
    if not present:
        return

    box_data    = []
    box_labels  = []
    box_colors  = []

    for mn, ml, c in present:
        errors = (all_preds[mn] - all_targets[mn]).abs().numpy().flatten()
        # Remove top-1% outliers for a readable y-axis
        clip   = float(np.percentile(errors, 99))
        errors = errors[errors <= clip]
        box_data.append(errors)
        box_labels.append(ml)
        box_colors.append(c)

    fig, ax = plt.subplots(figsize=(13, 6))
    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        notch=False,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticklabels(box_labels, fontsize=11)
    ax.set_ylabel("Absolute Error (ms)")
    ax.set_title(
        "Absolute Error Distribution per Model — all edges × conditions (99th pct clipped)",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "error_distribution.png")


# ---------------------------------------------------------------------------
# Figure 5 — EdgeWeightGNN α_e analysis
# ---------------------------------------------------------------------------
def plot_edge_weights(model: EdgeWeightGNN, test_data, device) -> None:
    model.eval()

    weights_list    = []
    interdomain_list = []
    bandwidth_list  = []

    subset = test_data[:20]
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if batch.u.dim() == 1:
                batch.u = batch.u.unsqueeze(0)
            model(
                x          = batch.x,
                edge_index = batch.edge_index,
                edge_attr  = batch.edge_attr,
                u          = batch.u,
                batch      = batch.batch,
            )
            w     = model.last_edge_weights.cpu().squeeze(1).numpy()
            inter = batch.edge_attr[:, 2].cpu().numpy()   # is_inter_domain
            bw    = batch.edge_attr[:, 0].cpu().numpy()   # bandwidth_mbps_norm

            weights_list.append(w)
            interdomain_list.append(inter.astype(bool))
            bandwidth_list.append(bw)

    weights    = np.concatenate(weights_list)
    interdomain = np.concatenate(interdomain_list)
    bandwidth  = np.concatenate(bandwidth_list)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("EdgeWeightGNN — Learned Edge Importance α_e (first 20 test topologies)",
                 fontsize=13, fontweight="bold")

    # Left: histogram by domain-crossing type
    ax = axes[0]
    ax.hist(weights[~interdomain], bins=40, alpha=0.7, label="Intra-domain", color=PALETTE[0])
    ax.hist(weights[interdomain],  bins=40, alpha=0.7, label="Inter-domain", color=PALETTE[3])
    ax.set_xlabel("α_e (edge importance gate)", fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Distribution by domain-crossing type")
    ax.legend()
    ax.grid(alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: scatter bandwidth_norm vs α_e
    ax = axes[1]
    ax.scatter(bandwidth[~interdomain], weights[~interdomain],
               alpha=0.25, s=8, color=PALETTE[0], label="Intra-domain")
    ax.scatter(bandwidth[interdomain],  weights[interdomain],
               alpha=0.5,  s=12, color=PALETTE[3], label="Inter-domain", zorder=3)
    ax.set_xlabel("Bandwidth (normalized)", fontsize=10)
    ax.set_ylabel("α_e (edge importance gate)", fontsize=10)
    ax.set_title("Edge importance vs. Bandwidth")
    ax.legend()
    ax.grid(alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "edge_weights.png")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _save(fig, filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    all_metrics: dict = {}
    all_preds:   dict = {}
    all_targets: dict = {}
    custom_model_obj  = None
    custom_test_data  = None

    for mn in MODEL_NAMES:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{mn}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {mn} — checkpoint not found")
            continue

        # Dataset routing
        if mn == "routenet":
            dp, bs = ROUTENET_DATASET_PATH, 1
        elif mn == "tgnn":
            dp, bs = TEMPORAL_DATASET_PATH, 1
        else:
            dp, bs = DATASET_PATH, BATCH_SIZE

        if not os.path.exists(dp):
            print(f"[SKIP] {mn} — dataset not found: {dp}")
            continue

        print(f"Evaluating {mn} ...", end="  ", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(mn, device)
        model.load_state_dict(ckpt["model_state"])

        dataset = load_dataset(dp)
        _, _, test_data = split_dataset(dataset)
        loader  = DataLoader(test_data, batch_size=bs, shuffle=False)

        preds, targets = collect_predictions(model, loader, device)
        metrics = compute_metrics(preds, targets)

        all_metrics[mn] = metrics
        all_preds[mn]   = preds.cpu()
        all_targets[mn] = targets.cpu()

        if mn == "custom":
            custom_model_obj = model
            custom_test_data = test_data

        print(f"MAE={metrics['average']['mae_ms']:.3f} ms  R²={metrics['average']['r2']:.4f}")

    if not all_metrics:
        print("\nNo checkpoints found — train the models first.")
        return

    print(f"\nGenerating figures → {FIGURES_DIR}/")
    plot_average_summary(all_metrics)
    plot_metric_comparison(all_metrics)
    plot_pred_vs_actual(all_preds, all_targets)
    plot_error_distribution(all_preds, all_targets)

    if custom_model_obj is not None and custom_test_data is not None:
        plot_edge_weights(custom_model_obj, custom_test_data, device)
    else:
        print("  [SKIP] edge_weights.png — EdgeWeightGNN checkpoint not loaded")

    print("\nDone.")


if __name__ == "__main__":
    main()
