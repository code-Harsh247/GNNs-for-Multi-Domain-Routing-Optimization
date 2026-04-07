"""
comprehensive_benchmark.py — Academic-Quality Model Comparison
================================================================
Loads all 6 trained checkpoints, runs inference on the 50-graph test set,
computes an extended set of metrics, performs per-topology-type breakdown,
runs Wilcoxon signed-rank tests against the GCN baseline, exports results
to CSV/JSON, and saves 6 publication-ready figures to docs/figures/.

Metrics
-------
  MAE (ms), RMSE (ms), R², MAPE (%), Spearman ρ, Max AE (ms),
  Inference speed (ms/graph), Parameter count, Checkpoint size (MB)

Output files
------------
  docs/benchmark_results.csv    — wide format, 1 row per model
  docs/benchmark_results.json   — full nested dict
  docs/figures/radar_chart.png
  docs/figures/topology_type_heatmap.png
  docs/figures/model_complexity.png
  docs/figures/condition_progression.png
  docs/figures/inference_speed.png
  docs/figures/improvement_over_baseline.png

Usage
-----
  python src/group11/training/comprehensive_benchmark.py

  # Train tgnn first if tgnn_best.pt is missing:
  python src/group11/training/train.py --model tgnn

Run from repo root.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats as scipy_stats
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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
DOCS_DIR      = "docs"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DOCS_DIR,    exist_ok=True)

MODEL_NAMES   = ["gcn", "gat", "mpnn", "routenet", "custom", "tgnn"]
MODEL_LABELS  = ["GCN",  "GAT", "MPNN", "RouteNet", "EdgeGNN", "ST-GNN"]
TOPO_TYPES    = ["random", "scale_free", "mesh", "ring", "hybrid"]
PALETTE       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

ALPHA_SIG     = 0.05   # Wilcoxon significance level


# ---------------------------------------------------------------------------
# Extended metric computation
# ---------------------------------------------------------------------------
def extended_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute MAPE, Spearman ρ, and Max AE on top of the standard metrics.
    Works for both (E, 4) and (E, 1) shaped tensors.
    """
    p = preds.numpy().flatten()
    t = targets.numpy().flatten()

    mape   = float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-8)) * 100)
    rho, _ = scipy_stats.spearmanr(p, t)
    max_ae = float(np.abs(p - t).max())

    return {"mape_pct": mape, "spearman_rho": float(rho), "max_ae_ms": max_ae}


def per_condition_mape(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """MAPE per load condition for 4-condition models."""
    out = {}
    for i, cond in enumerate(LOAD_CONDITIONS):
        p = preds[:, i].numpy()
        t = targets[:, i].numpy()
        out[cond] = float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-8)) * 100)
    return out


# ---------------------------------------------------------------------------
# Per-topology utilities
# ---------------------------------------------------------------------------
def get_topo_type_label(data) -> int:
    """Extract topology type index from u[0, 6:11].argmax()."""
    u = data.u
    if u.dim() == 1:
        u = u.unsqueeze(0)
    return int(u[0, 6:11].argmax().item())


def per_topology_errors(
    model, loader, device, is_tgnn: bool, is_routenet: bool
) -> list:
    """
    Returns a list of length N_test where each element is the mean absolute
    error for that graph (averaged over all edges × conditions).
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if batch.u.dim() == 1:
                batch.u = batch.u.unsqueeze(0)

            if is_tgnn:
                pred   = model(batch)
                target = batch.y_next
            elif is_routenet:
                pred   = model(batch)
                target = batch.y_edge
            else:
                pred = model(
                    x=batch.x, edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr, u=batch.u, batch=batch.batch,
                )
                target = batch.y_edge

            mae = (pred.cpu() - target.cpu()).abs().mean().item()
            errors.append(mae)

    return errors


def per_topology_type_mae(
    model, test_data, device, is_tgnn: bool, is_routenet: bool
) -> dict:
    """
    Returns dict: topology_type_name → mean MAE over all test graphs of that type.
    """
    model.eval()
    type_preds   = {i: [] for i in range(5)}
    type_targets = {i: [] for i in range(5)}

    with torch.no_grad():
        for data in test_data:
            ttype = get_topo_type_label(data)
            batch = data.to(device)
            if batch.u.dim() == 1:
                batch.u = batch.u.unsqueeze(0)

            if is_tgnn:
                pred   = model(batch)
                target = batch.y_next
            elif is_routenet:
                pred   = model(batch)
                target = batch.y_edge
            else:
                pred = model(
                    x=batch.x, edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr, u=batch.u, batch=batch.batch,
                )
                target = batch.y_edge

            mae = (pred.cpu() - target.cpu()).abs().mean().item()
            type_preds[ttype].append(mae)

    result = {}
    for i, name in enumerate(TOPO_TYPES):
        vals = type_preds[i]
        result[name] = float(np.mean(vals)) if vals else float("nan")
    return result


# ---------------------------------------------------------------------------
# Timed inference
# ---------------------------------------------------------------------------
def timed_inference(model, loader, device, is_tgnn, is_routenet) -> tuple:
    """Returns (preds, targets, ms_per_graph)."""
    model.eval()
    all_preds   = []
    all_targets = []
    n_graphs    = 0

    # Warm-up (avoid cold-start bias)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if batch.u.dim() == 1:
                batch.u = batch.u.unsqueeze(0)
            if is_tgnn:
                model(batch)
            elif is_routenet:
                model(batch)
            else:
                model(x=batch.x, edge_index=batch.edge_index,
                      edge_attr=batch.edge_attr, u=batch.u, batch=batch.batch)
            break  # single warm-up pass

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if batch.u.dim() == 1:
                batch.u = batch.u.unsqueeze(0)

            if is_tgnn:
                pred   = model(batch)
                target = batch.y_next
            elif is_routenet:
                pred   = model(batch)
                target = batch.y_edge
            else:
                pred = model(
                    x=batch.x, edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr, u=batch.u, batch=batch.batch,
                )
                target = batch.y_edge

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            n_graphs += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    preds   = torch.cat(all_preds,   dim=0)
    targets = torch.cat(all_targets, dim=0)
    ms_per_graph = elapsed_ms / max(n_graphs, 1)

    return preds, targets, ms_per_graph


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1 — Radar / spider chart
# ---------------------------------------------------------------------------
def plot_radar(summary: dict) -> None:
    # 5 normalized axes (all rescaled so that 1.0 = best model on that metric)
    metric_keys   = ["mae_ms", "rmse_ms", "mape_pct", "max_ae_ms", "spearman_rho"]
    metric_labels = ["MAE↓", "RMSE↓", "MAPE↓", "Max AE↓", "Spearman↑"]
    invert        = [True,    True,     True,    True,       False]   # True → lower is better

    present = [(mn, ml, PALETTE[i])
               for i, (mn, ml) in enumerate(zip(MODEL_NAMES, MODEL_LABELS))
               if mn in summary]
    if len(present) < 2:
        print("  [SKIP] radar_chart.png — need ≥2 models")
        return

    # Build raw value matrix
    raw = np.array([[summary[mn][k] for k in metric_keys] for mn, *_ in present])

    # Normalize per column: scale to [0, 1]; invert where lower=better
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col    = raw[:, j]
        lo, hi = col.min(), col.max()
        if hi == lo:
            norm[:, j] = 1.0
        else:
            scaled = (col - lo) / (hi - lo)
            norm[:, j] = (1.0 - scaled) if invert[j] else scaled

    # Radar angles
    N     = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7, color="grey")
    ax.set_ylim(0, 1)

    for idx, (mn, ml, c) in enumerate(present):
        vals   = norm[idx].tolist() + [norm[idx][0]]
        ax.plot(angles, vals, color=c, linewidth=2, label=ml)
        ax.fill(angles, vals, color=c, alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)
    ax.set_title("Model Comparison — Normalized Metrics\n(1.0 = best on each axis)",
                 size=12, fontweight="bold", pad=20)

    _save(fig, "radar_chart.png")


# ---------------------------------------------------------------------------
# Figure 2 — Topology-type MAE heatmap
# ---------------------------------------------------------------------------
def plot_topo_heatmap(topo_mae: dict) -> None:
    present = [(mn, ml) for mn, ml in zip(MODEL_NAMES, MODEL_LABELS) if mn in topo_mae]
    if not present:
        return

    data_matrix = np.array([[topo_mae[mn].get(t, float("nan"))
                              for t in TOPO_TYPES] for mn, _ in present])

    fig, ax = plt.subplots(figsize=(11, len(present) * 0.8 + 1.5))
    im = ax.imshow(data_matrix, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(len(TOPO_TYPES)))
    ax.set_xticklabels([t.replace("_", "\n") for t in TOPO_TYPES], fontsize=10)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([ml for _, ml in present], fontsize=10)
    ax.set_title("Per-Topology-Type MAE (ms) — Test Set", fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            v = data_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black")

    fig.colorbar(im, ax=ax, label="MAE (ms)", shrink=0.8)
    plt.tight_layout()
    _save(fig, "topology_type_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 3 — Model complexity tradeoff
# ---------------------------------------------------------------------------
def plot_complexity(summary: dict) -> None:
    present = [(mn, ml, PALETTE[i])
               for i, (mn, ml) in enumerate(zip(MODEL_NAMES, MODEL_LABELS))
               if mn in summary]
    if not present:
        return

    params     = [summary[mn]["param_count"]    / 1e3 for mn, *_ in present]  # K params
    mae        = [summary[mn]["mae_ms"]         for mn, *_ in present]
    speed      = [summary[mn]["ms_per_graph"]   for mn, *_ in present]
    labels     = [ml for _, ml, _ in present]
    colors     = [c for *_, c in present]

    # Bubble size proportional to inference speed (ms/graph)
    max_speed  = max(speed) if max(speed) > 0 else 1
    sizes      = [max(30, (s / max_speed) * 800) for s in speed]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(params, mae, s=sizes, c=colors, alpha=0.8, edgecolors="white",
                         linewidths=1.5)

    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (params[i], mae[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.set_xlabel("Parameter Count (K)", fontsize=11)
    ax.set_ylabel("Average MAE (ms)", fontsize=11)
    ax.set_title("Model Complexity vs. Accuracy\n(bubble size = inference time ms/graph)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Speed legend
    legend_speeds = [min(speed), np.median(speed), max(speed)]
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=np.sqrt(max(30, (s / max_speed) * 800) / np.pi),
               label=f"{s:.1f} ms/graph")
        for s in legend_speeds
    ]
    ax.legend(handles=legend_handles, title="Inference speed",
              loc="upper right", fontsize=9)

    plt.tight_layout()
    _save(fig, "model_complexity.png")


# ---------------------------------------------------------------------------
# Figure 4 — Per-condition MAE progression (static models only)
# ---------------------------------------------------------------------------
def plot_condition_progression(per_cond_mae: dict) -> None:
    # Only static models (4-condition)
    static = [(mn, ml, PALETTE[i])
              for i, (mn, ml) in enumerate(zip(MODEL_NAMES, MODEL_LABELS))
              if mn in per_cond_mae and per_cond_mae[mn] is not None]
    if not static:
        return

    x      = np.arange(len(LOAD_CONDITIONS))
    labels = [c.capitalize() for c in LOAD_CONDITIONS]

    fig, ax = plt.subplots(figsize=(10, 6))
    for mn, ml, c in static:
        vals = [per_cond_mae[mn][cond] for cond in LOAD_CONDITIONS]
        ax.plot(x, vals, marker="o", linewidth=2, markersize=7,
                color=c, label=ml)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Load Condition", fontsize=11)
    ax.set_ylabel("MAE (ms)", fontsize=11)
    ax.set_title("MAE Progression Across Load Conditions\n(higher load = harder prediction)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "condition_progression.png")


# ---------------------------------------------------------------------------
# Figure 5 — Inference speed + parameter count dual axis
# ---------------------------------------------------------------------------
def plot_inference_speed(summary: dict) -> None:
    present = [(mn, ml, PALETTE[i])
               for i, (mn, ml) in enumerate(zip(MODEL_NAMES, MODEL_LABELS))
               if mn in summary]
    if not present:
        return

    labels = [ml for _, ml, _ in present]
    params = [summary[mn]["param_count"] / 1e3 for mn, *_ in present]
    speeds = [summary[mn]["ms_per_graph"]      for mn, *_ in present]
    colors = [c for *_, c in present]
    x      = np.arange(len(present))
    w      = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w / 2, params, w, color=colors, alpha=0.75,
                    label="Params (K)", edgecolor="white")
    bars2 = ax2.bar(x + w / 2, speeds, w, color=colors, alpha=0.45,
                    hatch="//", label="Inference (ms/graph)", edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Parameter Count (K)", fontsize=11)
    ax2.set_ylabel("Inference Time (ms / graph)", fontsize=11)
    ax1.set_title("Model Size vs. Inference Speed", fontsize=12, fontweight="bold")

    # Annotate bars
    for bar, v in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}K", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars2, speeds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "inference_speed.png")


# ---------------------------------------------------------------------------
# Figure 6 — % MAE improvement over GCN baseline
# ---------------------------------------------------------------------------
def plot_improvement_over_gcn(summary: dict) -> None:
    if "gcn" not in summary:
        print("  [SKIP] improvement_over_baseline.png — GCN checkpoint not available")
        return

    gcn_mae = summary["gcn"]["mae_ms"]
    models_except_gcn = [(mn, ml, PALETTE[i])
                         for i, (mn, ml) in enumerate(zip(MODEL_NAMES, MODEL_LABELS))
                         if mn in summary and mn != "gcn"]
    if not models_except_gcn:
        return

    improvements = [(ml, (gcn_mae - summary[mn]["mae_ms"]) / gcn_mae * 100, c)
                    for mn, ml, c in models_except_gcn]
    # Sort descending
    improvements.sort(key=lambda x: x[1], reverse=True)

    labels = [x[0] for x in improvements]
    values = [x[1] for x in improvements]
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, edgecolor="white")

    ax.axvline(0, color="black", linewidth=1.0, linestyle="-")
    ax.set_xlabel("% MAE Improvement over GCN baseline", fontsize=11)
    ax.set_title("MAE Improvement vs. GCN Baseline\n(positive = better than GCN)",
                 fontsize=12, fontweight="bold")

    for bar, val in zip(bars, values):
        offset = 0.3 if val >= 0 else -0.3
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
                fontsize=10)

    handles = [mpatches.Patch(color="#55A868", label="Better than GCN"),
               mpatches.Patch(color="#C44E52", label="Worse than GCN")]
    ax.legend(handles=handles, fontsize=9)
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, "improvement_over_baseline.png")


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------
def _row(label, values, widths, fmt):
    row = f"  {label:<{widths[0]}}"
    for v, w in zip(values, widths[1:]):
        if isinstance(v, str):
            row += f"  {v:<{w}}"
        else:
            row += f"  {v:{w}{fmt}}"
    return row


def print_summary_table(summary: dict) -> None:
    cols   = ["MAE↓", "RMSE↓", "R²↑", "MAPE↓%", "Spearman↑", "MaxAE↓", "Params", "ms/graph"]
    widths = [10] + [10] * len(cols)
    sep    = "  " + "-" * (widths[0] + sum(w + 2 for w in widths[1:]))

    print("\n" + "=" * 88)
    print("  EXTENDED SUMMARY — All Models (Test Set)")
    print("=" * 88)
    header = f"  {'Model':<{widths[0]}}" + "".join(f"  {c:<10}" for c in cols)
    print(header)
    print(sep)

    for mn, ml in zip(MODEL_NAMES, MODEL_LABELS):
        if mn not in summary:
            continue
        s   = summary[mn]
        tag = "†" if mn == "tgnn" else " "
        row = (
            f"  {(ml+tag):<{widths[0]}}"
            f"  {s['mae_ms']:<10.3f}"
            f"  {s['rmse_ms']:<10.3f}"
            f"  {s['r2']:<10.4f}"
            f"  {s['mape_pct']:<10.2f}"
            f"  {s['spearman_rho']:<10.4f}"
            f"  {s['max_ae_ms']:<10.2f}"
            f"  {s['param_count']//1000:<10}K"
            f"  {s['ms_per_graph']:<10.2f}"
        )
        print(row)

    print(sep)
    print("  † ST-GNN solves a different task (temporal forecasting, single next-step target).")
    print("    Its metrics are not directly comparable to the 4-condition static models.\n")


def print_per_condition_table(per_cond_mae: dict, per_cond_r2: dict) -> None:
    print("=" * 72)
    print("  PER-CONDITION BREAKDOWN — MAE (ms) / R²")
    print("=" * 72)
    w_model = 10
    w_cond  = 16
    header  = f"  {'Model':<{w_model}}" + "".join(f"  {c.capitalize():<{w_cond}}" for c in LOAD_CONDITIONS)
    print(header)
    print("  " + "-" * (w_model + (w_cond + 2) * len(LOAD_CONDITIONS)))

    for mn, ml in zip(MODEL_NAMES, MODEL_LABELS):
        if mn not in per_cond_mae:
            continue
        if per_cond_mae[mn] is None:
            # tgnn
            v_mae = per_cond_r2.get(mn, {}).get("next_step", {}).get("mae_ms", float("nan"))
            v_r2  = per_cond_r2.get(mn, {}).get("next_step", {}).get("r2",     float("nan"))
            cell  = f"{v_mae:.3f}/{v_r2:.3f}"
            print(f"  {'ST-GNN†':<{w_model}}  {cell:<{w_cond}}" + "  (next-step only)")
        else:
            cells = ""
            for cond in LOAD_CONDITIONS:
                mae = per_cond_mae[mn].get(cond, float("nan"))
                r2  = per_cond_r2[mn].get(cond, float("nan"))
                cells += f"  {mae:.3f}/{r2:.4f}  "
            print(f"  {ml:<{w_model}}{cells}")

    print("  Format: MAE (ms) / R²\n")


def print_topo_table(topo_mae: dict) -> None:
    print("=" * 72)
    print("  PER-TOPOLOGY-TYPE MAE (ms)")
    print("=" * 72)
    w_model = 10
    w_type  = 12
    header  = f"  {'Model':<{w_model}}" + "".join(f"  {t.replace('_',' '):<{w_type}}" for t in TOPO_TYPES)
    print(header)
    print("  " + "-" * (w_model + (w_type + 2) * len(TOPO_TYPES)))

    for mn, ml in zip(MODEL_NAMES, MODEL_LABELS):
        if mn not in topo_mae:
            continue
        row = f"  {ml:<{w_model}}"
        for t in TOPO_TYPES:
            v = topo_mae[mn].get(t, float("nan"))
            row += f"  {v:<{w_type}.3f}"
        print(row)
    print()


def print_significance_table(wilcoxon: dict) -> None:
    if not wilcoxon:
        return
    print("=" * 60)
    print("  STATISTICAL SIGNIFICANCE vs. GCN BASELINE")
    print("  (Wilcoxon signed-rank test, per-topology avg abs error)")
    print("=" * 60)
    print(f"  {'Model':<12}  {'p-value':>12}  {'Significant?':>14}  {'Direction':>12}")
    print("  " + "-" * 54)
    for mn, res in wilcoxon.items():
        ml  = MODEL_LABELS[MODEL_NAMES.index(mn)]
        sig = "YES *" if res["p_value"] < ALPHA_SIG else "no"
        dir_ = res["direction"]
        print(f"  {ml:<12}  {res['p_value']:>12.4f}  {sig:>14}  {dir_:>12}")
    print(f"\n  α = {ALPHA_SIG}  (two-sided Wilcoxon signed-rank test)\n")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_csv(summary: dict, topo_mae: dict, wilcoxon: dict) -> None:
    import csv
    path = os.path.join(DOCS_DIR, "benchmark_results.csv")
    topo_cols = [f"mae_topo_{t}" for t in TOPO_TYPES]
    fieldnames = (
        ["model", "label", "mae_ms", "rmse_ms", "r2", "mape_pct",
         "spearman_rho", "max_ae_ms", "param_count", "ckpt_mb", "ms_per_graph"]
        + topo_cols
        + ["wilcoxon_p", "significant_vs_gcn"]
    )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mn, ml in zip(MODEL_NAMES, MODEL_LABELS):
            if mn not in summary:
                continue
            s   = summary[mn]
            row = {
                "model":       mn,
                "label":       ml,
                "mae_ms":      s["mae_ms"],
                "rmse_ms":     s["rmse_ms"],
                "r2":          s["r2"],
                "mape_pct":    s["mape_pct"],
                "spearman_rho":s["spearman_rho"],
                "max_ae_ms":   s["max_ae_ms"],
                "param_count": s["param_count"],
                "ckpt_mb":     s["ckpt_mb"],
                "ms_per_graph":s["ms_per_graph"],
            }
            for t in TOPO_TYPES:
                row[f"mae_topo_{t}"] = topo_mae.get(mn, {}).get(t, "")
            if mn in wilcoxon:
                row["wilcoxon_p"]          = wilcoxon[mn]["p_value"]
                row["significant_vs_gcn"]  = wilcoxon[mn]["p_value"] < ALPHA_SIG
            else:
                row["wilcoxon_p"]         = ""
                row["significant_vs_gcn"] = ""
            writer.writerow(row)

    print(f"  Saved: {path}")


def export_json(summary: dict, topo_mae: dict, wilcoxon: dict,
                per_cond_full: dict) -> None:
    path = os.path.join(DOCS_DIR, "benchmark_results.json")
    out  = {
        "summary":              summary,
        "per_topology_type_mae":topo_mae,
        "wilcoxon_vs_gcn":      wilcoxon,
        "per_condition_full":   per_cond_full,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    summary      : dict = {}   # model → flat metric dict
    topo_mae     : dict = {}   # model → {topo_type: mae}
    per_cond_mae : dict = {}   # model → {cond: mae}  (None for tgnn)
    per_cond_r2  : dict = {}   # model → {cond: r2}   (None for tgnn)
    per_cond_full: dict = {}   # model → full metrics dict from compute_metrics
    topo_errors  : dict = {}   # model → list[float] length 50 (for Wilcoxon)

    # -----------------------------------------------------------------------
    # Data collection loop
    # -----------------------------------------------------------------------
    for mn in MODEL_NAMES:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{mn}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {mn} — checkpoint not found at {ckpt_path}")
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

        print(f"Evaluating {mn} ...", flush=True)

        # Load model
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(mn, device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Parameter / checkpoint metadata
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ckpt_mb     = os.path.getsize(ckpt_path) / 1e6

        # Dataset split (reproducible same 50-graph test set)
        dataset               = load_dataset(dp)
        _, _, test_data       = split_dataset(dataset)

        is_tgnn     = (mn == "tgnn")
        is_routenet = (mn == "routenet")
        loader      = DataLoader(test_data, batch_size=bs, shuffle=False)

        # Timed inference
        preds, targets, ms_per_graph = timed_inference(
            model, loader, device, is_tgnn, is_routenet
        )

        # Standard metrics (MAE, RMSE, R²)
        std_metrics = compute_metrics(preds, targets)
        per_cond_full[mn] = std_metrics

        # Extended metrics (MAPE, Spearman, Max AE)
        ext = extended_metrics(preds, targets)

        # Flat summary entry
        summary[mn] = {
            "mae_ms":       std_metrics["average"]["mae_ms"],
            "rmse_ms":      std_metrics["average"]["rmse_ms"],
            "r2":           std_metrics["average"]["r2"],
            "mape_pct":     ext["mape_pct"],
            "spearman_rho": ext["spearman_rho"],
            "max_ae_ms":    ext["max_ae_ms"],
            "param_count":  param_count,
            "ckpt_mb":      ckpt_mb,
            "ms_per_graph": ms_per_graph,
        }

        # Per-condition breakdown
        if is_tgnn:
            per_cond_mae[mn] = None
            per_cond_r2[mn]  = None
        else:
            per_cond_mae[mn] = {
                cond: std_metrics["per_condition"][cond]["mae_ms"]
                for cond in LOAD_CONDITIONS
            }
            per_cond_r2[mn] = {
                cond: std_metrics["per_condition"][cond]["r2"]
                for cond in LOAD_CONDITIONS
            }

        # Per-topology type MAE (batch_size=1 loop over test_data)
        loader1 = DataLoader(test_data, batch_size=1, shuffle=False)
        topo_mae[mn] = per_topology_type_mae(model, test_data, device,
                                              is_tgnn, is_routenet)

        # Per-topology avg abs error (for Wilcoxon)
        topo_errors[mn] = per_topology_errors(model, loader1, device,
                                               is_tgnn, is_routenet)

        print(
            f"  MAE={summary[mn]['mae_ms']:.3f} ms  "
            f"R²={summary[mn]['r2']:.4f}  "
            f"MAPE={summary[mn]['mape_pct']:.2f}%  "
            f"Spearman={summary[mn]['spearman_rho']:.4f}  "
            f"Params={param_count//1000}K  "
            f"Speed={ms_per_graph:.1f}ms/graph"
        )

    if not summary:
        print("\nNo checkpoints found. Train all models first.")
        return

    # -----------------------------------------------------------------------
    # Wilcoxon signed-rank tests vs. GCN baseline
    # -----------------------------------------------------------------------
    wilcoxon: dict = {}
    if "gcn" in topo_errors:
        gcn_errs = np.array(topo_errors["gcn"])
        for mn in MODEL_NAMES:
            if mn == "gcn" or mn not in topo_errors:
                continue
            other_errs = np.array(topo_errors[mn])
            n = min(len(gcn_errs), len(other_errs))
            if n < 5:
                continue
            try:
                stat, p = scipy_stats.wilcoxon(gcn_errs[:n], other_errs[:n])
                direction = (
                    "better" if other_errs[:n].mean() < gcn_errs[:n].mean()
                    else "worse"
                )
                wilcoxon[mn] = {"statistic": float(stat), "p_value": float(p),
                                 "direction": direction}
            except ValueError:
                # Identical distributions (e.g., difference all zero)
                wilcoxon[mn] = {"statistic": 0.0, "p_value": 1.0, "direction": "equal"}

    # -----------------------------------------------------------------------
    # Print tables
    # -----------------------------------------------------------------------
    print_summary_table(summary)
    print_per_condition_table(per_cond_mae, per_cond_full)
    print_topo_table(topo_mae)
    print_significance_table(wilcoxon)

    # -----------------------------------------------------------------------
    # Export CSV + JSON
    # -----------------------------------------------------------------------
    print(f"Exporting results → {DOCS_DIR}/")
    export_csv(summary, topo_mae, wilcoxon)
    export_json(summary, topo_mae, wilcoxon, per_cond_full)

    # -----------------------------------------------------------------------
    # Generate 6 figures
    # -----------------------------------------------------------------------
    print(f"\nGenerating figures → {FIGURES_DIR}/")
    plot_radar(summary)
    plot_topo_heatmap(topo_mae)
    plot_complexity(summary)
    plot_condition_progression(per_cond_mae)
    plot_inference_speed(summary)
    plot_improvement_over_gcn(summary)

    print("\nDone.")


if __name__ == "__main__":
    main()
