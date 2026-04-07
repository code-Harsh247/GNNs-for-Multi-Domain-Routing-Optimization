"""
build_temporal_dataset.py  —  Temporal GNN Dataset Builder
============================================================
Reads the 6,000 raw snapshot files (500 topologies × 12 timesteps) at
data/raw/snapshots/topology_XXX_tYY_snapshot.json and assembles them into
sliding-window forecasting samples for the SpatialTemporalGNN.

Task
----
Given the last k=6 timesteps of per-edge traffic observations, predict the
per-edge latency at the NEXT timestep (t+k).

Samples per topology
--------------------
  12 timesteps total, k=6 history → windows at t_start ∈ {0,1,2,3,4,5}
  → 6 samples per topology × 500 topologies = 3,000 total samples.

New fields per TemporalData object
-----------------------------------
  edge_seq         (M, k, 4)   Per-edge temporal features for each history step:
                                 [0] load_norm        = load_mbps / bandwidth_mbps
                                 [1] utilization      (already normalised 0..1)
                                 [2] queue_norm       = queue_length_pkts / 100.0
                                 [3] is_bottleneck    (0.0 or 1.0)
  u_seq            (k, 5)      Global temporal features for each history step:
                                 [0] time_of_day      (0.0..1.0, normalised /24)
                                 [1] diurnal_factor   (0.0..1.0)
                                 [2] flash_event      (0.0 or 1.0)
                                 [3] flash_mult_norm  = flash_multiplier / 5.0
                                 [4] network_load_norm = network_load_mbps / 10000.0
  y_next           (M, 1)      Target: per-edge latency_ms at timestep t+k
  y_edge           (M, 4)      Static 4-condition labels (carried from gnn_dataset.pt)
  topology_id      scalar
  timestep_target  scalar      Index of the timestep being predicted (0-based)

Split convention
----------------
  Split is performed by topology_id (not by sample) to prevent leakage.
  400 train / 50 val / 50 test topologies → 2400 / 300 / 300 samples.

Output
------
  data/processed/dataset/temporal_dataset.pt   — list of TemporalData objects

Usage
-----
  python src/group11/dataset_assembly/build_temporal_dataset.py

Run from repo root.
"""

import json
import os
import sys

import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_SNAPSHOTS_DIR  = os.path.join("data", "raw", "snapshots")
PROC_ROOT          = os.path.join("data", "processed")
GNN_DATASET_PATH   = os.path.join(PROC_ROOT, "dataset", "gnn_dataset.pt")
TEMPORAL_PATH      = os.path.join(PROC_ROOT, "dataset", "temporal_dataset.pt")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K              = 6     # number of history timesteps
NUM_TIMESTEPS  = 12    # total snapshots per topology


# ---------------------------------------------------------------------------
# TemporalData — thin wrapper; no __inc__ overrides needed (no new COO indices)
# ---------------------------------------------------------------------------
class TemporalData(Data):
    """PyG Data subclass for temporal forecasting samples.

    Carries all standard fields (x, edge_index, edge_attr, u, y_edge) from
    the static dataset plus the temporal fields (edge_seq, u_seq, y_next).
    No custom __inc__ needed: edge_seq / u_seq / y_next do not contain node
    or edge index tensors that PyG's collator would need to offset.
    """
    pass


# ---------------------------------------------------------------------------
# Load a single snapshot JSON
# ---------------------------------------------------------------------------
def _load_snapshot(topology_id: int, t: int) -> dict:
    fname = f"topology_{topology_id:03d}_t{t:02d}_snapshot.json"
    fpath = os.path.join(RAW_SNAPSHOTS_DIR, fname)
    with open(fpath, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build per-edge temporal feature row from one snapshot
# edge_labels list must be ordered by edge_id (0-based, matches edge_index)
# ---------------------------------------------------------------------------
def _edge_features_from_snapshot(snap: dict, M: int) -> torch.Tensor:
    """Returns (M, 4) float32 tensor: [load_norm, utilization, queue_norm, is_bottleneck]."""
    row = torch.zeros(M, 4, dtype=torch.float)
    for e in snap["edge_labels"]:
        eid = int(e["edge_id"])
        if eid >= M:
            continue
        bw = float(e.get("bandwidth_mbps", 1.0))
        bw = bw if bw > 0 else 1.0
        load_norm    = float(e.get("load_mbps", 0.0)) / bw
        utilization  = float(e.get("utilization", 0.0))
        queue_norm   = float(e.get("queue_length_pkts", 0.0)) / 100.0
        is_bottleneck = float(e.get("is_bottleneck", 0))
        row[eid] = torch.tensor([load_norm, utilization, queue_norm, is_bottleneck])
    return row


# ---------------------------------------------------------------------------
# Build global temporal feature row from one snapshot
# ---------------------------------------------------------------------------
def _global_features_from_snapshot(snap: dict) -> torch.Tensor:
    """Returns (5,) float32 tensor."""
    gf = snap.get("global_features", {})
    time_of_day       = float(gf.get("time_of_day", 0.0)) / 24.0
    diurnal_factor    = float(gf.get("diurnal_factor", 1.0))
    flash_event       = float(gf.get("flash_event", 0))
    flash_mult_norm   = float(gf.get("flash_multiplier", 1.0)) / 5.0
    network_load_norm = float(gf.get("network_load_mbps", 0.0)) / 10000.0
    return torch.tensor(
        [time_of_day, diurnal_factor, flash_event, flash_mult_norm, network_load_norm],
        dtype=torch.float,
    )


# ---------------------------------------------------------------------------
# Build target latency tensor from one snapshot
# ---------------------------------------------------------------------------
def _target_from_snapshot(snap: dict, M: int) -> torch.Tensor:
    """Returns (M, 1) float32 tensor of latency_ms values."""
    row = torch.zeros(M, 1, dtype=torch.float)
    for e in snap["edge_labels"]:
        eid = int(e["edge_id"])
        if eid >= M:
            continue
        row[eid, 0] = float(e.get("latency_ms", 0.0))
    return row


# ---------------------------------------------------------------------------
# Build all TemporalData samples for one topology
# ---------------------------------------------------------------------------
def _build_samples(base_data: Data, topology_id: int) -> list:
    """Generate K sliding-window forecasting samples for one topology."""
    M = int(base_data.edge_index.size(1))

    # Ensure u is 2-D
    u = base_data.u
    if u.dim() == 1:
        u = u.unsqueeze(0)

    # Load all 12 snapshots upfront
    snaps = []
    for t in range(NUM_TIMESTEPS):
        try:
            snaps.append(_load_snapshot(topology_id, t))
        except FileNotFoundError:
            return []  # skip topology if any snapshot is missing

    samples = []
    num_windows = NUM_TIMESTEPS - K  # = 6 valid starting points

    for t_start in range(num_windows):
        # History window: t_start .. t_start + K - 1
        # Target timestep: t_start + K
        history_snaps = snaps[t_start : t_start + K]
        target_snap   = snaps[t_start + K]

        # edge_seq: (M, K, 4)
        edge_frames = [_edge_features_from_snapshot(s, M) for s in history_snaps]
        edge_seq = torch.stack(edge_frames, dim=1)  # (M, K, 4)

        # u_seq: (K, 5)
        global_frames = [_global_features_from_snapshot(s) for s in history_snaps]
        u_seq = torch.stack(global_frames, dim=0)   # (K, 5)

        # y_next: (M, 1)
        y_next = _target_from_snapshot(target_snap, M)

        sample = TemporalData(
            x               = base_data.x,
            edge_index      = base_data.edge_index,
            edge_attr       = base_data.edge_attr,
            u               = u,
            y_edge          = base_data.y_edge,
            edge_seq        = edge_seq,
            u_seq           = u_seq,
            y_next          = y_next,
            topology_id     = base_data.topology_id,
            num_nodes       = base_data.num_nodes,
            timestep_target = t_start + K,
        )
        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(TEMPORAL_PATH), exist_ok=True)

    print(f"Loading base dataset from {GNN_DATASET_PATH} ...")
    base_dataset = torch.load(GNN_DATASET_PATH, weights_only=False)
    print(f"  {len(base_dataset)} base graphs loaded.\n")

    all_samples      = []
    skipped          = 0

    for i, data in enumerate(base_dataset):
        tid = int(data.topology_id)
        samples = _build_samples(data, tid)

        if not samples:
            skipped += 1
            continue

        all_samples.extend(samples)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(base_dataset)} topologies "
                  f"({len(all_samples)} samples so far) ...")

    print(f"\nDataset assembly complete.")
    print(f"  Topologies processed : {len(base_dataset) - skipped}")
    print(f"  Topologies skipped   : {skipped}")
    print(f"  Total samples        : {len(all_samples)}")
    print(f"  Samples per topology : {K} (windows)")
    if all_samples:
        s0 = all_samples[0]
        print(f"\n  Sample[0] fields:")
        print(f"    x         : {s0.x.shape}")
        print(f"    edge_attr : {s0.edge_attr.shape}")
        print(f"    edge_seq  : {s0.edge_seq.shape}")
        print(f"    u_seq     : {s0.u_seq.shape}")
        print(f"    y_next    : {s0.y_next.shape}")
        print(f"    y_edge    : {s0.y_edge.shape}")

    print(f"\nSaving to {TEMPORAL_PATH} ...")
    torch.save(all_samples, TEMPORAL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
