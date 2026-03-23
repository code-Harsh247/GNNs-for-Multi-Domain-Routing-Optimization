"""
assemble_dataset.py  —  Phase D: Dataset Assembly
===================================================
Merges all processed artefacts into a single PyTorch Geometric dataset and
writes two output files to data/processed/dataset/:

    gnn_dataset.pt       — list of torch_geometric.data.Data objects (torch.save)
    dataset_index.json   — metadata index

Run from the repo root:
    python src/group11/dataset_assembly/assemble_dataset.py

PyG Data object fields per graph
---------------------------------
  x          (num_nodes, 12)  — node features
  edge_index (2, num_edges)   — COO connectivity
  edge_attr  (num_edges, 5)   — edge features (static, NO latency)
  u          (1, 11)          — global features
  y_edge     (num_edges, 4)   — targets: [low, medium, high, flash] latency
  topology_id  scalar
  num_nodes    scalar

NOTE: torch_geometric is imported lazily — if it is not installed the script
falls back to saving raw dicts so you can still validate the structure without
a full PyTorch installation.
"""

import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ROOT      = os.path.join("data", "raw")
PROC_ROOT     = os.path.join("data", "processed")
FEATURES_DIR  = os.path.join(PROC_ROOT, "features")
LABELS_DIR    = os.path.join(PROC_ROOT, "labels")
DATASET_DIR   = os.path.join(PROC_ROOT, "dataset")
INDEX_PATH    = os.path.join(RAW_ROOT, "topology_index.json")
JSON_DIR      = os.path.join(RAW_ROOT, "topologies", "json")
RAW_LABELS_DIR = os.path.join(RAW_ROOT, "labels")

LOAD_CONDITIONS = ["low", "medium", "high", "flash"]


# ---------------------------------------------------------------------------
# y_edge builder  (num_edges, 4)  — from Group 3's raw label file
# ---------------------------------------------------------------------------
def _build_y_edge(edges: list, raw_labels: dict) -> np.ndarray:
    """
    For every edge (in the order they appear in the topology JSON),
    stack the 4 load-condition latencies into a (num_edges, 4) array.
    column order: [low, medium, high, flash]
    """
    m      = len(edges)
    y_edge = np.zeros((m, 4), dtype=np.float32)

    for i, e in enumerate(edges):
        eid_str = str(e["edge_id"])
        for j, cond in enumerate(LOAD_CONDITIONS):
            y_edge[i, j] = raw_labels["load_conditions"][cond]["edge_latencies_ms"][eid_str]

    return y_edge


# ---------------------------------------------------------------------------
# edge_index builder  (2, num_edges)
# ---------------------------------------------------------------------------
def _build_edge_index(edges: list) -> np.ndarray:
    """COO format: row 0 = sources, row 1 = targets."""
    sources = [e["source"] for e in edges]
    targets = [e["target"] for e in edges]
    return np.array([sources, targets], dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Try importing PyTorch / PyG
    try:
        import torch
        from torch_geometric.data import Data
        use_pyg = True
        print("PyTorch Geometric detected — building proper Data objects.")
    except ImportError:
        use_pyg = False
        print("WARNING: torch / torch_geometric not installed.")
        print("         Saving raw numpy dicts instead of .pt file.")
        print("         Install with: pip install torch torch-geometric")

    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    total = len(index["topologies"])
    print(f"\nAssembling dataset from {total} topology/ies ...")

    data_list      = []
    index_entries  = []

    for entry in index["topologies"]:
        base = entry["filename_base"]
        tid  = entry["topology_id"]

        # Load topology JSON (for edge list + node list)
        with open(os.path.join(JSON_DIR, f"{base}.json"), encoding="utf-8") as f:
            topology = json.load(f)

        # Load raw labels (for y_edge)
        with open(os.path.join(RAW_LABELS_DIR, f"{base}_labels.json"), encoding="utf-8") as f:
            raw_labels = json.load(f)

        edges      = topology["edges"]
        nodes      = topology["nodes"]
        num_nodes  = len(nodes)
        num_edges  = len(edges)

        # Load processed feature arrays
        x       = np.load(os.path.join(FEATURES_DIR, f"{base}_node_features.npy"))
        ea      = np.load(os.path.join(FEATURES_DIR, f"{base}_edge_features.npy"))
        u_arr   = np.load(os.path.join(FEATURES_DIR, f"{base}_global_features.npy"))

        edge_index = _build_edge_index(edges)
        y_edge     = _build_y_edge(edges, raw_labels)
        u_2d       = u_arr.reshape(1, -1)   # (1, 11) for PyG convention

        if use_pyg:
            data = Data(
                x          = torch.tensor(x,          dtype=torch.float),
                edge_index = torch.tensor(edge_index, dtype=torch.long),
                edge_attr  = torch.tensor(ea,         dtype=torch.float),
                y_edge     = torch.tensor(y_edge,     dtype=torch.float),
                u          = torch.tensor(u_2d,       dtype=torch.float),
                topology_id = tid,
                num_nodes   = num_nodes,
            )
        else:
            # Fallback: plain dict
            data = {
                "x":           x,
                "edge_index":  edge_index,
                "edge_attr":   ea,
                "y_edge":      y_edge,
                "u":           u_2d,
                "topology_id": tid,
                "num_nodes":   num_nodes,
            }

        data_list.append(data)
        index_entries.append({
            "topology_id":  tid,
            "num_nodes":    num_nodes,
            "num_edges":    num_edges,
            "topology_type": entry["topology_type"],
        })

        print(f"  {base}  x{x.shape} ea{ea.shape} y_edge{y_edge.shape}")

    # --- Save dataset ---
    dataset_path = os.path.join(DATASET_DIR, "gnn_dataset.pt")
    if use_pyg:
        import torch
        torch.save(data_list, dataset_path)
        print(f"\nSaved PyG dataset -> {dataset_path}")
    else:
        import pickle
        fallback_path = os.path.join(DATASET_DIR, "gnn_dataset_fallback.pkl")
        with open(fallback_path, "wb") as f:
            pickle.dump(data_list, f)
        print(f"\nSaved fallback dataset -> {fallback_path}")

    # --- Save index ---
    dataset_index = {
        "total_graphs":       len(data_list),
        "node_feature_dim":   12,
        "edge_feature_dim":   5,
        "global_feature_dim": 11,
        "target_dim":         4,
        "load_conditions":    LOAD_CONDITIONS,
        "graphs":             index_entries,
    }
    idx_path = os.path.join(DATASET_DIR, "dataset_index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(dataset_index, f, indent=2)

    print(f"Saved dataset index  -> {idx_path}")
    print(f"\nDataset assembly complete. {len(data_list)} graphs total.")


if __name__ == "__main__":
    main()
