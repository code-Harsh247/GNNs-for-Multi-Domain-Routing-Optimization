"""
augment_dataset.py  —  RouteNet-Fermi Dataset Augmentation
===========================================================
Reads the existing gnn_dataset.pt (500 PyG Data objects) and enriches each
graph with path-level data drawn from the pre-computed enriched label files
at data/processed/labels/topology_XXX_enriched_labels.json.

New fields added per graph
--------------------------
  path_link_index  (2, E_pl)   COO tensor mapping path↔link memberships.
                               Row 0 = 0-based path indices (P values).
                               Row 1 = 0-based edge indices (M values).
  path_attr        (P, 2)      Per-path structural features:
                                 [0] path_length_norm  = (num_hops) / (num_nodes-1)
                                 [1] is_cross_domain   = 1.0 if any link on path
                                                         has edge_attr[e, 2] == 1.0
  num_paths        scalar      Number of src-dst paths in this graph.

Output
------
  data/processed/dataset/routenet_dataset.pt   — list of RouteNetData objects

The original gnn_dataset.pt is NOT modified.

Usage
-----
  python src/group11/dataset_assembly/augment_dataset.py

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
PROC_ROOT         = os.path.join("data", "processed")
DATASET_PATH      = os.path.join(PROC_ROOT, "dataset", "gnn_dataset.pt")
ROUTENET_PATH     = os.path.join(PROC_ROOT, "dataset", "routenet_dataset.pt")
LABELS_DIR        = os.path.join(PROC_ROOT, "labels")
RAW_TOPO_DIR      = os.path.join("data", "raw", "topologies", "json")


# ---------------------------------------------------------------------------
# RouteNetData — Data subclass with custom __inc__ for batching
# ---------------------------------------------------------------------------
class RouteNetData(Data):
    """PyG Data subclass that correctly auto-increments the path-link COO
    index during mini-batch collation.

    path_link_index row 0 (path indices) must be incremented by the cumulative
    number of paths from preceding graphs in the batch.
    path_link_index row 1 (edge indices) is already handled by the standard
    edge_index increment applied to all edge-indexed tensors; we replicate
    that here by incrementing by num_edges (== edge_index.size(1)).
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "path_link_index":
            return torch.tensor([[self.num_paths], [self.edge_index.size(1)]])
        return super().__inc__(key, value, *args, **kwargs)


# ---------------------------------------------------------------------------
# Build path tensors from enriched label JSON
# ---------------------------------------------------------------------------
def _build_path_tensors(
    path_latencies: list,
    edge_attr: torch.Tensor,
    num_nodes: int,
    node_traffic: dict,
):
    """
    Args:
        path_latencies: list of dicts from enriched_labels JSON
        edge_attr:      (M, 5) edge feature tensor from the base Data object
        num_nodes:      number of nodes in the topology
        node_traffic:   dict mapping node_id -> historical_traffic (0-1)

    Returns:
        path_link_index (2, E_pl) int64
        path_attr       (P, 4)   float32
            [0] path_length_norm    num_hops / (num_nodes - 1)
            [1] is_cross_domain     1.0 if any inter-domain link on path
            [2] demand_norm         mean historical_traffic of src & dst nodes
            [3] avg_bw_norm         mean edge_attr[eid, 0] (bandwidth) on path
        num_paths       int
    """
    path_ids  = []
    link_ids  = []
    pa_rows   = []

    denom = max(num_nodes - 1, 1)  # avoid div-by-zero for degenerate graphs

    for p_idx, entry in enumerate(path_latencies):
        edge_ids  = entry["edge_ids_on_path"]
        hop_path  = entry["hop_path"]
        num_hops  = len(edge_ids)

        # [0] path_length_norm
        path_length_norm = num_hops / denom

        # [1] is_cross_domain
        is_cross = 0.0
        for eid in edge_ids:
            if eid < edge_attr.size(0) and edge_attr[eid, 2].item() == 1.0:
                is_cross = 1.0
                break

        # [2] demand_norm: average historical_traffic of src and dst only
        # (using only endpoints avoids diluting the signal with transit nodes)
        if hop_path and len(hop_path) >= 2 and node_traffic:
            src_t = node_traffic.get(hop_path[0], 0.0)
            dst_t = node_traffic.get(hop_path[-1], 0.0)
            demand_norm = (src_t + dst_t) / 2.0
        elif hop_path and node_traffic:
            demand_norm = node_traffic.get(hop_path[0], 0.0)
        else:
            demand_norm = 0.0

        # [3] avg_bw_norm: mean normalised bandwidth of edges on path
        if edge_ids:
            valid_bws = [
                edge_attr[eid, 0].item()
                for eid in edge_ids
                if eid < edge_attr.size(0)
            ]
            avg_bw_norm = sum(valid_bws) / len(valid_bws) if valid_bws else 0.0
        else:
            avg_bw_norm = 0.0

        # COO entries
        for eid in edge_ids:
            path_ids.append(p_idx)
            link_ids.append(eid)

        pa_rows.append([path_length_norm, is_cross, demand_norm, avg_bw_norm])

    num_paths = len(path_latencies)

    if num_paths == 0 or len(path_ids) == 0:
        path_link_index = torch.zeros((2, 0), dtype=torch.long)
        path_attr       = torch.zeros((0, 4), dtype=torch.float)
        return path_link_index, path_attr, num_paths

    path_link_index = torch.tensor([path_ids, link_ids], dtype=torch.long)
    path_attr       = torch.tensor(pa_rows, dtype=torch.float)

    return path_link_index, path_attr, num_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(ROUTENET_PATH), exist_ok=True)

    print(f"Loading base dataset from {DATASET_PATH} ...")
    base_dataset = torch.load(DATASET_PATH, weights_only=False)
    print(f"  {len(base_dataset)} graphs loaded.\n")

    augmented = []
    total_paths = 0
    total_path_links = 0

    for i, data in enumerate(base_dataset):
        tid = int(data.topology_id)
        label_file = os.path.join(LABELS_DIR, f"topology_{tid:03d}_enriched_labels.json")

        if not os.path.exists(label_file):
            raise FileNotFoundError(
                f"Missing enriched label file: {label_file}\n"
                f"Run: python src/group11/ground_truth/enrich_labels.py"
            )

        with open(label_file, encoding="utf-8") as f:
            enriched = json.load(f)

        path_latencies = enriched["path_latencies"]

        # Load raw topology JSON to get per-node historical_traffic
        topo_file = os.path.join(RAW_TOPO_DIR, f"topology_{tid:03d}.json")
        with open(topo_file, encoding="utf-8") as f:
            topo_json = json.load(f)
        node_traffic = {
            n["node_id"]: n.get("historical_traffic", 0.0)
            for n in topo_json["nodes"]
        }

        # Ensure u is 2-D
        u = data.u
        if u.dim() == 1:
            u = u.unsqueeze(0)

        path_link_index, path_attr, num_paths = _build_path_tensors(
            path_latencies, data.edge_attr, int(data.num_nodes), node_traffic
        )

        rn_data = RouteNetData(
            x               = data.x,
            edge_index      = data.edge_index,
            edge_attr       = data.edge_attr,
            u               = u,
            y_edge          = data.y_edge,
            topology_id     = data.topology_id,
            num_nodes       = data.num_nodes,
            path_link_index = path_link_index,
            path_attr       = path_attr,
            num_paths       = num_paths,
        )

        augmented.append(rn_data)
        total_paths      += num_paths
        total_path_links += path_link_index.size(1)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(base_dataset)} graphs ...")

    print(f"\nAugmentation complete.")
    print(f"  Total graphs      : {len(augmented)}")
    print(f"  Total paths       : {total_paths}")
    print(f"  Total path-link   : {total_path_links}")
    print(f"  Avg paths/graph   : {total_paths / len(augmented):.1f}")

    torch.save(augmented, ROUTENET_PATH)
    print(f"\nSaved -> {ROUTENET_PATH}")

    # Quick sanity check on first graph
    d = augmented[0]
    print(f"\nSanity check — graph 0 (topology_id={int(d.topology_id)}):")
    print(f"  x               : {tuple(d.x.shape)}")
    print(f"  edge_index      : {tuple(d.edge_index.shape)}")
    print(f"  edge_attr       : {tuple(d.edge_attr.shape)}")
    print(f"  u               : {tuple(d.u.shape)}")
    print(f"  y_edge          : {tuple(d.y_edge.shape)}")
    print(f"  path_link_index : {tuple(d.path_link_index.shape)}")
    print(f"  path_attr       : {tuple(d.path_attr.shape)}")
    print(f"  num_paths       : {d.num_paths}")


if __name__ == "__main__":
    main()
