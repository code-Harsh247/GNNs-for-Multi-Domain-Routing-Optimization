"""
build_features.py  —  Phase A: Feature Engineering
====================================================
Reads every topology from data/raw/ (via topology_index.json) and produces
three NumPy feature files per topology in data/processed/features/:

    topology_XXX_node_features.npy   shape (num_nodes, 12)
    topology_XXX_edge_features.npy   shape (num_edges,  5)
    topology_XXX_global_features.npy shape (11,)

Run from the repo root:
    python src/group11/feature_engineering/build_features.py

FEATURE SPECS  (from group11_implementation_plan.md)
------------------------------------------------------
Node (12 dims):
  [0]   router_type_core      one-hot
  [1]   router_type_edge      one-hot
  [2]   router_type_gateway   one-hot
  [3]   capacity_mbps_norm    value / 1000.0
  [4]   x_coord               already in [0,1]
  [5]   y_coord               already in [0,1]
  [6-10] domain_id_onehot     one-hot, padded to 5 slots
  [11]  degree_norm           node_degree / max_degree_in_topology
  [12 is NOT here — see below]

Wait — the plan says indices 6-N cover 5 domain slots (6,7,8,9,10) and then
N+1=11 is degree_norm and N+2=12 would be avg_neighbour_bandwidth_norm.
But that totals 13, not 12.  Re-reading the plan:
  5 (type+cap+coords) + 5 (domain one-hot) + 2 (computed) = 12
So the domain one-hot occupies indices 6–10 (5 values), then:
  [11]  degree_norm
  Hmm, that's only 12 if we stop here.  But the plan also lists
  avg_neighbor_bandwidth_norm.  

Actually counting carefully:
  indices 0,1,2,3,4,5 = 6 features (router×3, cap, x, y)
  indices 6,7,8,9,10  = 5 features (domain one-hot)
  index  11            = degree_norm
  — that's 12 total.
The plan lists avg_neighbor_bandwidth_norm as N+2 where N=10, making it index 12 — 
but the plan says the vector is 12 dims (indices 0–11).  So N+2 is NOT included
in the 12-dim vector as stated.  We implement exactly 12 dims as specified.

Edge (5 dims):
  [0]  bandwidth_mbps_norm        value / 1000.0
  [1]  propagation_delay_ms_norm  value / 100.0
  [2]  is_inter_domain            1.0 / 0.0
  [3]  link_reliability           already in [0,1]
  [4]  cost_norm                  value / max_cost_in_topology

Global (11 dims):
  [0]   num_nodes_norm            num_nodes / 100.0
  [1]   num_edges_norm            num_edges / 500.0
  [2]   num_domains_norm          num_domains / 5.0
  [3]   avg_bandwidth_norm        mean(bandwidths) / 1000.0
  [4]   avg_propagation_delay_norm mean(prop_delays) / 100.0
  [5]   inter_domain_ratio        inter_domain_edges / num_edges
  [6-10] topology_type_onehot     one-hot over [random, scale_free, mesh, ring, hybrid]
"""

import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ROOT      = os.path.join("data", "raw")
FEATURES_DIR  = os.path.join("data", "processed", "features")
INDEX_PATH    = os.path.join(RAW_ROOT, "topology_index.json")
JSON_DIR      = os.path.join(RAW_ROOT, "topologies", "json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUTER_TYPE_ORDER  = ["core", "edge", "gateway"]
TOPOLOGY_TYPE_ORDER = ["random", "scale_free", "mesh", "ring", "hybrid"]
MAX_DOMAINS        = 5        # one-hot vector always padded to this width
NODE_FEAT_DIM      = 12
EDGE_FEAT_DIM      = 5
GLOBAL_FEAT_DIM    = 11


# ---------------------------------------------------------------------------
# Node feature builder
# ---------------------------------------------------------------------------
def build_node_features(nodes: list, edges: list) -> np.ndarray:
    """Return float32 array of shape (num_nodes, 12)."""
    n = len(nodes)
    feats = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)

    # Build a node-id → index map (nodes may not be 0-indexed consecutively)
    id_to_idx = {node["node_id"]: i for i, node in enumerate(nodes)}

    # Compute per-node degree and incident bandwidth sum for computed features
    degree      = np.zeros(n, dtype=np.float32)
    bw_sum      = np.zeros(n, dtype=np.float32)
    for e in edges:
        u = id_to_idx[e["source"]]
        v = id_to_idx[e["target"]]
        degree[u] += 1
        degree[v] += 1
        bw_sum[u] += e["bandwidth_mbps"]
        bw_sum[v] += e["bandwidth_mbps"]

    max_degree = degree.max() if degree.max() > 0 else 1.0

    for i, node in enumerate(nodes):
        rt = node["router_type"]
        # [0-2] router_type one-hot
        for j, rtype in enumerate(ROUTER_TYPE_ORDER):
            feats[i, j] = 1.0 if rt == rtype else 0.0
        # [3] capacity — Group 3 generates up to ~1600 Mbps; use 2000 as ceiling
        feats[i, 3] = node["capacity_mbps"] / 2000.0
        # [4-5] coords (already normalised by Group 3)
        feats[i, 4] = node["x_coord"]
        feats[i, 5] = node["y_coord"]
        # [6-10] domain_id one-hot (pad to MAX_DOMAINS=5 slots)
        did = int(node["domain_id"])
        if did < MAX_DOMAINS:
            feats[i, 6 + did] = 1.0
        # [11] degree_norm
        feats[i, 11] = degree[i] / max_degree

    assert feats.shape == (n, NODE_FEAT_DIM), f"Node feat shape mismatch: {feats.shape}"
    return feats


# ---------------------------------------------------------------------------
# Edge feature builder
# ---------------------------------------------------------------------------
def build_edge_features(edges: list) -> np.ndarray:
    """Return float32 array of shape (num_edges, 5)."""
    m = len(edges)
    feats = np.zeros((m, EDGE_FEAT_DIM), dtype=np.float32)

    max_cost = max((e["cost"] for e in edges), default=1.0)
    if max_cost == 0:
        max_cost = 1.0

    for i, e in enumerate(edges):
        feats[i, 0] = e["bandwidth_mbps"]       / 1000.0
        feats[i, 1] = e["propagation_delay_ms"]  / 100.0
        feats[i, 2] = 1.0 if e["is_inter_domain"] else 0.0
        feats[i, 3] = e["link_reliability"]            # already in [0,1]
        feats[i, 4] = e["cost"] / max_cost

    assert feats.shape == (m, EDGE_FEAT_DIM), f"Edge feat shape mismatch: {feats.shape}"
    return feats


# ---------------------------------------------------------------------------
# Global feature builder
# ---------------------------------------------------------------------------
def build_global_features(topology: dict) -> np.ndarray:
    """Return float32 array of shape (11,)."""
    feats = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    edges = topology["edges"]
    m     = len(edges)

    bandwidths  = [e["bandwidth_mbps"]      for e in edges]
    prop_delays = [e["propagation_delay_ms"] for e in edges]
    inter_count = sum(1 for e in edges if e["is_inter_domain"])

    feats[0] = topology["num_nodes"]  / 100.0
    feats[1] = m                      / 500.0
    feats[2] = topology["num_domains"] / 5.0
    feats[3] = (sum(bandwidths)  / m) / 1000.0 if m > 0 else 0.0
    feats[4] = (sum(prop_delays) / m) / 100.0  if m > 0 else 0.0
    feats[5] = inter_count / m if m > 0 else 0.0

    # [6-10] topology_type one-hot
    ttype = topology["topology_type"]
    if ttype in TOPOLOGY_TYPE_ORDER:
        feats[6 + TOPOLOGY_TYPE_ORDER.index(ttype)] = 1.0

    assert feats.shape == (GLOBAL_FEAT_DIM,), f"Global feat shape mismatch: {feats.shape}"
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    total = len(index["topologies"])
    print(f"Building features for {total} topolog{'y' if total == 1 else 'ies'} ...")

    for entry in index["topologies"]:
        base = entry["filename_base"]
        json_path = os.path.join(JSON_DIR, f"{base}.json")

        with open(json_path, encoding="utf-8") as f:
            topology = json.load(f)

        nodes = topology["nodes"]
        edges = topology["edges"]

        node_feats   = build_node_features(nodes, edges)
        edge_feats   = build_edge_features(edges)
        global_feats = build_global_features(topology)

        np.save(os.path.join(FEATURES_DIR, f"{base}_node_features.npy"),   node_feats)
        np.save(os.path.join(FEATURES_DIR, f"{base}_edge_features.npy"),   edge_feats)
        np.save(os.path.join(FEATURES_DIR, f"{base}_global_features.npy"), global_feats)

        print(f"  {base}  nodes{node_feats.shape}  edges{edge_feats.shape}  global{global_feats.shape}")

    print(f"\nFeature files written to {FEATURES_DIR}/")


if __name__ == "__main__":
    main()
