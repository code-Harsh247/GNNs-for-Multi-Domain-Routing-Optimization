"""
enrich_labels.py  —  Phase C: Ground Truth Enrichment
======================================================
For every topology, computes end-to-end path latency and bottleneck
identification for all source-destination pairs under all 4 load conditions,
writing topology_XXX_enriched_labels.json to data/processed/labels/.

Run from the repo root:
    python src/group11/ground_truth/enrich_labels.py

ALGORITHM
---------
1. Load topology JSON → build a NetworkX graph with edges keyed by (src, tgt).
2. Load labels JSON   → build a lookup: edge_id → {low, medium, high, flash}
3. For every (source, destination) pair where source != destination:
     a. Find the hop-count shortest path with nx.shortest_path
     b. Walk consecutive node pairs on the path to find the edge_id for each hop
     c. Sum per-link latencies for each load condition → end_to_end_latency_ms
     d. Find the edge_id with maximum latency → bottleneck_edge_id
4. Write enriched labels JSON
"""

import json
import os

import networkx as nx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ROOT       = os.path.join("data", "raw")
LABELS_RAW_DIR = os.path.join(RAW_ROOT, "labels")
LABELS_OUT_DIR = os.path.join("data", "processed", "labels")
INDEX_PATH     = os.path.join(RAW_ROOT, "topology_index.json")
JSON_DIR       = os.path.join(RAW_ROOT, "topologies", "json")

LOAD_CONDITIONS = ["low", "medium", "high", "flash"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_graph(edges: list) -> nx.Graph:
    """Build undirected NetworkX graph with edge_id stored on each edge."""
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"], edge_id=e["edge_id"])
    return G


def _build_edge_pair_lookup(edges: list) -> dict:
    """Return dict: (min(u,v), max(u,v)) → edge_id (handles undirected lookup)."""
    return {(min(e["source"], e["target"]), max(e["source"], e["target"])): e["edge_id"]
            for e in edges}


def _build_latency_lookup(labels: dict) -> dict:
    """Return dict: edge_id (int) → {condition: latency_ms}."""
    lookup = {}
    for cond in LOAD_CONDITIONS:
        for eid_str, lat in labels["load_conditions"][cond]["edge_latencies_ms"].items():
            eid = int(eid_str)
            if eid not in lookup:
                lookup[eid] = {}
            lookup[eid][cond] = lat
    return lookup


def _path_to_edge_ids(path: list, pair_lookup: dict) -> list:
    """Convert a node-hop list like [0, 3, 7] to a list of edge_ids."""
    edge_ids = []
    for u, v in zip(path[:-1], path[1:]):
        key = (min(u, v), max(u, v))
        edge_ids.append(pair_lookup[key])
    return edge_ids


def enrich_topology(topology: dict, labels: dict) -> dict:
    nodes     = topology["nodes"]
    edges     = topology["edges"]
    tid       = topology["topology_id"]
    node_ids  = [n["node_id"] for n in nodes]

    G           = _build_graph(edges)
    pair_lookup = _build_edge_pair_lookup(edges)
    lat_lookup  = _build_latency_lookup(labels)

    path_latencies = []

    for src in node_ids:
        for dst in node_ids:
            if src == dst:
                continue
            if not nx.has_path(G, src, dst):
                continue

            path     = nx.shortest_path(G, src, dst)          # hop-count shortest
            edge_ids = _path_to_edge_ids(path, pair_lookup)

            e2e = {}
            bottleneck = {}
            for cond in LOAD_CONDITIONS:
                latencies_on_path = [lat_lookup[eid][cond] for eid in edge_ids]
                e2e[cond]        = round(sum(latencies_on_path), 6)
                max_lat          = max(latencies_on_path)
                bottleneck[cond] = edge_ids[latencies_on_path.index(max_lat)]

            path_latencies.append({
                "source":                src,
                "destination":           dst,
                "hop_path":              path,
                "edge_ids_on_path":      edge_ids,
                "end_to_end_latency_ms": e2e,
                "bottleneck_edge_id":    bottleneck,
            })

    return {"topology_id": tid, "path_latencies": path_latencies}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(LABELS_OUT_DIR, exist_ok=True)

    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    total = len(index["topologies"])
    print(f"Enriching labels for {total} topology/ies ...")

    for entry in index["topologies"]:
        base = entry["filename_base"]

        with open(os.path.join(JSON_DIR, f"{base}.json"), encoding="utf-8") as f:
            topology = json.load(f)

        with open(os.path.join(LABELS_RAW_DIR, f"{base}_labels.json"), encoding="utf-8") as f:
            labels = json.load(f)

        enriched = enrich_topology(topology, labels)

        out_path = os.path.join(LABELS_OUT_DIR, f"{base}_enriched_labels.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2)

        n_pairs = len(enriched["path_latencies"])
        print(f"  {base}  {n_pairs} source-destination pairs enriched")

    print(f"\nEnriched label files written to {LABELS_OUT_DIR}/")


if __name__ == "__main__":
    main()
