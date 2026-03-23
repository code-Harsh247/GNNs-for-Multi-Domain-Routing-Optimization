"""
generate_mock_data.py
=====================
Generates 5 mock topologies that exactly match Group 3's schema contract
(defined in group3_implementation_plan.md).

Run from the repo root:
    python src/group11/generate_mock_data.py

Output written to:
    data/raw/topologies/json/     → topology_00X.json
    data/raw/topologies/graphml/  → topology_00X.graphml
    data/raw/topologies/csv/      → topology_00X_nodes.csv, topology_00X_edges.csv
    data/raw/labels/              → topology_00X_labels.json
    data/raw/topology_index.json

HOW IT WORKS
------------
1. Graph structure
   Each topology type maps to a NetworkX generator:
     - random     → erdos_renyi_graph (random edges with fixed probability)
     - scale_free → barabasi_albert_graph (hubs emerge naturally)
     - mesh       → grid_2d_graph trimmed to target node count
     - ring       → cycle_graph
     - hybrid     → barabasi_albert core + cycle_graph periphery, joined by cross-edges

2. Node attributes
   - router_type: randomly sampled from [core, edge, gateway]
   - capacity_mbps: one of [100, 250, 500, 1000] Mbps
   - x_coord, y_coord: uniform [0, 1] (already normalized)
   - domain_id: assigned by node_id % num_domains  (round-robin, gives even partition)
   All RNG calls use the topology's generation_seed for reproducibility.

3. Edge attributes
   - bandwidth_mbps: 10–100 Mbps for inter-domain, 100–1000 for intra-domain
     (matches Group 3's spec: inter-domain links are narrower)
   - propagation_delay_ms: uniform [1, 20] ms
   - is_inter_domain: True when source and target nodes are in different domains
   - link_reliability: uniform [0.95, 1.0]
   - cost: uniform [1, 10]

4. M/M/1 latency labels (Group 3's queueing formula)
   For each load condition (low=0.20, medium=0.50, high=0.80, flash=0.95):
     μ       = bandwidth_mbps / 0.001   (service rate, packets/ms)
     ρ       = utilization_fraction      (= λ/μ, always < 1)
     W_queue = (ρ / (μ × (1 − ρ))) × 1000   (ms)
     latency = propagation_delay_ms + W_queue
   The flash condition caps utilization at 0.95 to avoid division by zero.
"""

import csv
import json
import os
import random

import networkx as nx

# ---------------------------------------------------------------------------
# Configuration — 5 mock topologies covering all 5 topology types
# ---------------------------------------------------------------------------
TOPOLOGIES_CONFIG = [
    {"topology_id": 1, "filename_base": "topology_001", "num_nodes": 10,
     "topology_type": "random",     "num_domains": 2, "generation_seed": 42},
    {"topology_id": 2, "filename_base": "topology_002", "num_nodes": 20,
     "topology_type": "scale_free", "num_domains": 3, "generation_seed": 17},
    {"topology_id": 3, "filename_base": "topology_003", "num_nodes": 16,
     "topology_type": "mesh",       "num_domains": 2, "generation_seed": 99},
    {"topology_id": 4, "filename_base": "topology_004", "num_nodes": 10,
     "topology_type": "ring",       "num_domains": 2, "generation_seed": 55},
    {"topology_id": 5, "filename_base": "topology_005", "num_nodes": 30,
     "topology_type": "hybrid",     "num_domains": 4, "generation_seed": 73},
]

LOAD_CONDITIONS = {
    "low":    0.20,
    "medium": 0.50,
    "high":   0.80,
    "flash":  0.95,   # already capped — no extra clamping needed
}

ROUTER_TYPES = ["core", "edge", "gateway"]
CAPACITY_OPTIONS = [100.0, 250.0, 500.0, 1000.0]
AVG_PACKET_SIZE_MB = 0.001   # 1 KB — standard Ethernet frame approximation

# ---------------------------------------------------------------------------
# Output paths (relative to repo root — run script from there)
# ---------------------------------------------------------------------------
RAW_ROOT    = os.path.join("data", "raw")
JSON_DIR    = os.path.join(RAW_ROOT, "topologies", "json")
GRAPHML_DIR = os.path.join(RAW_ROOT, "topologies", "graphml")
CSV_DIR     = os.path.join(RAW_ROOT, "topologies", "csv")
LABELS_DIR  = os.path.join(RAW_ROOT, "labels")


# ---------------------------------------------------------------------------
# Step 1 — Graph generation
# ---------------------------------------------------------------------------
def _generate_nx_graph(topology_type: str, num_nodes: int, seed: int) -> nx.Graph:
    """Return a connected undirected NetworkX graph for the requested type."""
    rng = random.Random(seed)

    if topology_type == "random":
        # erdos_renyi: each possible edge included with probability p=0.35
        G = nx.erdos_renyi_graph(num_nodes, 0.35, seed=seed)

    elif topology_type == "scale_free":
        # barabasi_albert: new nodes attach to m=2 existing nodes (power-law degree)
        G = nx.barabasi_albert_graph(num_nodes, 2, seed=seed)

    elif topology_type == "mesh":
        # Start from a square grid, trim extra nodes, relabel 0..N-1
        side = int(num_nodes ** 0.5) + 1
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
        excess = list(G.nodes())[num_nodes:]
        G.remove_nodes_from(excess)
        G = nx.convert_node_labels_to_integers(G)

    elif topology_type == "ring":
        G = nx.cycle_graph(num_nodes)

    elif topology_type == "hybrid":
        # Core: scale-free subgraph on half the nodes
        half = num_nodes // 2
        core = nx.barabasi_albert_graph(half, 2, seed=seed)
        # Periphery: ring on the remaining nodes, relabelled to avoid ID collision
        ring = nx.cycle_graph(num_nodes - half)
        ring = nx.relabel_nodes(ring, {n: n + half for n in ring.nodes()})
        G = nx.compose(core, ring)
        # Add 3 cross-edges so the two parts are connected
        for _ in range(3):
            u = rng.randint(0, half - 1)
            v = rng.randint(half, num_nodes - 1)
            G.add_edge(u, v)

    else:
        raise ValueError(f"Unknown topology_type: {topology_type}")

    # Guarantee connectivity by stitching disconnected components
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        anchor = next(iter(components[0]))
        for comp in components[1:]:
            G.add_edge(anchor, next(iter(comp)))

    return G


# ---------------------------------------------------------------------------
# Step 2 — Build node + edge attribute dicts
# ---------------------------------------------------------------------------
def _build_nodes(G: nx.Graph, num_domains: int, rng: random.Random) -> list:
    nodes = []
    for n in sorted(G.nodes()):
        nodes.append({
            "node_id":       n,
            "router_type":   rng.choice(ROUTER_TYPES),
            "capacity_mbps": rng.choice(CAPACITY_OPTIONS),
            "x_coord":       round(rng.random(), 4),
            "y_coord":       round(rng.random(), 4),
            "domain_id":     n % num_domains,   # round-robin domain assignment
        })
    return nodes


def _build_edges(G: nx.Graph, nodes: list, rng: random.Random) -> list:
    domain_map = {n["node_id"]: n["domain_id"] for n in nodes}
    edges = []
    for eid, (u, v) in enumerate(sorted(G.edges())):
        is_inter = domain_map[u] != domain_map[v]
        # Inter-domain links are narrower (10–100 Mbps) per the spec
        bw = rng.uniform(10, 100) if is_inter else rng.uniform(100, 1000)
        edges.append({
            "edge_id":              eid,
            "source":               u,
            "target":               v,
            "bandwidth_mbps":       round(bw, 2),
            "propagation_delay_ms": round(rng.uniform(1.0, 20.0), 2),
            "is_inter_domain":      is_inter,
            "link_reliability":     round(rng.uniform(0.95, 1.0), 4),
            "cost":                 round(rng.uniform(1.0, 10.0), 2),
        })
    return edges


# ---------------------------------------------------------------------------
# Step 3 — M/M/1 latency labels
# ---------------------------------------------------------------------------
def _mm1_latency(bandwidth_mbps: float, propagation_delay_ms: float,
                 utilization: float) -> float:
    """
    M/M/1 total link latency (ms).

    μ       = bandwidth_mbps / AVG_PACKET_SIZE_MB   (service rate in packets/ms)
    ρ       = utilization  (arrival rate / service rate, must be < 1)
    W_queue = (ρ / (μ × (1 − ρ))) × 1000            (queueing delay in ms)
    latency = propagation_delay_ms + W_queue
    """
    mu = bandwidth_mbps / AVG_PACKET_SIZE_MB
    rho = min(utilization, 0.95)           # safety cap
    w_queue = (rho / (mu * (1.0 - rho))) * 1000.0
    return round(propagation_delay_ms + w_queue, 6)


def _build_labels(topology_id: int, edges: list) -> dict:
    labels = {"topology_id": topology_id, "load_conditions": {}}
    for condition, util in LOAD_CONDITIONS.items():
        edge_latencies = {
            str(e["edge_id"]): _mm1_latency(
                e["bandwidth_mbps"], e["propagation_delay_ms"], util
            )
            for e in edges
        }
        labels["load_conditions"][condition] = {
            "utilization_fraction": util,
            "edge_latencies_ms":    edge_latencies,
        }
    return labels


# ---------------------------------------------------------------------------
# Step 4 — Build domain_policies (every domain allows transit through all others)
# ---------------------------------------------------------------------------
def _build_domain_policies(num_domains: int) -> dict:
    return {
        str(d): [x for x in range(num_domains) if x != d]
        for d in range(num_domains)
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _makedirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _write_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_graphml(path: str, topology: dict):
    """Write GraphML with all required <key> declarations and <data> elements."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/graphml">',
        # Node key declarations
        '  <key id="node_id" for="node" attr.name="node_id" attr.type="int"/>',
        '  <key id="router_type" for="node" attr.name="router_type" attr.type="string"/>',
        '  <key id="capacity_mbps" for="node" attr.name="capacity_mbps" attr.type="double"/>',
        '  <key id="x_coord" for="node" attr.name="x_coord" attr.type="double"/>',
        '  <key id="y_coord" for="node" attr.name="y_coord" attr.type="double"/>',
        '  <key id="domain_id" for="node" attr.name="domain_id" attr.type="int"/>',
        # Edge key declarations — edge_id must be explicit <data>, not just the id= attribute
        '  <key id="edge_id" for="edge" attr.name="edge_id" attr.type="int"/>',
        '  <key id="bandwidth_mbps" for="edge" attr.name="bandwidth_mbps" attr.type="double"/>',
        '  <key id="propagation_delay_ms" for="edge" attr.name="propagation_delay_ms" attr.type="double"/>',
        '  <key id="is_inter_domain" for="edge" attr.name="is_inter_domain" attr.type="boolean"/>',
        '  <key id="link_reliability" for="edge" attr.name="link_reliability" attr.type="double"/>',
        '  <key id="cost" for="edge" attr.name="cost" attr.type="double"/>',
        f'  <graph id="topology_{topology["topology_id"]:03d}" edgedefault="undirected">',
    ]

    for n in topology["nodes"]:
        lines += [
            f'    <node id="{n["node_id"]}">',
            f'      <data key="node_id">{n["node_id"]}</data>',
            f'      <data key="router_type">{n["router_type"]}</data>',
            f'      <data key="capacity_mbps">{n["capacity_mbps"]}</data>',
            f'      <data key="x_coord">{n["x_coord"]}</data>',
            f'      <data key="y_coord">{n["y_coord"]}</data>',
            f'      <data key="domain_id">{n["domain_id"]}</data>',
            f'    </node>',
        ]

    for e in topology["edges"]:
        inter_str = "true" if e["is_inter_domain"] else "false"
        lines += [
            f'    <edge id="e{e["edge_id"]}" source="{e["source"]}" target="{e["target"]}">',
            f'      <data key="edge_id">{e["edge_id"]}</data>',
            f'      <data key="bandwidth_mbps">{e["bandwidth_mbps"]}</data>',
            f'      <data key="propagation_delay_ms">{e["propagation_delay_ms"]}</data>',
            f'      <data key="is_inter_domain">{inter_str}</data>',
            f'      <data key="link_reliability">{e["link_reliability"]}</data>',
            f'      <data key="cost">{e["cost"]}</data>',
            f'    </edge>',
        ]

    lines += ["  </graph>", "</graphml>"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_csv_nodes(path: str, nodes: list):
    fieldnames = ["node_id", "router_type", "capacity_mbps", "x_coord", "y_coord", "domain_id"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nodes)


def _write_csv_edges(path: str, edges: list):
    fieldnames = [
        "edge_id", "source", "target", "bandwidth_mbps",
        "propagation_delay_ms", "is_inter_domain", "link_reliability", "cost",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edges)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _makedirs(JSON_DIR, GRAPHML_DIR, CSV_DIR, LABELS_DIR)

    index_entries = []

    for cfg in TOPOLOGIES_CONFIG:
        tid        = cfg["topology_id"]
        base       = cfg["filename_base"]
        seed       = cfg["generation_seed"]
        num_nodes  = cfg["num_nodes"]
        num_domains = cfg["num_domains"]
        ttype      = cfg["topology_type"]

        print(f"  Generating {base}  ({ttype}, {num_nodes} nodes, {num_domains} domains) ...", end=" ")

        rng = random.Random(seed)

        # Build graph structure and attributes
        G       = _generate_nx_graph(ttype, num_nodes, seed)
        nodes   = _build_nodes(G, num_domains, rng)
        edges   = _build_edges(G, nodes, rng)
        policies = _build_domain_policies(num_domains)
        num_edges = len(edges)

        # Assemble the JSON topology object
        topology_json = {
            "topology_id":    tid,
            "num_nodes":      num_nodes,
            "num_edges":      num_edges,
            "num_domains":    num_domains,
            "topology_type":  ttype,
            "generation_seed": seed,
            "domain_policies": policies,
            "nodes":          nodes,
            "edges":          edges,
        }

        labels = _build_labels(tid, edges)

        # Write all 4 output formats
        _write_json(   os.path.join(JSON_DIR,    f"{base}.json"),         topology_json)
        _write_graphml(os.path.join(GRAPHML_DIR, f"{base}.graphml"),      topology_json)
        _write_csv_nodes(os.path.join(CSV_DIR,   f"{base}_nodes.csv"),    nodes)
        _write_csv_edges(os.path.join(CSV_DIR,   f"{base}_edges.csv"),    edges)
        _write_json(   os.path.join(LABELS_DIR,  f"{base}_labels.json"),  labels)

        index_entries.append({
            "topology_id":    tid,
            "filename_base":  base,
            "num_nodes":      num_nodes,
            "num_edges":      num_edges,
            "topology_type":  ttype,
            "num_domains":    num_domains,
            "generation_seed": seed,
        })

        print(f"done  ({num_edges} edges)")

    # Write master index
    index = {"total_topologies": len(index_entries), "topologies": index_entries}
    _write_json(os.path.join(RAW_ROOT, "topology_index.json"), index)
    print(f"\ntopology_index.json written  ({len(index_entries)} entries)")
    print("\nAll mock data written to data/raw/")


if __name__ == "__main__":
    main()
