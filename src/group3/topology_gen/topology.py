"""
Generates synthetic multi-domain (AS-like) network topologies per spec.

Topology types and node ranges
    random     : 10-100 nodes  (Erdős-Rényi)
    scale_free : 20-100 nodes  (Barabási-Albert)
    mesh       : 10-50  nodes  (2-D grid)
    ring       : 10-50  nodes  (cycle + random chords)
    hybrid     : 30-100 nodes  (Watts-Strogatz small-world)

Domain model
    2-5 domains per topology.
    Intra-domain link capacity : 100-1000 Mbps
    Inter-domain link capacity :  10-100  Mbps

Node attributes (GraphML key names match spec exactly)
    node_id, router_type, capacity_mbps, x_coord, y_coord,
    domain_id, historical_traffic

Edge attributes (GraphML key names match spec exactly)
    edge_id, bandwidth_mbps, propagation_delay_ms,
    is_inter_domain, link_reliability, cost
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from group3.labeling.config import DatasetConfig


# ---------------------------------------------------------------------------
# Router type -> node capacity scale factor
# ---------------------------------------------------------------------------
ROUTER_PROFILES: Dict[str, float] = {
    "core":    1.6,
    "edge":    0.8,
    "gateway": 1.2,
}

# ---------------------------------------------------------------------------
# Node-range limits per topology type
# ---------------------------------------------------------------------------
TYPE_NODE_RANGES: Dict[str, Tuple[int, int]] = {
    "random":     (10, 100),
    "scale_free": (20, 100),
    "mesh":       (10,  50),
    "ring":       (10,  50),
    "hybrid":     (30, 100),
}


@dataclass
class DomainLayout:
    domain_id: int
    nodes: List[int]


# ---------------------------------------------------------------------------
# Helper: partition N nodes into D roughly-equal groups
# ---------------------------------------------------------------------------
def _partition_nodes(num_nodes: int, num_domains: int, rng: np.random.Generator) -> List[int]:
    base = np.ones(num_domains, dtype=int)
    for _ in range(num_nodes - num_domains):
        base[rng.integers(0, num_domains)] += 1
    rng.shuffle(base)
    return base.tolist()


# ---------------------------------------------------------------------------
# Helper: 2-D centre positions for domains (normalised to [0.1, 0.9])
# ---------------------------------------------------------------------------
def _domain_centres(num_domains: int, rng: np.random.Generator) -> Dict[int, Tuple[float, float]]:
    return {
        d: (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)))
        for d in range(num_domains)
    }


# ---------------------------------------------------------------------------
# Intra-domain topology builders
# ---------------------------------------------------------------------------

def _build_random_domain(size: int, rng: np.random.Generator) -> nx.Graph:
    if size == 1:
        g = nx.Graph(); g.add_node(0); return g
    p = min(0.6, max(0.25, 4.0 / size))
    g = nx.erdos_renyi_graph(size, p, seed=int(rng.integers(0, 2**31)))
    return g if nx.is_connected(g) else nx.minimum_spanning_tree(nx.complete_graph(size))


def _build_scale_free_domain(size: int, rng: np.random.Generator) -> nx.Graph:
    if size <= 2:
        return nx.complete_graph(size)
    m = max(1, min(3, size // 4))
    return nx.barabasi_albert_graph(size, m, seed=int(rng.integers(0, 2**31)))


def _build_mesh_domain(size: int, rng: np.random.Generator) -> nx.Graph:
    cols = max(2, int(math.ceil(math.sqrt(size))))
    rows = max(2, int(math.ceil(size / cols)))
    g = nx.grid_2d_graph(rows, cols)
    mapping = {n: i for i, n in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, mapping)
    nodes_to_remove = list(g.nodes())[size:]
    g.remove_nodes_from(nodes_to_remove)
    return g if nx.is_connected(g) else nx.minimum_spanning_tree(nx.complete_graph(size))


def _build_ring_domain(size: int, rng: np.random.Generator) -> nx.Graph:
    if size <= 2:
        return nx.complete_graph(size)
    g = nx.cycle_graph(size)
    # Add ~10 % extra chords for robustness
    extra = max(0, int(size * 0.10))
    nodes = list(g.nodes())
    for _ in range(extra):
        u, v = int(rng.choice(nodes)), int(rng.choice(nodes))
        if u != v:
            g.add_edge(u, v)
    return g


def _build_hybrid_domain(size: int, rng: np.random.Generator) -> nx.Graph:
    if size <= 2:
        return nx.complete_graph(size)
    k = min(4, max(2, size // 4))
    if k % 2 == 1:
        k = max(2, k - 1)
    try:
        return nx.connected_watts_strogatz_graph(
            size, k=k, p=0.3, tries=20, seed=int(rng.integers(0, 2**31))
        )
    except nx.NetworkXError:
        return nx.cycle_graph(size)


_DOMAIN_BUILDERS = {
    "random":     _build_random_domain,
    "scale_free": _build_scale_free_domain,
    "mesh":       _build_mesh_domain,
    "ring":       _build_ring_domain,
    "hybrid":     _build_hybrid_domain,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_topology(
    cfg: DatasetConfig,
    rng: np.random.Generator,
    topology_id: int,
    topology_type: str,
    generation_seed: int,
) -> nx.Graph:
    """
    Build a single multi-domain topology.

    Parameters
    ----------
    cfg             : DatasetConfig
    rng             : seeded numpy Generator
    topology_id     : 1-indexed integer (matches filename topology_XXX)
    topology_type   : one of random / scale_free / mesh / ring / hybrid
    generation_seed : recorded in the graph for reproducibility

    Returns
    -------
    nx.Graph with all node and edge attributes set per spec.
    """
    # ---- Node count (clamped to type-specific range) ----
    type_min, type_max = TYPE_NODE_RANGES[topology_type]
    lo = max(cfg.min_nodes, type_min)
    hi = min(cfg.max_nodes, type_max)
    if lo > hi:
        lo, hi = type_min, type_max
    num_nodes = int(rng.integers(lo, hi + 1))

    # ---- Domain count: 2-5 per spec ----
    max_d = min(cfg.max_domains, max(cfg.min_domains, num_nodes // 3))
    max_d = max(cfg.min_domains, max_d)
    num_domains = int(rng.integers(cfg.min_domains, max_d + 1))
    domain_sizes = _partition_nodes(num_nodes, num_domains, rng)
    centres = _domain_centres(num_domains, rng)

    builder = _DOMAIN_BUILDERS[topology_type]
    graph = nx.Graph(
        topology_id=topology_id,
        num_domains=num_domains,
        topology_type=topology_type,
        generation_seed=generation_seed,
    )

    layouts: List[DomainLayout] = []
    current_node = 0

    # ---- Build each domain ----
    for domain_id, size in enumerate(domain_sizes):
        domain_nodes = list(range(current_node, current_node + size))
        layouts.append(DomainLayout(domain_id=domain_id, nodes=domain_nodes))
        current_node += size

        local = builder(size, rng)
        if size > 1 and not nx.is_connected(local):
            local = nx.minimum_spanning_tree(nx.complete_graph(size))

        mapping = {old: domain_nodes[old] for old in local.nodes}
        local = nx.relabel_nodes(local, mapping)
        graph.add_nodes_from(local.nodes)
        graph.add_edges_from(local.edges, is_inter_domain=False)

        cx, cy = centres[domain_id]
        for n in domain_nodes:
            router_type = str(rng.choice(
                list(ROUTER_PROFILES.keys()), p=[0.20, 0.50, 0.30]
            ))
            cap_scale    = ROUTER_PROFILES[router_type]
            node_cap     = float(np.clip(rng.uniform(100.0, 1000.0) * cap_scale, 100.0, 2000.0))
            x = float(np.clip(cx + rng.normal(0, 0.08), 0.0, 1.0))
            y = float(np.clip(cy + rng.normal(0, 0.08), 0.0, 1.0))
            # historical_traffic: per-node baseline load ratio [0.1, 0.8]
            # Used by traffic.py to scale outgoing demand realistically
            hist_traffic = float(rng.uniform(0.1, 0.8))
            graph.nodes[n].update({
                "node_id":           int(n),
                "router_type":       router_type,
                "capacity_mbps":     node_cap,
                "x_coord":           x,
                "y_coord":           y,
                "domain_id":         int(domain_id),
                "historical_traffic":hist_traffic,   # ← node feature for GNN + traffic sim
            })

    # ---- Inter-domain edges (AS-like sparse backbone) ----
    for left in range(num_domains - 1):
        u = int(rng.choice(layouts[left].nodes))
        v = int(rng.choice(layouts[left + 1].nodes))
        graph.add_edge(u, v, is_inter_domain=True)

    extra = int(rng.integers(num_domains, max(num_domains + 1, 2 * num_domains)))
    for _ in range(extra):
        d1, d2 = rng.choice(num_domains, size=2, replace=False)
        u = int(rng.choice(layouts[int(d1)].nodes))
        v = int(rng.choice(layouts[int(d2)].nodes))
        graph.add_edge(u, v, is_inter_domain=True)

    # ---- Static edge attributes ----
    for eidx, (u, v) in enumerate(graph.edges):
        is_inter = bool(graph.edges[u, v].get("is_inter_domain", False))
        ux, uy = graph.nodes[u]["x_coord"], graph.nodes[u]["y_coord"]
        vx, vy = graph.nodes[v]["x_coord"], graph.nodes[v]["y_coord"]
        distance      = float(np.hypot(ux - vx, uy - vy))
        prop_delay_ms = float(max(0.1, distance * 20.0 + rng.uniform(0.1, 2.0)))

        # Capacity per spec
        bandwidth_mbps = (
            float(rng.uniform(10.0, 100.0))    # inter-domain: 10-100 Mbps
            if is_inter else
            float(rng.uniform(100.0, 1000.0))  # intra-domain: 100-1000 Mbps
        )
        reliability = float(rng.uniform(0.94, 0.999))
        cost        = float(prop_delay_ms + 500.0 / max(bandwidth_mbps, 1.0))

        graph.edges[u, v].update({
            "edge_id":             int(eidx),
            "bandwidth_mbps":      bandwidth_mbps,
            "propagation_delay_ms":prop_delay_ms,
            "is_inter_domain":     is_inter,
            "link_reliability":    reliability,
            "cost":                cost,
        })

    # ---- Graph-level attributes ----
    domain_policy: Dict[str, List[int]] = {}
    for d in range(num_domains):
        others  = [o for o in range(num_domains) if o != d]
        k       = int(rng.integers(1, max(2, len(others) + 1)))
        allowed = [int(x) for x in rng.choice(others, size=min(k, len(others)), replace=False)]
        domain_policy[str(d)] = sorted(allowed)

    graph.graph.update({
        "topology_id":     int(topology_id),
        "num_nodes":       int(graph.number_of_nodes()),
        "num_edges":       int(graph.number_of_edges()),
        "num_domains":     int(num_domains),
        "topology_type":   topology_type,
        "generation_seed": int(generation_seed),
        "domain_policies": json.dumps(domain_policy),
        "description":     "Synthetic multi-domain network topology",
    })

    return graph