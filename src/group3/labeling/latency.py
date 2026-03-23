"""
Provides two distinct latency-computation modes:

LAYER 1 - Static ground-truth labels
    compute_all_load_conditions()
    Given a graph and 4 fixed utilization fractions, applies the M/M/1
    formula to every edge and returns per-edge latency for each load condition.

    Formula:
        μ  = bandwidth_mbps / avg_packet_size_mb        avg_packet_size_mb = 0.001
        p  = utilization_fraction  (capped at 0.95)
        W  = (p / (μ * (1 - p))) * 1000               [ms]
        latency = propagation_delay_ms + W              [ms]

LAYER 2 - Temporal simulation labels
    simulate_snapshot()
    Given a graph and a real traffic matrix (from traffic.py), routes each
    (src, dst) demand via shortest-path, accumulates per-link load, then
    computes per-link metrics under either M/M/1 or M/G/1 queuing models:

        Edge outputs per timestep:
            load_mbps, utilization, queue_length_pkts,
            latency_ms (prop + queue + saturation penalty),
            is_bottleneck  (top-10 % utilization flag)

        Path outputs per timestep (sampled up to max_paths):
            source, target, demand_mbps, path_hops,
            e2e_latency_ms, path_bottleneck_count

        Aggregate outputs:
            avg_utilization, max_utilization
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


# Average packet size in Mb (1500 bytes ≈ 0.001 Mb)
_AVG_PKT_MB: float = 0.001
# Bits per average packet (for service rate in packets/s)
_AVG_PKT_BITS: float = _AVG_PKT_MB * 1e6


# ===========================================================================
# LAYER 1 - Static labels
# ===========================================================================

def _mm1_queue_delay_ms(
    bandwidth_mbps: float,
    utilization: float,
    avg_packet_size_mb: float,
) -> float:
    """
    M/M/1 queueing delay in milliseconds.
    utilization is capped at 0.95 to prevent division by zero.
    """
    rho = min(utilization, 0.95)
    mu  = bandwidth_mbps / avg_packet_size_mb   # Mbps / Mb = packets per second... but
    # Actually: mu is service rate in [Mbps / Mb = 1/s ... treat unit as 'flows/s'
    # The formula gives W in seconds, then * 1000 -> ms
    w_s = rho / (mu * max(1.0 - rho, 1e-9))    # seconds
    return float(w_s * 1000.0)


def compute_link_latencies(
    graph: nx.Graph,
    utilization_fraction: float,
    avg_packet_size_mb: float = _AVG_PKT_MB,
) -> Dict[int, float]:
    """Return {edge_id → latency_ms} for a single uniform utilization fraction."""
    result: Dict[int, float] = {}
    for u, v in graph.edges:
        a = graph.edges[u, v]
        edge_id      = int(a["edge_id"])
        bandwidth    = float(a["bandwidth_mbps"])
        prop_delay   = float(a["propagation_delay_ms"])
        w_queue_ms   = _mm1_queue_delay_ms(bandwidth, utilization_fraction, avg_packet_size_mb)
        result[edge_id] = round(prop_delay + w_queue_ms, 6)
    return result


def compute_all_load_conditions(
    graph: nx.Graph,
    load_condition_names: List[str],
    load_conditions: List[float],
    avg_packet_size_mb: float = _AVG_PKT_MB,
) -> Dict[str, Dict[str, object]]:
    """
    Compute static ground-truth latency for all 4 load conditions.

    Returns:
        {
          "low":    {"utilization_fraction": 0.20,
                     "edge_latencies_ms": {"0": 2.51, ...}},
          "medium": {...},
          "high":   {...},
          "flash":  {...},
        }
    """
    result: Dict[str, Dict[str, object]] = {}
    for name, frac in zip(load_condition_names, load_conditions):
        lats = compute_link_latencies(graph, frac, avg_packet_size_mb)
        result[name] = {
            "utilization_fraction": float(frac),
            "edge_latencies_ms":    {str(eid): lat for eid, lat in lats.items()},
        }
    return result


# ===========================================================================
# LAYER 2 - Temporal simulation
# ===========================================================================

def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _temporal_queue_ms(
    utilization: float,
    service_rate_pps: float,
    model: str,
) -> Tuple[float, float]:
    """
    Compute (queueing_delay_ms, queue_length_packets) for M/M/1 or M/G/1.

    Parameters
    ----------
    utilization      : p (clamped to [1e-6, 0.995])
    service_rate_pps : μ in packets/second  = bandwidth_mbps * 1e6 / _AVG_PKT_BITS
    model            : "mm1" or "mg1"

    M/M/1
        W = p / (μ(1-p))      E[N] = p² / (1-p)
    M/G/1  (Pollaczek-Khinchine, SCV = 1.5 for bursty traffic)
        W = ((1+SCV)/2) * p / (μ(1-p))
        E[N] = λ * W
    """
    rho = min(max(utilization, 1e-6), 0.995)
    lam = rho * service_rate_pps

    if model == "mm1":
        w_s    = rho / max(service_rate_pps - lam, 1e-9)
        e_n    = (rho * rho) / max(1.0 - rho, 1e-6)
    else:   # mg1
        scv    = 1.5                    # squared coefficient of variation > 1 → bursty
        w_s    = ((1.0 + scv) / 2.0) * (rho / max(service_rate_pps - lam, 1e-9))
        e_n    = lam * w_s

    return float(w_s * 1000.0), float(e_n)     # (ms, packets)


def simulate_snapshot(
    graph: nx.Graph,
    traffic_matrix: np.ndarray,
    queue_model: str,
    max_paths: int,
) -> Dict[str, object]:
    """
    Route all non-zero demands, accumulate link loads, then compute per-link
    and per-path metrics.

    Parameters
    ----------
    graph          : topology (needs bandwidth_mbps, propagation_delay_ms, edge_id)
    traffic_matrix : np.ndarray shape (N, N), demand_mbps[i, j]
    queue_model    : "mm1" or "mg1"
    max_paths      : maximum number of (src, dst) pairs to record path metrics for

    Returns
    -------
    dict with keys:
        "edge_rows"       : List[dict] - one dict per edge
        "path_rows"       : List[dict] - one dict per sampled path
        "avg_utilization" : float
        "max_utilization" : float
    """
    nodes = list(graph.nodes)

    # ---- Initialise per-edge load accumulator ----
    edge_load: Dict[Tuple[int, int], float] = {
        _edge_key(int(u), int(v)): 0.0 for u, v in graph.edges
    }

    sampled_paths: List[Dict[str, object]] = []
    sampled_count = 0

    # ---- Route each (src, dst) pair via congestion-aware shortest path ----
    # Weight = prop_delay + serialisation time at 1 % load, penalises narrow links
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if i == j:
                continue
            demand = float(traffic_matrix[i, j])
            if demand <= 0.0:
                continue

            try:
                path = nx.shortest_path(
                    graph, source=src, target=dst,
                    weight=lambda u, v, d: float(
                        d["propagation_delay_ms"]
                        + 300.0 / max(d["bandwidth_mbps"], 1.0)
                    ),
                )
            except nx.NetworkXNoPath:
                continue

            for pu, pv in zip(path[:-1], path[1:]):
                edge_load[_edge_key(int(pu), int(pv))] += demand

            if sampled_count < max_paths:
                sampled_paths.append({
                    "source":      int(src),
                    "target":      int(dst),
                    "demand_mbps": demand,
                    "path":        path,
                })
                sampled_count += 1

    # ---- Compute per-edge metrics ----
    edge_rows: List[Dict[str, object]] = []
    edge_latency_map: Dict[Tuple[int, int], float] = {}
    utilizations: List[float] = []

    for u, v in graph.edges:
        k           = _edge_key(int(u), int(v))
        load        = edge_load[k]
        a           = graph.edges[u, v]
        edge_id     = int(a["edge_id"])
        bandwidth   = float(a["bandwidth_mbps"])
        prop_delay  = float(a["propagation_delay_ms"])
        reliability = float(a.get("link_reliability", 0.99))
        cost        = float(a.get("cost", 1.0))
        is_inter    = bool(a.get("is_inter_domain", False))

        util = min(load / max(bandwidth, 1e-6), 1.2)    # allow slight overload
        service_rate_pps = (bandwidth * 1e6) / _AVG_PKT_BITS

        queue_ms, queue_len = _temporal_queue_ms(util, service_rate_pps, queue_model)

        # Saturation penalty: sharp rise above 98 % utilization
        sat_penalty = 0.0
        if util >= 0.98:
            sat_penalty = 10.0 + 120.0 * (util - 0.98)

        latency_ms = prop_delay + queue_ms + sat_penalty
        edge_latency_map[k] = latency_ms
        utilizations.append(util)

        edge_rows.append({
            "edge_id":             edge_id,
            "u":                   int(u),
            "v":                   int(v),
            "bandwidth_mbps":      bandwidth,
            "load_mbps":           float(load),
            "utilization":         float(util),
            "queue_length_pkts":   float(queue_len),
            "latency_ms":          float(latency_ms),
            "propagation_delay_ms":prop_delay,
            "link_reliability":    reliability,
            "cost_metric":         cost,
            "is_bottleneck":       0,       # filled below
            "is_inter_domain":     is_inter,
        })

    # ---- Mark bottlenecks (top-10 % utilization threshold, minimum 85 %) ----
    bottleneck_threshold = float(np.quantile(utilizations, 0.9)) if utilizations else 0.9
    bottleneck_threshold = max(0.85, bottleneck_threshold)
    for row in edge_rows:
        row["is_bottleneck"] = int(float(row["utilization"]) >= bottleneck_threshold)

    # Build a lookup from edge_id → is_bottleneck for path computation
    eid_bottleneck: Dict[int, int] = {
        int(r["edge_id"]): int(r["is_bottleneck"]) for r in edge_rows
    }
    # Build a lookup from edge (u,v) → edge_id
    uv_to_eid: Dict[Tuple[int, int], int] = {
        _edge_key(int(r["u"]), int(r["v"])): int(r["edge_id"]) for r in edge_rows
    }

    # ---- Compute per-path metrics ----
    path_rows: List[Dict[str, object]] = []
    for sp in sampled_paths:
        links = list(zip(sp["path"][:-1], sp["path"][1:]))
        e2e_latency = sum(
            edge_latency_map[_edge_key(int(pu), int(pv))] for pu, pv in links
        )
        bottleneck_count = sum(
            eid_bottleneck.get(uv_to_eid.get(_edge_key(int(pu), int(pv)), -1), 0)
            for pu, pv in links
        )
        path_rows.append({
            "source":               int(sp["source"]),
            "target":               int(sp["target"]),
            "demand_mbps":          float(sp["demand_mbps"]),
            "path_hops":            int(max(len(sp["path"]) - 1, 0)),
            "e2e_latency_ms":       float(e2e_latency),
            "path_bottleneck_count":int(bottleneck_count),
        })

    avg_util = float(np.mean(utilizations)) if utilizations else 0.0
    max_util = float(np.max(utilizations))  if utilizations else 0.0

    return {
        "edge_rows":       edge_rows,
        "path_rows":       path_rows,
        "avg_utilization": avg_util,
        "max_utilization": max_util,
    }