"""
Generates realistic, time-varying traffic matrices for a given network topology.

Features
--------
* Diurnal pattern      - sinusoidal load variation over a simulated 24-hour day
* Flash crowds         - random burst events that multiply demand 1.8-3.4x
* Application profiles - web (bursty/light), video (steady/heavy), bulk (large/smooth)
* Inter-domain boost   - cross-domain flows carry slightly higher base demand
* Per-node historical  - nodes with higher historical_traffic generate more outgoing load

Output
------
generate_traffic_matrix() returns:
    matrix      : np.ndarray  shape (N, N)  - demand in Mbps for every (src, dst) pair
    global_meta : dict with global features for the snapshot:
                    time_of_day        [0, 24)  hours
                    diurnal_factor     [0.25, 1.15]
                    flash_event        0 or 1
                    flash_multiplier   1.0 or 1.8-3.4
                    traffic_profile    "web" | "video" | "bulk"
                    network_load_mbps  total traffic in Mbps
"""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Application traffic profiles:  (mean_scale, log-normal sigma)
# ---------------------------------------------------------------------------
APP_PROFILES: Dict[str, Tuple[float, float]] = {
    "web":   (0.5, 0.6),    # many small HTTP flows, high variance
    "video": (1.1, 0.4),    # large steady streams, low variance
    "bulk":  (0.9, 1.0),    # large occasional transfers, high variance
}
APP_PROFILE_PROBS = [0.55, 0.25, 0.20]   # web, video, bulk selection probabilities


def _diurnal_factor(timestep: int, total_steps: int) -> float:
    """
    Map a discrete timestep index onto a 24-hour diurnal load multiplier.

    Shape: sinusoidal, trough (~0.25) at t=0 (≈ 4 AM), peak (~1.15) at midday.
    """
    phase = 2.0 * np.pi * (timestep % total_steps) / max(total_steps, 1)
    raw = np.sin(phase - np.pi / 2.0)              # ∈ [-1, 1]
    return float(0.70 + 0.45 * (1.0 + raw) / 2.0) # ∈ [0.25, 1.15]


def generate_traffic_matrix(
    graph: nx.Graph,
    timestep: int,
    total_steps: int,
    rng: np.random.Generator,
    flash_crowd_probability: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Build an NxN traffic demand matrix for one timestep.

    Parameters
    ----------
    graph                   : topology graph; nodes must have 'domain_id';
                              'historical_traffic' is used if present.
    timestep                : current time index (0 … total_steps-1)
    total_steps             : total timesteps in the series for this topology
    rng                     : seeded numpy Generator
    flash_crowd_probability : probability this timestep is a flash-crowd event

    Returns
    -------
    matrix      : np.ndarray float64, shape (N, N), demand_mbps[i, j]
    global_meta : dict of global snapshot features
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=np.float64)

    # ---- Global multipliers ----
    diurnal = _diurnal_factor(timestep, total_steps)

    is_flash = bool(rng.uniform(0.0, 1.0) < flash_crowd_probability)
    flash_multiplier = float(rng.uniform(1.8, 3.4)) if is_flash else 1.0

    app_name = str(rng.choice(list(APP_PROFILES.keys()), p=APP_PROFILE_PROBS))
    app_mean_scale, app_sigma = APP_PROFILES[app_name]

    # ---- Per-flow demand ----
    for i, src in enumerate(nodes):
        src_domain = int(graph.nodes[src].get("domain_id", 0))
        # Nodes with higher historical load send proportionally more traffic
        src_hist = float(graph.nodes[src].get("historical_traffic", 0.4))

        for j, dst in enumerate(nodes):
            if src == dst:
                continue
            dst_domain = int(graph.nodes[dst].get("domain_id", 0))

            # Cross-domain flows are heavier (aggregation / peering effect)
            inter_boost = 1.3 if src_domain != dst_domain else 1.0

            # Lognormal base demand
            base = float(rng.lognormal(mean=0.1, sigma=app_sigma))

            demand_mbps = (
                base
                * app_mean_scale
                * diurnal
                * flash_multiplier
                * inter_boost
                * (0.5 + src_hist)   # node historical traffic scaling [0.5, 1.3]
            )
            matrix[i, j] = max(0.0, demand_mbps)

    global_meta: Dict[str, object] = {
        "time_of_day":       float((24.0 * timestep) / max(total_steps, 1)),
        "diurnal_factor":    float(diurnal),
        "flash_event":       int(is_flash),
        "flash_multiplier":  float(flash_multiplier),
        "traffic_profile":   app_name,
        "network_load_mbps": float(matrix.sum()),
    }
    return matrix, global_meta