"""
simulate_traffic.py  —  Phase B: Traffic Simulation
=====================================================
For every topology in topology_index.json, generates 16 traffic matrices
(4 application types × 4 load conditions) plus diurnal snapshots and an
optional flash-crowd event, and writes topology_XXX_traffic.json to
data/processed/traffic/.

Run from the repo root:
    python src/group11/traffic_simulation/simulate_traffic.py

DESIGN
------
Traffic matrix T[i][j] = traffic demand in Mbps from node i to node j.
  - Diagonal always 0
  - Scaled by load condition multiplier to produce low/medium/high/flash matrices

Each application type has its own base rate range and distribution:
  video_streaming : Uniform [5, 25] Mbps  — steady
  web_browsing    : Exponential mean=2    — bursty; capped at 10 Mbps
  file_transfer   : Uniform [10, 100] Mbps — sustained
  voip            : Uniform [0.1, 0.5] Mbps — constant small

Load condition multipliers scale the base matrix:
  low    = 0.20
  medium = 0.50
  high   = 0.80
  flash  = 0.95  (near-saturation)
These map onto the same 4 utilization levels used by Group 3's M/M/1 model.

Diurnal snapshots: the medium-load video_streaming matrix is scaled by
the hourly multiplier for peak (18:00), off-peak (03:00), and business (10:00).

Flash crowd: applied to every 5th topology (topology_id % 5 == 0).
All traffic destined for a randomly chosen node is multiplied by 5.
"""

import json
import os
import random

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ROOT     = os.path.join("data", "raw")
TRAFFIC_DIR  = os.path.join("data", "processed", "traffic")
INDEX_PATH   = os.path.join(RAW_ROOT, "topology_index.json")
JSON_DIR     = os.path.join(RAW_ROOT, "topologies", "json")

# ---------------------------------------------------------------------------
# Diurnal multipliers — one per hour, index = hour of day
# ---------------------------------------------------------------------------
DIURNAL_MULTIPLIERS = [
    0.3, 0.2, 0.2, 0.2, 0.3, 0.4,   # 00:00 – 05:00
    0.6, 0.8, 1.0, 1.0, 0.9, 0.9,   # 06:00 – 11:00
    1.0, 1.0, 0.9, 0.9, 1.0, 1.1,   # 12:00 – 17:00
    1.2, 1.1, 1.0, 0.8, 0.6, 0.4,   # 18:00 – 23:00
]

LOAD_MULTIPLIERS = {
    "low":    0.20,
    "medium": 0.50,
    "high":   0.80,
    "flash":  0.95,
}


# ---------------------------------------------------------------------------
# Base traffic matrix generators
# ---------------------------------------------------------------------------
def _zero_diagonal(matrix: list) -> list:
    for i in range(len(matrix)):
        matrix[i][i] = 0.0
    return matrix


def _scale_matrix(matrix: list, factor: float) -> list:
    return [[round(v * factor, 4) for v in row] for row in matrix]


def _gen_video_streaming(n: int, rng: random.Random) -> list:
    """Uniform [5, 25] Mbps — steady baseline."""
    return _zero_diagonal(
        [[round(rng.uniform(5.0, 25.0), 4) for _ in range(n)] for _ in range(n)]
    )


def _gen_web_browsing(n: int, rng: random.Random) -> list:
    """Exponential (mean=2) — bursty, capped at 10 Mbps."""
    return _zero_diagonal(
        [[round(min(rng.expovariate(1 / 2.0), 10.0), 4) for _ in range(n)]
         for _ in range(n)]
    )


def _gen_file_transfer(n: int, rng: random.Random) -> list:
    """Uniform [10, 100] Mbps — sustained throughput."""
    return _zero_diagonal(
        [[round(rng.uniform(10.0, 100.0), 4) for _ in range(n)] for _ in range(n)]
    )


def _gen_voip(n: int, rng: random.Random) -> list:
    """Uniform [0.1, 0.5] Mbps — constant small packets."""
    return _zero_diagonal(
        [[round(rng.uniform(0.1, 0.5), 4) for _ in range(n)] for _ in range(n)]
    )


APP_GENERATORS = {
    "video_streaming": _gen_video_streaming,
    "web_browsing":    _gen_web_browsing,
    "file_transfer":   _gen_file_transfer,
    "voip":            _gen_voip,
}


# ---------------------------------------------------------------------------
# Per-topology builder
# ---------------------------------------------------------------------------
def build_traffic(topology_id: int, num_nodes: int,
                  generation_seed: int) -> dict:
    rng = random.Random(generation_seed + 1000)   # offset from topology seed

    # --- 16 traffic matrices (4 apps × 4 load conditions) ---
    traffic_matrices = {}
    for app_name, gen_fn in APP_GENERATORS.items():
        base = gen_fn(num_nodes, rng)
        traffic_matrices[app_name] = {
            cond: _scale_matrix(base, mult)
            for cond, mult in LOAD_MULTIPLIERS.items()
        }

    # --- Diurnal snapshots (based on medium-load video_streaming matrix) ---
    medium_matrix = traffic_matrices["video_streaming"]["medium"]
    diurnal_snapshots = {
        "peak_18h": {
            "multiplier":     DIURNAL_MULTIPLIERS[18],
            "traffic_matrix": _scale_matrix(medium_matrix, DIURNAL_MULTIPLIERS[18]),
        },
        "offpeak_03h": {
            "multiplier":     DIURNAL_MULTIPLIERS[3],
            "traffic_matrix": _scale_matrix(medium_matrix, DIURNAL_MULTIPLIERS[3]),
        },
        "business_10h": {
            "multiplier":     DIURNAL_MULTIPLIERS[10],
            "traffic_matrix": _scale_matrix(medium_matrix, DIURNAL_MULTIPLIERS[10]),
        },
    }

    # --- Flash crowd (every 5th topology by ID) ---
    apply_flash = (topology_id % 5 == 0)
    if apply_flash:
        target_node = rng.randint(0, num_nodes - 1)
        flash_crowd = {
            "applied":         True,
            "target_node_id":  target_node,
            "spike_multiplier": 5.0,
        }
    else:
        flash_crowd = {"applied": False}

    return {
        "topology_id":       topology_id,
        "traffic_matrices":  traffic_matrices,
        "diurnal_snapshots": diurnal_snapshots,
        "flash_crowd":       flash_crowd,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(TRAFFIC_DIR, exist_ok=True)

    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)

    total = len(index["topologies"])
    print(f"Simulating traffic for {total} topology/ies ...")

    for entry in index["topologies"]:
        base     = entry["filename_base"]
        tid      = entry["topology_id"]
        n_nodes  = entry["num_nodes"]
        seed     = entry["generation_seed"]

        traffic = build_traffic(tid, n_nodes, seed)

        out_path = os.path.join(TRAFFIC_DIR, f"{base}_traffic.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(traffic, f, indent=2)

        flash_str = " [flash-crowd]" if traffic["flash_crowd"]["applied"] else ""
        print(f"  {base}  {n_nodes} nodes  16 matrices{flash_str}")

    print(f"\nTraffic files written to {TRAFFIC_DIR}/")


if __name__ == "__main__":
    main()
