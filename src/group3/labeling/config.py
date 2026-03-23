"""
Two layers of output are generated per topology:

  Layer 1 - Static topology files
      topology_XXX.graphml / .json / _nodes.csv / _edges.csv
      topology_XXX_labels.json   (4 fixed load conditions, M/M/1)

  Layer 2 - Temporal snapshot files
      snapshots/topology_XXX_t00_snapshot.json   (T timesteps)
      snapshots/topology_XXX_snapshots.csv        (flat CSV of all timesteps)
      Each snapshot carries the full traffic matrix, per-link utilization /
      queue length / latency (M/M/1 and M/G/1), path latency, and bottlenecks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Topology type distribution
# Random:150  Scale-free:100  Mesh:100  Ring:75  Hybrid:75  => 500 total
# ---------------------------------------------------------------------------
TOPOLOGY_TYPE_DISTRIBUTION = [
    ("random",     150),
    ("scale_free", 100),
    ("mesh",       100),
    ("ring",        75),
    ("hybrid",      75),
]

@dataclass
class DatasetConfig:
    # ---- Reproducibility ----
    random_seed: int = 42

    # ---- Topology count & type distribution ----
    num_topologies: int = 500
    use_distribution: bool = True   # follow the 150/100/100/75/75 split

    # ---- Topology size ----
    min_nodes: int = 10
    max_nodes: int = 100

    # ---- Domain model ----
    min_domains: int = 2
    max_domains: int = 5

    # ---- Output ----
    output_dir: Path = Path("data/raw")

    # ------------------------------------------------------------------
    # Layer 1 - Static labels (4 fixed load conditions, M/M/1)
    # ------------------------------------------------------------------
    load_conditions: List[float] = field(
        default_factory=lambda: [0.20, 0.50, 0.80, 0.95]
    )
    load_condition_names: List[str] = field(
        default_factory=lambda: ["low", "medium", "high", "flash"]
    )
    # M/M/1: average_packet_size_mb = 0.001  (≈ 1500 bytes per spec)
    avg_packet_size_mb: float = 0.001

    # ------------------------------------------------------------------
    # Layer 2 - Temporal simulation (traffic matrices + snapshots)
    # ------------------------------------------------------------------
    timesteps_per_topology: int = 12        # T time steps per topology
    queue_model: str = "mm1"                # "mm1" or "mg1"
    flash_crowd_probability: float = 0.08   # prob. of flash-crowd at any timestep
    max_paths_per_snapshot: int = 250       # cap sampled (src,dst) pairs for path labels

    def validate(self) -> None:
        if self.min_nodes < 2 or self.max_nodes < self.min_nodes:
            raise ValueError("Invalid node bounds")
        if self.min_domains < 2 or self.max_domains < self.min_domains:
            raise ValueError("Invalid domain bounds")
        if self.num_topologies <= 0:
            raise ValueError("num_topologies must be positive")
        if len(self.load_conditions) != len(self.load_condition_names):
            raise ValueError("load_conditions and load_condition_names must have equal length")
        for u in self.load_conditions:
            if not (0.0 < u <= 1.0):
                raise ValueError(f"utilization fraction {u} must be in (0, 1]")
        if self.timesteps_per_topology <= 0:
            raise ValueError("timesteps_per_topology must be positive")
        if self.queue_model not in {"mm1", "mg1"}:
            raise ValueError("queue_model must be 'mm1' or 'mg1'")
        if not (0.0 <= self.flash_crowd_probability <= 1.0):
            raise ValueError("flash_crowd_probability must be in [0, 1]")
        if self.max_paths_per_snapshot <= 0:
            raise ValueError("max_paths_per_snapshot must be positive")