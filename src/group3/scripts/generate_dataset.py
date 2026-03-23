"""
CLI entry point for the dataset generator.

Generates both output layers:

  Layer 1 - Static topology files
      topology_XXX.graphml / .json / _nodes.csv / _edges.csv
      topology_XXX_labels.json   (4 fixed load conditions, M/M/1)
      topology_index.json

  Layer 2 - Temporal snapshots (assignment: realistic traffic + full labeling)
      snapshots/topology_XXX_tTT_snapshot.json
      snapshots/topology_XXX_edge_snapshots.csv
      snapshots/topology_XXX_path_snapshots.csv
      snapshots/topology_XXX_global_snapshots.csv
      snapshots/all_snapshots.csv

Examples
--------
Full run (500 topologies * 12 timesteps, both M/M/1 models):
    python scripts/generate_dataset.py

Smoke test (5 topologies * 4 timesteps):
    python scripts/generate_dataset.py \\
        --num-topologies 5 \\
        --min-nodes 10 --max-nodes 30 \\
        --timesteps 4 \\
        --no-distribution \\
        --output-dir outputs/smoke_dataset

M/G/1 model (bursty traffic):
    python scripts/generate_dataset.py --queue-model mg1

High flash-crowd probability:
    python scripts/generate_dataset.py --flash-crowd-prob 0.20
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from group3.labeling.config import DatasetConfig
from group3.labeling.dataset import generate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-domain GNN routing dataset (Group 3 spec)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Topology ----
    parser.add_argument(
        "--num-topologies", type=int, default=500,
        help="Total number of topologies to generate",
    )
    parser.add_argument(
        "--min-nodes", type=int, default=10,
        help="Minimum nodes per topology",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=100,
        help="Maximum nodes per topology",
    )
    parser.add_argument(
        "--min-domains", type=int, default=2,
        help="Minimum AS domains per topology (spec: 2)",
    )
    parser.add_argument(
        "--max-domains", type=int, default=5,
        help="Maximum AS domains per topology (spec: 5)",
    )
    parser.add_argument(
        "--no-distribution", action="store_true",
        help="Disable 150/100/100/75/75 type distribution; pick types uniformly",
    )

    # ---- Temporal simulation (Layer 2) ----
    parser.add_argument(
        "--timesteps", type=int, default=12,
        help="Number of timesteps per topology (temporal snapshots)",
    )
    parser.add_argument(
        "--queue-model", choices=["mm1", "mg1"], default="mm1",
        help="Queueing model for temporal simulation: M/M/1 or M/G/1",
    )
    parser.add_argument(
        "--flash-crowd-prob", type=float, default=0.08,
        help="Probability of a flash-crowd event at each timestep",
    )
    parser.add_argument(
        "--max-paths", type=int, default=250,
        help="Max (src, dst) paths sampled per snapshot for path-level labels",
    )

    # ---- Misc ----
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw"),
        help="Root output directory",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = DatasetConfig(
        random_seed=args.seed,
        num_topologies=args.num_topologies,
        use_distribution=not args.no_distribution,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        min_domains=args.min_domains,
        max_domains=args.max_domains,
        timesteps_per_topology=args.timesteps,
        queue_model=args.queue_model,
        flash_crowd_probability=args.flash_crowd_prob,
        max_paths_per_snapshot=args.max_paths,
        output_dir=args.output_dir,
    )
    generate_dataset(cfg)


if __name__ == "__main__":
    main()