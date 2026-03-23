"""
End-to-end pipeline orchestrating both output layers per topology.

LAYER 1 - Static topology files
    topology_XXX.graphml  /  .json  /  _nodes.csv  /  _edges.csv
    topology_XXX_labels.json   (4 fixed load conditions via M/M/1)
    topology_index.json

LAYER 2 - Temporal snapshots (realistic traffic + full labeling)
    snapshots/topology_XXX_tTT_snapshot.json      (one per timestep)
    snapshots/topology_XXX_edge_snapshots.csv      (all T edge rows for this topology)
    snapshots/topology_XXX_path_snapshots.csv      (all T path rows)
    snapshots/topology_XXX_global_snapshots.csv    (all T global summary rows)
    snapshots/all_snapshots.csv                    (master flat CSV across all topologies)

Output root: data/raw/   (configurable via DatasetConfig.output_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total=0, desc="", unit=""): pass
        def update(self, n: int = 1) -> None: pass
        def close(self) -> None: pass

from .config import DatasetConfig, TOPOLOGY_TYPE_DISTRIBUTION
from .exporters import (
    ensure_dirs,
    topology_filename,
    snapshot_filename,
    write_graphml,
    write_topology_json,
    write_nodes_csv,
    write_edges_csv,
    write_label_json,
    write_snapshot_json,
    append_snapshot_csv,
    append_master_snapshot_row,
    write_topology_index,
    SNAPSHOT_GLOBAL_CSV_FIELDS,
)
from .latency import compute_all_load_conditions, simulate_snapshot
from group3.topology_gen.topology import generate_topology
from group3.domain_modeling.traffic import generate_traffic_matrix


# ---------------------------------------------------------------------------
# Helper: build the ordered list of topology types to generate
# ---------------------------------------------------------------------------

def _build_type_list(cfg: DatasetConfig) -> List[str]:
    if not cfg.use_distribution:
        return []
    total_spec = sum(c for _, c in TOPOLOGY_TYPE_DISTRIBUTION)
    target     = cfg.num_topologies
    types: List[str] = []
    if target == total_spec:
        for ttype, count in TOPOLOGY_TYPE_DISTRIBUTION:
            types.extend([ttype] * count)
    else:
        for ttype, count in TOPOLOGY_TYPE_DISTRIBUTION:
            scaled = max(1, round(count * target / total_spec))
            types.extend([ttype] * scaled)
        while len(types) > target:
            types.pop()
        while len(types) < target:
            largest = max(TOPOLOGY_TYPE_DISTRIBUTION, key=lambda x: x[1])[0]
            types.append(largest)
    return types


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_dataset(cfg: DatasetConfig) -> Path:
    cfg.validate()
    root = Path(cfg.output_dir)
    dirs = ensure_dirs(root)

    master_rng  = np.random.default_rng(cfg.random_seed)
    type_list   = _build_type_list(cfg)
    all_types   = [t for t, _ in TOPOLOGY_TYPE_DISTRIBUTION]

    index_entries: List[Dict[str, Any]] = []

    # Master snapshot CSV path (across all topologies)
    master_csv = dirs["snapshots"] / "all_snapshots.csv"

    total_snapshots = cfg.num_topologies * cfg.timesteps_per_topology
    progress = tqdm(
        total=cfg.num_topologies + total_snapshots,
        desc="Generating dataset",
        unit="step",
    )

    for i in range(cfg.num_topologies):
        topology_id     = i + 1     # 1-indexed per spec
        generation_seed = int(master_rng.integers(0, 2**31))
        topo_rng        = np.random.default_rng(generation_seed)

        # Topology type
        ttype = (type_list[i] if cfg.use_distribution and type_list
                 else str(master_rng.choice(all_types)))

        # ----------------------------------------------------------------
        # Generate static topology graph
        # ----------------------------------------------------------------
        graph = generate_topology(
            cfg=cfg,
            rng=topo_rng,
            topology_id=topology_id,
            topology_type=ttype,
            generation_seed=generation_seed,
        )
        fname_base = topology_filename(topology_id)

        # ----------------------------------------------------------------
        # LAYER 1 - Static file exports
        # ----------------------------------------------------------------
        write_graphml(graph,       dirs["graphml"] / f"{fname_base}.graphml")
        write_topology_json(graph, dirs["json"]    / f"{fname_base}.json")
        write_nodes_csv(graph,     dirs["csv"]     / f"{fname_base}_nodes.csv")
        write_edges_csv(graph,     dirs["csv"]     / f"{fname_base}_edges.csv")

        # Static ground-truth labels (4 fixed load conditions, M/M/1)
        load_data = compute_all_load_conditions(
            graph=graph,
            load_condition_names=cfg.load_condition_names,
            load_conditions=cfg.load_conditions,
            avg_packet_size_mb=cfg.avg_packet_size_mb,
        )
        write_label_json(
            topology_id=topology_id,
            load_conditions_data=load_data,
            path=dirs["labels"] / f"{fname_base}_labels.json",
        )

        # Per-topology snapshot CSV paths
        edge_csv   = dirs["snapshots"] / f"{fname_base}_edge_snapshots.csv"
        path_csv   = dirs["snapshots"] / f"{fname_base}_path_snapshots.csv"
        global_csv = dirs["snapshots"] / f"{fname_base}_global_snapshots.csv"

        # Accumulate index entry
        index_entries.append({
            "topology_id":     int(topology_id),
            "filename_base":   fname_base,
            "num_nodes":       int(graph.graph["num_nodes"]),
            "num_edges":       int(graph.graph["num_edges"]),
            "topology_type":   ttype,
            "num_domains":     int(graph.graph["num_domains"]),
            "generation_seed": int(generation_seed),
            "num_timesteps":   int(cfg.timesteps_per_topology),
        })
        progress.update(1)

        # ----------------------------------------------------------------
        # LAYER 2 - Temporal snapshot simulation
        # ----------------------------------------------------------------
        for t in range(cfg.timesteps_per_topology):
            # Traffic matrix with diurnal + flash-crowd + app profile variation
            traffic_matrix, global_meta = generate_traffic_matrix(
                graph=graph,
                timestep=t,
                total_steps=cfg.timesteps_per_topology,
                rng=topo_rng,
                flash_crowd_probability=cfg.flash_crowd_probability,
            )

            # Simulate routing and compute per-link + per-path metrics
            sim_results = simulate_snapshot(
                graph=graph,
                traffic_matrix=traffic_matrix,
                queue_model=cfg.queue_model,
                max_paths=cfg.max_paths_per_snapshot,
            )

            # Write per-timestep JSON snapshot (full ground truth)
            write_snapshot_json(
                topology_id=topology_id,
                timestep=t,
                global_meta=global_meta,
                sim_results=sim_results,
                queue_model=cfg.queue_model,
                path=dirs["snapshots"] / f"{snapshot_filename(topology_id, t)}_snapshot.json",
            )

            # Append to per-topology flat CSVs
            append_snapshot_csv(
                topology_id=topology_id,
                timestep=t,
                global_meta=global_meta,
                sim_results=sim_results,
                queue_model=cfg.queue_model,
                edge_csv=edge_csv,
                path_csv=path_csv,
                global_csv=global_csv,
            )

            # Append one row to the master all_snapshots.csv
            snap_id = snapshot_filename(topology_id, t)
            master_row: Dict[str, object] = {
                "snapshot_id":      snap_id,
                "topology_id":      topology_id,
                "timestep":         t,
                "time_of_day":      global_meta.get("time_of_day", 0.0),
                "diurnal_factor":   global_meta.get("diurnal_factor", 1.0),
                "flash_event":      global_meta.get("flash_event", 0),
                "flash_multiplier": global_meta.get("flash_multiplier", 1.0),
                "traffic_profile":  global_meta.get("traffic_profile", ""),
                "network_load_mbps":global_meta.get("network_load_mbps", 0.0),
                "avg_utilization":  sim_results["avg_utilization"],
                "max_utilization":  sim_results["max_utilization"],
                "queue_model":      cfg.queue_model,
            }
            append_master_snapshot_row(master_row, master_csv)
            progress.update(1)

    progress.close()

    # ----------------------------------------------------------------
    # Write topology index
    # ----------------------------------------------------------------
    write_topology_index(root, index_entries)

    total_snaps = cfg.num_topologies * cfg.timesteps_per_topology
    print(f"\nDataset written to : {root}")
    print(f"  Topologies       : {len(index_entries)}")
    print(f"  Timesteps/topo   : {cfg.timesteps_per_topology}")
    print(f"  Total snapshots  : {total_snaps}")
    print(f"  Queue model      : {cfg.queue_model}")
    print(f"  Index            : {root / 'topology_index.json'}")
    print(f"  Master CSV       : {master_csv}")

    return root