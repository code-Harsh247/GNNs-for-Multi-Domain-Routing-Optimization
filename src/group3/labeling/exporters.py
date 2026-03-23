"""
All file-writing utilities for the dataset pipeline.

Static topology files:
    data/raw/topologies/graphml/topology_001.graphml
    data/raw/topologies/json/topology_001.json
    data/raw/topologies/csv/topology_001_nodes.csv
    data/raw/topologies/csv/topology_001_edges.csv
    data/raw/labels/topology_001_labels.json
    data/raw/topology_index.json

Temporal snapshot files:
    data/raw/snapshots/topology_001_t00_snapshot.json
    data/raw/snapshots/topology_001_t01_snapshot.json   ...
    data/raw/snapshots/topology_001_snapshots.csv       (flat CSV for all T steps)
    data/raw/snapshots/all_snapshots.csv                (master flat CSV)

Naming: topology_XXX  (3-digit, 1-indexed, zero-padded)
        timestep suffix _tTT  (2-digit, 0-indexed)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

def ensure_dirs(root: Path) -> Dict[str, Path]:
    """Create required subdirectories and return a name→Path map."""
    dirs: Dict[str, Path] = {
        "graphml":   root / "topologies" / "graphml",
        "json":      root / "topologies" / "json",
        "csv":       root / "topologies" / "csv",
        "labels":    root / "labels",
        "snapshots": root / "snapshots",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def topology_filename(topology_id: int) -> str:
    """Return the base filename stem, e.g. 'topology_042'."""
    return f"topology_{topology_id:03d}"


def snapshot_filename(topology_id: int, timestep: int) -> str:
    """Return a snapshot stem, e.g. 'topology_042_t07'."""
    return f"topology_{topology_id:03d}_t{timestep:02d}"


# ---------------------------------------------------------------------------
# GraphML export
# ---------------------------------------------------------------------------

def write_graphml(graph: nx.Graph, path: Path) -> None:
    """
    Export topology graph to GraphML with full <key> declarations per spec.
    Written manually to guarantee exact attribute names and types required by Group 11.
    """
    import xml.etree.ElementTree as ET

    NS = "http://graphml.graphdrawing.org/graphml"
    ET.register_namespace("", NS)
    root_el = ET.Element("graphml", {"xmlns": NS})

    # ---- <key> declarations ----
    key_defs = [
        # (id, for, attr.name, attr.type)
        # node
        ("node_id",             "node",  "node_id",             "int"),
        ("router_type",         "node",  "router_type",         "string"),
        ("capacity_mbps",       "node",  "capacity_mbps",       "double"),
        ("x_coord",             "node",  "x_coord",             "double"),
        ("y_coord",             "node",  "y_coord",             "double"),
        ("domain_id",           "node",  "domain_id",           "int"),
        ("historical_traffic",  "node",  "historical_traffic",  "double"),
        # edge
        ("edge_id",             "edge",  "edge_id",             "int"),
        ("bandwidth_mbps",      "edge",  "bandwidth_mbps",      "double"),
        ("propagation_delay_ms","edge",  "propagation_delay_ms","double"),
        ("is_inter_domain",     "edge",  "is_inter_domain",     "boolean"),
        ("link_reliability",    "edge",  "link_reliability",    "double"),
        ("cost",                "edge",  "cost",                "double"),
        # graph
        ("topology_id",         "graph", "topology_id",         "int"),
        ("num_nodes",           "graph", "num_nodes",           "int"),
        ("num_edges",           "graph", "num_edges",           "int"),
        ("num_domains",         "graph", "num_domains",         "int"),
        ("topology_type",       "graph", "topology_type",       "string"),
        ("generation_seed",     "graph", "generation_seed",     "int"),
        ("domain_policies",     "graph", "domain_policies",     "string"),
    ]
    for kid, for_, name, atype in key_defs:
        ET.SubElement(root_el, "key", {
            "id":        kid,
            "for":       for_,
            "attr.name": name,
            "attr.type": atype,
        })

    topo_id  = graph.graph.get("topology_id", 0)
    graph_el = ET.SubElement(root_el, "graph", {
        "id":          topology_filename(topo_id),
        "edgedefault": "undirected",
    })

    # Graph-level data
    for key_id, val in [
        ("topology_id",    str(graph.graph.get("topology_id", 0))),
        ("num_nodes",      str(graph.graph.get("num_nodes", graph.number_of_nodes()))),
        ("num_edges",      str(graph.graph.get("num_edges", graph.number_of_edges()))),
        ("num_domains",    str(graph.graph.get("num_domains", 1))),
        ("topology_type",  str(graph.graph.get("topology_type", "random"))),
        ("generation_seed",str(graph.graph.get("generation_seed", 0))),
        ("domain_policies",str(graph.graph.get("domain_policies", "{}"))),
    ]:
        _data_el(graph_el, key_id, val)

    # Nodes
    for n in sorted(graph.nodes):
        a = graph.nodes[n]
        ne = ET.SubElement(graph_el, "node", {"id": str(n)})
        _data_el(ne, "node_id",            str(int(a.get("node_id", n))))
        _data_el(ne, "router_type",        str(a.get("router_type", "edge")))
        _data_el(ne, "capacity_mbps",      f"{float(a.get('capacity_mbps', 100.0)):.6f}")
        _data_el(ne, "x_coord",            f"{float(a.get('x_coord', 0.0)):.6f}")
        _data_el(ne, "y_coord",            f"{float(a.get('y_coord', 0.0)):.6f}")
        _data_el(ne, "domain_id",          str(int(a.get("domain_id", 0))))
        _data_el(ne, "historical_traffic", f"{float(a.get('historical_traffic', 0.0)):.6f}")

    # Edges
    for eidx, (u, v) in enumerate(graph.edges):
        a  = graph.edges[u, v]
        ee = ET.SubElement(graph_el, "edge", {
            "id":     f"e{int(a.get('edge_id', eidx))}",
            "source": str(u),
            "target": str(v),
        })
        _data_el(ee, "edge_id",             str(int(a.get("edge_id", eidx))))
        _data_el(ee, "bandwidth_mbps",      f"{float(a.get('bandwidth_mbps', 100.0)):.6f}")
        _data_el(ee, "propagation_delay_ms",f"{float(a.get('propagation_delay_ms', 1.0)):.6f}")
        _data_el(ee, "is_inter_domain",     "true" if a.get("is_inter_domain", False) else "false")
        _data_el(ee, "link_reliability",    f"{float(a.get('link_reliability', 0.99)):.6f}")
        _data_el(ee, "cost",                f"{float(a.get('cost', 1.0)):.6f}")

    _write_xml(root_el, path)


def _data_el(parent: Any, key: str, text: str) -> Any:
    import xml.etree.ElementTree as ET
    el = ET.SubElement(parent, "data", {"key": key})
    el.text = text
    return el


def _write_xml(root_el: Any, path: Path) -> None:
    import xml.etree.ElementTree as ET
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root_el)
    ET.indent(tree, space="  ")
    with path.open("wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)


# ---------------------------------------------------------------------------
# JSON topology export
# ---------------------------------------------------------------------------

def write_topology_json(graph: nx.Graph, path: Path) -> None:
    """Write topology_XXX.json per Group 11 spec structure."""
    topo_id = int(graph.graph.get("topology_id", 0))
    dp_raw  = graph.graph.get("domain_policies", "{}")
    try:
        domain_policies = json.loads(dp_raw)
    except (json.JSONDecodeError, TypeError):
        domain_policies = {}

    nodes = []
    for n in sorted(graph.nodes):
        a = graph.nodes[n]
        nodes.append({
            "node_id":            int(a.get("node_id", n)),
            "router_type":        str(a.get("router_type", "edge")),
            "capacity_mbps":      float(a.get("capacity_mbps", 100.0)),
            "x_coord":            float(a.get("x_coord", 0.0)),
            "y_coord":            float(a.get("y_coord", 0.0)),
            "domain_id":          int(a.get("domain_id", 0)),
            "historical_traffic": float(a.get("historical_traffic", 0.0)),
        })

    edges = []
    for eidx, (u, v) in enumerate(graph.edges):
        a = graph.edges[u, v]
        edges.append({
            "edge_id":             int(a.get("edge_id", eidx)),
            "source":              int(u),
            "target":              int(v),
            "bandwidth_mbps":      float(a.get("bandwidth_mbps", 100.0)),
            "propagation_delay_ms":float(a.get("propagation_delay_ms", 1.0)),
            "is_inter_domain":     bool(a.get("is_inter_domain", False)),
            "link_reliability":    float(a.get("link_reliability", 0.99)),
            "cost":                float(a.get("cost", 1.0)),
        })

    _write_json(path, {
        "topology_id":     topo_id,
        "num_nodes":       int(graph.graph.get("num_nodes", graph.number_of_nodes())),
        "num_edges":       int(graph.graph.get("num_edges", graph.number_of_edges())),
        "num_domains":     int(graph.graph.get("num_domains", 1)),
        "topology_type":   str(graph.graph.get("topology_type", "random")),
        "generation_seed": int(graph.graph.get("generation_seed", 0)),
        "domain_policies": domain_policies,
        "nodes":           nodes,
        "edges":           edges,
    })


# ---------------------------------------------------------------------------
# CSV topology exports
# ---------------------------------------------------------------------------

NODE_CSV_FIELDS = [
    "node_id", "router_type", "capacity_mbps",
    "x_coord", "y_coord", "domain_id", "historical_traffic",
]
EDGE_CSV_FIELDS = [
    "edge_id", "source", "target", "bandwidth_mbps",
    "propagation_delay_ms", "is_inter_domain", "link_reliability", "cost",
]


def write_nodes_csv(graph: nx.Graph, path: Path) -> None:
    rows = []
    for n in sorted(graph.nodes):
        a = graph.nodes[n]
        rows.append({
            "node_id":            int(a.get("node_id", n)),
            "router_type":        str(a.get("router_type", "edge")),
            "capacity_mbps":      float(a.get("capacity_mbps", 100.0)),
            "x_coord":            float(a.get("x_coord", 0.0)),
            "y_coord":            float(a.get("y_coord", 0.0)),
            "domain_id":          int(a.get("domain_id", 0)),
            "historical_traffic": float(a.get("historical_traffic", 0.0)),
        })
    _write_csv(path, NODE_CSV_FIELDS, rows)


def write_edges_csv(graph: nx.Graph, path: Path) -> None:
    rows = []
    for eidx, (u, v) in enumerate(graph.edges):
        a = graph.edges[u, v]
        rows.append({
            "edge_id":             int(a.get("edge_id", eidx)),
            "source":              int(u),
            "target":              int(v),
            "bandwidth_mbps":      float(a.get("bandwidth_mbps", 100.0)),
            "propagation_delay_ms":float(a.get("propagation_delay_ms", 1.0)),
            "is_inter_domain":     bool(a.get("is_inter_domain", False)),
            "link_reliability":    float(a.get("link_reliability", 0.99)),
            "cost":                float(a.get("cost", 1.0)),
        })
    _write_csv(path, EDGE_CSV_FIELDS, rows)


# ---------------------------------------------------------------------------
# Static label file
# ---------------------------------------------------------------------------

def write_label_json(
    topology_id: int,
    load_conditions_data: Dict[str, Dict[str, object]],
    path: Path,
) -> None:
    """Write topology_XXX_labels.json per Group 11 spec."""
    _write_json(path, {
        "topology_id":     int(topology_id),
        "load_conditions": load_conditions_data,
    })


# ---------------------------------------------------------------------------
# Temporal snapshot exports
# ---------------------------------------------------------------------------

# Per-timestep edge metrics columns
SNAPSHOT_EDGE_CSV_FIELDS = [
    "snapshot_id", "topology_id", "timestep",
    "edge_id", "u", "v",
    "bandwidth_mbps", "load_mbps", "utilization",
    "queue_length_pkts", "latency_ms", "propagation_delay_ms",
    "link_reliability", "cost_metric", "is_bottleneck", "is_inter_domain",
]

# Per-timestep path metrics columns
SNAPSHOT_PATH_CSV_FIELDS = [
    "snapshot_id", "topology_id", "timestep",
    "source", "target", "demand_mbps",
    "path_hops", "e2e_latency_ms", "path_bottleneck_count",
]

# Per-timestep global summary columns
SNAPSHOT_GLOBAL_CSV_FIELDS = [
    "snapshot_id", "topology_id", "timestep",
    "time_of_day", "diurnal_factor", "flash_event", "flash_multiplier",
    "traffic_profile", "network_load_mbps",
    "avg_utilization", "max_utilization",
    "queue_model",
]


def write_snapshot_json(
    topology_id: int,
    timestep: int,
    global_meta: Dict[str, object],
    sim_results: Dict[str, object],
    queue_model: str,
    path: Path,
) -> None:
    """
    Write topology_XXX_tTT_snapshot.json - the rich per-timestep ground truth.

    Contains:
        snapshot_id, topology_id, timestep,
        global_features (time_of_day, diurnal_factor, flash_event, ...),
        queue_model,
        edge_labels  (per-edge utilization / queue / latency / bottleneck),
        path_labels  (per-path e2e latency / bottleneck count),
        summary      (avg_utilization, max_utilization)
    """
    snap_id = snapshot_filename(topology_id, timestep)
    _write_json(path, {
        "snapshot_id":     snap_id,
        "topology_id":     int(topology_id),
        "timestep":        int(timestep),
        "queue_model":     queue_model,
        "global_features": global_meta,
        "edge_labels":     sim_results["edge_rows"],
        "path_labels":     sim_results["path_rows"],
        "summary": {
            "avg_utilization": float(sim_results["avg_utilization"]),
            "max_utilization": float(sim_results["max_utilization"]),
        },
    })


def append_snapshot_csv(
    topology_id: int,
    timestep: int,
    global_meta: Dict[str, object],
    sim_results: Dict[str, object],
    queue_model: str,
    edge_csv: Path,
    path_csv: Path,
    global_csv: Path,
) -> None:
    """
    Append one timestep's worth of rows to the three flat CSVs:
        topology_XXX_edge_snapshots.csv
        topology_XXX_path_snapshots.csv
        topology_XXX_global_snapshots.csv
    """
    snap_id = snapshot_filename(topology_id, timestep)

    # Edge rows
    edge_rows = []
    for r in sim_results["edge_rows"]:
        row = dict(r)
        row["snapshot_id"] = snap_id
        row["topology_id"] = topology_id
        row["timestep"]    = timestep
        edge_rows.append(row)
    _append_csv(edge_csv, SNAPSHOT_EDGE_CSV_FIELDS, edge_rows)

    # Path rows
    path_rows = []
    for r in sim_results["path_rows"]:
        row = dict(r)
        row["snapshot_id"] = snap_id
        row["topology_id"] = topology_id
        row["timestep"]    = timestep
        path_rows.append(row)
    _append_csv(path_csv, SNAPSHOT_PATH_CSV_FIELDS, path_rows)

    # Global row
    global_row = {
        "snapshot_id":     snap_id,
        "topology_id":     topology_id,
        "timestep":        timestep,
        "time_of_day":     global_meta.get("time_of_day", 0.0),
        "diurnal_factor":  global_meta.get("diurnal_factor", 1.0),
        "flash_event":     global_meta.get("flash_event", 0),
        "flash_multiplier":global_meta.get("flash_multiplier", 1.0),
        "traffic_profile": global_meta.get("traffic_profile", ""),
        "network_load_mbps":global_meta.get("network_load_mbps", 0.0),
        "avg_utilization": sim_results["avg_utilization"],
        "max_utilization": sim_results["max_utilization"],
        "queue_model":     queue_model,
    }
    _append_csv(global_csv, SNAPSHOT_GLOBAL_CSV_FIELDS, [global_row])


def append_master_snapshot_row(
    row: Dict[str, object],
    master_csv: Path,
) -> None:
    """Append a single global summary row to the master all_snapshots.csv."""
    _append_csv(master_csv, SNAPSHOT_GLOBAL_CSV_FIELDS, [row])


# ---------------------------------------------------------------------------
# Topology index
# ---------------------------------------------------------------------------

def write_topology_index(root: Path, index_entries: List[Dict[str, Any]]) -> None:
    _write_json(root / "topology_index.json", {
        "total_topologies": len(index_entries),
        "topologies":       index_entries,
    })


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """Append rows to a CSV, writing header only if the file does not yet exist."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)