"""
Pre-handoff validation script for dataset.

Checks every requirement from both the Group 3 Implementation Plan and the
broader assignment scope:

  Phase A  - topology generation (file existence, naming, counts)
  Phase B  - multi-domain modelling (domain attributes, policies, capacities)
  Phase C  - static ground truth labels (4 load conditions, edge ID consistency)
  Phase D  - temporal snapshots (diurnal variation, flash crowds, app profiles,
              per-link utilization / queue / latency, per-path labels)
  Format   - GraphML keys, JSON structure, CSV column order
  Index    - topology_index.json completeness

Usage
-----
    python scripts/validate_dataset.py                       # validate data/raw
    python scripts/validate_dataset.py --data-dir outputs/smoke_dataset
    python scripts/validate_dataset.py --data-dir data/raw --quiet
    python scripts/validate_dataset.py --data-dir data/raw --skip-temporal
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Spec constants
# ---------------------------------------------------------------------------
REQUIRED_NODE_ATTRS  = {
    "node_id", "router_type", "capacity_mbps",
    "x_coord", "y_coord", "domain_id", "historical_traffic",
}
REQUIRED_EDGE_ATTRS  = {
    "edge_id", "bandwidth_mbps", "propagation_delay_ms",
    "is_inter_domain", "link_reliability", "cost",
}
REQUIRED_GRAPH_ATTRS = {
    "topology_id", "num_nodes", "num_edges", "num_domains",
    "topology_type", "generation_seed", "domain_policies",
}
VALID_ROUTER_TYPES   = {"core", "edge", "gateway"}
VALID_TOPOLOGY_TYPES = {"random", "scale_free", "mesh", "ring", "hybrid"}
LOAD_CONDITIONS      = {"low": 0.20, "medium": 0.50, "high": 0.80, "flash": 0.95}

NODE_CSV_COLS = [
    "node_id", "router_type", "capacity_mbps",
    "x_coord", "y_coord", "domain_id", "historical_traffic",
]
EDGE_CSV_COLS = [
    "edge_id", "source", "target", "bandwidth_mbps",
    "propagation_delay_ms", "is_inter_domain", "link_reliability", "cost",
]

REQUIRED_SNAPSHOT_KEYS    = {
    "snapshot_id", "topology_id", "timestep", "queue_model",
    "global_features", "edge_labels", "path_labels", "summary",
}
REQUIRED_GLOBAL_FEAT_KEYS = {
    "time_of_day", "diurnal_factor", "flash_event",
    "flash_multiplier", "traffic_profile", "network_load_mbps",
}
REQUIRED_EDGE_LABEL_KEYS  = {
    "edge_id", "utilization", "queue_length_pkts", "latency_ms",
    "load_mbps", "is_bottleneck", "propagation_delay_ms", "is_inter_domain",
}
REQUIRED_PATH_LABEL_KEYS  = {
    "source", "target", "demand_mbps", "path_hops",
    "e2e_latency_ms", "path_bottleneck_count",
}

GRAPHML_NS = "http://graphml.graphdrawing.org/graphml"


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

class Results:
    def __init__(self, quiet: bool = False) -> None:
        self.errors:   List[str] = []
        self.warnings: List[str] = []
        self.passed:   int = 0
        self.quiet     = quiet

    def ok(self, msg: str) -> None:
        self.passed += 1
        if not self.quiet:
            print(f"  ✓ {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  ⚠ {msg}")

    def err(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"  ✗ {msg}")

    def section(self, title: str) -> None:
        print(f"\n{'─'*64}")
        print(f"  {title}")
        print(f"{'─'*64}")

    def summary(self) -> bool:
        print(f"\n{'='*64}")
        print(f"  VALIDATION SUMMARY")
        print(f"{'='*64}")
        print(f"  Passed  : {self.passed}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Errors  : {len(self.errors)}")
        if self.errors:
            print("\n  ERRORS:")
            for e in self.errors:
                print(f"    ✗ {e}")
        if self.warnings:
            print("\n  WARNINGS:")
            for w in self.warnings:
                print(f"    ⚠ {w}")
        ok = len(self.errors) == 0
        status = "READY FOR HANDOFF ✓" if ok else "NOT READY - fix errors above ✗"
        print(f"\n  {status}")
        print(f"{'='*64}\n")
        return ok


# ---------------------------------------------------------------------------
# GraphML helpers
# ---------------------------------------------------------------------------

def _graphml_declared_keys(tree: ET.ElementTree) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for k in tree.findall(f"{{{GRAPHML_NS}}}key"):
        name = k.get("attr.name", "")
        kid  = k.get("id", "")
        if name:
            keys[name] = kid
    return keys


# ---------------------------------------------------------------------------
# Per-topology validation - Phases A, B, C
# ---------------------------------------------------------------------------

def validate_topology(
    topo_id: int,
    base: str,
    dirs: Dict[str, Path],
    r: Results,
) -> Dict:
    """Validate one topology's static files. Returns parsed JSON topology."""
    tag = f"[{base}]"

    # ---- GraphML ----
    gml_path = dirs["graphml"] / f"{base}.graphml"
    if not gml_path.exists():
        r.err(f"{tag} GraphML file missing")
        return {}
    try:
        tree = ET.parse(gml_path)
    except ET.ParseError as exc:
        r.err(f"{tag} GraphML parse error: {exc}")
        return {}

    declared = _graphml_declared_keys(tree)
    all_required = REQUIRED_NODE_ATTRS | REQUIRED_EDGE_ATTRS | REQUIRED_GRAPH_ATTRS
    missing_keys = all_required - set(declared.keys())
    if missing_keys:
        for k in sorted(missing_keys):
            r.err(f"{tag} GraphML missing <key>: {k}")
    else:
        r.ok(f"{tag} GraphML: all {len(all_required)} required keys declared")

    # ---- JSON topology ----
    json_path = dirs["json"] / f"{base}.json"
    if not json_path.exists():
        r.err(f"{tag} JSON topology file missing")
        return {}
    try:
        topo = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        r.err(f"{tag} JSON parse error: {exc}")
        return {}

    for field in ["topology_id", "num_nodes", "num_edges", "num_domains",
                  "topology_type", "generation_seed", "domain_policies",
                  "nodes", "edges"]:
        if field not in topo:
            r.err(f"{tag} JSON missing field: {field}")

    ttype = topo.get("topology_type", "")
    if ttype not in VALID_TOPOLOGY_TYPES:
        r.err(f"{tag} Invalid topology_type: '{ttype}'")
    else:
        r.ok(f"{tag} topology_type='{ttype}'")

    if topo.get("topology_id") != topo_id:
        r.err(f"{tag} JSON topology_id={topo.get('topology_id')} ≠ {topo_id}")
    else:
        r.ok(f"{tag} topology_id matches filename")

    nd = topo.get("num_domains", 0)
    if not (2 <= nd <= 5):
        r.err(f"{tag} num_domains={nd} outside spec range [2,5]")
    else:
        r.ok(f"{tag} num_domains={nd} ∈ [2,5]")

    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])

    # Node attributes
    node_attr_ok = True
    for n in nodes:
        missing = (REQUIRED_NODE_ATTRS - {"historical_traffic"}) - set(n.keys())
        if missing:
            r.err(f"{tag} node {n.get('node_id','?')} missing: {missing}")
            node_attr_ok = False
        if "historical_traffic" not in n:
            r.warn(f"{tag} node {n.get('node_id')} missing historical_traffic")
        rt = n.get("router_type", "")
        if rt not in VALID_ROUTER_TYPES:
            r.err(f"{tag} node {n.get('node_id')} invalid router_type: '{rt}'")
            node_attr_ok = False
        x, y = n.get("x_coord", -1), n.get("y_coord", -1)
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            r.warn(f"{tag} node {n.get('node_id')} coords out of [0,1]: "
                   f"x={x:.3f} y={y:.3f}")
        if n.get("domain_id") is None:
            r.err(f"{tag} node {n.get('node_id')} missing domain_id")
            node_attr_ok = False
    if node_attr_ok:
        r.ok(f"{tag} all {len(nodes)} nodes have required attributes")

    # Edge attributes + capacity ranges
    intra_ok = inter_ok = True
    seen_eids: set = set()
    for e in edges:
        eid = e.get("edge_id")
        if eid in seen_eids:
            r.err(f"{tag} duplicate edge_id: {eid}")
        seen_eids.add(eid)
        missing = REQUIRED_EDGE_ATTRS - set(e.keys())
        if missing:
            r.err(f"{tag} edge {eid} missing attrs: {missing}")
        bw       = e.get("bandwidth_mbps", 0)
        is_inter = e.get("is_inter_domain", False)
        if is_inter:
            if not (10.0 <= bw <= 100.0):
                r.err(f"{tag} inter-domain edge {eid} bw={bw:.1f} ∉ [10,100] Mbps")
                inter_ok = False
        else:
            if not (100.0 <= bw <= 1000.0):
                r.err(f"{tag} intra-domain edge {eid} bw={bw:.1f} ∉ [100,1000] Mbps")
                intra_ok = False
        rel = e.get("link_reliability", -1)
        if not (0.0 <= rel <= 1.0):
            r.err(f"{tag} edge {eid} link_reliability={rel} ∉ [0,1]")
    if intra_ok:
        r.ok(f"{tag} all intra-domain bandwidths ∈ [100,1000] Mbps")
    if inter_ok:
        r.ok(f"{tag} all inter-domain bandwidths ∈ [10,100] Mbps")

    # domain_policies
    dp = topo.get("domain_policies", {})
    if isinstance(dp, dict):
        r.ok(f"{tag} domain_policies: valid dict ({len(dp)} domain entries)")
    else:
        r.err(f"{tag} domain_policies is not a dict")

    # ---- Node CSV ----
    nodes_csv = dirs["csv"] / f"{base}_nodes.csv"
    if not nodes_csv.exists():
        r.err(f"{tag} nodes CSV missing")
    else:
        with nodes_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            if cols != NODE_CSV_COLS:
                r.err(f"{tag} nodes CSV wrong columns: {cols}")
            else:
                r.ok(f"{tag} nodes CSV column order correct")
            csv_rows = list(reader)
            if len(csv_rows) != topo.get("num_nodes", -1):
                r.err(f"{tag} nodes CSV {len(csv_rows)} rows ≠ num_nodes={topo.get('num_nodes')}")
            else:
                r.ok(f"{tag} nodes CSV row count = {len(csv_rows)}")

    # ---- Edge CSV ----
    edges_csv = dirs["csv"] / f"{base}_edges.csv"
    if not edges_csv.exists():
        r.err(f"{tag} edges CSV missing")
    else:
        with edges_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = list(reader.fieldnames or [])
            if cols != EDGE_CSV_COLS:
                r.err(f"{tag} edges CSV wrong columns: {cols}")
            else:
                r.ok(f"{tag} edges CSV column order correct")
            csv_rows = list(reader)
            if len(csv_rows) != topo.get("num_edges", -1):
                r.err(f"{tag} edges CSV {len(csv_rows)} rows ≠ num_edges={topo.get('num_edges')}")
            else:
                r.ok(f"{tag} edges CSV row count = {len(csv_rows)}")

    # ---- Label file ----
    lbl_path = dirs["labels"] / f"{base}_labels.json"
    if not lbl_path.exists():
        r.err(f"{tag} label file missing")
        return topo
    try:
        lbl = json.loads(lbl_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        r.err(f"{tag} label parse error: {exc}")
        return topo

    if lbl.get("topology_id") != topo_id:
        r.err(f"{tag} label topology_id={lbl.get('topology_id')} ≠ {topo_id}")
    else:
        r.ok(f"{tag} label topology_id matches")

    load_conds    = lbl.get("load_conditions", {})
    edge_ids_topo = {str(e["edge_id"]) for e in edges}

    for cond_name, expected_frac in LOAD_CONDITIONS.items():
        if cond_name not in load_conds:
            r.err(f"{tag} label missing condition: '{cond_name}'")
            continue
        cond = load_conds[cond_name]
        frac = cond.get("utilization_fraction", -1)
        if abs(frac - expected_frac) > 1e-9:
            r.err(f"{tag} label/{cond_name} utilization_fraction={frac} ≠ {expected_frac}")
        else:
            r.ok(f"{tag} label/{cond_name}: utilization_fraction={frac}")
        latencies  = cond.get("edge_latencies_ms", {})
        missing_ids = edge_ids_topo - set(latencies.keys())
        if missing_ids:
            r.err(f"{tag} label/{cond_name}: missing edge IDs {sorted(missing_ids)[:5]}")
        else:
            r.ok(f"{tag} label/{cond_name}: all {len(latencies)} edge IDs present")

    return topo


# ---------------------------------------------------------------------------
# Phase D - temporal snapshot validation
# ---------------------------------------------------------------------------

def validate_snapshots(
    topo_id: int,
    base: str,
    topo: Dict,
    snap_dir: Path,
    num_timesteps: int,
    r: Results,
) -> None:
    tag = f"[{base}/snap]"

    if not topo:
        r.warn(f"{tag} skipping (topology JSON unavailable)")
        return

    edge_ids_topo = {e["edge_id"] for e in topo.get("edges", [])}
    diurnal_vals: List[float] = []
    flash_events: List[int]   = []
    profiles:     List[str]   = []

    for t in range(num_timesteps):
        snap_path = snap_dir / f"{base}_t{t:02d}_snapshot.json"
        if not snap_path.exists():
            r.err(f"{tag} t={t:02d}: snapshot JSON missing")
            continue
        try:
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            r.err(f"{tag} t={t:02d}: parse error: {exc}")
            continue

        missing_top = REQUIRED_SNAPSHOT_KEYS - set(snap.keys())
        if missing_top:
            r.err(f"{tag} t={t:02d}: missing keys: {missing_top}")
            continue

        # Global features
        gf = snap.get("global_features", {})
        missing_gf = REQUIRED_GLOBAL_FEAT_KEYS - set(gf.keys())
        if missing_gf:
            r.err(f"{tag} t={t:02d}: global_features missing: {missing_gf}")
        else:
            r.ok(f"{tag} t={t:02d}: global_features OK "
                 f"(diurnal={gf.get('diurnal_factor',0):.3f}, "
                 f"flash={gf.get('flash_event',0)}, "
                 f"profile={gf.get('traffic_profile','')})")

        df = float(gf.get("diurnal_factor", -1))
        diurnal_vals.append(df)
        flash_events.append(int(gf.get("flash_event", -1)))
        profiles.append(str(gf.get("traffic_profile", "")))

        if not (0.20 <= df <= 1.20):
            r.warn(f"{tag} t={t:02d}: diurnal_factor={df:.3f} outside [0.20,1.20]")

        fe  = int(gf.get("flash_event", 0))
        fmx = float(gf.get("flash_multiplier", 1.0))
        if fe == 1 and not (1.8 <= fmx <= 3.4):
            r.warn(f"{tag} t={t:02d}: flash_event=1 but multiplier={fmx:.2f} ∉ [1.8,3.4]")

        tp = gf.get("traffic_profile", "")
        if tp not in {"web", "video", "bulk"}:
            r.err(f"{tag} t={t:02d}: unknown traffic_profile='{tp}'")

        # Edge labels
        edge_labels = snap.get("edge_labels", [])
        if len(edge_labels) != len(edge_ids_topo):
            r.err(f"{tag} t={t:02d}: {len(edge_labels)} edge labels ≠ "
                  f"{len(edge_ids_topo)} topology edges")
        else:
            r.ok(f"{tag} t={t:02d}: {len(edge_labels)} edge labels match topology")

        edge_ok = True
        for el in edge_labels:
            missing_el = REQUIRED_EDGE_LABEL_KEYS - set(el.keys())
            if missing_el:
                r.err(f"{tag} t={t:02d}: edge label missing: {missing_el}")
                edge_ok = False
                break
            if el.get("edge_id") not in edge_ids_topo:
                r.err(f"{tag} t={t:02d}: unknown edge_id={el.get('edge_id')}")
                edge_ok = False
                break
            if float(el.get("latency_ms", -1)) <= 0:
                r.err(f"{tag} t={t:02d}: edge {el.get('edge_id')} latency_ms ≤ 0")
                edge_ok = False
                break
        if edge_ok:
            r.ok(f"{tag} t={t:02d}: all edge labels valid "
                 f"(avg_util={snap.get('summary',{}).get('avg_utilization',0):.3f})")

        # Path labels
        path_labels = snap.get("path_labels", [])
        if path_labels:
            pl0 = path_labels[0]
            missing_pl = REQUIRED_PATH_LABEL_KEYS - set(pl0.keys())
            if missing_pl:
                r.err(f"{tag} t={t:02d}: path label missing: {missing_pl}")
            else:
                r.ok(f"{tag} t={t:02d}: {len(path_labels)} path labels OK")
        else:
            r.warn(f"{tag} t={t:02d}: no path labels recorded")

        # Summary
        summ = snap.get("summary", {})
        if "avg_utilization" not in summ or "max_utilization" not in summ:
            r.err(f"{tag} t={t:02d}: summary incomplete")
        else:
            r.ok(f"{tag} t={t:02d}: summary OK "
                 f"(avg={summ['avg_utilization']:.3f} "
                 f"max={summ['max_utilization']:.3f})")

    # ---- Temporal variation checks ----
    if len(diurnal_vals) > 1:
        n_unique = len(set(round(d, 4) for d in diurnal_vals))
        if n_unique == 1:
            r.err(f"{tag} diurnal_factor constant across all {len(diurnal_vals)} "
                  f"timesteps - temporal variation missing!")
        else:
            r.ok(f"{tag} diurnal_factor varies: {[round(d,3) for d in diurnal_vals]}")

    if any(fe == 1 for fe in flash_events):
        r.ok(f"{tag} flash-crowd observed: timesteps {[i for i,fe in enumerate(flash_events) if fe==1]}")
    else:
        r.warn(f"{tag} no flash-crowd in {len(flash_events)} steps "
               f"(expected ~8 % per step; may be fine for short series)")

    r.ok(f"{tag} traffic profiles seen: {sorted(set(profiles))}")

    # ---- Per-topology flat CSVs ----
    for csv_stem in ["edge_snapshots", "path_snapshots", "global_snapshots"]:
        csv_path = snap_dir / f"{base}_{csv_stem}.csv"
        if not csv_path.exists():
            r.err(f"{tag} snapshot CSV missing: {csv_path.name}")
        else:
            with csv_path.open(encoding="utf-8") as f:
                n_rows = sum(1 for _ in csv.reader(f)) - 1
            r.ok(f"{tag} {csv_path.name}: {n_rows} data rows")


# ---------------------------------------------------------------------------
# Master all_snapshots.csv
# ---------------------------------------------------------------------------

def validate_master_csv(snap_dir: Path, total_snapshots: int, r: Results) -> None:
    master = snap_dir / "all_snapshots.csv"
    if not master.exists():
        r.err(f"all_snapshots.csv missing")
        return
    with master.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == total_snapshots:
        r.ok(f"all_snapshots.csv: {len(rows)} rows ✓")
    else:
        r.warn(f"all_snapshots.csv: {len(rows)} rows ≠ expected {total_snapshots}")

    # Diurnal must vary within each topology's rows
    by_topo: Dict[str, List[float]] = {}
    for row in rows:
        tid = row.get("topology_id", "?")
        by_topo.setdefault(tid, []).append(float(row.get("diurnal_factor", 0)))
    varying = sum(
        1 for vals in by_topo.values()
        if len(set(round(v, 4) for v in vals)) > 1
    )
    r.ok(f"all_snapshots.csv: diurnal varies in {varying}/{len(by_topo)} topologies")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Group 3 dataset (Layer 1 spec + Layer 2 temporal)"
    )
    parser.add_argument("--data-dir",      type=Path, default=Path("data/raw"))
    parser.add_argument("--quiet",         action="store_true")
    parser.add_argument("--skip-temporal", action="store_true",
                        help="Skip Phase D temporal snapshot checks")
    args = parser.parse_args()

    root = args.data_dir
    r    = Results(quiet=args.quiet)

    dirs: Dict[str, Path] = {
        "graphml":   root / "topologies" / "graphml",
        "json":      root / "topologies" / "json",
        "csv":       root / "topologies" / "csv",
        "labels":    root / "labels",
        "snapshots": root / "snapshots",
    }

    # ---- Phase A: directories ----
    r.section("Phase A - Directory structure")
    for name, p in dirs.items():
        if p.is_dir():
            r.ok(f"Directory exists: {p}")
        else:
            r.err(f"Directory missing: {p}")

    # ---- topology_index.json ----
    r.section("topology_index.json")
    idx_path = root / "topology_index.json"
    if not idx_path.exists():
        r.err(f"topology_index.json missing at {idx_path}")
        r.summary(); sys.exit(1)
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        r.err(f"topology_index.json parse error: {exc}")
        r.summary(); sys.exit(1)

    total        = idx.get("total_topologies", 0)
    entries      = idx.get("topologies", [])
    num_timesteps = entries[0].get("num_timesteps", 12) if entries else 12

    if total != len(entries):
        r.err(f"total_topologies={total} ≠ len(topologies)={len(entries)}")
    else:
        r.ok(f"total_topologies={total} matches entry count")
    if total < 500:
        r.warn(f"total_topologies={total} < 500 (spec requires 500+)")
    else:
        r.ok(f"total_topologies={total} ≥ 500")

    required_idx_fields = {
        "topology_id", "filename_base", "num_nodes", "num_edges",
        "topology_type", "num_domains", "generation_seed", "num_timesteps",
    }
    for entry in entries[:5]:
        missing = required_idx_fields - set(entry.keys())
        if missing:
            r.err(f"Index entry topology_{entry.get('topology_id',0):03d} missing: {missing}")
        else:
            r.ok(f"Index entry topology_{entry['topology_id']:03d}: all required fields present")

    # ---- Type distribution ----
    r.section("Phase A - Topology type distribution")
    type_counts: Dict[str, int] = {}
    for entry in entries:
        tt = entry.get("topology_type", "unknown")
        type_counts[tt] = type_counts.get(tt, 0) + 1
    for tt in VALID_TOPOLOGY_TYPES:
        count = type_counts.get(tt, 0)
        if count == 0:
            r.err(f"No topologies of type '{tt}'")
        else:
            r.ok(f"Type '{tt}': {count} topologies")

    # ---- Phases B + C: per-topology static validation ----
    r.section("Phase B + C - Static file validation")
    topology_ids = sorted(e["topology_id"] for e in entries)
    topo_cache: Dict[int, Dict] = {}
    for topo_id in topology_ids:
        base = f"topology_{topo_id:03d}"
        topo_cache[topo_id] = validate_topology(topo_id, base, dirs, r)

    # ---- Edge ID cross-check ----
    r.section("Phase C - Edge ID cross-check (sample 10)")
    for topo_id in topology_ids[:10]:
        base     = f"topology_{topo_id:03d}"
        topo     = topo_cache.get(topo_id, {})
        lbl_path = dirs["labels"] / f"{base}_labels.json"
        if not (topo and lbl_path.exists()):
            continue
        lbl       = json.loads(lbl_path.read_text())
        topo_eids = {str(e["edge_id"]) for e in topo.get("edges", [])}
        for cond in LOAD_CONDITIONS:
            lbl_eids = set(
                lbl.get("load_conditions", {})
                   .get(cond, {})
                   .get("edge_latencies_ms", {})
                   .keys()
            )
            if topo_eids == lbl_eids:
                r.ok(f"[{base}] edge IDs match topology↔label ({cond})")
            else:
                r.err(f"[{base}] edge ID mismatch topology↔label ({cond})")

    # ---- Phase D: temporal snapshots ----
    if not args.skip_temporal:
        r.section("Phase D - Temporal snapshot validation")
        for topo_id in topology_ids:
            base = f"topology_{topo_id:03d}"
            validate_snapshots(
                topo_id=topo_id,
                base=base,
                topo=topo_cache.get(topo_id, {}),
                snap_dir=dirs["snapshots"],
                num_timesteps=num_timesteps,
                r=r,
            )
        r.section("Phase D - Master all_snapshots.csv")
        validate_master_csv(dirs["snapshots"], total * num_timesteps, r)
    else:
        r.section("Phase D - Temporal snapshots (SKIPPED)")
        r.warn("Temporal snapshot validation skipped via --skip-temporal")

    sys.exit(0 if r.summary() else 1)


if __name__ == "__main__":
    main()