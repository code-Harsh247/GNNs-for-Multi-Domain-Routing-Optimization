"""
validate_group11.py  —  Phase 1 Validation Script
===================================================
Checks every requirement from the Group 11 implementation plan's
"Validation Scripts" and "Validation Checklist Before Declaring Phase 1
Complete" sections.

Run from the repo root using the venv:
    .venv/Scripts/python tests/validate_group11.py

Exit code 0  = all checks passed.
Exit code 1  = one or more checks failed (errors printed to stdout).
"""

import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ROOT       = os.path.join("data", "raw")
PROC_ROOT      = os.path.join("data", "processed")
FEATURES_DIR   = os.path.join(PROC_ROOT, "features")
TRAFFIC_DIR    = os.path.join(PROC_ROOT, "traffic")
LABELS_DIR     = os.path.join(PROC_ROOT, "labels")
DATASET_DIR    = os.path.join(PROC_ROOT, "dataset")
INDEX_PATH     = os.path.join(RAW_ROOT, "topology_index.json")
JSON_DIR       = os.path.join(RAW_ROOT, "topologies", "json")
RAW_LABELS_DIR = os.path.join(RAW_ROOT, "labels")

DATASET_PT     = os.path.join(DATASET_DIR, "gnn_dataset.pt")
DATASET_INDEX  = os.path.join(DATASET_DIR, "dataset_index.json")

APP_TYPES       = ["video_streaming", "web_browsing", "file_transfer", "voip"]
LOAD_CONDITIONS = ["low", "medium", "high", "flash"]

NODE_FEAT_DIM   = 12
EDGE_FEAT_DIM   = 5
GLOBAL_FEAT_DIM = 11
TARGET_DIM      = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
errors   = []
warnings = []


def fail(msg: str):
    errors.append(f"  FAIL  {msg}")


def ok(msg: str):
    print(f"  PASS  {msg}")


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Check 1 — topology_index.json exists
# ---------------------------------------------------------------------------
def check_index_exists() -> list:
    section("Check 1 — topology_index.json exists")
    if not os.path.isfile(INDEX_PATH):
        fail(f"topology_index.json not found at {INDEX_PATH}")
        return []
    ok("topology_index.json found")
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = json.load(f)
    ok(f"topology_index.json parsed — {index['total_topologies']} topologies listed")
    return index["topologies"]


# ---------------------------------------------------------------------------
# Check 2 — Feature files: existence, shape, value range, no NaN/Inf
# ---------------------------------------------------------------------------
def check_features(topologies: list):
    section("Check 2 — Feature files (existence, shape, range, no NaN/Inf)")
    for entry in topologies:
        base      = entry["filename_base"]
        n_nodes   = entry["num_nodes"]

        # Load topology JSON to get actual edge count
        json_path = os.path.join(JSON_DIR, f"{base}.json")
        with open(json_path, encoding="utf-8") as f:
            topo = json.load(f)
        n_edges = len(topo["edges"])

        # --- node features ---
        nf_path = os.path.join(FEATURES_DIR, f"{base}_node_features.npy")
        if not os.path.isfile(nf_path):
            fail(f"{base}: node_features.npy missing")
        else:
            nf = np.load(nf_path)
            if nf.shape != (n_nodes, NODE_FEAT_DIM):
                fail(f"{base}: node_features shape {nf.shape} != ({n_nodes},{NODE_FEAT_DIM})")
            elif np.any(np.isnan(nf)) or np.any(np.isinf(nf)):
                fail(f"{base}: node_features contains NaN or Inf")
            elif nf.min() < -1e-6 or nf.max() > 1.0 + 1e-6:
                fail(f"{base}: node_features out of [0,1] range  min={nf.min():.4f} max={nf.max():.4f}")
            else:
                ok(f"{base}: node_features {nf.shape} OK")

        # --- edge features ---
        ef_path = os.path.join(FEATURES_DIR, f"{base}_edge_features.npy")
        if not os.path.isfile(ef_path):
            fail(f"{base}: edge_features.npy missing")
        else:
            ef = np.load(ef_path)
            if ef.shape != (n_edges, EDGE_FEAT_DIM):
                fail(f"{base}: edge_features shape {ef.shape} != ({n_edges},{EDGE_FEAT_DIM})")
            elif np.any(np.isnan(ef)) or np.any(np.isinf(ef)):
                fail(f"{base}: edge_features contains NaN or Inf")
            elif ef.min() < -1e-6 or ef.max() > 1.0 + 1e-6:
                fail(f"{base}: edge_features out of [0,1] range  min={ef.min():.4f} max={ef.max():.4f}")
            else:
                ok(f"{base}: edge_features {ef.shape} OK")

        # --- global features ---
        gf_path = os.path.join(FEATURES_DIR, f"{base}_global_features.npy")
        if not os.path.isfile(gf_path):
            fail(f"{base}: global_features.npy missing")
        else:
            gf = np.load(gf_path)
            if gf.shape != (GLOBAL_FEAT_DIM,):
                fail(f"{base}: global_features shape {gf.shape} != ({GLOBAL_FEAT_DIM},)")
            elif np.any(np.isnan(gf)) or np.any(np.isinf(gf)):
                fail(f"{base}: global_features contains NaN or Inf")
            else:
                ok(f"{base}: global_features {gf.shape} OK")


# ---------------------------------------------------------------------------
# Check 3 — Traffic files: existence, all app types, all load conditions
# ---------------------------------------------------------------------------
def check_traffic(topologies: list):
    section("Check 3 — Traffic files (existence, all apps, all conditions)")
    for entry in topologies:
        base     = entry["filename_base"]
        tf_path  = os.path.join(TRAFFIC_DIR, f"{base}_traffic.json")
        if not os.path.isfile(tf_path):
            fail(f"{base}: traffic file missing")
            continue
        with open(tf_path, encoding="utf-8") as f:
            tf = json.load(f)

        missing_apps = [a for a in APP_TYPES if a not in tf.get("traffic_matrices", {})]
        if missing_apps:
            fail(f"{base}: traffic missing app types: {missing_apps}")
            continue

        bad_conds = []
        for app in APP_TYPES:
            for cond in LOAD_CONDITIONS:
                if cond not in tf["traffic_matrices"][app]:
                    bad_conds.append(f"{app}.{cond}")
        if bad_conds:
            fail(f"{base}: traffic missing conditions: {bad_conds}")
        else:
            ok(f"{base}: traffic file OK (4 apps × 4 conditions)")


# ---------------------------------------------------------------------------
# Check 4 — Enriched label files: existence, edge_ids valid
# ---------------------------------------------------------------------------
def check_enriched_labels(topologies: list):
    section("Check 4 — Enriched label files (existence, valid edge IDs)")
    for entry in topologies:
        base    = entry["filename_base"]
        el_path = os.path.join(LABELS_DIR, f"{base}_enriched_labels.json")
        if not os.path.isfile(el_path):
            fail(f"{base}: enriched_labels file missing")
            continue

        # Load valid edge IDs from topology JSON
        with open(os.path.join(JSON_DIR, f"{base}.json"), encoding="utf-8") as f:
            topo = json.load(f)
        valid_edge_ids = {e["edge_id"] for e in topo["edges"]}

        with open(el_path, encoding="utf-8") as f:
            el = json.load(f)

        bad_refs = []
        for entry_pl in el.get("path_latencies", []):
            for eid in entry_pl.get("edge_ids_on_path", []):
                if eid not in valid_edge_ids:
                    bad_refs.append(eid)

        if bad_refs:
            fail(f"{base}: enriched labels reference invalid edge IDs: {set(bad_refs)}")
        elif not el.get("path_latencies"):
            fail(f"{base}: enriched labels has no path_latencies entries")
        else:
            ok(f"{base}: enriched labels OK ({len(el['path_latencies'])} paths)")


# ---------------------------------------------------------------------------
# Check 5 — gnn_dataset.pt: loads, correct graph count, required fields
# ---------------------------------------------------------------------------
def check_dataset(topologies: list):
    section("Check 5 — gnn_dataset.pt (loads, graph count, field shapes)")
    if not os.path.isfile(DATASET_PT):
        fail(f"gnn_dataset.pt not found at {DATASET_PT}")
        return

    try:
        import torch
        data_list = torch.load(DATASET_PT, weights_only=False)
    except Exception as e:
        fail(f"gnn_dataset.pt failed to load: {e}")
        return

    expected = len(topologies)
    if len(data_list) != expected:
        fail(f"gnn_dataset.pt contains {len(data_list)} graphs, expected {expected}")
    else:
        ok(f"gnn_dataset.pt loaded — {len(data_list)} graphs")

    required_fields = ["x", "edge_index", "edge_attr", "y_edge", "u"]
    for i, data in enumerate(data_list):
        missing = [f for f in required_fields if not hasattr(data, f)]
        if missing:
            fail(f"Graph {i}: missing fields {missing}")
            continue

        shape_errors = []
        n = data.num_nodes
        m = data.edge_index.shape[1]

        if tuple(data.x.shape)         != (n, NODE_FEAT_DIM):
            shape_errors.append(f"x={tuple(data.x.shape)} expected ({n},{NODE_FEAT_DIM})")
        if tuple(data.edge_attr.shape) != (m, EDGE_FEAT_DIM):
            shape_errors.append(f"edge_attr={tuple(data.edge_attr.shape)} expected ({m},{EDGE_FEAT_DIM})")
        if tuple(data.y_edge.shape)    != (m, TARGET_DIM):
            shape_errors.append(f"y_edge={tuple(data.y_edge.shape)} expected ({m},{TARGET_DIM})")
        if tuple(data.u.shape)         != (1, GLOBAL_FEAT_DIM):
            shape_errors.append(f"u={tuple(data.u.shape)} expected (1,{GLOBAL_FEAT_DIM})")

        # NaN / Inf check
        for field in ["x", "edge_attr", "y_edge", "u"]:
            t = getattr(data, field)
            if torch.isnan(t).any() or torch.isinf(t).any():
                shape_errors.append(f"{field} contains NaN or Inf")

        if shape_errors:
            fail(f"Graph {i} (topology_id={data.topology_id}): {'; '.join(shape_errors)}")
        else:
            ok(f"Graph {i} (topology_id={data.topology_id}): all fields and shapes OK")


# ---------------------------------------------------------------------------
# Check 6 — dataset_index.json matches actual dataset
# ---------------------------------------------------------------------------
def check_dataset_index(topologies: list):
    section("Check 6 — dataset_index.json consistency")
    if not os.path.isfile(DATASET_INDEX):
        fail(f"dataset_index.json not found at {DATASET_INDEX}")
        return
    with open(DATASET_INDEX, encoding="utf-8") as f:
        idx = json.load(f)

    expected_total = len(topologies)
    if idx.get("total_graphs") != expected_total:
        fail(f"dataset_index total_graphs={idx.get('total_graphs')} expected {expected_total}")
    else:
        ok(f"total_graphs={idx['total_graphs']} matches topology count")

    dim_checks = [
        ("node_feature_dim",   NODE_FEAT_DIM),
        ("edge_feature_dim",   EDGE_FEAT_DIM),
        ("global_feature_dim", GLOBAL_FEAT_DIM),
        ("target_dim",         TARGET_DIM),
    ]
    for key, expected in dim_checks:
        if idx.get(key) != expected:
            fail(f"dataset_index {key}={idx.get(key)} expected {expected}")
        else:
            ok(f"dataset_index {key}={idx[key]} correct")

    indexed_ids = {g["topology_id"] for g in idx.get("graphs", [])}
    actual_ids  = {t["topology_id"] for t in topologies}
    if indexed_ids != actual_ids:
        fail(f"dataset_index graph IDs {indexed_ids} != topology IDs {actual_ids}")
    else:
        ok(f"dataset_index graph IDs match topology_index")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\nGroup 11 — Phase 1 Validation")
    print("=" * 60)

    topologies = check_index_exists()
    if not topologies:
        print("\nCannot continue without topology_index.json.")
        sys.exit(1)

    check_features(topologies)
    check_traffic(topologies)
    check_enriched_labels(topologies)
    check_dataset(topologies)
    check_dataset_index(topologies)

    print(f"\n{'='*60}")
    if errors:
        print(f"  {len(errors)} ERROR(S) FOUND:")
        for e in errors:
            print(e)
        print("=" * 60)
        sys.exit(1)
    else:
        print("  ALL CHECKS PASSED — Phase 1 is complete.")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
