# Phase 1 Report — GNN Multi-Domain Routing Optimization

> **Status:** Complete — All validation checks passed  
> **Branch:** `dev` (merges `group-3` + `group-11`)  
> **Date completed:** March 23, 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Team Responsibilities](#2-team-responsibilities)
3. [Group 3 — Dataset Generation](#3-group-3--dataset-generation)
4. [Group 11 — Feature Engineering & Dataset Assembly](#4-group-11--feature-engineering--dataset-assembly)
5. [Integration & Git Workflow](#5-integration--git-workflow)
6. [Bugs Found and Fixed](#6-bugs-found-and-fixed)
7. [Final Dataset Statistics](#7-final-dataset-statistics)
8. [Feature Specifications](#8-feature-specifications)
9. [File Structure Reference](#9-file-structure-reference)
10. [How to Reproduce](#10-how-to-reproduce)
11. [Validation Results](#11-validation-results)
12. [Known Limitations & Notes for Phase 2](#12-known-limitations--notes-for-phase-2)

---

## 1. Project Overview

The goal of this project is to train a **Graph Neural Network (GNN)** that predicts per-link latency in multi-domain computer networks under varying traffic loads. Networks are modelled as graphs where routers are nodes and physical links are edges.

Phase 1 covers everything up to and including dataset assembly — producing the `gnn_dataset.pt` file that Phase 2 model training will consume.

The dataset spans **500 synthetic network topologies** across 5 topology types, with per-edge latency ground truth computed under 4 distinct load conditions using an M/M/1 queueing model.

---

## 2. Team Responsibilities

| Group | Role | Deliverables |
|---|---|---|
| **Group 3** | Topology generation & ground-truth labelling | Raw topology files (JSON, GraphML, CSV), static latency labels, temporal snapshots |
| **Group 11** | Feature engineering, traffic simulation & dataset assembly | Processed feature arrays (.npy), traffic matrices, enriched labels, final PyG dataset |

Both groups worked on separate branches (`group-3` and `group-11`) and were merged into a shared `dev` branch upon Phase 1 completion.

---

## 3. Group 3 — Dataset Generation

### 3.1 What was built

Group 3 implemented a complete **two-layer dataset generation pipeline** producing 500 synthetic network topologies with full ground-truth labels.

**Entry point:** `src/group3/scripts/generate_dataset.py`

### 3.2 Topology Generation (`topology_gen/topology.py`)

Five topology types are generated with the following distribution:

| Type | Count | Node Range | Description |
|---|---|---|---|
| `random` | 150 | 10–100 | Erdős–Rényi random graphs |
| `scale_free` | 100 | 20–100 | Barabási–Albert preferential attachment (hub-spoke structure) |
| `mesh` | 100 | 10–100 | High-connectivity grid-like graphs |
| `ring` | 75 | 10–100 | Circular topology — exactly 2 paths between any pair |
| `hybrid` | 75 | 20–100 | Scale-free core with ring periphery |

Each topology has:
- **2–5 AS-like domains** with domain-aware edge capacity tiers
- Node attributes: `router_type` (core/edge/gateway), `capacity_mbps`, `x_coord`, `y_coord`, `domain_id`, `historical_traffic`
- Edge attributes: `bandwidth_mbps`, `propagation_delay_ms`, `is_inter_domain`, `link_reliability`, `cost`, `edge_id`
- Capacity tiers: intra-domain 100–1000 Mbps, inter-domain 10–100 Mbps

### 3.3 Traffic Modeling (`domain_modeling/traffic.py`)

Temporal traffic matrices are generated with:
- **24-hour diurnal sinusoidal load curve** (multiplier range: 0.25–1.15×)
- **Flash-crowd events** (8% probability per timestep, 1.8–3.4× multiplier)
- **3 application profiles:** web, video, bulk — each with distinct flow size distributions
- **Inter-domain traffic boost:** 1.3× for cross-AS flows
- **12 timesteps per topology** for temporal snapshots

### 3.4 Latency Labelling (`labeling/latency.py`)

Two-layer labelling system:

**Layer 1 — Static labels (4 load conditions):**
- Utilization fractions: `low=0.20`, `medium=0.50`, `high=0.80`, `flash=0.95`
- M/M/1 queueing formula: $W_{queue} = \dfrac{\rho}{\mu(1 - \rho)} \times 1000$ ms
- Service rate: $\mu = \dfrac{\text{bandwidth\_mbps}}{0.001}$
- Utilization capped at 0.95 to prevent division by zero
- Saturation penalty applied at ≥ 98% utilization: $10 + 120(\rho - 0.98)$ ms

**Layer 2 — Temporal labels (12 timesteps per topology):**
- Both M/M/1 and M/G/1 (Pollaczek-Khinchine, SCV=1.5) models
- Congestion-aware shortest path routing per timestep
- Bottleneck detection: top-10% by utilization, minimum threshold 85%

### 3.5 Export Pipeline (`labeling/exporters.py`)

Each topology is exported in four formats:

| Format | Location | Contents |
|---|---|---|
| **JSON** | `data/raw/topologies/json/` | Full node/edge attributes including `domain_policies` as a parsed dict |
| **GraphML** | `data/raw/topologies/graphml/` | Full XML with `<key>` declarations per spec |
| **CSV (nodes)** | `data/raw/topologies/csv/` | One row per node with all attributes |
| **CSV (edges)** | `data/raw/topologies/csv/` | One row per edge with all attributes |
| **Static labels** | `data/raw/labels/` | `edge_id → {low, medium, high, flash}` latency mapping |
| **Temporal snapshots** | `data/raw/snapshots/` | Per-timestep JSON + aggregated CSV files |

**Master index:** `data/raw/topology_index.json` — lists all 500 topologies with metadata.

### 3.6 Group 3 Output File Count

| File Type | Count |
|---|---|
| JSON topology files | 500 |
| GraphML files | 500 |
| Node CSV files | 500 |
| Edge CSV files | 500 |
| Static label files | 500 |
| Temporal snapshot JSON files | 6,000 (12 per topology) |
| Snapshot CSV files | 1,500 (3 per topology) |
| `all_snapshots.csv` (master) | 1 (6,000 data rows) |
| `topology_index.json` | 1 |
| **Total** | **~10,002** |

---

## 4. Group 11 — Feature Engineering & Dataset Assembly

Group 11 implements a **four-phase pipeline** that reads Group 3's raw outputs and produces a ML-ready PyTorch Geometric dataset.

### Phase A — Feature Engineering

**Script:** `src/group11/feature_engineering/build_features.py`  
**Output directory:** `data/processed/features/`  
**Files produced:** 1,500 `.npy` files (3 per topology × 500 topologies)

Reads each topology's JSON file and produces three NumPy arrays:

#### Node Features — shape `(num_nodes, 12)`

| Index | Feature | Source | Range | Method |
|---|---|---|---|---|
| 0 | `router_type_core` | Group 3 | {0, 1} | One-hot |
| 1 | `router_type_edge` | Group 3 | {0, 1} | One-hot |
| 2 | `router_type_gateway` | Group 3 | {0, 1} | One-hot |
| 3 | `capacity_mbps_norm` | Group 3 | [0, 0.8] | `capacity_mbps / 2000.0` |
| 4 | `x_coord` | Group 3 | [0, 1] | Already normalized |
| 5 | `y_coord` | Group 3 | [0, 1] | Already normalized |
| 6–10 | `domain_id_[0–4]` | Group 3 | {0, 1} | One-hot, padded to 5 slots |
| 11 | `degree_norm` | Computed | [0, 1] | `degree / max_degree_in_topology` |

#### Edge Features — shape `(num_edges, 5)`

| Index | Feature | Source | Range | Method |
|---|---|---|---|---|
| 0 | `bandwidth_mbps_norm` | Group 3 | [0, 1] | `bandwidth_mbps / 1000.0` |
| 1 | `propagation_delay_ms_norm` | Group 3 | [0, 1] | `propagation_delay_ms / 100.0` |
| 2 | `is_inter_domain` | Group 3 | {0, 1} | Binary flag |
| 3 | `link_reliability` | Group 3 | [0.95, 1.0] | Already in [0, 1] |
| 4 | `cost_norm` | Group 3 | [0, 1] | `cost / max_cost_in_topology` |

> **Note:** Latency values are intentionally excluded from edge features — they are the prediction *targets* in `y_edge`. Including them in inputs would cause target leakage during training.

#### Global Features — shape `(11,)`

| Index | Feature | Source | Range | Method |
|---|---|---|---|---|
| 0 | `num_nodes_norm` | Group 3 | [0.1, 1.0] | `num_nodes / 100.0` |
| 1 | `num_edges_norm` | Group 3 | [0.02, 1.0] | `num_edges / 500.0` |
| 2 | `num_domains_norm` | Group 3 | [0.2, 1.0] | `num_domains / 5.0` |
| 3 | `avg_bandwidth_norm` | Computed | [0, 1] | `mean(bandwidths) / 1000.0` |
| 4 | `avg_propagation_delay_norm` | Computed | [0, 1] | `mean(prop_delays) / 100.0` |
| 5 | `inter_domain_ratio` | Computed | [0, 1] | `inter_domain_edges / num_edges` |
| 6–10 | `topology_type_[random/scale_free/mesh/ring/hybrid]` | Group 3 | {0, 1} | One-hot |

---

### Phase B — Traffic Simulation

**Script:** `src/group11/traffic_simulation/simulate_traffic.py`  
**Output directory:** `data/processed/traffic/`  
**Files produced:** 500 `.json` files

Generates traffic demand matrices for four application types at four load levels.

| Application | Traffic Distribution | Characteristics |
|---|---|---|
| `video_streaming` | Exponential with high mean | High, bursty flows |
| `web_browsing` | Exponential with medium mean | Short, frequent flows |
| `file_transfer` | Uniform, large values | Sustained high-volume flows |
| `voip` | Uniform, small values | Constant low-bitrate flows |

**Load conditions:**

| Condition | Target Utilization |
|---|---|
| `low` | 20% |
| `medium` | 50% |
| `high` | 80% |
| `flash` | 95% |

Each traffic file contains **16 matrices** (4 apps × 4 conditions) plus diurnal hourly snapshots (peak at 18:00, off-peak at 03:00, business hours at 10:00). Flash-crowd events are injected on topologies where `topology_id % 5 == 0` (topologies 5, 10, 15, …, 500) with a 5× surge to a randomly selected target node.

---

### Phase C — Ground Truth Enrichment

**Script:** `src/group11/ground_truth/enrich_labels.py`  
**Output directory:** `data/processed/labels/`  
**Files produced:** 500 `.json` files

For every source–destination pair in each topology:

1. Builds an undirected NetworkX graph from the topology's edge list
2. Computes the **hop-count shortest path** using `nx.shortest_path`
3. Walks consecutive node pairs along the path to look up `edge_id`s
4. **Sums per-link latencies** for each of the 4 load conditions → `end_to_end_latency_ms`
5. Identifies the **bottleneck edge** (highest latency link) per condition → `bottleneck_edge_id`

With 500 topologies averaging 48 nodes, this enriches approximately **1.1 million source-destination pairs** in total.

---

### Phase D — Dataset Assembly

**Script:** `src/group11/dataset_assembly/assemble_dataset.py`  
**Output directory:** `data/processed/dataset/`  
**Files produced:** `gnn_dataset.pt`, `dataset_index.json`

Combines all Phase A, B, and C outputs into **PyTorch Geometric `Data` objects** — one per topology.

#### PyG Data Object Structure

| Field | Shape | Type | Contents |
|---|---|---|---|
| `x` | `(num_nodes, 12)` | `float32` | Node feature matrix |
| `edge_index` | `(2, num_edges)` | `int64` | COO-format adjacency (source, target pairs) |
| `edge_attr` | `(num_edges, 5)` | `float32` | Edge feature matrix |
| `u` | `(1, 11)` | `float32` | Global (graph-level) feature vector |
| `y_edge` | `(num_edges, 4)` | `float32` | Target latency (ms) under 4 load conditions |
| `topology_id` | scalar | `int` | Topology identifier |
| `num_nodes` | scalar | `int` | Number of nodes |

The `y_edge` targets are **raw latency in milliseconds** — not normalized — so the Phase 2 model must account for scale variation across load conditions.

The full dataset is serialized with `torch.save` and loaded with `torch.load`.

---

## 5. Integration & Git Workflow

### Branch structure

```
main          ← stable releases only
└── dev       ← integration branch (current working branch)
    ├── group-3   ← Group 3's dataset generation (complete)
    └── group-11  ← Group 11's pipeline (complete)
```

### Merge history

1. `group-3` and `group-11` were both merged into `dev`
2. **Conflict:** Both branches had added files to `data/raw/` — Group 11 had 5-topology mock data, Group 3 had the real 500-topology dataset. Resolved by keeping Group 3's authoritative versions for all `data/raw/` files.
3. Group 3's full dataset was explicitly restored via `git checkout group-3 -- data/raw/` after the initial conflict resolution incorrectly kept Group 11's mock data.
4. All processed outputs (`data/processed/`) were regenerated on the real 500-topology dataset.

### Ownership convention going forward

| Directory | Owner | Notes |
|---|---|---|
| `data/raw/` | Group 3 | Never overwrite — source of truth |
| `src/group3/` | Group 3 | Dataset generation scripts |
| `data/processed/` | Group 11 | Regenerated by running the pipeline |
| `src/group11/` | Group 11 | Feature engineering & assembly scripts |

---

## 6. Bugs Found and Fixed

### Bug 1 — Node capacity normalization out of range (Group 11 pipeline)

**File:** `src/group11/feature_engineering/build_features.py`  
**Problem:** `capacity_mbps` was divided by `1000.0` for normalization. Group 3 generates node capacities up to ~1600 Mbps (intra-domain routers can exceed 1000 Mbps), causing feature values up to 1.6 — outside the required `[0, 1]` range. The validation script flagged this as a FAIL on every topology.  
**Fix:** Changed divisor from `1000.0` to `2000.0`, which comfortably bounds all values at ≤ 0.8 and correctly represents relative capacity differences.

```python
# Before (incorrect):
feats[i, 3] = node["capacity_mbps"] / 1000.0

# After (fixed):
feats[i, 3] = node["capacity_mbps"] / 2000.0
```

### Bug 2 — Unicode character crash on Windows (Group 11 pipeline)

**File:** `src/group11/dataset_assembly/assemble_dataset.py`  
**Problem:** Three `print()` statements used the Unicode arrow character `→` (`\u2192`). On Windows, the terminal uses cp1252 encoding which cannot encode this character, causing a `UnicodeEncodeError` that crashed `assemble_dataset.py` after completing all 500 graphs.  
**Fix:** Replaced all `→` with the ASCII equivalent `->`.

### Bug 3 — Filename typo in Group 3's validation script

**File:** `src/group3/scripts/validata_dataset.py`  
**Problem:** The validation script was saved with a typo (`validata` instead of `validate`).  
**Fix:** File renamed to `validate_dataset.py`.

---

## 7. Final Dataset Statistics

| Metric | Value |
|---|---|
| Total topologies | 500 |
| Total graphs in `gnn_dataset.pt` | 500 |
| `gnn_dataset.pt` file size | 4.4 MB |
| Node feature dimension | 12 |
| Edge feature dimension | 5 |
| Global feature dimension | 11 |
| Target dimension (`y_edge`) | 4 |
| Load conditions | `low`, `medium`, `high`, `flash` |

### Topology type distribution

| Type | Count | % |
|---|---|---|
| `random` | 150 | 30% |
| `scale_free` | 100 | 20% |
| `mesh` | 100 | 20% |
| `ring` | 75 | 15% |
| `hybrid` | 75 | 15% |

### Size distribution (across all 500 topologies)

| Metric | Min | Max | Average |
|---|---|---|---|
| Nodes per topology | 10 | 100 | 48.4 |
| Edges per topology | 13 | 579 | 109.0 |
| Domains per topology | 2 | 5 | — |

### Processed file counts

| File type | Count |
|---|---|
| Node feature files (`.npy`) | 500 |
| Edge feature files (`.npy`) | 500 |
| Global feature files (`.npy`) | 500 |
| Traffic matrix files (`.json`) | 500 |
| Enriched label files (`.json`) | 500 |
| **Total processed files** | **2,500** |

---

## 8. Feature Specifications

### 8.1 Prediction targets (`y_edge`)

The GNN is trained to predict **per-edge latency in milliseconds** under 4 load conditions.
Targets are computed from Group 3's M/M/1 queueing model:

$$W_{queue} = \frac{\rho}{\mu(1 - \rho)} \times 1000 \text{ ms}$$

where $\mu = \dfrac{\text{bandwidth\_mbps}}{0.001}$ and $\rho$ is the utilization fraction.

Saturation penalty at ≥ 98% utilization: $10 + 120(\rho - 0.98)$ ms.

### 8.2 Why latency is excluded from edge features

Latency labels (`y_edge`) are the prediction target. If they were also included in `edge_attr` (inputs), the model could trivially copy the input to achieve zero loss — learning nothing about the underlying physics. This is called **target leakage** and is avoided by design.

### 8.3 Normalization conventions

All input features are bounded to `[0, 1]` to ensure stable gradient flow during training:

| Feature | Normalization strategy |
|---|---|
| `capacity_mbps` | Divide by 2000.0 (observed max ~1600 Mbps) |
| `bandwidth_mbps` | Divide by 1000.0 (max intra-domain bandwidth) |
| `propagation_delay_ms` | Divide by 100.0 (assumed max physical delay) |
| `cost` | Divide by `max_cost_in_topology` (per-topology) |
| `num_nodes` | Divide by 100.0 (max nodes per spec) |
| `num_edges` | Divide by 500.0 (conservative upper bound) |
| `num_domains` | Divide by 5.0 (max per spec) |
| Coordinates, reliability | Already in [0, 1] — passed through |
| One-hot encodings | {0.0, 1.0} by definition |
| `y_edge` targets | **Not normalized** — raw ms values |

---

## 9. File Structure Reference

```
data/
├── raw/                                  ← Group 3 output (do not modify)
│   ├── topology_index.json               # Master index of all 500 topologies
│   ├── topologies/
│   │   ├── json/topology_XXX.json        # Full node/edge attributes
│   │   ├── graphml/topology_XXX.graphml  # GraphML with <key> declarations
│   │   └── csv/topology_XXX_{nodes,edges}.csv
│   ├── labels/
│   │   └── topology_XXX_labels.json      # M/M/1 latency per edge per condition
│   └── snapshots/                        # Temporal 12-step snapshots
│
└── processed/                            ← Group 11 output (auto-generated)
    ├── features/
    │   ├── topology_XXX_node_features.npy    # shape (num_nodes, 12)
    │   ├── topology_XXX_edge_features.npy    # shape (num_edges, 5)
    │   └── topology_XXX_global_features.npy  # shape (11,)
    ├── traffic/
    │   └── topology_XXX_traffic.json     # 16 matrices (4 apps × 4 conditions)
    ├── labels/
    │   └── topology_XXX_enriched_labels.json # All S-D pair path latencies
    └── dataset/
        ├── gnn_dataset.pt                # Final PyG dataset (500 graphs, 4.4 MB)
        └── dataset_index.json            # Graph count, dimensions, metadata

src/
├── group3/                               # Dataset generation (Group 3)
│   ├── topology_gen/topology.py
│   ├── domain_modeling/traffic.py
│   ├── labeling/{config,latency,exporters,dataset}.py
│   └── scripts/{generate_dataset,validate_dataset}.py
│
└── group11/                              # Feature engineering & assembly (Group 11)
    ├── feature_engineering/build_features.py
    ├── traffic_simulation/simulate_traffic.py
    ├── ground_truth/enrich_labels.py
    ├── dataset_assembly/assemble_dataset.py
    └── generate_mock_data.py             # No longer needed (Group 3 delivered)

tests/
└── validate_group11.py                   # Phase 1 validation suite (exit 0 = all pass)
```

---

## 10. How to Reproduce

### Prerequisites

```bash
# Activate virtual environment (from repo root)
.venv\Scripts\Activate.ps1        # Windows PowerShell
source .venv/bin/activate          # Linux/macOS
```

### Step 1 — Run Group 3's dataset generator (if regenerating from scratch)

```bash
python src/group3/scripts/generate_dataset.py --num-topologies 500
```

> Skip this step if `data/raw/` is already populated (it is after cloning `dev`).

### Step 2 — Feature Engineering (Phase A)

```bash
python src/group11/feature_engineering/build_features.py
```

Output: 1,500 `.npy` files in `data/processed/features/`

### Step 3 — Traffic Simulation (Phase B)

```bash
python src/group11/traffic_simulation/simulate_traffic.py
```

Output: 500 `.json` files in `data/processed/traffic/`

### Step 4 — Ground Truth Enrichment (Phase C)

```bash
python src/group11/ground_truth/enrich_labels.py
```

Output: 500 `.json` files in `data/processed/labels/`

### Step 5 — Dataset Assembly (Phase D)

```bash
python src/group11/dataset_assembly/assemble_dataset.py
```

Output: `gnn_dataset.pt` and `dataset_index.json` in `data/processed/dataset/`

### Step 6 — Validate

```bash
python tests/validate_group11.py              # Group 11 pipeline validation
python src/group3/scripts/validate_dataset.py # Group 3 dataset validation
```

Both scripts exit with code `0` on success.

> **Note:** Steps 2–4 are independent of each other and can be run in any order, but all must complete before Step 5. All scripts must be run from the **repo root directory**.

---

## 11. Validation Results

### Group 11 Validation (`tests/validate_group11.py`)

| Check | Scope | Result |
|---|---|---|
| Check 1 — `topology_index.json` exists and is parseable | 1 file | PASS |
| Check 2 — Feature file existence, shape, range `[0,1]`, no NaN/Inf | 500 × 3 files | PASS (all 500) |
| Check 3 — Traffic files: all 4 app types, all 4 load conditions | 500 files | PASS (all 500) |
| Check 4 — Enriched label files: valid edge IDs | 500 files | PASS (all 500) |
| Check 5 — `gnn_dataset.pt` loads, all required PyG fields present | 1 file, 500 graphs | PASS |
| Check 6 — `dataset_index.json` matches dataset (graph count, dimensions) | 1 file | PASS |

**Final result: ALL CHECKS PASSED — Phase 1 is complete.**

---

## 12. Known Limitations & Notes for Phase 2

### For the Phase 2 model development team

1. **`y_edge` is not normalized.** Raw latency values in milliseconds. At low load (`ρ = 0.20`) typical values are 1–25 ms; at flash load (`ρ = 0.95`) they can exceed 200 ms. Consider using a log-scale loss or per-condition normalization.

2. **`edge_index` is undirected.** Each undirected link appears once. If your GNN message-passing assumes directed edges, use PyG's `to_directed()` utility or `torch_geometric.utils.to_undirected` appropriately.

3. **Variable graph sizes.** Topologies range from 10 to 100 nodes and 13 to 579 edges. Use PyG's `DataLoader` with `batch_size` tuning to manage memory, especially for the large topologies.

4. **Domain one-hot is zero-padded to 5 slots.** Topologies with fewer than 5 domains will have trailing zeros in features `[6–10]`. This is correct behavior — do not interpret zero as a domain membership.

5. **Temporal snapshots are available but not in `gnn_dataset.pt`.** Group 3 generated 12 temporal snapshots per topology in `data/raw/snapshots/`. If Phase 2 wants to train a temporal model (e.g., a graph RNN), these can be incorporated in a future pipeline extension.

6. **`generate_mock_data.py` is no longer needed.** This script was written to bootstrap Group 11's pipeline before Group 3's data was available. It produces 5 mock topologies that do not reflect the real dataset. Do not run it — it will overwrite `data/raw/` with mock data.

7. **Capacity normalization is set to `/2000.0`.** This was updated from the original `/1000.0` after discovering Group 3 generates capacities up to ~1600 Mbps. The effective range of the normalized feature is `[0.07, 0.80]` — values will never reach 1.0.
