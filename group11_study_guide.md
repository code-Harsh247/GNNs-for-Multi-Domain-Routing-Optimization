# Group 11 — Comprehensive Study Guide
## GNN Data Pipeline for Multi-Domain Network Routing

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Pipeline](#2-architecture--pipeline)
3. [Data Layer](#3-data-layer)
4. [Script 1 — `generate_mock_data.py`](#4-script-1--generate_mock_datapy)
5. [Script 2 — `build_features.py`](#5-script-2--build_featurespy)
6. [Script 3 — `simulate_traffic.py`](#6-script-3--simulate_trafficpy)
7. [Script 4 — `enrich_labels.py`](#7-script-4--enrich_labelspy)
8. [Script 5 — `assemble_dataset.py`](#8-script-5--assemble_datasetpy)
9. [Script 6 — `validate_group11.py`](#9-script-6--validate_group11py)
10. [Feature Specifications](#10-feature-specifications)
11. [Design Decisions](#11-key-design-decisions)
12. [End-to-End Data Flow](#12-end-to-end-data-flow)
13. [Dependencies](#13-dependencies)

---

## 1. Project Overview

Group 11 is responsible for the **data engineering phase** of a larger GNN-based network routing project. Their work sits between raw topology generation (done by Group 3) and model training.

The goal is to take raw network graph files and produce a fully-formed, ready-to-train **PyTorch Geometric (PyG) dataset** containing:
- Normalized node, edge, and global feature matrices
- Traffic demand simulations
- Enriched ground-truth labels (end-to-end path latencies)
- A final `.pt` dataset file consumed by the GNN trainer

All 5 scripts form a **strictly sequential pipeline** — each script's outputs are the inputs to the next.

---

## 2. Architecture & Pipeline

The pipeline runs in this exact order:

```
Step 0:  generate_mock_data.py   →  creates all raw data (topologies + labels)
                                        ↓
Step A:  build_features.py       →  extracts and normalizes node/edge/global features
Step B:  simulate_traffic.py     →  generates 16 traffic matrices per topology
Step C:  enrich_labels.py        →  computes all-pairs shortest paths + latencies
                                        ↓
Step D:  assemble_dataset.py     →  merges everything into PyG Data objects
                                        ↓
         validate_group11.py     →  runs 35 automated checks, exits 0 or 1
```

### Directory Layout (Generated Files)

```
data/
├── raw/
│   ├── topology_index.json                  ← master registry of all topologies
│   ├── topologies/
│   │   ├── json/topology_00X.json           ← full node + edge attributes
│   │   ├── graphml/topology_00X.graphml     ← GraphML format (for other tools)
│   │   └── csv/topology_00X_{nodes,edges}.csv
│   └── labels/
│       └── topology_00X_labels.json         ← M/M/1 latency per edge × 4 conditions
└── processed/
    ├── features/
    │   ├── topology_00X_node_features.npy   ← shape (N, 12)
    │   ├── topology_00X_edge_features.npy   ← shape (M, 5)
    │   └── topology_00X_global_features.npy ← shape (11,)
    ├── traffic/
    │   └── topology_00X_traffic.json        ← 16 demand matrices + diurnal + flash
    ├── labels/
    │   └── topology_00X_enriched_labels.json ← all-pairs paths + e2e latency + bottleneck
    └── dataset/
        ├── gnn_dataset.pt                   ← final PyG dataset (5 Data objects)
        └── dataset_index.json               ← metadata + dimension info
```

---

## 3. Data Layer

### `data/raw/topology_index.json`

The master registry. Every script reads this file first — **no script ever hardcodes topology IDs or scans the filesystem**. Contains one entry per topology:

| Field | Type | Description |
|---|---|---|
| `topology_id` | int | Unique integer ID (1–5) |
| `filename_base` | string | e.g. `"topology_001"` |
| `num_nodes` | int | Node count for the graph |
| `num_edges` | int | Edge count for the graph |
| `topology_type` | string | One of: `random`, `scale_free`, `mesh`, `ring`, `hybrid` |
| `num_domains` | int | Number of routing domains |
| `generation_seed` | int | Random seed for reproducibility |

**The 5 mock topologies:**

| ID | Type | Nodes | Edges | Domains |
|---|---|---|---|---|
| 1 | random | 10 | 19 | 2 |
| 2 | scale_free | 20 | 36 | 3 |
| 3 | mesh | 16 | 23 | 2 |
| 4 | ring | 10 | 10 | 2 |
| 5 | hybrid | 30 | 44 | 4 |

### `data/processed/dataset/dataset_index.json`

Metadata index for the assembled dataset. Key fields:

| Field | Value |
|---|---|
| `total_graphs` | 5 |
| `node_feature_dim` | 12 |
| `edge_feature_dim` | 5 |
| `global_feature_dim` | 11 |
| `target_dim` | 4 |
| `load_conditions` | `["low","medium","high","flash"]` |

---

## 4. Script 1 — `generate_mock_data.py`

**File:** [src/group11/generate_mock_data.py](src/group11/generate_mock_data.py)

**Purpose:** Bootstrap the pipeline. This script is the first to run and it creates all raw data from scratch. Since Group 3 (topology generators) works independently, this script **replicates Group 3's output schema exactly** so Group 11's pipeline can be developed and tested without dependency on Group 3's delivery.

**Run:** `python src/group11/generate_mock_data.py`

---

### Constants / Configuration

| Constant | Value | Purpose |
|---|---|---|
| `TOPOLOGIES_CONFIG` | list of 5 dicts | One config per topology (type, nodes, domains, seed) |
| `LOAD_CONDITIONS` | `{low:0.20, medium:0.50, high:0.80, flash:0.95}` | Network utilization fractions |
| `ROUTER_TYPES` | `["core","edge","gateway"]` | Possible router classifications |
| `CAPACITY_OPTIONS` | `[100, 250, 500, 1000]` Mbps | Possible node capacities |
| `AVG_PACKET_SIZE_MB` | `0.001` (1 KB) | Used in M/M/1 queuing formula |

---

### Functions

#### `_generate_nx_graph(topology_type, num_nodes, seed) → nx.Graph`

Creates a connected undirected NetworkX graph matching the requested topology type:

| Topology Type | NetworkX Generator Used | Notes |
|---|---|---|
| `random` | `erdos_renyi_graph(p=0.35)` | Random edges with 35% probability |
| `scale_free` | `barabasi_albert_graph(m=2)` | Power-law degree distribution |
| `mesh` | `grid_2d_graph` trimmed | Lattice structure, trimmed to exact N |
| `ring` | `cycle_graph` | Circular topology |
| `hybrid` | BA-core + ring periphery + 3 cross-edges | Half nodes use BA, other half ring |

After generation, **connectivity is guaranteed** by stitching disconnected components together.

---

#### `_build_nodes(G, num_domains, rng) → list[dict]`

Assigns attributes to every node:

| Attribute | Generation |
|---|---|
| `router_type` | Random choice from `ROUTER_TYPES` |
| `capacity_mbps` | Random choice from `CAPACITY_OPTIONS` |
| `x_coord`, `y_coord` | Uniform random in `[0, 1]` |
| `domain_id` | `node_id % num_domains` (round-robin assignment) |

---

#### `_build_edges(G, nodes, rng) → list[dict]`

Assigns attributes to every edge:

| Attribute | Generation |
|---|---|
| `bandwidth_mbps` | `[10,100]` if inter-domain, `[100,1000]` if intra-domain |
| `propagation_delay_ms` | Uniform `[1, 20]` ms |
| `is_inter_domain` | `True` if source and target are in different domains |
| `link_reliability` | Uniform `[0.95, 1.0]` |
| `cost` | Uniform `[1, 10]` |
| `edge_id` | Sequential integer starting at 0 |

---

#### `_mm1_latency(bandwidth_mbps, propagation_delay_ms, utilization) → float`

Computes total link latency using the **M/M/1 queuing model**:

```
μ  = bandwidth_mbps / AVG_PACKET_SIZE_MB     (service rate in packets/s)
ρ  = min(utilization, 0.95)                  (capped to prevent infinite queue)
W_queue = (ρ / (μ × (1 − ρ))) × 1000        (queuing delay in ms)
total_latency = propagation_delay_ms + W_queue
```

This is the **physics of the simulation** — it models how network congestion increases at higher utilization.

---

#### `_build_labels(topology_id, edges) → dict`

Calls `_mm1_latency` for every **(edge × load condition)** combination — 4 load conditions × M edges = `4M` latency values.

Output structure:
```json
{
  "topology_id": 1,
  "load_conditions": {
    "low":    { "utilization_fraction": 0.20, "edge_latencies_ms": { "0": 1.04, ... } },
    "medium": { "utilization_fraction": 0.50, "edge_latencies_ms": { "0": 1.10, ... } },
    "high":   { "utilization_fraction": 0.80, "edge_latencies_ms": { "0": 1.52, ... } },
    "flash":  { "utilization_fraction": 0.95, "edge_latencies_ms": { "0": 5.90, ... } }
  }
}
```

---

#### `_build_domain_policies(num_domains) → dict`

Creates a routing policy where every domain allows transit through all other domains. Used as metadata.

---

#### Writer Helper Functions

| Function | What it writes |
|---|---|
| `_makedirs(*paths)` | Creates output directories with `exist_ok=True` |
| `_write_json(path, data)` | Indented JSON file |
| `_write_graphml(path, topology)` | Standards-compliant GraphML with `<key>` declarations and `<data>` elements |
| `_write_csv_nodes(path, nodes)` | CSV: `node_id, router_type, capacity_mbps, x_coord, y_coord, domain_id` |
| `_write_csv_edges(path, edges)` | CSV: `edge_id, source, target, bandwidth_mbps, propagation_delay_ms, is_inter_domain, link_reliability, cost` |

---

#### `main()`

Orchestrates the full generation:
1. Iterates over all 5 topology configs
2. Calls `_generate_nx_graph` → `_build_nodes` → `_build_edges` → `_build_labels`
3. Writes all output formats (JSON, GraphML, 2 CSVs, labels JSON)
4. Accumulates index entries and writes `topology_index.json` at the end

**Inputs:** None (pure generation from config and random seeds)

**Outputs:**
- `data/raw/topologies/json/topology_00X.json`
- `data/raw/topologies/graphml/topology_00X.graphml`
- `data/raw/topologies/csv/topology_00X_nodes.csv`
- `data/raw/topologies/csv/topology_00X_edges.csv`
- `data/raw/labels/topology_00X_labels.json`
- `data/raw/topology_index.json`

---

## 5. Script 2 — `build_features.py`

**File:** [src/group11/feature_engineering/build_features.py](src/group11/feature_engineering/build_features.py)

**Purpose:** Phase A of the pipeline. Reads every raw topology JSON and produces three NumPy feature arrays per topology. These arrays are the **input tensors** that the GNN will see — they must be normalized to `[0,1]`.

**Run:** `python src/group11/feature_engineering/build_features.py`

---

### Constants

| Constant | Value | Purpose |
|---|---|---|
| `ROUTER_TYPE_ORDER` | `["core","edge","gateway"]` | Defines one-hot encoding order |
| `TOPOLOGY_TYPE_ORDER` | `["random","scale_free","mesh","ring","hybrid"]` | Defines one-hot encoding order |
| `MAX_DOMAINS` | `5` | Fixed padded size for domain one-hot |
| `NODE_FEAT_DIM` | `12` | Expected node feature width |
| `EDGE_FEAT_DIM` | `5` | Expected edge feature width |
| `GLOBAL_FEAT_DIM` | `11` | Expected global feature width |

---

### Functions

#### `build_node_features(nodes, edges) → np.ndarray of shape (N, 12)`

Builds a feature vector for each node. Computed in two passes:
1. **First pass over edges:** compute per-node `degree` and `bandwidth_sum`
2. **Second pass over nodes:** fill the 12-element feature vector

| Index | Feature | How computed |
|---|---|---|
| 0–2 | `router_type` one-hot | e.g. `core` → `[1,0,0]` |
| 3 | `capacity_norm` | `capacity_mbps / 1000` |
| 4 | `x_coord` | Already in `[0,1]` |
| 5 | `y_coord` | Already in `[0,1]` |
| 6–10 | `domain_id` one-hot (5 slots) | e.g. domain 2 → `[0,0,1,0,0]` |
| 11 | `degree_norm` | `node_degree / max_degree_in_topology` |

All values are verified to be in `[0,1]`. Shape is asserted to be `(N, 12)`.

---

#### `build_edge_features(edges) → np.ndarray of shape (M, 5)`

Builds a feature vector for each edge:

| Index | Feature | How computed |
|---|---|---|
| 0 | `bw_norm` | `bandwidth_mbps / 1000` |
| 1 | `delay_norm` | `propagation_delay_ms / 100` |
| 2 | `is_inter_domain` | Directly as float (`0.0` or `1.0`) |
| 3 | `link_reliability` | Already in `[0.95, 1.0]` ⊆ `[0,1]` |
| 4 | `cost_norm` | `cost / max_cost_in_topology` |

> **Critical:** Latency is **not included** here to prevent target leakage into input features.

---

#### `build_global_features(topology) → np.ndarray of shape (11,)`

Builds a single feature vector for the entire graph:

| Index | Feature | How computed |
|---|---|---|
| 0 | `num_nodes_norm` | `num_nodes / 100` |
| 1 | `num_edges_norm` | `num_edges / 500` |
| 2 | `num_domains_norm` | `num_domains / 5` |
| 3 | `mean_bw_norm` | `mean(bandwidth_mbps) / 1000` |
| 4 | `mean_delay_norm` | `mean(propagation_delay_ms) / 100` |
| 5 | `inter_domain_ratio` | `count(inter_domain edges) / num_edges` |
| 6–10 | `topology_type` one-hot | e.g. `mesh` → `[0,0,1,0,0]` |

---

#### `main()`

Reads `topology_index.json`, calls all three builder functions per topology, saves `.npy` files, prints shape summaries.

**Inputs:** `data/raw/topology_index.json`, `data/raw/topologies/json/topology_00X.json`

**Outputs:**
- `data/processed/features/topology_00X_node_features.npy` — `(N, 12)`
- `data/processed/features/topology_00X_edge_features.npy` — `(M, 5)`
- `data/processed/features/topology_00X_global_features.npy` — `(11,)`

---

## 6. Script 3 — `simulate_traffic.py`

**File:** [src/group11/traffic_simulation/simulate_traffic.py](src/group11/traffic_simulation/simulate_traffic.py)

**Purpose:** Phase B. Generates **16 traffic demand matrices** per topology (4 app types × 4 load conditions) plus diurnal snapshots and optional flash-crowd events. This captures realistic time-varying and application-specific network load.

**Run:** `python src/group11/traffic_simulation/simulate_traffic.py`

---

### Constants

#### `DIURNAL_MULTIPLIERS` (24-element array)
Models time-of-day traffic variation. Values range from `0.2` (3am off-peak) to `1.2` (6pm peak). Indexed by hour of day.

#### `LOAD_MULTIPLIERS`
```python
{"low": 0.20, "medium": 0.50, "high": 0.80, "flash": 0.95}
```

---

### Traffic Generation Functions (all return N×N demand matrix)

| Function | Traffic Behavior | Distribution |
|---|---|---|
| `_gen_video_streaming(n, rng)` | Steady baseline | Uniform `[5, 25]` Mbps |
| `_gen_web_browsing(n, rng)` | Bursty | Exponential(mean=2), capped at 10 Mbps |
| `_gen_file_transfer(n, rng)` | Sustained high-throughput | Uniform `[10, 100]` Mbps |
| `_gen_voip(n, rng)` | Constant, tiny | Uniform `[0.1, 0.5]` Mbps |

Helper functions:
- `_zero_diagonal(matrix)` — sets `T[i][i] = 0` (no self-traffic)
- `_scale_matrix(matrix, factor)` — element-wise multiply by `factor`

---

### Core function: `build_traffic(topology_id, num_nodes, generation_seed) → dict`

This is the main traffic builder. Uses `rng = Random(seed + 1000)` for reproducibility.

**Steps:**
1. Generates one **base matrix** per app type (4 matrices)
2. For each app type, scales by each of 4 load multipliers → **16 matrices total**
3. Computes 3 **diurnal snapshots** from the medium `video_streaming` matrix:
   - `peak_18h` — multiplier 1.2 (6pm peak)
   - `offpeak_03h` — multiplier 0.2 (3am valley)
   - `business_10h` — multiplier 0.9 (10am business hours)
4. Adds flash-crowd event: **only applied to topology 5** (`topology_id % 5 == 0`), picks a random target node and applies 5× spike

---

#### Output JSON structure per topology

```json
{
  "topology_id": 1,
  "traffic_matrices": {
    "video_streaming": {
      "low":    [[0.0, 4.1, ...], ...],
      "medium": [[0.0, 10.2, ...], ...],
      "high":   [[0.0, 16.4, ...], ...],
      "flash":  [[0.0, 19.5, ...], ...]
    },
    "web_browsing":  { "low": ..., "medium": ..., "high": ..., "flash": ... },
    "file_transfer": { "low": ..., "medium": ..., "high": ..., "flash": ... },
    "voip":          { "low": ..., "medium": ..., "high": ..., "flash": ... }
  },
  "diurnal_snapshots": {
    "peak_18h":     { "multiplier": 1.2, "traffic_matrix": [[...]] },
    "offpeak_03h":  { "multiplier": 0.2, "traffic_matrix": [[...]] },
    "business_10h": { "multiplier": 0.9, "traffic_matrix": [[...]] }
  },
  "flash_crowd": {
    "applied": true,
    "target_node_id": 7,
    "spike_multiplier": 5
  }
}
```

**Inputs:** `data/raw/topology_index.json`

**Outputs:** `data/processed/traffic/topology_00X_traffic.json`

---

## 7. Script 4 — `enrich_labels.py`

**File:** [src/group11/ground_truth/enrich_labels.py](src/group11/ground_truth/enrich_labels.py)

**Purpose:** Phase C. Generates ground-truth routing labels. For every source→destination pair in each topology, it finds the **shortest hop path**, sums per-link latencies from the raw labels, identifies the **bottleneck link**, and records all of this under all 4 load conditions.

**Run:** `python src/group11/ground_truth/enrich_labels.py`

---

### Functions

#### `_build_graph(edges) → nx.Graph`

Builds an undirected NetworkX graph from the edge list. Stores `edge_id` as an edge attribute so edges can later be looked up by their node pair.

---

#### `_build_edge_pair_lookup(edges) → dict`

Creates a mapping: `(min(u,v), max(u,v)) → edge_id`

This enables **undirected-safe** edge lookup — regardless of which direction a path traverses an edge, the correct `edge_id` is found.

---

#### `_build_latency_lookup(labels) → dict`

Parses the raw labels file into a lookup: `edge_id (int) → { "low": ms, "medium": ms, "high": ms, "flash": ms }`

---

#### `_path_to_edge_ids(path, pair_lookup) → list[int]`

Converts a node-hop list (e.g. `[0, 3, 7]`) into a list of edge IDs (e.g. `[edge_id_of_0-3, edge_id_of_3-7]`).

---

#### `enrich_topology(topology, labels) → dict`

The main logic. For each `(source, destination)` pair where `source ≠ destination` and a path exists:

1. Calls `nx.shortest_path(G, source, destination)` — uses **hop-count** as the weight (not latency, to avoid leakage)
2. Converts node path to `edge_ids_on_path`
3. For each load condition: sums per-link latencies → `end_to_end_latency_ms`
4. For each load condition: finds the maximum-latency link → `bottleneck_edge_id`

---

#### Output JSON structure (per path entry)

```json
{
  "source": 0,
  "destination": 5,
  "hop_path": [0, 2, 5],
  "edge_ids_on_path": [3, 11],
  "end_to_end_latency_ms": {
    "low": 2.1,
    "medium": 3.4,
    "high": 8.7,
    "flash": 45.2
  },
  "bottleneck_edge_id": {
    "low": 11,
    "medium": 11,
    "high": 11,
    "flash": 11
  }
}
```

**Inputs:**
- `data/raw/topology_index.json`
- `data/raw/topologies/json/topology_00X.json`
- `data/raw/labels/topology_00X_labels.json`

**Outputs:** `data/processed/labels/topology_00X_enriched_labels.json`

---

## 8. Script 5 — `assemble_dataset.py`

**File:** [src/group11/dataset_assembly/assemble_dataset.py](src/group11/dataset_assembly/assemble_dataset.py)

**Purpose:** Phase D — the final assembly step. Merges node features, edge features, global features, edge connectivity, and target latency values into PyTorch Geometric `Data` objects, and saves the complete dataset to disk.

**Run:** `python src/group11/dataset_assembly/assemble_dataset.py`

---

### Functions

#### `_build_y_edge(edges, raw_labels) → np.ndarray of shape (M, 4)`

Constructs the **target tensor** for model training. For each edge in topology order, reads the 4 load-condition latencies from the raw label file.

Column order: `[low, medium, high, flash]` (latency in ms)

> This is the **only place** raw latency values enter the assembled dataset — and they go only into `y_edge` (the training target), never into `edge_attr` (the input features).

---

#### `_build_edge_index(edges) → np.ndarray of shape (2, M)`

Produces a COO (coordinate) format edge connectivity matrix:
- Row 0: source node indices
- Row 1: target node indices
- Column order matches the topology JSON edge order

This is the standard format required by PyTorch Geometric.

---

#### `main()`

Full assembly logic:

1. Attempts `import torch` and `from torch_geometric.data import Data`
   - **If PyG is available:** saves PyG `Data` objects via `torch.save`
   - **If PyG is unavailable:** falls back to saving plain Python dicts as pickle
2. For each topology:
   - Loads raw topology JSON + raw labels JSON
   - Loads 3 `.npy` feature files (node, edge, global)
   - Calls `_build_edge_index` and `_build_y_edge`
   - Reshapes global features to `(1, 11)`
   - Constructs `Data(x, edge_index, edge_attr, y_edge, u, topology_id, num_nodes)`
3. Saves full list to `gnn_dataset.pt`
4. Writes `dataset_index.json` with dimension metadata

**Inputs:**
- `data/raw/topology_index.json`
- `data/raw/topologies/json/topology_00X.json`
- `data/raw/labels/topology_00X_labels.json`
- `data/processed/features/topology_00X_{node,edge,global}_features.npy`

**Outputs:**
- `data/processed/dataset/gnn_dataset.pt`
- `data/processed/dataset/dataset_index.json`

---

### PyG `Data` Object — Field Reference

This is what the GNN model receives as a training sample:

| Field | Shape | Type | Contents |
|---|---|---|---|
| `x` | `(N, 12)` | float tensor | Node feature matrix |
| `edge_index` | `(2, M)` | long tensor | COO edge connectivity |
| `edge_attr` | `(M, 5)` | float tensor | Edge feature matrix (no latency) |
| `u` | `(1, 11)` | float tensor | Global feature vector |
| `y_edge` | `(M, 4)` | float tensor | Target latencies: `[low, medium, high, flash]` in ms |
| `topology_id` | scalar | int | Integer topology ID |
| `num_nodes` | scalar | int | Number of nodes |

---

## 9. Script 6 — `validate_group11.py`

**File:** [tests/validate_group11.py](tests/validate_group11.py)

**Purpose:** Automated end-to-end validation of all pipeline outputs. Runs **35 checks** across all processed data files. Exits with code `0` on full pass, `1` if any check fails.

**Run:** `python tests/validate_group11.py`

---

### Helper Functions

| Function | Purpose |
|---|---|
| `fail(msg)` | Appends error message to `errors[]` list |
| `ok(msg)` | Prints `PASS: msg` |
| `section(title)` | Prints header separator |

---

### Check Functions

#### `check_index_exists() → list`
Verifies `topology_index.json` exists and is valid JSON. Returns the topology list.
- If this fails, execution stops immediately (all other checks depend on the index).

---

#### `check_features(topologies)`
For each topology, verifies:
- Node `.npy` file has shape `(N, 12)` where N matches the index
- Node features: all values in `[0,1]`, no `NaN`, no `Inf`
- Edge `.npy` file has shape `(M, 5)` where M matches the index
- Edge features: all values in `[0,1]`, no `NaN`, no `Inf`
- Global `.npy` file has shape `(11,)`, no `NaN`, no `Inf`

---

#### `check_traffic(topologies)`
For each topology, verifies:
- Traffic JSON file exists
- Contains all 4 application types: `video_streaming`, `web_browsing`, `file_transfer`, `voip`
- Each application type has all 4 load conditions: `low`, `medium`, `high`, `flash`

---

#### `check_enriched_labels(topologies)`
For each topology, verifies:
- Enriched labels JSON file exists
- At least one path entry exists
- Every `edge_id` referenced in any path entry is a valid ID in the topology

---

#### `check_dataset(topologies)`
Verifies the assembled PyG dataset:
- `gnn_dataset.pt` loads successfully
- Contains exactly `N` graph objects
- Each graph has all required fields: `x`, `edge_index`, `edge_attr`, `y_edge`, `u`
- Shape checks: `x=(N,12)`, `edge_attr=(M,5)`, `y_edge=(M,4)`, `u=(1,11)`
- No `NaN` or `Inf` values in any tensor

---

#### `check_dataset_index(topologies)`
Verifies `dataset_index.json`:
- File exists and is valid JSON
- `total_graphs` matches number of topologies
- Dimension fields match expected values: `node_feature_dim=12`, `edge_feature_dim=5`, `global_feature_dim=11`, `target_dim=4`
- All graph IDs match those in the topology index

---

### `main()` Execution Flow

```
1. check_index_exists()  → if fail → exit immediately
2. check_features()
3. check_traffic()
4. check_enriched_labels()
5. check_dataset()
6. check_dataset_index()
   ↓
Collect all errors → print summary → exit 0 (all pass) or exit 1 (any fail)
```

---

## 10. Feature Specifications

### Node Features — `(N, 12)` — All values in `[0,1]`

| Slot | Meaning |
|---|---|
| 0 | `router_type == "core"` (one-hot) |
| 1 | `router_type == "edge"` (one-hot) |
| 2 | `router_type == "gateway"` (one-hot) |
| 3 | `capacity_mbps / 1000` |
| 4 | `x_coord` (geographic position) |
| 5 | `y_coord` (geographic position) |
| 6 | `domain_id == 0` (one-hot) |
| 7 | `domain_id == 1` (one-hot) |
| 8 | `domain_id == 2` (one-hot) |
| 9 | `domain_id == 3` (one-hot) |
| 10 | `domain_id == 4` (one-hot) |
| 11 | `degree / max_degree` |

### Edge Features — `(M, 5)` — All values in `[0,1]`

| Slot | Meaning |
|---|---|
| 0 | `bandwidth_mbps / 1000` |
| 1 | `propagation_delay_ms / 100` |
| 2 | `is_inter_domain` (0 or 1) |
| 3 | `link_reliability` (already in `[0.95,1.0]`) |
| 4 | `cost / max_cost` |

### Global Features — `(11,)` — All values in `[0,1]`

| Slot | Meaning |
|---|---|
| 0 | `num_nodes / 100` |
| 1 | `num_edges / 500` |
| 2 | `num_domains / 5` |
| 3 | `mean_bandwidth_mbps / 1000` |
| 4 | `mean_propagation_delay_ms / 100` |
| 5 | `inter_domain_edge_count / num_edges` |
| 6 | `topology_type == "random"` (one-hot) |
| 7 | `topology_type == "scale_free"` (one-hot) |
| 8 | `topology_type == "mesh"` (one-hot) |
| 9 | `topology_type == "ring"` (one-hot) |
| 10 | `topology_type == "hybrid"` (one-hot) |

### Target — `y_edge (M, 4)` — Raw values in ms (NOT normalized)

| Slot | Meaning |
|---|---|
| 0 | Edge latency at `low` load (20% utilization) |
| 1 | Edge latency at `medium` load (50% utilization) |
| 2 | Edge latency at `high` load (80% utilization) |
| 3 | Edge latency at `flash` load (95% utilization) |

---

## 11. Key Design Decisions

### No Target Leakage
The most important design constraint: **latency values never appear in input features**. Latency only exists in `y_edge`. Edge features contain bandwidth, delay (propagation), reliability, and cost — but not computed latency. This ensures the GNN must learn to predict latency, not just copy it.

### Normalized Inputs, Raw Targets
All feature values in `x`, `edge_attr`, and `u` are normalized to `[0,1]`. The target `y_edge` is kept in raw milliseconds so the model learns the actual physical scale of the problem.

### Seeded Reproducibility
Every topology uses a fixed `generation_seed`. Traffic simulation uses `seed + 1000`. This ensures the entire dataset can be regenerated identically at any time.

### Index-Driven Architecture
No script ever hardcodes topology IDs or scans the filesystem. All scripts begin by reading `topology_index.json`. This means adding new topologies only requires updating that one file.

### Hop-Count Routing in Labels
`enrich_labels.py` uses hop-count (not latency) for shortest paths. This is intentional — the GNN's job is to predict latency and learn which paths are actually fast, not to be told by already knowing the latencies.

### GraphML Schema Compliance
The `generate_mock_data.py` script explicitly writes conformant GraphML with proper `<key>` declarations and `<data>` elements. This fixes a known schema bug mentioned in the project plan.

### PyG Fallback
`assemble_dataset.py` gracefully falls back to saving plain Python dicts as pickle if PyTorch Geometric is not installed, making the pipeline runnable in constrained environments.

---

## 12. End-to-End Data Flow

```
generate_mock_data.py
│
│  Produces: data/raw/
│    ├── topology_index.json
│    ├── topologies/json/topology_00X.json   ← node+edge attributes
│    ├── topologies/graphml/topology_00X.graphml
│    ├── topologies/csv/topology_00X_{nodes,edges}.csv
│    └── labels/topology_00X_labels.json     ← M/M/1 latency (4 conditions × M edges)
│
├──▶ build_features.py
│       Reads:  topology JSON
│       Writes: data/processed/features/
│                 topology_00X_node_features.npy    (N, 12)
│                 topology_00X_edge_features.npy    (M, 5)
│                 topology_00X_global_features.npy  (11,)
│
├──▶ simulate_traffic.py
│       Reads:  topology_index.json
│       Writes: data/processed/traffic/
│                 topology_00X_traffic.json  (16 demand matrices + diurnal + flash)
│
├──▶ enrich_labels.py
│       Reads:  topology JSON + raw labels JSON
│       Writes: data/processed/labels/
│                 topology_00X_enriched_labels.json (all-pairs paths + e2e latency + bottleneck)
│
└──▶ assemble_dataset.py
        Reads:  all of the above
        Writes: data/processed/dataset/
                  gnn_dataset.pt          ← 5 PyG Data objects ready for training
                  dataset_index.json      ← metadata + dimension specs

        validate_group11.py
          Reads: all processed outputs
          Output: exit 0 (35/35 checks pass) or exit 1 + error list
```

---

## 13. Dependencies

From [requirements.txt](requirements.txt):

| Package | Role |
|---|---|
| `numpy` | Feature array construction and `.npy` I/O |
| `pandas` | CSV handling |
| `networkx` | Graph construction, topology generation, shortest paths |
| `scikit-learn` | Utility functions |
| `torch` | Tensor operations, `torch.save` / `torch.load` |
| `torchvision`, `torchaudio` | PyTorch ecosystem (CPU build) |
| `torch-geometric` | `Data` object format for GNN training |
| `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv` | PyG dependencies |

---

*This study guide covers all 6 scripts in the Group 11 pipeline. The pipeline is designed to scale from the current 5 mock topologies to 500+ real topologies by simply updating `topology_index.json`.*
