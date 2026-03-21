# Group 11 — Implementation Plan
## GNN Multi-Domain Routing: Feature Engineering, Traffic Simulation & Dataset Assembly

> **This document is the source of truth for Group 11's deliverables.**
> Group 11 reads from Group 3's output files. The exact structure of those files is defined in
> `group3_implementation_plan.md`. Always refer to that document for input file schemas.
> The final assembled dataset produced by Group 11 is what Phase 2 (GNN development) will consume.

---

## Overview of Responsibilities

Group 11 is responsible for:
1. Reading Group 3's topology files and engineering node, edge, and global feature vectors
2. Simulating realistic traffic patterns on each topology
3. Generating ground truth for end-to-end path latency and bottleneck identification
4. Assembling the complete, final dataset in PyTorch Geometric (PyG) compatible format
5. Writing validation scripts and a data card documenting everything

---

## Checklist

### Phase A — Feature Engineering
- [ ] Read all topology files from `data/raw/` using `topology_index.json`
- [ ] Generate node feature vectors for every router in every topology
- [ ] One-hot encode categorical node features (`router_type`, `domain_id`)
- [ ] Normalize all continuous node feature values to [0, 1] range
- [ ] Generate edge feature vectors for every link in every topology
- [ ] Ensure edge feature vector dimension is consistent across all 500+ topologies
- [ ] Define and generate global (graph-level) feature vectors for every topology
- [ ] Save intermediate feature files to `data/processed/features/`

### Phase B — Traffic Simulation
- [ ] Implement traffic generators for 4 application types: video streaming, web browsing, file transfer, VoIP
- [ ] Generate traffic matrices for every topology under each application type
- [ ] Implement diurnal traffic patterns (24-hour traffic variation cycle)
- [ ] Implement flash crowd scenarios (sudden localized traffic spikes)
- [ ] Save all traffic matrices to `data/processed/traffic/`

### Phase C — Ground Truth Enrichment
- [ ] Combine traffic matrices with Group 3's per-link latency labels
- [ ] Compute end-to-end path latency for all source-destination pairs in every topology
- [ ] Identify and label the bottleneck link for each source-destination path
- [ ] Save enriched ground truth to `data/processed/labels/`

### Phase D — Dataset Assembly
- [ ] Merge node features + edge features + global features + enriched labels into one unified structure
- [ ] Convert the unified structure into PyTorch Geometric `Data` objects
- [ ] Save the full dataset as a PyG `InMemoryDataset`
- [ ] Write validation scripts to check for missing values, dimension mismatches, and label completeness
- [ ] Write the data card documenting every feature

---

## File Structure

All output files must be placed in the following directory structure:

```
data/
└── processed/
    ├── features/
    │   ├── topology_001_node_features.npy
    │   ├── topology_001_edge_features.npy
    │   ├── topology_001_global_features.npy
    │   └── ...
    ├── traffic/
    │   ├── topology_001_traffic.json
    │   ├── topology_002_traffic.json
    │   └── ...
    ├── labels/
    │   ├── topology_001_enriched_labels.json
    │   ├── topology_002_enriched_labels.json
    │   └── ...
    └── dataset/
        ├── gnn_dataset.pt              # Final PyTorch Geometric dataset
        └── dataset_index.json          # Index of all graphs in the dataset
```

---

## Naming Conventions

- Feature files: `topology_XXX_node_features.npy`, `topology_XXX_edge_features.npy`, `topology_XXX_global_features.npy`
- Traffic files: `topology_XXX_traffic.json`
- Enriched label files: `topology_XXX_enriched_labels.json`
- All `XXX` indices must match exactly with Group 3's topology index

---

## Input Files (from Group 3)

Group 11 reads from the following files produced by Group 3. Do not modify these files.

| File | What to read from it |
|---|---|
| `data/raw/topology_index.json` | List of all topology IDs and filename bases |
| `data/raw/topologies/json/topology_XXX.json` | Node list, edge list, domain policies |
| `data/raw/labels/topology_XXX_labels.json` | Per-link latency under 4 load conditions |

Always use `topology_index.json` to iterate — never hardcode topology IDs or scan the filesystem directly.

---

## Phase A — Feature Engineering (Detailed)

### Node Features

For every node in every topology, construct a feature vector in this exact column order:

| Index | Feature Name | Source | Type | Notes |
|---|---|---|---|---|
| 0 | `router_type_core` | Group 3 JSON | float | One-hot: 1.0 if router_type == "core" else 0.0 |
| 1 | `router_type_edge` | Group 3 JSON | float | One-hot: 1.0 if router_type == "edge" else 0.0 |
| 2 | `router_type_gateway` | Group 3 JSON | float | One-hot: 1.0 if router_type == "gateway" else 0.0 |
| 3 | `capacity_mbps_norm` | Group 3 JSON | float | Normalized: value / 1000.0 (max capacity is 1000 Mbps) |
| 4 | `x_coord` | Group 3 JSON | float | Already normalized [0, 1] by Group 3 |
| 5 | `y_coord` | Group 3 JSON | float | Already normalized [0, 1] by Group 3 |
| 6–N | `domain_id_onehot` | Group 3 JSON | float | One-hot over domain IDs — size = max number of domains across dataset (use 5, pad with 0s) |
| N+1 | `degree_norm` | Computed | float | Node degree / max degree in this topology |
| N+2 | `avg_neighbor_bandwidth_norm` | Computed | float | Mean bandwidth of incident edges / 1000.0 |

> **Node feature vector size: 5 (router type + capacity + coords) + 5 (domain one-hot) + 2 (computed) = 12 dimensions**

All nodes across all topologies must have a feature vector of exactly **12 dimensions**.

**Saving node features:**
Save as a 2D NumPy array of shape `(num_nodes, 12)` to `topology_XXX_node_features.npy`.

---

### Edge Features

For every edge in every topology, construct a feature vector in this exact column order:

| Index | Feature Name | Source | Type | Notes |
|---|---|---|---|---|
| 0 | `bandwidth_mbps_norm` | Group 3 JSON | float | value / 1000.0 |
| 1 | `propagation_delay_ms_norm` | Group 3 JSON | float | value / 100.0 (assume max 100ms) |
| 2 | `is_inter_domain` | Group 3 JSON | float | 1.0 if True, 0.0 if False |
| 3 | `link_reliability` | Group 3 JSON | float | Already in [0, 1] |
| 4 | `cost_norm` | Group 3 JSON | float | value / max cost in this topology |
| 5 | `latency_low_norm` | Group 3 Labels | float | low load latency / 200.0 (cap at 200ms) |
| 6 | `latency_medium_norm` | Group 3 Labels | float | medium load latency / 200.0 |
| 7 | `latency_high_norm` | Group 3 Labels | float | high load latency / 200.0 |
| 8 | `latency_flash_norm` | Group 3 Labels | float | flash load latency / 200.0 |

> **Edge feature vector size: 9 dimensions**

All edges across all topologies must have a feature vector of exactly **9 dimensions**.

**Joining labels to edges:** Use `edge_id` as the join key between `topology_XXX.json` and `topology_XXX_labels.json`.

**Saving edge features:**
Save as a 2D NumPy array of shape `(num_edges, 9)` to `topology_XXX_edge_features.npy`.

---

### Global (Graph-Level) Features

One feature vector per topology describing the entire graph:

| Index | Feature Name | How to Compute | Notes |
|---|---|---|---|
| 0 | `num_nodes_norm` | num_nodes / 100.0 | Normalized by max possible nodes |
| 1 | `num_edges_norm` | num_edges / 500.0 | Normalized by assumed max edges |
| 2 | `num_domains_norm` | num_domains / 5.0 | Normalized by max domains |
| 3 | `avg_bandwidth_norm` | mean(all edge bandwidths) / 1000.0 | |
| 4 | `avg_propagation_delay_norm` | mean(all edge propagation delays) / 100.0 | |
| 5 | `inter_domain_ratio` | num_inter_domain_edges / num_edges | Fraction of cross-domain links |
| 6–9 | `topology_type_onehot` | One-hot over 5 types | random, scale_free, mesh, ring, hybrid |

> **Global feature vector size: 10 dimensions**

**Saving global features:**
Save as a 1D NumPy array of shape `(10,)` to `topology_XXX_global_features.npy`.

---

## Phase B — Traffic Simulation (Detailed)

### Application Types

Implement traffic generators for these 4 application types. Each has a different traffic profile:

| Application | Avg Rate (Mbps) | Burstiness | Pattern |
|---|---|---|---|
| Video Streaming | 5–25 | Low | Steady, slowly varying |
| Web Browsing | 0.1–10 | High | Bursty, short flows |
| File Transfer | 10–100 | Very Low | Sustained max throughput |
| VoIP | 0.1–0.5 | Very Low | Constant small packets |

### Traffic Matrix

A traffic matrix `T` for a topology with `N` nodes is an `N × N` matrix where `T[i][j]` is the traffic demand (in Mbps) from node `i` to node `j`. Diagonal values are always 0.

Generate one traffic matrix per topology per application type per load condition. That is, for each topology: 4 app types × 4 load conditions = **16 traffic matrices**.

### Diurnal Pattern

Scale traffic matrices by a time-of-day multiplier. Use this 24-value array (one multiplier per hour):

```python
diurnal_multipliers = [
    0.3, 0.2, 0.2, 0.2, 0.3, 0.4,   # 00:00 – 05:00 (night, low)
    0.6, 0.8, 1.0, 1.0, 0.9, 0.9,   # 06:00 – 11:00 (morning ramp)
    1.0, 1.0, 0.9, 0.9, 1.0, 1.1,   # 12:00 – 17:00 (business hours)
    1.2, 1.1, 1.0, 0.8, 0.6, 0.4    # 18:00 – 23:00 (evening peak, taper)
]
```

Store 3 representative time snapshots per topology: **peak (18:00)**, **off-peak (03:00)**, and **business (10:00)**.

### Flash Crowd Scenario

A flash crowd is a sudden 5× spike in traffic directed at a single randomly selected destination node, lasting for a short burst. Apply to 20% of topologies (every 5th topology by index). Mark the affected destination node and the time of the spike in the traffic file.

### Traffic File Format (`topology_XXX_traffic.json`)

```json
{
  "topology_id": 1,
  "traffic_matrices": {
    "video_streaming": {
      "low": [[0, 5.2, 3.1], [4.8, 0, 6.0], [2.9, 5.5, 0]],
      "medium": [[0, 12.1, 8.4], [11.0, 0, 13.2], [7.8, 12.5, 0]],
      "high": [[0, 18.5, 14.2], [17.1, 0, 19.0], [13.5, 18.8, 0]],
      "flash": [[0, 22.0, 17.1], [20.5, 0, 23.0], [16.2, 22.5, 0]]
    },
    "web_browsing": { "...": "..." },
    "file_transfer": { "...": "..." },
    "voip": { "...": "..." }
  },
  "diurnal_snapshots": {
    "peak_18h": { "multiplier": 1.2, "traffic_matrix": [[0, 6.24, 3.72], "..."] },
    "offpeak_03h": { "multiplier": 0.2, "traffic_matrix": [[0, 1.04, 0.62], "..."] },
    "business_10h": { "multiplier": 1.0, "traffic_matrix": [[0, 5.2, 3.1], "..."] }
  },
  "flash_crowd": {
    "applied": true,
    "target_node_id": 7,
    "spike_multiplier": 5.0
  }
}
```

> For topologies where flash crowd is not applied, set `"applied": false` and omit the other flash crowd fields.

---

## Phase C — Ground Truth Enrichment (Detailed)

### End-to-End Path Latency

For every source-destination pair in every topology:

1. Find the shortest path by hop count using NetworkX (`nx.shortest_path`)
2. Sum the per-link latencies along that path from Group 3's label file
3. Do this for all 4 load conditions

This gives you a ground truth end-to-end latency for every (source, destination, load_condition) triple.

### Bottleneck Identification

The bottleneck link on a path is the link with the highest latency under the given load condition. Identify and record the `edge_id` of the bottleneck for every path.

### Enriched Label File Format (`topology_XXX_enriched_labels.json`)

```json
{
  "topology_id": 1,
  "path_latencies": [
    {
      "source": 0,
      "destination": 2,
      "hop_path": [0, 1, 2],
      "edge_ids_on_path": [0, 3],
      "end_to_end_latency_ms": {
        "low": 5.61,
        "medium": 10.75,
        "high": 30.60,
        "flash": 209.80
      },
      "bottleneck_edge_id": {
        "low": 3,
        "medium": 3,
        "high": 0,
        "flash": 0
      }
    }
  ]
}
```

---

## Phase D — Dataset Assembly (Detailed)

### PyTorch Geometric Data Object

For each topology, create one `torch_geometric.data.Data` object with these fields:

| Field | Shape | Description |
|---|---|---|
| `x` | `(num_nodes, 12)` | Node feature matrix |
| `edge_index` | `(2, num_edges)` | Edge connectivity in COO format |
| `edge_attr` | `(num_edges, 9)` | Edge feature matrix |
| `u` | `(1, 10)` | Global feature vector |
| `y_edge` | `(num_edges, 4)` | Target: per-link latency under 4 load conditions |
| `topology_id` | scalar | Topology index |
| `num_nodes` | scalar | Number of nodes |

**Building `edge_index`:** This is a `(2, num_edges)` tensor where `edge_index[0]` contains source node IDs and `edge_index[1]` contains target node IDs, in the same order as the edge list in Group 3's JSON.

**Building `y_edge`:** Stack the 4 load condition latencies from `topology_XXX_labels.json` into a `(num_edges, 4)` tensor in this column order: `[low, medium, high, flash]`.

### Saving the Dataset

```python
import torch
from torch_geometric.data import InMemoryDataset

# After building all Data objects into a list called `data_list`:
torch.save(data_list, 'data/processed/dataset/gnn_dataset.pt')
```

### Dataset Index File (`dataset_index.json`)

```json
{
  "total_graphs": 500,
  "node_feature_dim": 12,
  "edge_feature_dim": 9,
  "global_feature_dim": 10,
  "target_dim": 4,
  "load_conditions": ["low", "medium", "high", "flash"],
  "graphs": [
    {
      "topology_id": 1,
      "num_nodes": 20,
      "num_edges": 35,
      "topology_type": "random"
    }
  ]
}
```

---

## Validation Scripts

Write a script `tests/validate_group11.py` that checks:

- [ ] Every topology in `topology_index.json` has a corresponding node, edge, and global feature file
- [ ] Node feature arrays are shape `(num_nodes, 12)` — no topology has a different dimension
- [ ] Edge feature arrays are shape `(num_edges, 9)` — no topology has a different dimension
- [ ] Global feature arrays are shape `(10,)` for every topology
- [ ] No NaN or Inf values in any feature array
- [ ] All values in node and edge feature arrays are in [0, 1] range (post-normalization)
- [ ] Every topology has a traffic file with all 4 application types and all 4 load conditions
- [ ] Every topology has an enriched label file
- [ ] `edge_ids_on_path` in enriched labels only reference edge IDs that exist in the topology
- [ ] `gnn_dataset.pt` loads without errors and contains the correct number of graphs
- [ ] `dataset_index.json` matches the actual dataset

Run this script and fix all errors before declaring Phase 1 complete.

---

## Data Card

Write `docs/data_card.md` documenting the following for every feature:

- Feature name
- Which file it comes from (Group 3 or computed by Group 11)
- Data type
- Value range after normalization
- How it was computed or normalized
- Why it is included (what information it gives the GNN)

---

## Suggested Tools & Libraries

```
torch                    # PyTorch
torch-geometric          # PyTorch Geometric for graph dataset
numpy                    # Feature arrays
pandas                   # Tabular processing
networkx                 # Path finding (shortest_path for end-to-end latency)
json                     # Reading Group 3 files
scikit-learn             # MinMaxScaler if needed for normalization
```

---

## Milestone Schedule

| Milestone | Deadline | Dependency |
|---|---|---|
| Schema agreement with Group 3 | End of Week 1 | Must happen before any coding |
| Feature engineering code ready (runs on sample 5 topologies) | End of Week 2 | Needs Group 3's sample files from Week 1 |
| Traffic simulation complete for all topologies | End of Week 3 | Needs Group 3's full dataset from Week 3 |
| Enriched labels + full dataset assembled | End of Week 4 | Needs Group 3's full dataset |
| Validation pass + data card written | End of Week 4 | Final deliverable for Phase 1 |

---

## Dependency on Group 3

Group 11's work is downstream of Group 3. The table below shows exactly what you need from them and when:

| What you need | When you need it | Why |
|---|---|---|
| 5 sample topologies in JSON format | End of Week 1 | To write and test feature engineering code |
| `topology_index.json` (partial) | End of Week 2 | To iterate over available topologies |
| All 500+ topologies + label files | End of Week 3 | To run full pipeline |

If Group 3 is delayed, immediately flag it — do not wait. You can write and test all code on the 5 sample topologies and run the full pipeline only in Week 4 if needed.

---

## Validation Checklist Before Declaring Phase 1 Complete

- [ ] `validate_group11.py` runs with zero errors on full dataset
- [ ] `gnn_dataset.pt` loads correctly and contains 500+ graphs
- [ ] All graphs have `x`, `edge_index`, `edge_attr`, `u`, `y_edge` fields
- [ ] Feature dimensions are consistent: node=12, edge=9, global=10, target=4
- [ ] No NaN or Inf values anywhere in the dataset
- [ ] `dataset_index.json` is complete and accurate
- [ ] `docs/data_card.md` is written and covers all features
- [ ] All traffic files are present with all 4 application types
- [ ] All enriched label files are present with bottleneck IDs

---

*Last updated: Phase 1 planning. This document should be treated as a contract — Group 11's output is what Phase 2 (GNN model development) will load directly.*
