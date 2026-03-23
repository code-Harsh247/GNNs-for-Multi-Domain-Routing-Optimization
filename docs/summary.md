# Project Summary — GNN Multi-Domain Routing Optimization

## What this project is

The goal is to train a Graph Neural Network (GNN) that can predict how long it takes for data to travel along each link in a computer network, under different amounts of traffic load.

The project is split between two groups:

- **Group 3** — generates synthetic network topologies and computes latency values using a mathematical queuing model.
- **Group 11** — takes Group 3's output, builds machine-learning-ready features from it, and assembles a dataset that a GNN can be trained on.

---

## What was done

### 1. Reviewed both implementation plans

Both `group3_implementation_plan.md` and `group11_implementation_plan.md` were read and checked for correctness. Three bugs were found and fixed directly in the plan files.

**Bug 1 — Wrong feature vector size (Group 11 plan)**
The global feature vector was documented as having 10 values, but there are 5 topology types that each need their own slot in a one-hot encoding, so the correct size is 11. All references in the plan were updated.

**Bug 2 — Latency values stored in the wrong place (Group 11 plan)**
Latency was listed as part of `edge_attr` (the input features) and also in `y_edge` (the prediction targets). Storing a value in both the inputs and the targets means the model can just copy the input instead of learning anything. The latency columns were removed from `edge_attr`. The edge feature dimension was corrected from 9 to 5 everywhere in the plan.

**Bug 3 — Missing data element in GraphML format (Group 3 plan)**
The `edge_id` field (a unique number identifying each link) was shown in the plan as an XML attribute on the `<edge>` element. However, Group 11 needs to look up edges by `edge_id` using the `<data>` element system that GraphML uses. Without a `<key>` declaration and a `<data key="edge_id">` element inside each `<edge>`, the value would not be readable. Both were added to the plan.

---

### 2. Generated mock data

Group 11's pipeline depends on topology files that Group 3 produces. Rather than waiting for Group 3, a script was written to generate realistic mock topologies that match Group 3's exact file format.

**Script:** `src/group11/generate_mock_data.py`

Five topologies were generated, one of each type:
- `topology_001` — random (10 nodes)
- `topology_002` — scale-free (20 nodes)
- `topology_003` — mesh (16 nodes)
- `topology_004` — ring (10 nodes)
- `topology_005` — hybrid (30 nodes)

For each topology the script writes:
- A JSON file with full node and edge attributes
- A GraphML file
- Three CSV files (node list, edge list, adjacency matrix)
- A JSON file with latency labels (one value per edge per load condition, computed using an M/M/1 queuing formula)
- A `topology_index.json` listing all topologies

All outputs land in `data/raw/`.

---

### 3. Built the four-phase pipeline (Group 11 Phase 1)

#### Phase A — Feature engineering
**Script:** `src/group11/feature_engineering/build_features.py`

Reads each topology JSON and produces three NumPy array files per topology:

- **Node features** — shape `(num_nodes, 12)`: router type (3 one-hot slots), normalised capacity, x/y coordinates, domain membership (5 one-hot slots), normalised degree.
- **Edge features** — shape `(num_edges, 5)`: normalised bandwidth, normalised propagation delay, whether the link crosses a domain boundary, link reliability, normalised cost.
- **Global features** — shape `(11,)`: normalised node count, normalised edge count, normalised domain count, average bandwidth, average propagation delay, fraction of inter-domain links, topology type (5 one-hot slots).

All values are in the range `[0, 1]`. Output goes to `data/processed/features/`.

#### Phase B — Traffic simulation
**Script:** `src/group11/traffic_simulation/simulate_traffic.py`

Generates traffic demand matrices for four application types (video streaming, web browsing, file transfer, VoIP) at four load levels (low=20%, medium=50%, high=80%, flash=95%). Each matrix has one demand value per source-destination pair.

Additional outputs include hourly snapshots showing how traffic changes across a 24-hour period, and a flash-crowd event on `topology_005` where one source-destination pair receives a sudden surge of demand.

Output goes to `data/processed/traffic/`.

#### Phase C — Ground truth enrichment
**Script:** `src/group11/ground_truth/enrich_labels.py`

For every source-destination pair in each topology:
- Finds the shortest path (by hop count) using NetworkX.
- Sums the per-link latencies along that path under each load condition to get an end-to-end latency.
- Identifies the bottleneck link (the link with the highest latency on the path) for each load condition.

Output goes to `data/processed/labels/`.

#### Phase D — Dataset assembly
**Script:** `src/group11/dataset_assembly/assemble_dataset.py`

Combines all the outputs from Phases A, B, and C into PyTorch Geometric `Data` objects — the format that GNN training code expects. Each object contains:

| Field | Shape | Contents |
|---|---|---|
| `x` | `(N, 12)` | Node features |
| `edge_index` | `(2, M)` | Source and destination node indices for each edge |
| `edge_attr` | `(M, 5)` | Edge features |
| `u` | `(1, 11)` | Global features |
| `y_edge` | `(M, 4)` | Target latency under 4 load conditions |

All 5 graph objects are saved to `data/processed/dataset/gnn_dataset.pt`. A `dataset_index.json` file lists each graph with its topology ID, node count, edge count, and type.

---

### 4. Set up the Python environment

A virtual environment was created at `.venv/` and the following packages were installed:

- `numpy` 2.4.3
- `pandas` 3.0.1
- `networkx` 3.6.1
- `scikit-learn` 1.8.0
- `torch` 2.10.0 (CPU build)
- `torch-geometric` 2.7.0

A `requirements.txt` file was written at the repo root so any team member can recreate the same environment.

---

### 5. Wrote the validation script

**Script:** `tests/validate_group11.py`

Runs 35 automated checks across all Phase 1 outputs:

1. `topology_index.json` exists and is valid JSON.
2. Every topology has node, edge, and global feature files with the correct shapes and values in `[0, 1]`.
3. Every topology has a traffic file containing all 4 application types and all 4 load conditions.
4. Every topology has an enriched label file, and every edge ID referenced in path lists exists in the topology.
5. `gnn_dataset.pt` loads correctly, contains 5 graphs, and every field has the correct shape with no NaN or infinite values.
6. `dataset_index.json` reports the correct graph count and feature dimensions, and its graph IDs match the topology index.

All 35 checks pass (exit code 0).

---

### 6. Wrote the data card

**File:** `docs/data_card.md`

Documents every feature in the dataset:
- All 12 node features: name, source file, value range, how it is computed, and why it is included.
- All 5 edge features: same detail, plus a note explaining why latency is deliberately excluded from this group.
- All 11 global features: same detail.
- All 4 prediction targets in `y_edge`: the load condition, utilisation value, and the M/M/1 formula used to compute each value.
- A table showing which files come from Group 3 and which are computed by Group 11.

---

## Current state of the repository

```
data/
  raw/
    topology_index.json           — lists all 5 topologies
    topologies/json/              — 5 topology JSON files
    topologies/graphml/           — 5 GraphML files
    topologies/csv/               — 15 CSV files (node list, edge list, adjacency per topology)
    labels/                       — 5 latency label files
  processed/
    features/                     — 15 NumPy files (node, edge, global per topology)
    traffic/                      — 5 traffic JSON files
    labels/                       — 5 enriched label JSON files
    dataset/
      gnn_dataset.pt              — 5 PyG Data objects
      dataset_index.json          — metadata for all 5 graphs

src/group11/
  generate_mock_data.py
  feature_engineering/build_features.py
  traffic_simulation/simulate_traffic.py
  ground_truth/enrich_labels.py
  dataset_assembly/assemble_dataset.py

tests/
  validate_group11.py             — 35 checks, all passing

docs/
  data_card.md                    — feature documentation
  summary.md                      — this file

requirements.txt
.venv/                            — Python virtual environment
```

## What comes next

- **Group 3** delivers real topology data. Run `build_features.py`, `simulate_traffic.py`, `enrich_labels.py`, and `assemble_dataset.py` in order to process it. Then re-run `validate_group11.py` to confirm the full dataset passes all checks.
- **Phase 2** (GNN model development) can begin using `gnn_dataset.pt` immediately with the mock data, since all field names, shapes, and units are already finalised.
