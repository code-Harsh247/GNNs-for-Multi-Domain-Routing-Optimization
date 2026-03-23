# GNN Multi-Domain Routing - Dataset Generation Pipeline
### Group 3 · Module 1 of 2

> **Objective** - Generate 500+ synthetic multi-domain network topologies with fully labeled ground-truth latency data under both static load conditions and realistic time-varying traffic, ready for consumption by Group 11's GNN training pipeline.

---

## Table of Contents

1. [Project Context](#1-project-context)
2. [Repository Structure](#2-repository-structure)
3. [Architecture Overview](#3-architecture-overview)
4. [Output Structure](#4-output-structure)
5. [Data Layers Explained](#5-data-layers-explained)
6. [File Format Specifications](#6-file-format-specifications)
7. [Node & Edge Feature Reference](#7-node--edge-feature-reference)
8. [Traffic Simulation Model](#8-traffic-simulation-model)
9. [Queueing Models](#9-queueing-models)
10. [Topology Types & Distribution](#10-topology-types--distribution)
11. [Setup & Installation](#11-setup--installation)
12. [Generating the Dataset](#12-generating-the-dataset)
13. [CLI Reference](#13-cli-reference)
14. [Validating the Dataset](#14-validating-the-dataset)
15. [Reproducibility](#15-reproducibility)
16. [Module Reference](#16-module-reference)
17. [Notes for Group 11](#17-notes-for-group-11)

---

## 1. Project Context

This repository implements **Module 1** of a two-group project on Graph Neural Networks for Multi-Domain Routing Optimization. The full project scope is:

- **Module 1 (this repo - Group 3):** Dataset creation, network modeling, traffic simulation, ground-truth labeling, and export.
- **Module 2 (Group 11):** GNN model development (GCN, GAT, MPNN, RouteNet-Fermi), congestion-aware routing optimizer, comparison against Dijkstra/static weighted routing, visualization.

Group 11 consumes the files produced here directly. The file formats, naming conventions, and attribute schemas in this document are the **contract between the two groups**.

---

## 2. Repository Structure

```
IAP_TermProject/
├── src/
│   └── gnn_routing/
│       ├── __init__.py          # Package exports
│       ├── config.py            # DatasetConfig dataclass + type distribution table
│       ├── topology.py          # 5 topology type builders + multi-domain partitioning
│       ├── traffic.py           # Time-varying traffic matrix generator
│       ├── latency.py           # M/M/1 static labels + M/M/1/M/G/1 temporal simulation
│       ├── exporters.py         # All file writers (GraphML, JSON, CSV, labels, snapshots)
│       └── dataset.py           # End-to-end pipeline orchestrator
│
├── scripts/
│   ├── generate_dataset.py      # CLI entry point
│   └── validate_dataset.py      # Pre-handoff spec compliance checker
│
├── data/
│   └── raw/                     # Default output directory (generated, not committed)
│
├── requirements.txt
└── README.md
```

---

## 3. Architecture Overview

The pipeline runs in a single loop over all topologies. For each topology it performs two sequential phases:

```
For each topology_id in 1..N:
│
├── generate_topology()                     topology.py
│     ├── Choose type from distribution (random/scale_free/mesh/ring/hybrid)
│     ├── Build intra-domain subgraph per type
│     ├── Connect domains with inter-domain edges (AS-like backbone)
│     └── Assign all node/edge static attributes
│
├── LAYER 1 - Static exports                exporters.py + latency.py
│     ├── write_graphml()
│     ├── write_topology_json()
│     ├── write_nodes_csv() + write_edges_csv()
│     └── compute_all_load_conditions() → write_label_json()
│           Uses M/M/1 at ρ = 0.20 / 0.50 / 0.80 / 0.95
│
└── LAYER 2 - Temporal loop (T timesteps)   traffic.py + latency.py + exporters.py
      For t in 0 .. T-1:
        ├── generate_traffic_matrix()
        │     ├── Diurnal factor (sinusoidal 24-hour curve)
        │     ├── Flash crowd event (random burst, 8 % probability)
        │     ├── Application profile selection (web / video / bulk)
        │     └── Per-flow lognormal demand scaled by node history
        │
        ├── simulate_snapshot()
        │     ├── Route all (src,dst) demands via shortest-path
        │     ├── Accumulate per-link load_mbps
        │     ├── Compute utilization, queue delay, saturation penalty
        │     ├── Flag bottleneck edges (top-10 %, min threshold 85 %)
        │     └── Compute end-to-end path latency + bottleneck count
        │
        └── write_snapshot_json() + append_snapshot_csv()

After all topologies:
    write_topology_index()       → topology_index.json
    all_snapshots.csv            (master flat CSV: all topologies × all timesteps)
```

---

## 4. Output Structure

All output lives under `data/raw/` (configurable via `--output-dir`):

```
data/raw/
│
├── topology_index.json                        ← master index for Group 11
│
├── topologies/
│   ├── graphml/
│   │   ├── topology_001.graphml
│   │   └── ...  (500 files)
│   ├── json/
│   │   ├── topology_001.json
│   │   └── ...  (500 files)
│   └── csv/
│       ├── topology_001_nodes.csv
│       ├── topology_001_edges.csv
│       └── ...  (1,000 files - 2 per topology)
│
├── labels/
│   ├── topology_001_labels.json
│   └── ...  (500 files)
│
└── snapshots/
    ├── all_snapshots.csv                      ← master flat CSV (500 × 12 = 6,000 rows)
    │
    ├── topology_001_t00_snapshot.json         ← per-timestep rich ground truth
    ├── topology_001_t01_snapshot.json
    ├── ...
    ├── topology_001_t11_snapshot.json         (12 per topology = 6,000 total)
    │
    ├── topology_001_edge_snapshots.csv        ← all 12 timesteps, edge rows
    ├── topology_001_path_snapshots.csv        ← all 12 timesteps, path rows
    ├── topology_001_global_snapshots.csv      ← all 12 timesteps, global row
    └── ...  (3 CSVs per topology = 1,500 files)
```

**File counts for a full 500-topology, 12-timestep run:**

| Artifact | Count |
|---|---|
| GraphML topology files | 500 |
| JSON topology files | 500 |
| Node + edge CSV files | 1,000 |
| Static label files | 500 |
| Per-timestep snapshot JSONs | 6,000 |
| Per-topology snapshot CSVs | 1,500 |
| Master all_snapshots.csv | 1 |
| topology_index.json | 1 |
| **Total files** | **~10,002** |

---

## 5. Data Layers Explained

### Layer 1 - Static Topology Files

These are the primary deliverable for Group 11's GNN training pipeline. They capture the **structural properties** of each topology - the graph itself, its attributes, and ground-truth edge latency at four canonical utilization levels.

The four load conditions apply the M/M/1 formula uniformly across all links. They answer the question "what would latency be if the whole network were at X % utilization?" - giving Group 11 clean, unambiguous training targets for each structural configuration without needing to run a traffic simulation.

### Layer 2 - Temporal Snapshot Files

These capture **dynamic network state** across T simulated timesteps per topology. Each timestep is driven by a full traffic simulation:

- A 24-hour diurnal load curve (sinusoidal, peaks at midday)
- Random flash-crowd events (demand spikes up to 3.4×)
- Application-profile selection (web / video / bulk, each with distinct flow size distribution)
- Per-node historical traffic scaling

The traffic matrix is then routed through the topology, each link's actual load is computed, and per-link latency is derived from real utilization. These labels are suitable for temporal GNN architectures (T-GNNs) that must predict congestion from observed traffic patterns, and for training the congestion-aware routing optimizer in Module 2.

---

## 6. File Format Specifications

### GraphML (`topology_XXX.graphml`)

Standard GraphML with all node, edge, and graph attributes declared as `<key>` elements. Written with explicit XML to guarantee exact attribute names, types, and ordering required by Group 11.

**Required `<key>` declarations:**

```xml
<!-- Node keys -->
<key id="node_id"            for="node" attr.name="node_id"            attr.type="int"/>
<key id="router_type"        for="node" attr.name="router_type"        attr.type="string"/>
<key id="capacity_mbps"      for="node" attr.name="capacity_mbps"      attr.type="double"/>
<key id="x_coord"            for="node" attr.name="x_coord"            attr.type="double"/>
<key id="y_coord"            for="node" attr.name="y_coord"            attr.type="double"/>
<key id="domain_id"          for="node" attr.name="domain_id"          attr.type="int"/>
<key id="historical_traffic" for="node" attr.name="historical_traffic" attr.type="double"/>

<!-- Edge keys -->
<key id="edge_id"              for="edge" attr.name="edge_id"              attr.type="int"/>
<key id="bandwidth_mbps"       for="edge" attr.name="bandwidth_mbps"       attr.type="double"/>
<key id="propagation_delay_ms" for="edge" attr.name="propagation_delay_ms" attr.type="double"/>
<key id="is_inter_domain"      for="edge" attr.name="is_inter_domain"      attr.type="boolean"/>
<key id="link_reliability"     for="edge" attr.name="link_reliability"     attr.type="double"/>
<key id="cost"                 for="edge" attr.name="cost"                 attr.type="double"/>

<!-- Graph keys -->
<key id="topology_id"     for="graph" attr.name="topology_id"     attr.type="int"/>
<key id="num_nodes"       for="graph" attr.name="num_nodes"       attr.type="int"/>
<key id="num_edges"       for="graph" attr.name="num_edges"       attr.type="int"/>
<key id="num_domains"     for="graph" attr.name="num_domains"     attr.type="int"/>
<key id="topology_type"   for="graph" attr.name="topology_type"   attr.type="string"/>
<key id="generation_seed" for="graph" attr.name="generation_seed" attr.type="int"/>
<key id="domain_policies" for="graph" attr.name="domain_policies" attr.type="string"/>
```

---

### JSON Topology (`topology_XXX.json`)

Mirrors the GraphML structure. `domain_policies` is a **parsed dict** here (not a JSON string as in GraphML).

```json
{
  "topology_id": 1,
  "num_nodes": 42,
  "num_edges": 67,
  "num_domains": 3,
  "topology_type": "scale_free",
  "generation_seed": 191664964,
  "domain_policies": {
    "0": [1, 2],
    "1": [0],
    "2": [0, 1]
  },
  "nodes": [
    {
      "node_id": 0,
      "router_type": "core",
      "capacity_mbps": 1280.5,
      "x_coord": 0.42,
      "y_coord": 0.77,
      "domain_id": 0,
      "historical_traffic": 0.61
    }
  ],
  "edges": [
    {
      "edge_id": 0,
      "source": 0,
      "target": 1,
      "bandwidth_mbps": 450.0,
      "propagation_delay_ms": 2.5,
      "is_inter_domain": false,
      "link_reliability": 0.987,
      "cost": 3.61
    }
  ]
}
```

`domain_policies` keys are domain IDs as strings; values are lists of domain IDs that domain may route traffic through.

---

### Node CSV (`topology_XXX_nodes.csv`)

Columns **in this exact order**:

```
node_id, router_type, capacity_mbps, x_coord, y_coord, domain_id, historical_traffic
```

---

### Edge CSV (`topology_XXX_edges.csv`)

Columns **in this exact order**:

```
edge_id, source, target, bandwidth_mbps, propagation_delay_ms,
is_inter_domain, link_reliability, cost
```

---

### Static Label File (`topology_XXX_labels.json`)

One file per topology. Contains M/M/1 latency for every edge under all four canonical load conditions. **Edge IDs in `edge_latencies_ms` are string keys** matching `edge_id` in the topology files - this is the join key for Group 11.

```json
{
  "topology_id": 1,
  "load_conditions": {
    "low": {
      "utilization_fraction": 0.20,
      "edge_latencies_ms": {
        "0": 2.51,
        "1": 3.10,
        "2": 1.89
      }
    },
    "medium": { "utilization_fraction": 0.50, "edge_latencies_ms": { "0": 4.20, "1": 6.55 } },
    "high":   { "utilization_fraction": 0.80, "edge_latencies_ms": { "0": 12.40, "1": 18.20 } },
    "flash":  { "utilization_fraction": 0.95, "edge_latencies_ms": { "0": 89.30, "1": 120.50 } }
  }
}
```

---

### Snapshot JSON (`topology_XXX_tTT_snapshot.json`)

One file per topology per timestep (12 files per topology = 6,000 total at default settings). This is the richest output artifact, containing the full dynamic ground truth for one simulated time window.

```json
{
  "snapshot_id": "topology_001_t03",
  "topology_id": 1,
  "timestep": 3,
  "queue_model": "mm1",

  "global_features": {
    "time_of_day": 6.0,
    "diurnal_factor": 0.925,
    "flash_event": 0,
    "flash_multiplier": 1.0,
    "traffic_profile": "video",
    "network_load_mbps": 842.3
  },

  "edge_labels": [
    {
      "edge_id": 0,
      "u": 0, "v": 1,
      "bandwidth_mbps": 450.0,
      "load_mbps": 189.4,
      "utilization": 0.421,
      "queue_length_pkts": 2.3,
      "latency_ms": 4.18,
      "propagation_delay_ms": 2.5,
      "link_reliability": 0.987,
      "cost_metric": 3.61,
      "is_bottleneck": 0,
      "is_inter_domain": false
    }
  ],

  "path_labels": [
    {
      "source": 0,
      "target": 5,
      "demand_mbps": 12.4,
      "path_hops": 3,
      "e2e_latency_ms": 14.7,
      "path_bottleneck_count": 0
    }
  ],

  "summary": {
    "avg_utilization": 0.38,
    "max_utilization": 0.91
  }
}
```

---

### Snapshot CSVs

Three flat CSVs per topology across all T timesteps. All share `snapshot_id` as a join key.

**`topology_XXX_edge_snapshots.csv`** - one row per edge per timestep:
```
snapshot_id, topology_id, timestep, edge_id, u, v, bandwidth_mbps, load_mbps,
utilization, queue_length_pkts, latency_ms, propagation_delay_ms,
link_reliability, cost_metric, is_bottleneck, is_inter_domain
```

**`topology_XXX_path_snapshots.csv`** - one row per sampled path per timestep:
```
snapshot_id, topology_id, timestep, source, target, demand_mbps,
path_hops, e2e_latency_ms, path_bottleneck_count
```

**`topology_XXX_global_snapshots.csv`** - one row per timestep:
```
snapshot_id, topology_id, timestep, time_of_day, diurnal_factor, flash_event,
flash_multiplier, traffic_profile, network_load_mbps, avg_utilization,
max_utilization, queue_model
```

**`all_snapshots.csv`** - master file, same columns as global CSV, all topologies and all timesteps in one file. 6,000 rows for the default full run.

---

### Topology Index (`topology_index.json`)

Located at `data/raw/topology_index.json`. The single entry point for Group 11 to iterate over all topologies without scanning the filesystem.

```json
{
  "total_topologies": 500,
  "topologies": [
    {
      "topology_id": 1,
      "filename_base": "topology_001",
      "num_nodes": 42,
      "num_edges": 67,
      "topology_type": "scale_free",
      "num_domains": 3,
      "generation_seed": 191664964,
      "num_timesteps": 12
    }
  ]
}
```

---

## 7. Node & Edge Feature Reference

### Static Node Features

| Attribute | Type | Range | Description |
|---|---|---|---|
| `node_id` | int | ≥ 0 | Unique integer ID within the topology |
| `router_type` | string | `core`, `edge`, `gateway` | Functional role; scale factors: core 1.6×, gateway 1.2×, edge 0.8× |
| `capacity_mbps` | float | 100–2000 Mbps | Node processing capacity, base [100, 1000] × router type scale |
| `x_coord` | float | [0.0, 1.0] | Normalised X position; domain centre + Gaussian jitter (σ=0.08) |
| `y_coord` | float | [0.0, 1.0] | Normalised Y position |
| `domain_id` | int | 0 … D-1 | AS domain membership (0-indexed) |
| `historical_traffic` | float | [0.1, 0.8] | Per-node baseline load ratio; used by traffic simulator to scale outgoing demand |

### Static Edge Features

| Attribute | Type | Range | Description |
|---|---|---|---|
| `edge_id` | int | ≥ 0 | Unique integer ID per topology; join key with label files |
| `bandwidth_mbps` | float | 10–100 (inter) / 100–1000 (intra) Mbps | Maximum link capacity |
| `propagation_delay_ms` | float | > 0.1 ms | Fixed delay from Euclidean distance between node coords × 20 + jitter |
| `is_inter_domain` | bool | true/false | Whether the link crosses AS domain boundaries |
| `link_reliability` | float | [0.94, 0.999] | Static probability the link is available |
| `cost` | float | > 0 | Static routing cost = `prop_delay_ms + 500 / bandwidth_mbps` |

### Dynamic Edge Features (Layer 2 snapshots only)

| Attribute | Type | Description |
|---|---|---|
| `load_mbps` | float | Actual traffic accumulated by routing all demands through this link |
| `utilization` | float | `load_mbps / bandwidth_mbps`, capped at 1.2 to represent overloaded links |
| `queue_length_pkts` | float | Expected queue length in packets (from M/M/1 or M/G/1 formula) |
| `latency_ms` | float | Total link latency = propagation delay + queueing delay + saturation penalty |
| `is_bottleneck` | int | 1 if utilization ≥ max(85 %, 90th-percentile utilization) in this snapshot |

### Global Features (per snapshot)

| Attribute | Type | Description |
|---|---|---|
| `time_of_day` | float | Simulated hour [0, 24) |
| `diurnal_factor` | float | Load multiplier [0.25, 1.15] from the sinusoidal 24-hour curve |
| `flash_event` | int | 0 or 1 - whether a flash-crowd event occurred this timestep |
| `flash_multiplier` | float | 1.0 (no event) or uniform [1.8, 3.4] (event) |
| `traffic_profile` | string | `web`, `video`, or `bulk` |
| `network_load_mbps` | float | Sum of all (src, dst) demands in the traffic matrix |

---

## 8. Traffic Simulation Model

Traffic matrices are generated by `traffic.py:generate_traffic_matrix()`. Each timestep independently samples the following:

### Diurnal Pattern

A sinusoidal curve models the natural 24-hour traffic rhythm. Timestep 0 maps to ~4 AM (trough ≈ 0.25), the midpoint maps to midday (peak ≈ 1.15):

```
phase  = 2π × (t mod T) / T
factor = 0.70 + 0.45 × (1 + sin(phase − π/2)) / 2    ∈ [0.25, 1.15]
```

### Flash Crowds

With probability `flash_crowd_probability` (default 8 %), all demands in the timestep are multiplied by a uniform random factor in [1.8, 3.4]. This simulates sudden traffic spikes from viral events, DDoS reroutes, or large scheduled transfers.

### Application Profiles

One profile is active per timestep, drawn from a fixed probability distribution:

| Profile | Probability | Mean scale | Log-normal σ | Represents |
|---|---|---|---|---|
| `web` | 55 % | 0.5× | 0.6 | Many small HTTP/HTTPS flows, high variance |
| `video` | 25 % | 1.1× | 0.4 | Large steady streaming flows, low variance |
| `bulk` | 20 % | 0.9× | 1.0 | Occasional very large file transfers, extreme variance |

### Per-flow Demand Formula

Each (src, dst) pair's demand in Mbps:

```
base        ~ LogNormal(mean=0.1, sigma=app_sigma)
demand_mbps = base × app_scale × diurnal × flash_multiplier
                  × inter_domain_boost × (0.5 + historical_traffic_src)
```

Where:
- `inter_domain_boost` = 1.3 for cross-domain flows, 1.0 for intra-domain
- `historical_traffic_src` ∈ [0.1, 0.8] is the source node's stored baseline (busier nodes generate more traffic)

### Routing

Demands are routed via Dijkstra with a composite link weight that penalises narrow-bandwidth links:

```
link_weight = propagation_delay_ms + 300.0 / bandwidth_mbps
```

The `300 / bw` term adds ~3 ms penalty for a 100 Mbps link vs ~0.3 ms for 1000 Mbps, steering traffic towards high-capacity paths even when propagation delay is similar.

---

## 9. Queueing Models

### Layer 1 - M/M/1 Static Labels (Spec Formula)

Applied uniformly at 4 fixed utilization fractions. `avg_packet_size_mb = 0.001` (≈ 1500 bytes):

```
μ  = bandwidth_mbps / avg_packet_size_mb      [service rate, Mbps / Mb = 1/s]
ρ  = utilization_fraction                      [capped at 0.95]
W  = (ρ / (μ × (1 − ρ))) × 1000              [queueing delay, ms]

latency_ms = propagation_delay_ms + W
```

### Layer 2 - Temporal Simulation (M/M/1 or M/G/1)

Uses the **actual computed utilization** per link from traffic routing, not a uniform fraction.

**M/M/1** (Poisson arrivals, exponential service - default):
```
ρ_clamped  = clamp(utilization, 1e-6, 0.995)
μ_pps      = bandwidth_mbps × 1e6 / 12000     [packets per second, 1500-byte packets]
λ          = ρ_clamped × μ_pps
W_s        = ρ_clamped / (μ_pps − λ)          [seconds]
E[N]       = ρ² / (1 − ρ)                     [expected queue length in packets]
```

**M/G/1** (Poisson arrivals, general service - use `--queue-model mg1` for bursty traffic):

Uses the Pollaczek-Khinchine approximation with SCV = 1.5 (squared coefficient of variation > 1 models bursty packet sizes):
```
W_s  = ((1 + SCV) / 2) × ρ / (μ_pps − λ)    [seconds]
E[N] = λ × W_s
```

**Saturation penalty** (applied when utilization ≥ 98 %, captures severe congestion):
```
penalty = 10.0 + 120.0 × (utilization − 0.98)   [ms]
```

**Total link latency (both models):**
```
latency_ms = propagation_delay_ms + W_s × 1000 + penalty
```

**Bottleneck detection:** An edge is flagged `is_bottleneck = 1` if its utilization meets or exceeds `max(0.85, 90th-percentile utilization)` across all edges in that snapshot.

---

## 10. Topology Types & Distribution

The 500 topologies follow a fixed distribution matching the project spec. Each type uses a different graph-generation algorithm producing structurally distinct connectivity patterns:

| Type | Count | Node Range | Algorithm | Real-world analogue |
|---|---|---|---|---|
| `random` | 150 | 10–100 | Erdős-Rényi (`erdos_renyi_graph`) | Research/campus networks, random peering |
| `scale_free` | 100 | 20–100 | Barabási-Albert (`barabasi_albert_graph`) | Internet AS topology, hub-and-spoke ISPs |
| `mesh` | 100 | 10–50 | 2-D grid (`grid_2d_graph`) | Data-centre interconnects, WDM optical rings |
| `ring` | 75 | 10–50 | Cycle + random chords (`cycle_graph`) | SONET/SDH rings, metro-Ethernet |
| `hybrid` | 75 | 30–100 | Watts-Strogatz small-world | Mixed enterprise / ISP topologies |

When `--num-topologies` differs from 500, each bucket is scaled proportionally, rounded to the nearest integer, and trimmed/padded to hit the exact target count.

**Domain model (applies to all types):**

Each topology is partitioned into 2–5 AS-like domains. Nodes within a domain are connected by the type-specific algorithm. Domains are connected via a sparse inter-domain backbone:
- One guaranteed link between each pair of consecutive domains
- A random number (1–2× num_domains) of additional random inter-domain links

Intra-domain links: 100–1000 Mbps. Inter-domain links: 10–100 Mbps (narrower, as in real inter-AS peering).

---

## 11. Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/DKsheeraj/IAP_TermProject.git
cd IAP_TermProject

# 2. Create and activate virtual environment
python -m venv iap_proj
source iap_proj/bin/activate          # Linux / macOS
# iap_proj\Scripts\activate           # Windows CMD
# $env:PYTHONPATH = "src"            # Windows PowerShell

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set PYTHONPATH so the package is importable
export PYTHONPATH=src                 # Linux / macOS
# set PYTHONPATH=src                  # Windows CMD
```

**`requirements.txt`:**
```
networkx>=3.2
numpy>=1.26
tqdm>=4.66
matplotlib>=3.8
plotly>=5.22
pyvis>=0.3
pypdf>=6.0
```

---

## 12. Generating the Dataset

### Full dataset (500 topologies × 12 timesteps)

```bash
python scripts/generate_dataset.py
```

Output goes to `data/raw/`. Produces 6,000 temporal snapshots plus 500 static topology packages. Estimated runtime: 10–20 minutes depending on hardware.

### Smoke test (~5 seconds, 5 topologies)

```bash
python scripts/generate_dataset.py \
    --num-topologies 5 \
    --min-nodes 10 --max-nodes 30 \
    --timesteps 4 \
    --no-distribution \
    --output-dir outputs/smoke_dataset
```

### M/G/1 model run (bursty traffic scenario)

```bash
python scripts/generate_dataset.py \
    --queue-model mg1 \
    --flash-crowd-prob 0.12 \
    --output-dir data/raw_mg1
```

### Python API

```python
from gnn_routing.config import DatasetConfig
from gnn_routing.dataset import generate_dataset

cfg = DatasetConfig(
    random_seed=42,
    num_topologies=500,
    use_distribution=True,       # follow 150/100/100/75/75 type split
    min_nodes=10,
    max_nodes=100,
    min_domains=2,
    max_domains=5,
    timesteps_per_topology=12,
    queue_model="mm1",            # "mm1" or "mg1"
    flash_crowd_probability=0.08,
    max_paths_per_snapshot=250,
    output_dir="data/raw",
)
generate_dataset(cfg)
```

---

## 13. CLI Reference

```
usage: generate_dataset.py [-h]
                            [--num-topologies N]
                            [--min-nodes N] [--max-nodes N]
                            [--min-domains N] [--max-domains N]
                            [--no-distribution]
                            [--timesteps N]
                            [--queue-model {mm1,mg1}]
                            [--flash-crowd-prob F]
                            [--max-paths N]
                            [--seed N]
                            [--output-dir PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--num-topologies` | 500 | Total topologies to generate |
| `--min-nodes` | 10 | Minimum nodes per topology |
| `--max-nodes` | 100 | Maximum nodes per topology |
| `--min-domains` | 2 | Minimum AS domains per topology |
| `--max-domains` | 5 | Maximum AS domains per topology |
| `--no-distribution` | off | Disable type distribution; pick types uniformly at random |
| `--timesteps` | 12 | Temporal timesteps per topology (Layer 2) |
| `--queue-model` | `mm1` | Queueing model: `mm1` (M/M/1) or `mg1` (M/G/1) |
| `--flash-crowd-prob` | 0.08 | Probability of a flash-crowd event per timestep |
| `--max-paths` | 250 | Max (src, dst) paths recorded per snapshot for path-level labels |
| `--seed` | 42 | Master random seed for full reproducibility |
| `--output-dir` | `data/raw` | Root output directory |

---

## 14. Validating the Dataset

Run the validator before declaring data ready for Group 11:

```bash
# Full validation (both layers)
python scripts/validate_dataset.py --data-dir data/raw

# Errors only - suppress the ✓ lines
python scripts/validate_dataset.py --data-dir data/raw --quiet

# Skip temporal snapshot checks (faster, validates Layer 1 only)
python scripts/validate_dataset.py --data-dir data/raw --skip-temporal

# Validate a smoke dataset
python scripts/validate_dataset.py --data-dir outputs/smoke_dataset --quiet
```

The validator runs four phases, exiting with code `0` (pass) or `1` (errors found):

**Phase A - Directory structure and index**
All required directories exist; `topology_index.json` is present and complete; `total_topologies` matches entry count; all 5 topology types are represented; all required index fields present.

**Phase B - Multi-domain modelling (per topology)**
All required node/edge/graph attributes present in JSON, GraphML, and CSV; router type valid; coordinates in [0, 1]; all nodes have `domain_id`; `num_domains` ∈ [2, 5]; intra-domain bandwidth ∈ [100, 1000] Mbps; inter-domain bandwidth ∈ [10, 100] Mbps; `link_reliability` ∈ [0, 1]; `domain_policies` valid; CSV column order exact.

**Phase C - Static ground-truth labels**
All four load conditions present; each `utilization_fraction` matches expected value exactly; every topology edge ID appears as a key in `edge_latencies_ms`; join key integrity between topology and label files confirmed.

**Phase D - Temporal snapshots**
All T snapshot JSONs exist per topology; all required top-level keys, global feature keys, edge label keys, and path label keys present; diurnal factor confirmed to vary across timesteps (verifies temporal simulation ran); flash multiplier consistent with flash event flag; traffic profile valid; edge label count matches topology edge count; per-topology CSVs exist; `all_snapshots.csv` row count matches expected total.

---

## 15. Reproducibility

Every topology stores its `generation_seed` in all three format files and in `topology_index.json`. To reproduce any single topology exactly:

```python
import numpy as np
from gnn_routing.config import DatasetConfig
from gnn_routing.topology import generate_topology

cfg = DatasetConfig()

# Retrieve generation_seed from topology_index.json for the topology you want
generation_seed = 191664964

rng = np.random.default_rng(generation_seed)
graph = generate_topology(
    cfg=cfg,
    rng=rng,
    topology_id=1,
    topology_type="scale_free",
    generation_seed=generation_seed,
)
```

The master RNG in the full pipeline is `numpy.random.default_rng(random_seed)`, and each topology's `generation_seed` is derived from it sequentially. Re-running `generate_dataset()` with the same `--seed` reproduces the entire dataset identically.

---

## 16. Module Reference

### `config.py` - `DatasetConfig`

Central configuration dataclass. All parameters have defaults matching the spec. Call `cfg.validate()` to check parameter bounds before use.

```python
@dataclass
class DatasetConfig:
    # Reproducibility
    random_seed: int = 42

    # Topology count and type distribution
    num_topologies: int = 500
    use_distribution: bool = True     # follow 150/100/100/75/75 split

    # Topology size
    min_nodes: int = 10
    max_nodes: int = 100

    # Domain model (spec: 2–5)
    min_domains: int = 2
    max_domains: int = 5

    # Output
    output_dir: Path = Path("data/raw")

    # Layer 1 - static M/M/1 labels
    load_conditions: List[float] = [0.20, 0.50, 0.80, 0.95]
    load_condition_names: List[str] = ["low", "medium", "high", "flash"]
    avg_packet_size_mb: float = 0.001

    # Layer 2 - temporal simulation
    timesteps_per_topology: int = 12
    queue_model: str = "mm1"            # or "mg1"
    flash_crowd_probability: float = 0.08
    max_paths_per_snapshot: int = 250
```

---

### `topology.py` - `generate_topology()`

Builds a single `nx.Graph` with all required attributes. Connectivity within each domain uses a type-specific builder:

| Builder | Algorithm |
|---|---|
| `_build_random_domain` | `nx.erdos_renyi_graph`, falls back to spanning tree if disconnected |
| `_build_scale_free_domain` | `nx.barabasi_albert_graph` with `m = max(1, min(3, size//4))` |
| `_build_mesh_domain` | `nx.grid_2d_graph` trimmed to exact node count |
| `_build_ring_domain` | `nx.cycle_graph` + ~10 % random chords |
| `_build_hybrid_domain` | `nx.connected_watts_strogatz_graph` (k=4, p=0.3), falls back to cycle |

---

### `traffic.py` - `generate_traffic_matrix()`

```python
matrix, global_meta = generate_traffic_matrix(
    graph,                       # nx.Graph with domain_id, historical_traffic on nodes
    timestep=t,                  # int, 0 … T-1
    total_steps=T,               # int
    rng=rng,                     # np.random.Generator
    flash_crowd_probability=0.08 # float
)
# matrix:       np.ndarray (N, N), demand_mbps[i, j]
# global_meta:  dict with time_of_day, diurnal_factor, flash_event, flash_multiplier,
#               traffic_profile, network_load_mbps
```

---

### `latency.py` - Two modes

**Layer 1 (static):**
```python
load_data = compute_all_load_conditions(
    graph,
    load_condition_names=["low", "medium", "high", "flash"],
    load_conditions=[0.20, 0.50, 0.80, 0.95],
    avg_packet_size_mb=0.001,
)
# Returns: {"low": {"utilization_fraction": 0.20, "edge_latencies_ms": {"0": 2.51, ...}}, ...}
```

**Layer 2 (temporal):**
```python
sim_results = simulate_snapshot(
    graph,
    traffic_matrix=matrix,   # np.ndarray (N, N)
    queue_model="mm1",        # or "mg1"
    max_paths=250,
)
# Returns: {
#   "edge_rows":       List[dict],   # one per edge
#   "path_rows":       List[dict],   # one per sampled (src, dst) pair
#   "avg_utilization": float,
#   "max_utilization": float,
# }
```

---

### `exporters.py` - File writers

| Function | Output file |
|---|---|
| `write_graphml(graph, path)` | `topology_XXX.graphml` |
| `write_topology_json(graph, path)` | `topology_XXX.json` |
| `write_nodes_csv(graph, path)` | `topology_XXX_nodes.csv` |
| `write_edges_csv(graph, path)` | `topology_XXX_edges.csv` |
| `write_label_json(topo_id, load_data, path)` | `topology_XXX_labels.json` |
| `write_snapshot_json(topo_id, t, global_meta, sim_results, model, path)` | `topology_XXX_tTT_snapshot.json` |
| `append_snapshot_csv(topo_id, t, global_meta, sim_results, model, edge_csv, path_csv, global_csv)` | Per-topology snapshot CSVs |
| `append_master_snapshot_row(row, master_csv)` | `all_snapshots.csv` |
| `write_topology_index(root, entries)` | `topology_index.json` |

---

### `dataset.py` - `generate_dataset(cfg)`

Orchestrates the full pipeline. Returns the `Path` to the output root. Prints a summary on completion:

```
Dataset written to : data/raw
  Topologies       : 500
  Timesteps/topo   : 12
  Total snapshots  : 6000
  Queue model      : mm1
  Index            : data/raw/topology_index.json
  Master CSV       : data/raw/snapshots/all_snapshots.csv
```

---

## 17. Notes for Group 11

### Loading topologies

```python
import json
from pathlib import Path

root = Path("data/raw")
index = json.loads((root / "topology_index.json").read_text())

for entry in index["topologies"]:
    base = entry["filename_base"]     # e.g. "topology_001"

    # Load structure
    topo = json.loads((root / "topologies" / "json" / f"{base}.json").read_text())

    # Load static labels - join to edges by edge_id (cast label key to int)
    lbl  = json.loads((root / "labels" / f"{base}_labels.json").read_text())
    high_latency = {
        int(k): v
        for k, v in lbl["load_conditions"]["high"]["edge_latencies_ms"].items()
    }
```

### Loading temporal snapshots

```python
import pandas as pd

# All global metadata in one DataFrame (6,000 rows)
all_snaps = pd.read_csv("data/raw/snapshots/all_snapshots.csv")

# Per-topology edge metrics across all timesteps
edge_df = pd.read_csv("data/raw/snapshots/topology_001_edge_snapshots.csv")

# Single snapshot - full dynamic ground truth
snap = json.loads(
    Path("data/raw/snapshots/topology_001_t06_snapshot.json").read_text()
)
global_feats = snap["global_features"]   # time_of_day, diurnal_factor, flash_event, ...
edge_labels  = snap["edge_labels"]        # list of dicts, one per edge
path_labels  = snap["path_labels"]        # list of dicts, one per sampled path
```

### Key join points

| Key | Type in topology | Type in labels | Description |
|---|---|---|---|
| `edge_id` | `int` | `str` (dict key) | Primary join between topology edges and label files |
| `snapshot_id` | - | `str` (e.g. `"topology_001_t06"`) | Join across the three snapshot CSV types |
| `topology_id` | `int` | `int` | Links everything back to the index |

### Loading GraphML with NetworkX

```python
import networkx as nx

G = nx.read_graphml("data/raw/topologies/graphml/topology_001.graphml")

# GraphML stores all values as strings - cast as needed:
for n, d in G.nodes(data=True):
    d["domain_id"]          = int(d["domain_id"])
    d["capacity_mbps"]      = float(d["capacity_mbps"])
    d["historical_traffic"] = float(d["historical_traffic"])

for u, v, d in G.edges(data=True):
    d["bandwidth_mbps"]       = float(d["bandwidth_mbps"])
    d["is_inter_domain"]      = d["is_inter_domain"] == "true"
    d["edge_id"]              = int(d["edge_id"])
    d["propagation_delay_ms"] = float(d["propagation_delay_ms"])

# Domain policies are a JSON-encoded string in GraphML:
import json
domain_policies = json.loads(G.graph["domain_policies"])
```

### Important type caveats

- `is_inter_domain` is the string `"true"` / `"false"` in GraphML (GraphML boolean type), but a Python `bool` in JSON and CSV. Always cast when loading from GraphML.
- `domain_policies` is a JSON-encoded string in GraphML and in the NetworkX graph object; it is a parsed dict in the JSON topology file.
- Edge IDs in `edge_latencies_ms` are **string keys** in JSON (JSON object keys are always strings). Cast with `int(k)` when joining to numeric edge IDs.

---

*Group 3 - IAP Term Project · Module 1 · Dataset Generation & Ground Truth Labeling*