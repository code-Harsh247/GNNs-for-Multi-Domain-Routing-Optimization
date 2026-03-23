# Group 3 — Implementation Plan
## GNN Multi-Domain Routing: Dataset Generation & Ground Truth Labeling

> **This document is the source of truth for Group 3's deliverables.**
> Group 11 will write their code against the file formats and naming conventions defined here.
> Do not deviate from the schemas without notifying Group 11 first.

---

## Overview of Responsibilities

Group 3 is responsible for:
1. Generating 500+ synthetic network topologies
2. Modeling multi-domain (AS-like) structure on each topology
3. Computing ground truth latency labels under multiple load conditions
4. Exporting everything in formats that Group 11 can directly consume

---

## Checklist

### Phase A — Topology Generation
- [ ] Write code to generate 500+ synthetic network topologies
- [ ] Vary topology sizes: 10, 20, 30, 50, 75, and 100 nodes
- [ ] Vary topology types: random, scale-free, mesh, ring, and hybrid
- [ ] Build a config system so topologies can be reproduced with a seed value
- [ ] Export each topology in GraphML, JSON, and CSV formats

### Phase B — Multi-Domain Modeling
- [ ] Partition routers in each topology into 2–5 domains (AS-like groups)
- [ ] Assign link capacities (intra-domain: 100–1000 Mbps, inter-domain: 10–100 Mbps)
- [ ] Define domain policies — which domains allow transit through which other domains
- [ ] Store domain membership as a node attribute and domain policies as a graph attribute

### Phase C — Ground Truth Labeling
- [ ] Simulate 4 load conditions per topology: low (20%), medium (50%), high (80%), flash (95%+)
- [ ] Compute per-link latency under each load condition using M/M/1 queueing formula
- [ ] Attach latency values as edge attributes in the exported files
- [ ] Validate that all links across all topologies have labels for all 4 load conditions

---

## File Structure

All output files must be placed in the following directory structure inside the repository:

```
data/
└── raw/
    ├── topologies/
    │   ├── graphml/
    │   │   ├── topology_001.graphml
    │   │   ├── topology_002.graphml
    │   │   └── ...
    │   ├── json/
    │   │   ├── topology_001.json
    │   │   ├── topology_002.json
    │   │   └── ...
    │   └── csv/
    │       ├── topology_001_nodes.csv
    │       ├── topology_001_edges.csv
    │       └── ...
    ├── labels/
    │   ├── topology_001_labels.json
    │   ├── topology_002_labels.json
    │   └── ...
    └── topology_index.json
```

---

## Naming Conventions

### Topology Files
- Format: `topology_XXX` where `XXX` is a zero-padded 3-digit index starting from `001`
- Examples: `topology_001`, `topology_042`, `topology_500`
- Each topology must exist in **all three formats** (GraphML, JSON, CSV)

### Label Files
- Format: `topology_XXX_labels.json`
- Must correspond 1-to-1 with a topology file of the same index
- Example: `topology_001.graphml` → `topology_001_labels.json`

### CSV Files (nodes and edges are separate)
- Node file: `topology_XXX_nodes.csv`
- Edge file: `topology_XXX_edges.csv`

---

## File Format Specifications

### 1. GraphML Format (`topology_XXX.graphml`)

GraphML is the primary format. All node and edge attributes must be declared as `<key>` elements in the header.

**Required node attributes:**

| Attribute Key | Type | Description |
|---|---|---|
| `node_id` | int | Unique integer ID for the node |
| `router_type` | string | One of: `core`, `edge`, `gateway` |
| `capacity_mbps` | float | Max processing capacity in Mbps |
| `x_coord` | float | X position (0.0 to 1.0, normalized) |
| `y_coord` | float | Y position (0.0 to 1.0, normalized) |
| `domain_id` | int | Domain this router belongs to (0-indexed) |

**Required edge attributes:**

| Attribute Key | Type | Description |
|---|---|---|
| `edge_id` | int | Unique integer ID for the edge |
| `bandwidth_mbps` | float | Maximum link bandwidth in Mbps |
| `propagation_delay_ms` | float | Fixed propagation delay in milliseconds |
| `is_inter_domain` | boolean | True if the link crosses domain boundaries |
| `link_reliability` | float | Value between 0.0 and 1.0 (1.0 = perfectly reliable) |
| `cost` | float | Static routing cost metric |

**Required graph attributes:**

| Attribute Key | Type | Description |
|---|---|---|
| `topology_id` | int | Matches the XXX index in the filename |
| `num_nodes` | int | Total number of nodes |
| `num_edges` | int | Total number of edges |
| `num_domains` | int | Number of domains in this topology |
| `topology_type` | string | One of: `random`, `scale_free`, `mesh`, `ring`, `hybrid` |
| `generation_seed` | int | Random seed used — for reproducibility |
| `domain_policies` | string | JSON-encoded dict of allowed inter-domain transit paths |

**Example GraphML snippet:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/graphml">
  <key id="node_id" for="node" attr.name="node_id" attr.type="int"/>
  <key id="router_type" for="node" attr.name="router_type" attr.type="string"/>
  <key id="capacity_mbps" for="node" attr.name="capacity_mbps" attr.type="double"/>
  <key id="x_coord" for="node" attr.name="x_coord" attr.type="double"/>
  <key id="y_coord" for="node" attr.name="y_coord" attr.type="double"/>
  <key id="domain_id" for="node" attr.name="domain_id" attr.type="int"/>
  <key id="edge_id" for="edge" attr.name="edge_id" attr.type="int"/>
  <key id="bandwidth_mbps" for="edge" attr.name="bandwidth_mbps" attr.type="double"/>
  <key id="propagation_delay_ms" for="edge" attr.name="propagation_delay_ms" attr.type="double"/>
  <key id="is_inter_domain" for="edge" attr.name="is_inter_domain" attr.type="boolean"/>
  <key id="link_reliability" for="edge" attr.name="link_reliability" attr.type="double"/>
  <key id="cost" for="edge" attr.name="cost" attr.type="double"/>
  <graph id="topology_001" edgedefault="undirected">
    <node id="0">
      <data key="node_id">0</data>
      <data key="router_type">core</data>
      <data key="capacity_mbps">1000.0</data>
      <data key="x_coord">0.42</data>
      <data key="y_coord">0.77</data>
      <data key="domain_id">0</data>
    </node>
    <edge id="e0" source="0" target="1">
      <data key="edge_id">0</data>
      <data key="bandwidth_mbps">500.0</data>
      <data key="propagation_delay_ms">2.5</data>
      <data key="is_inter_domain">false</data>
      <data key="link_reliability">0.99</data>
      <data key="cost">1.0</data>
    </edge>
  </graph>
</graphml>
```

---

### 2. JSON Format (`topology_XXX.json`)

The JSON file must mirror the GraphML exactly but in a graph JSON structure.

**Required structure:**
```json
{
  "topology_id": 1,
  "num_nodes": 20,
  "num_edges": 35,
  "num_domains": 3,
  "topology_type": "random",
  "generation_seed": 42,
  "domain_policies": {
    "0": [1, 2],
    "1": [0],
    "2": [0, 1]
  },
  "nodes": [
    {
      "node_id": 0,
      "router_type": "core",
      "capacity_mbps": 1000.0,
      "x_coord": 0.42,
      "y_coord": 0.77,
      "domain_id": 0
    }
  ],
  "edges": [
    {
      "edge_id": 0,
      "source": 0,
      "target": 1,
      "bandwidth_mbps": 500.0,
      "propagation_delay_ms": 2.5,
      "is_inter_domain": false,
      "link_reliability": 0.99,
      "cost": 1.0
    }
  ]
}
```

> **Note on `domain_policies`:** Keys are domain IDs as strings. Values are lists of domain IDs that the key domain is allowed to route traffic through. Example above: domain 0 can transit through domains 1 and 2.

---

### 3. CSV Format (`topology_XXX_nodes.csv` and `topology_XXX_edges.csv`)

Two separate CSV files per topology.

**Node CSV columns (in this exact order):**
```
node_id, router_type, capacity_mbps, x_coord, y_coord, domain_id
```

**Edge CSV columns (in this exact order):**
```
edge_id, source, target, bandwidth_mbps, propagation_delay_ms, is_inter_domain, link_reliability, cost
```

---

### 4. Label File (`topology_XXX_labels.json`)

One label file per topology. Contains latency values for every edge under all 4 load conditions.

**Required structure:**
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
    "medium": {
      "utilization_fraction": 0.50,
      "edge_latencies_ms": {
        "0": 4.20,
        "1": 6.55,
        "2": 2.90
      }
    },
    "high": {
      "utilization_fraction": 0.80,
      "edge_latencies_ms": {
        "0": 12.40,
        "1": 18.20,
        "2": 7.30
      }
    },
    "flash": {
      "utilization_fraction": 0.95,
      "edge_latencies_ms": {
        "0": 89.30,
        "1": 120.50,
        "2": 45.60
      }
    }
  }
}
```

> **Important:** Edge IDs in `edge_latencies_ms` must match the `edge_id` values in the topology files exactly. Group 11 will use the edge IDs as the join key.

---

### 5. Topology Index (`topology_index.json`)

One master index file at the root of `data/raw/`. This allows Group 11 to iterate over all topologies programmatically without scanning the filesystem.

**Required structure:**
```json
{
  "total_topologies": 500,
  "topologies": [
    {
      "topology_id": 1,
      "filename_base": "topology_001",
      "num_nodes": 20,
      "num_edges": 35,
      "topology_type": "random",
      "num_domains": 3,
      "generation_seed": 42
    },
    {
      "topology_id": 2,
      "filename_base": "topology_002",
      "num_nodes": 50,
      "num_edges": 92,
      "topology_type": "scale_free",
      "num_domains": 4,
      "generation_seed": 87
    }
  ]
}
```

---

## M/M/1 Latency Formula

Use the following formula for per-link latency calculation:

```
Given:
  μ  = link service rate = bandwidth_mbps / average_packet_size_mb
  λ  = arrival rate = μ × utilization_fraction
  ρ  = λ / μ  (must be < 1)

Queueing delay (ms):
  W_queue = (ρ / (μ × (1 - ρ))) × 1000

Total link latency (ms):
  latency = propagation_delay_ms + W_queue
```

Use `average_packet_size_mb = 0.001` (1500 bytes ≈ 1 KB, a standard Ethernet frame).

For the `flash` condition where utilization approaches 1, cap `utilization_fraction` at `0.95` to prevent division by zero.

---

## Distribution of Topologies

Generate the 500+ topologies according to this distribution:

| Topology Type | Count | Node Range |
|---|---|---|
| Random | 150 | 10–100 |
| Scale-free | 100 | 20–100 |
| Mesh | 100 | 10–50 |
| Ring | 75 | 10–50 |
| Hybrid | 75 | 30–100 |

---

## Suggested Tools & Libraries

```
networkx          # Topology generation and GraphML export
graph-tool        # Optional: for large-scale generation
numpy             # Numerical computations for queueing model
pandas            # CSV export
json              # JSON export (built-in)
random / numpy.random  # Seeded random generation
```

---

## Validation Checklist Before Handoff

Before telling Group 11 that data is ready, verify:

- [ ] Every `topology_XXX.graphml` has a matching `topology_XXX_labels.json`
- [ ] Every topology has node CSV and edge CSV files
- [ ] All 4 load conditions are present in every label file
- [ ] Edge IDs in label files match edge IDs in topology files
- [ ] `topology_index.json` lists all generated topologies
- [ ] No topology has a node with a missing `domain_id`
- [ ] No edge has a missing `bandwidth_mbps` or `propagation_delay_ms`
- [ ] Generation seeds are recorded and the same seed reproduces the same topology

---

*Last updated: Phase 1 planning. Any changes to this document must be communicated to Group 11 immediately.*
