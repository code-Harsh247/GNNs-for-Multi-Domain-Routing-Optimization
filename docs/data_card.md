# Data Card — GNN Multi-Domain Routing Dataset
## Group 11 Phase 1 Output

> **Audience:** Phase 2 team (GNN model developers).
> This card documents every feature in the assembled dataset so that model
> authors know exactly what each input represents and how to interpret it.

---

## Dataset Summary

| Property | Value |
|---|---|
| Total graphs (mock / dev) | 5 |
| Total graphs (full, after Group 3 delivery) | 500+ |
| Node feature dimension | 12 |
| Edge feature dimension | 5 |
| Global feature dimension | 11 |
| Target dimension (`y_edge`) | 4 |
| File format | PyTorch Geometric `Data` objects saved via `torch.save` |
| Dataset file | `data/processed/dataset/gnn_dataset.pt` |
| Index file | `data/processed/dataset/dataset_index.json` |

---

## Node Features — `x` shape `(num_nodes, 12)`

Each node represents a router in the network topology.

| Index | Feature Name | Source | Value Range | How Computed | Why Included |
|---|---|---|---|---|---|
| 0 | `router_type_core` | Group 3 JSON | {0.0, 1.0} | One-hot: 1.0 if `router_type == "core"` | Core routers handle high-volume backbone traffic; the GNN needs to distinguish them from edge/gateway nodes |
| 1 | `router_type_edge` | Group 3 JSON | {0.0, 1.0} | One-hot: 1.0 if `router_type == "edge"` | Edge routers are access points — typically lower capacity and higher congestion risk |
| 2 | `router_type_gateway` | Group 3 JSON | {0.0, 1.0} | One-hot: 1.0 if `router_type == "gateway"` | Gateway routers bridge domains; inter-domain latency hotspots usually pass through them |
| 3 | `capacity_mbps_norm` | Group 3 JSON | [0.1, 1.0] | `capacity_mbps / 1000.0` (max = 1000 Mbps) | Router processing capacity affects how quickly packets can be forwarded; higher capacity → less queuing at the node |
| 4 | `x_coord` | Group 3 JSON | [0.0, 1.0] | Already normalised by Group 3 | Geographic position is a proxy for propagation distance between nearby nodes |
| 5 | `y_coord` | Group 3 JSON | [0.0, 1.0] | Already normalised by Group 3 | Geographic position — combined with `x_coord` gives the GNN spatial layout awareness |
| 6 | `domain_id_0` | Group 3 JSON | {0.0, 1.0} | One-hot slot 0 of 5: 1.0 if `domain_id == 0` | Domain membership determines inter/intra-domain link policies and capacity tiers |
| 7 | `domain_id_1` | Group 3 JSON | {0.0, 1.0} | One-hot slot 1 of 5: 1.0 if `domain_id == 1` | See above |
| 8 | `domain_id_2` | Group 3 JSON | {0.0, 1.0} | One-hot slot 2 of 5: 1.0 if `domain_id == 2` | See above |
| 9 | `domain_id_3` | Group 3 JSON | {0.0, 1.0} | One-hot slot 3 of 5: 1.0 if `domain_id == 3` | See above |
| 10 | `domain_id_4` | Group 3 JSON | {0.0, 1.0} | One-hot slot 4 of 5: 1.0 if `domain_id == 4` | See above. Padded with 0s for topologies with fewer than 5 domains |
| 11 | `degree_norm` | Computed (Group 11) | [0.0, 1.0] | `node_degree / max_degree_in_topology` | Highly connected nodes are potential bottlenecks; degree is a structural property the GNN cannot derive from edge_index alone without explicit signalling |

---

## Edge Features — `edge_attr` shape `(num_edges, 5)`

Each edge represents a physical network link. **Latency values are NOT stored here** — they are the prediction targets in `y_edge`. Keeping them separate prevents target leakage during training.

| Index | Feature Name | Source | Value Range | How Computed | Why Included |
|---|---|---|---|---|---|
| 0 | `bandwidth_mbps_norm` | Group 3 JSON | [0.01, 1.0] | `bandwidth_mbps / 1000.0` (max = 1000 Mbps) | Bandwidth directly determines the M/M/1 service rate; low-bandwidth links become bottlenecks under load — the GNN needs this to predict latency |
| 1 | `propagation_delay_ms_norm` | Group 3 JSON | [0.01, 0.2] | `propagation_delay_ms / 100.0` (assumed max = 100 ms) | Propagation delay is the irreducible baseline latency regardless of load; it is always part of total link latency |
| 2 | `is_inter_domain` | Group 3 JSON | {0.0, 1.0} | 1.0 if the link crosses domain boundaries, else 0.0 | Inter-domain links are capacity-constrained (10–100 Mbps vs 100–1000 Mbps intra-domain) and subject to domain transit policies |
| 3 | `link_reliability` | Group 3 JSON | [0.95, 1.0] | Already in [0, 1] — no normalisation applied | Reliability affects effective throughput; a less reliable link effectively reduces available bandwidth |
| 4 | `cost_norm` | Group 3 JSON | [0.0, 1.0] | `cost / max_cost_in_topology` (per-topology normalisation) | Static routing cost is what traditional protocols use for path selection; including it lets the GNN learn when cost is a poor proxy for actual latency |

---

## Global (Graph-Level) Features — `u` shape `(1, 11)`

One vector per topology describing the graph as a whole. Used by GNN architectures that support global context (e.g., Graph Networks with global aggregation).

| Index | Feature Name | Source | Value Range | How Computed | Why Included |
|---|---|---|---|---|---|
| 0 | `num_nodes_norm` | Group 3 JSON | [0.1, 1.0] | `num_nodes / 100.0` | Network scale affects congestion dynamics; larger networks have more alternative paths but also more potential bottlenecks |
| 1 | `num_edges_norm` | Group 3 JSON | [0.02, 1.0] | `num_edges / 500.0` | Edge density (together with `num_nodes`) characterises connectivity; sparse graphs have fewer detour options |
| 2 | `num_domains_norm` | Group 3 JSON | [0.2, 1.0] | `num_domains / 5.0` | More domains → more inter-domain links → more low-capacity bottlenecks and policy constraints |
| 3 | `avg_bandwidth_norm` | Computed (Group 11) | [0.0, 1.0] | `mean(all edge bandwidths) / 1000.0` | Average link capacity is a coarse summary of the network's throughput ceiling |
| 4 | `avg_propagation_delay_norm` | Computed (Group 11) | [0.0, 1.0] | `mean(all propagation delays) / 100.0` | Average physical delay baseline before any queuing; useful for normalising predicted latencies in context |
| 5 | `inter_domain_ratio` | Computed (Group 11) | [0.0, 1.0] | `num_inter_domain_edges / num_edges` | High inter-domain ratio = more low-capacity links = higher congestion risk overall |
| 6 | `topology_type_random` | Group 3 JSON | {0.0, 1.0} | One-hot: 1.0 if `topology_type == "random"` | Random graphs have unpredictable path lengths and no hub structure |
| 7 | `topology_type_scale_free` | Group 3 JSON | {0.0, 1.0} | One-hot: 1.0 if `topology_type == "scale_free"` | Scale-free topologies have hub nodes — high-degree nodes are bottlenecks under load |
| 8 | `topology_type_mesh` | Group 3 JSON | {0.0, 1.0} | One-hot: 1.0 if `topology_type == "mesh"` | Mesh topologies have high redundancy and short maximum path lengths |
| 9 | `topology_type_ring` | Group 3 JSON | {0.0, 1.0} | One-hot: 1.0 if `topology_type == "ring"` | Ring topologies have exactly two paths between any pair — very constrained routing |
| 10 | `topology_type_hybrid` | Group 3 JSON | {0.0, 1.0} | One-hot: 1.0 if `topology_type == "hybrid"` | Hybrid topologies combine scale-free core with ring periphery — mixed bottleneck patterns |

---

## Prediction Targets — `y_edge` shape `(num_edges, 4)`

These are the ground-truth labels the GNN is trained to predict. They are built from Group 3's M/M/1 queueing model output and are **not** included in any input feature to avoid target leakage.

| Column | Condition | Utilization | How Computed |
|---|---|---|---|
| 0 | `low` | 20% | `propagation_delay_ms + W_queue` where `ρ = 0.20` |
| 1 | `medium` | 50% | `propagation_delay_ms + W_queue` where `ρ = 0.50` |
| 2 | `high` | 80% | `propagation_delay_ms + W_queue` where `ρ = 0.80` |
| 3 | `flash` | 95% | `propagation_delay_ms + W_queue` where `ρ = 0.95` (capped to avoid ÷0) |

M/M/1 formula: $W_{queue} = \dfrac{\rho}{\mu(1 - \rho)} \times 1000$ ms, where $\mu = \dfrac{\text{bandwidth\_mbps}}{0.001}$

Values are **not normalised** in `y_edge` — the model should predict raw latency in milliseconds and the loss function should account for scale variation across load conditions.

---

## Edge Connectivity — `edge_index` shape `(2, num_edges)`

Standard PyTorch Geometric COO format. `edge_index[0]` = source node IDs, `edge_index[1]` = target node IDs. Order matches the edge list in Group 3's `topology_XXX.json` exactly, so row `i` of `edge_attr` and `y_edge` corresponds to `edge_index[:, i]`.

---

## Source Files

| Feature group | Raw source | Produced by |
|---|---|---|
| Node attributes (router_type, capacity, coords, domain_id) | `data/raw/topologies/json/topology_XXX.json` | Group 3 |
| Edge attributes (bandwidth, delay, inter_domain, reliability, cost) | `data/raw/topologies/json/topology_XXX.json` | Group 3 |
| Topology metadata (type, num_domains, etc.) | `data/raw/topology_index.json` | Group 3 |
| Latency labels (y_edge) | `data/raw/labels/topology_XXX_labels.json` | Group 3 (M/M/1 model) |
| Computed node features (degree_norm) | Derived from edge list | Group 11 `build_features.py` |
| Computed global features (avg_bandwidth, avg_delay, inter_domain_ratio) | Derived from edge list | Group 11 `build_features.py` |
| Traffic matrices | Generated stochastically per app type | Group 11 `simulate_traffic.py` |
| Enriched path labels (end-to-end latency, bottleneck IDs) | Derived via NetworkX shortest path + label join | Group 11 `enrich_labels.py` |

---

*Last updated: Phase 1 completion. Maintained by Group 11.*
