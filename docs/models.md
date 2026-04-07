# GNN Models for Multi-Domain Routing Optimization

This document describes the three Graph Neural Network (GNN) models developed for per-edge latency prediction across multi-domain network topologies. All three models share a common input schema and a unified edge-level regression objective.

---

## Task Overview

Each model takes a **network topology represented as a graph** and predicts the **per-edge latency** (in milliseconds) under four traffic load conditions:

| Target Index | Load Condition |
|:---:|:---|
| 0 | Low |
| 1 | Medium |
| 2 | High |
| 3 | Flash (burst) |

---

## Shared Input Schema

All models consume a PyTorch Geometric `Data` object with the following fields:

| Field | Shape | Description |
|:---|:---:|:---|
| `x` | `(N, 12)` | Node features |
| `edge_index` | `(2, M)` | COO-format edge connectivity |
| `edge_attr` | `(M, 5)` | Edge features |
| `u` | `(1, 11)` | Global (graph-level) features, broadcast to all nodes |
| `y_edge` | `(M, 4)` | Ground-truth latency targets (training only) |

### Node Features — 12 dimensions

| Index | Feature | Description |
|:---:|:---|:---|
| 0 | `router_type_core` | One-hot: is a core router |
| 1 | `router_type_edge` | One-hot: is an edge router |
| 2 | `router_type_gateway` | One-hot: is a gateway router |
| 3 | `capacity_mbps_norm` | Router capacity normalized by 1000 |
| 4 | `x_coord` | Normalized x position in [0, 1] |
| 5 | `y_coord` | Normalized y position in [0, 1] |
| 6–10 | `domain_id_onehot` | One-hot domain ID (5 slots) |
| 11 | `degree_norm` | Node degree / max degree in topology |

### Edge Features — 5 dimensions

| Index | Feature | Description |
|:---:|:---|:---|
| 0 | `bandwidth_mbps_norm` | Link bandwidth / 1000 |
| 1 | `propagation_delay_ms_norm` | Propagation delay / 100 |
| 2 | `is_inter_domain` | 1.0 if the link crosses a domain boundary |
| 3 | `link_reliability` | Reliability score in [0, 1] |
| 4 | `cost_norm` | Link cost / max cost in topology |

### Global Features — 11 dimensions

| Index | Feature | Description |
|:---:|:---|:---|
| 0 | `num_nodes_norm` | Number of nodes / 100 |
| 1 | `num_edges_norm` | Number of edges / 500 |
| 2 | `num_domains_norm` | Number of domains / 5 |
| 3 | `avg_bandwidth_norm` | Mean link bandwidth / 1000 |
| 4 | `avg_propagation_delay_norm` | Mean propagation delay / 100 |
| 5 | `inter_domain_ratio` | Fraction of inter-domain edges |
| 6–10 | `topology_type_onehot` | One-hot over {random, scale_free, mesh, ring, hybrid} |

---

## Model 1 — GCN: Graph Convolutional Network

**File:** [src/group11/models/gcn.py](../src/group11/models/gcn.py)  
**Class:** `GCNLatencyPredictor`  
**Role:** Baseline model

### Overview

The GCN baseline uses standard **Graph Convolutional Network** layers ([Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907)). Each node aggregates information from its direct neighbours with uniform (normalized) weighting. It provides a simple, computationally efficient baseline against which the more expressive GAT and MPNN models are compared.

### Architecture

```
Input: x (N,12), edge_index (2,M), edge_attr (M,5), u (1,11)

1. Global Context Injection
   h = [x || u_broadcast]          → (N, 23)

2. GCN Layer 1
   h = ReLU(GCNConv(h, edge_index)) → (N, 64)
   h = Dropout(p=0.1)

3. GCN Layer 2
   h = ReLU(GCNConv(h, edge_index)) → (N, 64)
   h = Dropout(p=0.1)

4. Edge Decoder (MLP)
   e_in = [h_src || h_dst || edge_attr]   → (M, 133)
   Linear(133→64) → ReLU → Dropout
   Linear(64→32)  → ReLU
   Linear(32→4)                           → (M, 4)
```

### Key Design Choices

- **Global feature injection:** The 11-dimensional global feature vector `u` is broadcast and concatenated to every node embedding before message passing, giving each node awareness of the overall topology context.
- **Uniform aggregation:** GCNConv aggregates neighbour messages using symmetric normalized adjacency weights — all neighbours are treated equally.
- **Edge decoder:** After message passing, source and destination node embeddings are concatenated with the raw edge features to produce per-edge predictions.

### Hyperparameters

| Parameter | Value |
|:---|:---:|
| `hidden_dim` | 64 |
| `dropout` | 0.1 |
| `num_targets` | 4 |

---

## Model 2 — GAT: Graph Attention Network

**File:** [src/group11/models/gat.py](../src/group11/models/gat.py)  
**Class:** `GATLatencyPredictor`  
**Role:** Attention-based model

### Overview

The GAT model uses **Graph Attention Network** layers ([Veličković et al., 2018](https://arxiv.org/abs/1710.10903)). Unlike GCN, GAT learns a set of **attention coefficients** that dynamically weight each neighbour's contribution during aggregation. This is particularly beneficial for heterogeneous network topologies where nodes have varying numbers of neighbours (e.g., gateway nodes vs. leaf edge routers) and where the importance of a neighbour depends on the local structural context.

### Architecture

```
Input: x (N,12), edge_index (2,M), edge_attr (M,5), u (1,11)

1. Global Context Injection
   h = [x || u_broadcast]               → (N, 23)

2. GAT Layer 1  (4 heads, concat)
   h = ELU(GATConv(h, edge_index,
           heads=4, concat=True))        → (N, 128)   [32 × 4]
   h = Dropout(p=0.1)

3. GAT Layer 2  (4 heads, average)
   h = ELU(GATConv(h, edge_index,
           heads=4, concat=False))       → (N, 32)
   h = Dropout(p=0.1)

4. Edge Decoder (MLP)
   e_in = [h_src || h_dst || edge_attr]  → (M, 69)
   Linear(69→64) → ReLU → Dropout
   Linear(64→32) → ReLU
   Linear(32→4)                          → (M, 4)
```

### Key Design Choices

- **Multi-head attention (Layer 1, concat):** The first layer runs 4 independent attention heads and concatenates their outputs, expanding the representation space to `32 × 4 = 128` dimensions, capturing diverse relational patterns.
- **Averaged attention (Layer 2):** The second layer also uses 4 heads but averages their outputs, producing a compact 32-dimensional embedding that integrates the multi-head perspectives.
- **ELU activation:** Exponential Linear Unit (ELU) is used instead of ReLU to handle the smooth, learned attention-weighted aggregations more gracefully.
- **Heterogeneous topology handling:** Attention scores let the model learn that a gateway router's neighbours matter differently than a core router's neighbours.

### Hyperparameters

| Parameter | Value |
|:---|:---:|
| `hidden_dim` (per head) | 32 |
| `heads` | 4 |
| `dropout` | 0.1 |
| `num_targets` | 4 |

---

## Model 3 — MPNN: Message Passing Neural Network

**File:** [src/group11/models/mpnn.py](../src/group11/models/mpnn.py)  
**Class:** `MPNNLatencyPredictor`  
**Role:** Most expressive model

### Overview

The MPNN uses **NNConv** ([Gilmer et al., 2017](https://arxiv.org/abs/1704.01212)), an edge-conditioned message passing layer. Unlike GCN and GAT, which treat edge features only at the decoder stage, NNConv incorporates edge attributes **directly into the message-passing kernel**. A small edge-feature network (ENN) maps each edge's 5-dimensional feature vector to a full `hidden_dim × hidden_dim` weight matrix, enabling the model to modulate messages by link properties such as bandwidth, delay, and the inter-domain flag.

An additional **GRU cell** provides a recurrent update to the hidden state across multiple message-passing steps, giving the model temporal depth without increasing the number of distinct convolutional layers.

### Architecture

```
Input: x (N,12), edge_index (2,M), edge_attr (M,5), u (1,11)

1. Global Context Injection
   h_in = [x || u_broadcast]             → (N, 23)

2. Input Projection
   h = tanh(Linear(23→64))               → (N, 64)

3. MPNN Steps  ×3  (shared weights)
   For each step:
     ENN: Linear(5→64) → ReLU → Linear(64→64×64)
     m = ReLU(NNConv(h, edge_index, edge_attr))  → (N, 64)
     m = Dropout(p=0.1)
     h = GRUCell(m, h)                           → (N, 64)

4. Edge Decoder (MLP)
   e_in = [h_src || h_dst || edge_attr]   → (M, 133)
   Linear(133→64) → ReLU → Dropout
   Linear(64→32)  → ReLU
   Linear(32→4)                           → (M, 4)
```

### Key Design Choices

- **Edge-conditioned kernels (ENN):** The edge network maps `edge_attr ∈ ℝ⁵ → ℝ^{64×64}`, producing a unique linear transformation per edge. This means the message from node `v` to node `u` through edge `e` is explicitly shaped by `e`'s properties (e.g., a high-bandwidth edge produces a different message than a bottleneck inter-domain link).
- **GRU-based recurrent update:** Rather than stacking independent layers, the same `NNConv` + `GRUCell` block is applied for `steps=3` rounds with shared weights. The GRU gate mechanism controls information flow, preventing the over-smoothing that can degrade deep GCN stacks.
- **Input projection with tanh:** A linear projection from 23 → 64 dimensions with `tanh` activation initializes the hidden state `h` before the recurrent steps, ensuring the GRU has a well-scaled starting point.
- **Mean aggregation:** NNConv uses mean pooling over incoming edge-conditioned messages, making it robust to varying node degrees.

### Hyperparameters

| Parameter | Value |
|:---|:---:|
| `hidden_dim` | 64 |
| `steps` | 3 |
| `dropout` | 0.1 |
| `num_targets` | 4 |

---

## Training Setup

All three models are trained with identical hyperparameters for fair comparison:

| Setting | Value |
|:---|:---:|
| Dataset split | 400 train / 50 val / 50 test topologies |
| Batch size | 16 graphs (1 for RouteNet-Fermi — see below) |
| Epochs | 50 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Loss function | MSE (Mean Squared Error) |
| Random seed | 42 |

Checkpoints are saved to `data/processed/checkpoints/<model>_best.pt` based on the lowest validation loss.

**Training command:**
```bash
python src/group11/training/train.py --model gcn
python src/group11/training/train.py --model gat
python src/group11/training/train.py --model mpnn
python src/group11/training/train.py --model routenet
python src/group11/training/train.py --model custom
python src/group11/training/train.py --model tgnn
```

> **RouteNet-Fermi** requires the augmented dataset to be built first:
> ```bash
> python src/group11/dataset_assembly/augment_dataset.py
> ```
> RouteNet uses `batch_size=1` because the number of paths varies per graph, making path-index batching non-trivial. Each graph is a complete training sample.

> **SpatialTemporalGNN (tgnn)** requires the temporal dataset to be built first:
> ```bash
> python src/group11/dataset_assembly/build_temporal_dataset.py
> ```
> This assembles 3,000 temporal forecasting samples (500 topologies × 6 sliding windows of K=6 history steps) from the 6,000 raw snapshot JSON files in `data/raw/snapshots/`. The tgnn also uses `batch_size=1` because the number of edges `M` varies per topology and the temporal tensors `edge_seq (M,K,4)` cannot be naively stacked across topologies of different sizes.

---

## Model 4 — RouteNet-Fermi

**File:** [src/group11/models/routenet_fermi.py](../src/group11/models/routenet_fermi.py)  
**Class:** `RouteNetFermiPredictor`  
**Role:** Heterogeneous path-link GNN (most structurally expressive)  
**Reference:** Ferriol-Galmés et al., [arXiv:2212.12070](https://arxiv.org/abs/2212.12070), IEEE/ACM ToN 2023

### Overview

RouteNet-Fermi is fundamentally different from the three models above. Instead of operating solely on a node-edge graph, it constructs a **heterogeneous bipartite graph** between two entity types:

- **Links** (network edges) — `M` entities with a hidden state `h_link ∈ ℝ^d`
- **Paths** (source-destination routes) — `P` entities with a hidden state `h_path ∈ ℝ^d`

The membership relationship (which links belong to which paths) is encoded in `path_link_index (2, E_pl)` — a COO sparse tensor generated from the pre-computed enriched labels at `data/processed/labels/topology_XXX_enriched_labels.json`.

Both sets of hidden states are updated alternately through `K = 8` rounds of GRU-gated scatter message passing, allowing the model to reason about how traffic on one path affects the latency of shared links — which is the core physical phenomenon in routing optimization.

### Dataset Augmentation

RouteNet requires additional data fields not present in `gnn_dataset.pt`. The script [src/group11/dataset_assembly/augment_dataset.py](../src/group11/dataset_assembly/augment_dataset.py) reads the existing dataset and enriched labels to produce `routenet_dataset.pt` with three new fields per graph:

| Field | Shape | Description |
|:---|:---:|:---|
| `path_link_index` | `(2, E_pl)` | COO: row 0 = path index, row 1 = edge index |
| `path_attr` | `(P, 2)` | Per-path features: `[path_length_norm, is_cross_domain]` |
| `num_paths` | scalar | Number of src-dst paths in the graph |

The `RouteNetData` subclass (defined in `augment_dataset.py`) overrides `__inc__` so PyG's DataLoader correctly auto-increments both path and link indices during batch collation.

### Architecture

```
Input: edge_attr (M,5), u (B,11), path_link_index (2,E_pl), path_attr (P,2)

1. Link State Initialisation
   u_edge  = u[batch_e]                           (M, 11)
   h_link  = tanh(Linear([edge_attr || u_edge]))  (M, 64)   ← global context here

2. Path State Initialisation
   h_path  = tanh(Linear(path_attr))              (P, 64)

3. K=8 Alternating Message-Passing Steps  (shared weights)
   For each step:
     a. Links ← Paths
        msgs   = mpl_p2l(h_path[path_ids])        (E_pl, 64)
        m_link = scatter_mean(msgs, link_ids, M)   (M, 64)
        h_link = GRUCell(m_link, h_link)           (M, 64)

     b. Paths ← Links
        msgs   = mpl_l2p(h_link[link_ids])        (E_pl, 64)
        m_path = scatter_mean(msgs, path_ids, P)   (P, 64)
        h_path = GRUCell(m_path, h_path)           (P, 64)

     c. Dropout(p=0.1) on both h_link and h_path

4. Edge Decoder  (MLP)
   dec_in = [h_link || edge_attr]                 (M, 69)
   Linear(69→64) → ReLU → Dropout
   Linear(64→32) → ReLU
   Linear(32→4)                                   (M, 4)
```

### Key Design Choices

- **Explicit path entities:** Paths are first-class citizens in the computation graph. The model can directly learn how traffic on one path influences the state of shared links — something no single-graph homogeneous GNN can represent.
- **Global context in link init only:** The 11-dimensional global feature vector `u` is broadcast to each edge at initialisation time (before any message passing). Path states are initialised from structural features only, since no per-path traffic demand is available as input.
- **Shared GRU weights across K steps:** A single `GRUCell` is reused for all 8 rounds of link updates, and another for all 8 rounds of path updates. This gives the model recurrent depth without multiplying parameters.
- **Scatter-mean aggregation:** Consistent with the original paper; mean pooling is robust to paths of varying length and links with varying path-load counts.
- **`batch_size=1` training:** Because the number of paths `P` varies between topologies, batching requires a `path_batch` tensor for correct scatter indexing. RouteNet-Fermi uses `batch_size=1` to avoid this complexity while remaining fully correct. Each of the 500 topologies is an independent training sample.

### Hyperparameters

| Parameter | Value |
|:---|:---:|
| `hidden_dim` | 64 |
| `steps` (K) | 8 |
| `dropout` | 0.1 |
| `path_dim` | 2 |
| `num_targets` | 4 |

---

---

## Model 5 — Custom GNN with Edge-Weight Learning

**File:** [src/group11/models/edge_gnn.py](../src/group11/models/edge_gnn.py)  
**Class:** `EdgeWeightGNN`  
**Role:** Interpretable per-link importance gating

### Overview

`EdgeWeightGNN` introduces an **Edge Weight Network (EWN)** that computes a scalar importance gate $\alpha_e \in (0, 1)$ for every link in the topology. Unlike GAT (which derives attention weights from node-pair representations) or MPNN (which derives a full 64×64 matrix per edge), the EWN conditions on both **link-level features** `edge_attr (M, 5)` and **global topology context** `u_edge (M, 11)` to produce a lightweight, interpretable scalar that directly controls how much each link's message contributes to aggregation. After every forward pass, the weights are stored in `self.last_edge_weights` for per-topology analysis.

### Architecture

```
Input: x (N,12), edge_index (2,M), edge_attr (M,5), u (1,11)

1. Global Context Injection
   u_node = u_broadcast                           → (N, 11)
   u_edge = u_broadcast                           → (M, 11)
   h = tanh(Linear([x || u_node]))                → (N, 64)

2. Edge Weight Network (EWN)
   ewn_in = [edge_attr || u_edge]                 → (M, 16)
   α_e = Sigmoid(Linear(32→1)(ReLU(Linear(16→32)(ewn_in))))  → (M, 1)
   self.last_edge_weights = α_e                   ← stored for interpretability

3. K=3 Recurrent Steps  (shared weights)
   For each step:
     m = ReLU(msg_net(h[src]))                    → (M, 64)   messages from source nodes
     m_weighted = α_e * m                          → (M, 64)   gated by learned importance
     agg = scatter_mean(m_weighted, dst, N)        → (N, 64)   aggregate to destinations
     h = GRUCell(agg, h)                           → (N, 64)   recurrent state update
     h = Dropout(p=0.1)

4. Edge Decoder (MLP)
   e_in = [h_src || h_dst || edge_attr]           → (M, 133)
   Linear(133→64) → ReLU → Dropout
   Linear(64→32)  → ReLU
   Linear(32→4)                                   → (M, 4)
```

### Key Design Choices

- **Context-aware edge gates:** The EWN takes `[edge_attr || u_edge]` (16-dim) as input, so edge importance is conditioned on both the link's local properties (bandwidth, delay, inter-domain flag) and the global topology state (size, domain count, topology type). An inter-domain link in a large hybrid topology will receive a different gate value than the same link in a small ring topology.
- **Scalar vs. matrix:** MPNN's ENN produces a 64×64 matrix per edge (~4K values). The EWN produces a single scalar — far fewer parameters, and directly interpretable as a link importance score.
- **Sigmoid activation:** Output values are strictly in (0, 1), making them readable as probability-like importance weights. A value near 1 means the model relies heavily on that link; near 0 means it suppresses it.
- **`last_edge_weights` attribute:** Stored after every `forward()` call (detached from the computation graph). Enables post-hoc analysis: which links does the model consider important for a given topology? Are inter-domain links consistently weighted higher?
- **GRU recurrent update:** Same pattern as MPNN — shared weights across K=3 steps prevent over-smoothing and allow the gated signal to propagate iteratively through the graph.
- **Forward signature matches GCN/GAT/MPNN:** `forward(x, edge_index, edge_attr, u, batch=None)` — no `run_epoch()` dispatch changes required.

### Hyperparameters

| Parameter | Value |
|:---|:---:|
| `hidden_dim` | 64 |
| `steps` (K) | 3 |
| `dropout` | 0.1 |
| `ewn_hidden` | 32 |
| `num_targets` | 4 |

---

## Model 6 — Spatial-Temporal GNN (ST-GNN)

**File:** [src/group11/models/temporal_gnn.py](../src/group11/models/temporal_gnn.py)  
**Class:** `SpatialTemporalGNN`  
**Role:** Temporal traffic forecasting — predict next-timestep per-edge latency  
**Dataset:** `data/processed/dataset/temporal_dataset.pt` (3,000 `TemporalData` objects)

### Overview

`SpatialTemporalGNN` is a **forecasting model** — unlike Models 1–5 which predict latency under four static load conditions, ST-GNN takes a **sequence of K=6 historical traffic snapshots** for a topology and predicts the **per-edge latency at the next (7th) timestep**. This models the network operator's real-time inference problem: given the last 6 observations of link load, utilisation, queue depth, and bottleneck status, predict where congestion will occur next.

The architecture combines:
1. **GCNConv** (shared across time) for spatial message passing within each snapshot, and
2. **Transformer self-attention** over the K-step node sequence for temporal modelling.

The Transformer is preferred over a GRU for temporal encoding because self-attention can attend directly to any past snapshot — including a flash event 5 steps ago — without suffering from vanishing gradients.

### Dataset: TemporalData Fields

Each `TemporalData` sample extends a PyG `Data` object with the following additional fields:

| Field | Shape | Description |
|:---|:---:|:---|
| `x` | `(N, 12)` | Static node features (same as other models) |
| `edge_index` | `(2, M)` | Static COO edge connectivity |
| `edge_attr` | `(M, 5)` | Static edge features |
| `u` | `(1, 11)` | Static global features |
| `edge_seq` | `(M, K, 4)` | Per-edge temporal features over K=6 history steps |
| `u_seq` | `(K, 5)` | Global temporal features over K=6 history steps |
| `y_next` | `(M, 1)` | Target: latency_ms at the next (K+1-th) timestep |
| `y_edge` | `(M, 4)` | Carried from the static dataset (not used as training target) |

#### Temporal Edge Features — 4 dimensions per step

| Index | Feature | Description |
|:---:|:---|:---|
| 0 | `load_norm` | Link load / link bandwidth (utilisation proxy) |
| 1 | `utilization` | Direct utilisation value from snapshot |
| 2 | `queue_norm` | Queue length (packets) / 100 |
| 3 | `is_bottleneck` | Binary flag: link identified as bottleneck |

#### Temporal Global Features — 5 dimensions per step

| Index | Feature | Description |
|:---:|:---|:---|
| 0 | `time_of_day_norm` | Hour of day / 24 |
| 1 | `diurnal_factor` | Multiplicative diurnal traffic scaling |
| 2 | `flash_event` | Binary: flash-crowd event active |
| 3 | `flash_multiplier_norm` | Flash multiplier / 5 |
| 4 | `network_load_norm` | Total network load (Mbps) / 10,000 |

### Architecture

```
Input per sample:
  x          (N, 12)     Static node features
  edge_index (2, M)      COO connectivity
  edge_attr  (M, 5)      Static edge features
  u          (1, 11)     Static global features
  edge_seq   (M, K, 4)   Temporal edge features  (K=6)
  u_seq      (K, 5)      Temporal global features (K=6)

─── Spatial Encoding (shared weights across K steps) ───────────────────────

For each history step s ∈ {0 .. K-1}:
  1. Build per-node input:
       node_in = [x || u_broadcast || u_seq[s].expand(N,-1)]  → (N, 28)
                  12         11               5
  2. Input projection:
       h_proj = tanh(Linear(28→64))                           → (N, 64)
  3. Spatial message passing:
       h_s    = ReLU(GCNConv(h_proj, edge_index))             → (N, 64)
       h_s    = Dropout(p=0.1)

Stack → h_all = [h_0 || h_1 || ... || h_{K-1}]               → (N, K, 64)

─── Temporal Encoding (Transformer self-attention) ─────────────────────────

  h_temporal = TransformerEncoderLayer(h_all)                 → (N, K, 64)
               (d_model=64, nhead=4, dim_feedforward=128,
                dropout=0.1, batch_first=True)
  h_final    = h_temporal[:, -1, :]                           → (N, 64)
               (last token = "present" state after attending to history)

─── Edge Decoder ────────────────────────────────────────────────────────────

  e_in = [h_final[src] || h_final[dst] || edge_attr || edge_seq[:,-1,:]]
                   64             64            5               4
  e_in                                                         → (M, 137)

  Linear(137→64) → ReLU → Dropout
  Linear(64→32)  → ReLU
  Linear(32→1)                                                 → (M, 1)
```

### Key Design Choices

- **Shared spatial encoder across K steps:** Network topology is fixed; only traffic changes between timesteps. Sharing `GCNConv` weights across steps reflects this inductive bias and dramatically reduces parameters compared to K independent spatial layers.
- **Static + temporal global at every step:** Both the topology-level context `u (1,11)` and the timestep-specific conditions `u_seq[s] (5,)` are fed as node inputs at each step. This ensures the spatial encoder knows both what kind of network it is processing and what the global traffic conditions were at each timestep.
- **Transformer over GRU for temporal depth:** Self-attention with nhead=4 over K=6 steps allows direct attention to any past event (e.g., a flash-crowd onset 5 steps ago). GRU would need to carry that signal through all intermediate steps.
- **Last-step edge features in decoder:** `edge_seq[:, -1, :]` — the most recent traffic observation — is concatenated directly into the decoder input alongside the aggregated temporal representation `h_final`. This ensures the decoder always sees the raw current state in addition to the temporal summary.
- **Forecasting vs. regression task shift:** Unlike Models 1–5 which see static per-condition targets `y_edge (M,4)`, the ST-GNN targets `y_next (M,1)` — latency at the next real timestep. Training loss is `MSE(pred, batch.y_next)`.
- **`batch_size=1`:** Like RouteNet-Fermi, the variable-`M` temporal tensors `edge_seq (M,K,4)` cannot be zero-padded and batched without a mask, so each topology is processed independently.

### Hyperparameters

| Parameter | Value |
|:---|:---:|
| `node_dim` | 12 |
| `edge_dim` | 5 |
| `global_dim` | 11 |
| `edge_temporal_dim` | 4 |
| `global_temporal_dim` | 5 |
| `hidden_dim` | 64 |
| `steps` (K) | 6 |
| `nhead` | 4 |
| `dropout` | 0.1 |
| `num_targets` | 1 |

---

## Architecture Comparison

| Feature | GCN | GAT | MPNN | RouteNet-Fermi | EdgeWeightGNN | ST-GNN |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Aggregation type | Uniform (normalized) | Learned attention | Edge-conditioned kernel | Scatter-mean (path↔link) | **Scalar-gated mean** | Uniform (normalized) |
| Uses edge features in message passing | No | No | **Yes** | **Yes (at init)** | **Yes (gate)** | No (in decoder) |
| Attention mechanism | No | **Yes** | No | No | No | **Yes (Transformer)** |
| Recurrent state update | No | No | **Yes (GRU)** | **Yes (GRU ×2)** | **Yes (GRU)** | No |
| Temporal modelling | No | No | No | No | No | **Yes (K=6 steps)** |
| Explicit path entities | No | No | No | **Yes** | No | No |
| Number of MP layers / steps | 2 | 2 | 3 (shared) | 8 (shared) | 3 (shared) | 6 (shared, spatial) |
| Node embedding dim (final) | 64 | 32 | 64 | 64 (link) | 64 | 64 |
| Graph type | Node-edge | Node-edge | Node-edge | Path-link bipartite | Node-edge | Node-edge |
| Edge weight interpretability | No | No | No | No | **Yes (α_e)** | No |
| Task | Static regression | Static regression | Static regression | Static regression | Static regression | **Temporal forecasting** |
| Output shape | `(M, 4)` | `(M, 4)` | `(M, 4)` | `(M, 4)` | `(M, 4)` | **`(M, 1)`** |
| Expressiveness | Baseline | Medium | High | Highest | Medium–High | High (spatial+temporal) |
| Relative complexity | Low | Medium | High | Very High | Medium | High |

---

## Output

**Models 1–5** output a tensor of shape `(M, 4)`, where `M` is the number of edges and the 4 columns correspond to predicted latency (ms) for the **low**, **medium**, **high**, and **flash** load conditions respectively.

**Model 6 (ST-GNN)** outputs a tensor of shape `(M, 1)` — predicted latency (ms) at the **next timestep** given K=6 historical observations. The model is evaluated with `--model tgnn` and reports a single condition (`next_step`) in the evaluation tables.
