# Group 11 — Progress Report Presentation Content
## GNN Multi-Domain Routing: Data Engineering Pipeline

> **Instructions for presenter:** This file contains the full content for the Group 11 progress report PPT. Each `---` block is one slide. Speaker notes are in *italics* beneath each slide.

---

---

## SLIDE 1 — Title Slide

**Title:** Group 11: Data Engineering for GNN-Based Multi-Domain Routing

**Subtitle:** Progress Report — Phase 1 Complete

**Content:**
- Team: Group 11
- Role: Feature Engineering · Traffic Simulation · Dataset Assembly
- Date: March 2026

---

---

## SLIDE 2 — Our Role in the Project

**Title:** Where Group 11 Fits

**Content:**
```
Group 3                    Group 11                   Phase 2
─────────────────          ─────────────────          ─────────────────
Raw Network                Data Engineering           GNN Model
Topology Files      ──▶    Pipeline             ──▶   Training
                           (our work)
```

- Group 3 generates raw network topologies and per-link latency labels
- **Group 11 transforms that raw data into a GNN-ready dataset**
- Phase 2 (GNN team) will consume our final `gnn_dataset.pt` file

*Emphasize that Group 11 is the bridge — without clean, correctly formatted data, the GNN cannot train.*

---

---

## SLIDE 3 — Responsibilities Overview

**Title:** What Group 11 Was Tasked With

**Content (4 bullets):**

1. **Feature Engineering** — Extract and normalize node, edge, and graph-level features from raw topology files
2. **Traffic Simulation** — Generate realistic traffic demand matrices for 4 application types under 4 load conditions
3. **Ground Truth Enrichment** — Compute end-to-end path latencies and identify bottleneck links for all source-destination pairs
4. **Dataset Assembly** — Package everything into PyTorch Geometric `Data` objects ready for model training
5. **Validation** — Automated 35-check test script to verify all outputs are correct

---

---

---

## SLIDE 3b — Roles

**Title:** Who Did What — Group 11

**Content:**

| Member | Responsibilities |
|---|---|
| [Your Name] | Feature Engineering (`build_features.py`) · Traffic Simulation (`simulate_traffic.py`) · Validation (`validate_group11.py`) |
| [Member 2 Name] | Mock Data Generation (`generate_mock_data.py`) |
| [Member 3 Name] | Ground Truth Enrichment (`enrich_labels.py`) |
| [Member 4 Name] | Dataset Assembly (`assemble_dataset.py`) |

**Summary of contributions:**
- **Feature Engineering** — Defined and built the 3 NumPy arrays (node, edge, global) consumed by the GNN
- **Traffic Simulation** — Generated 16 demand matrices per topology with diurnal and flash crowd overlays
- **Validation** — Designed and implemented the 35-check automated test suite certifying all 500 datasets

*Update names before presenting. This slide shows clear ownership of each deliverable and ensures accountability.*

---

---

## SLIDE 4 — The Pipeline (Architecture Diagram)

**Title:** Our 4-Phase Pipeline

**Content:**

```
Phase 0                Phase A               Phase B
generate_mock_data  →  build_features    →   simulate_traffic
        │                    │                     │
        │         node (N,12)│            16 traffic│
        │         edge (M,5) │             matrices │
        │         global(11,)│                      │
        ▼                    ▼                      ▼

                      Phase C                Phase D
                   enrich_labels    →    assemble_dataset
                         │                     │
              90 paths   │          gnn_dataset│.pt
              per topo   │          (5 PyG     │Data objects)
                         ▼                     ▼

                                    validate_group11.py
                                    35 checks → exit 0
```

*Walk through left to right. Every phase consumes the previous phase's outputs. Nothing is hardcoded — everything is driven by topology_index.json.*

---

---

## SLIDE 5 — The Data We Work With

**Title:** Input Data (From Group 3)

**Content — Two columns:**

**Left: Sample of Group 3 Topologies (first 5 of 500)**
| ID | Type | Nodes | Edges | Domains | Timesteps |
|---|---|---|---|---|---|
| 001 | Random | 82 | 421 | 2 | 12 |
| 002 | Random | 51 | 98 | 3 | 12 |
| 003 | Random | 59 | 135 | 5 | 12 |
| 004 | Random | 67 | 191 | 3 | 12 |
| 005 | Random | 44 | 122 | 2 | 12 |
| ... | ... | ... | ... | ... | ... |
| **Total** | **Random** | **12–95+** | **15–421+** | **2–5** | **12** |

**Right: Per-topology inputs (× 500)**
- `topology_00X.json` — node list, edge list, coordinates, domain info
- `topology_00X_labels.json` — per-link latency × 4 load conditions
- `topology_index.json` — master registry of all 500 topologies (all scripts start here)

**Dataset scale:** 500 topologies, all randomly generated, varying in size and domain count.

*Group 3 delivered 500 real topologies. Our pipeline processed all 500 — zero code changes were needed thanks to the index-driven architecture.*

---

---

## SLIDE 6 — Phase 0: Mock Data Generation

**Title:** Phase 0 — `generate_mock_data.py`

**Why it exists:**
- Group 3 and Group 11 worked in parallel
- We needed realistic topology files to develop and test against before Group 3 delivered
- This script replicates Group 3's exact output schema

**What it generates:**
- Synthetic topologies using real graph models: Erdős–Rényi, Barabási–Albert, Grid, Cycle, Hybrid
- Node attributes: router type, capacity, coordinates, domain ID
- Edge attributes: bandwidth, propagation delay, inter-domain flag, reliability, cost
- **Per-link latency using the M/M/1 queuing model** across 4 utilization levels

**M/M/1 Formula:**
```
total_latency = propagation_delay + queuing_delay

queuing_delay = (ρ / (μ × (1 − ρ))) × 1000 ms
where:  μ = bandwidth / packet_size
        ρ = min(utilization, 0.95)
```

*Once Group 3 delivered the real 500 topologies, the pipeline ran against them without any modifications — the mock data served its purpose as a development scaffold.*

---

---

## SLIDE 7 — Phase A: Feature Engineering

**Title:** Phase A — `build_features.py`

**Purpose:** Convert raw node/edge attributes into normalized numerical arrays the GNN can process.

**Three arrays produced per topology:**

| Array | Shape | Contents |
|---|---|---|
| Node features | `(N, 12)` | Router type (one-hot), capacity, coordinates, domain (one-hot), degree |
| Edge features | `(M, 5)` | Bandwidth, propagation delay, inter-domain flag, reliability, cost |
| Global features | `(11,)` | Node/edge/domain counts, avg bandwidth, avg delay, topology type (one-hot) |

**Key design rule:** All feature values normalized to `[0, 1]`. Latency is **deliberately excluded** from edge features to prevent target leakage.

**Example — Node 8 in topology_001:**
```
router_type = "core"  →  [1, 0, 0]
capacity = 1000 Mbps  →  1.000
x=0.097, y=0.848      →  [0.097, 0.848]
domain_id = 0         →  [1, 0, 0, 0, 0]
degree = 4, max = 5   →  0.80
─────────────────────────────────────────
Feature vector: [1,0,0, 1.0, 0.097, 0.848, 1,0,0,0,0, 0.80]
```

---

---

## SLIDE 8 — Phase B: Traffic Simulation

**Title:** Phase B — `simulate_traffic.py`

**Purpose:** Generate realistic network demand to simulate different usage scenarios.

**16 Traffic Matrices per topology** (4 app types × 4 load conditions):

| App Type | Traffic Pattern | Rate Range |
|---|---|---|
| Video Streaming | Steady baseline | 5–25 Mbps |
| Web Browsing | Bursty (exponential) | 0–10 Mbps |
| File Transfer | Sustained throughput | 10–100 Mbps |
| VoIP | Tiny constant flows | 0.1–0.5 Mbps |

**Load Conditions:** Low (20%), Medium (50%), High (80%), Flash (95%)

**Additional outputs:**
- **3 Diurnal snapshots** — same matrix scaled by time-of-day multiplier (3am / 10am / 6pm)
- **Flash crowd event** — applied to topology_005: one target node receives 5× traffic spike

**Example (topology_001, video_streaming, T[0][1]):**
```
base = 12.103 Mbps
low   = 12.103 × 0.20 =  2.42 Mbps
medium= 12.103 × 0.50 =  6.05 Mbps
high  = 12.103 × 0.80 =  9.68 Mbps
flash = 12.103 × 0.95 = 11.50 Mbps
```

---

---

## SLIDE 9 — Phase C: Ground Truth Enrichment

**Title:** Phase C — `enrich_labels.py`

**Purpose:** Produce the training labels — end-to-end path latency and bottleneck links for every source→destination pair in every topology.

**Process for each (src, dst) pair:**

```
1. Find shortest path by hop count (NetworkX)
   e.g. node 0 → node 1:  path = [0, 2, 1]

2. Convert to edge IDs:
   (0,2) → edge 0    (2,1) → edge 4
   edge_ids_on_path = [0, 4]

3. Sum per-link latencies from raw labels:
   low:  19.490 + 7.934 = 27.424 ms
   flash: 19.523 + 8.213 = 27.736 ms

4. Find bottleneck (slowest link):
   edge 0 dominates in all conditions → bottleneck = edge 0
```

**Why hop-count routing?** Using latency as path weight would pre-solve the answer the GNN is supposed to learn.

**Scale:** topology_005 (30 nodes) → **870 paths × 4 conditions = 3,480 latency values**

---

---

## SLIDE 10 — Phase D: Dataset Assembly

**Title:** Phase D — `assemble_dataset.py`

**Purpose:** Merge all processed artefacts into PyTorch Geometric `Data` objects — one per topology.

**Each `Data` object contains:**

| Field | Shape | Description |
|---|---|---|
| `x` | `(N, 12)` | Node feature matrix — **model INPUT** |
| `edge_index` | `(2, M)` | COO edge connectivity — **model INPUT** |
| `edge_attr` | `(M, 5)` | Edge feature matrix — **model INPUT** |
| `u` | `(1, 11)` | Global feature vector — **model INPUT** |
| `y_edge` | `(M, 4)` | Target latencies `[low, med, high, flash]` — **model TARGET** |

**GNN Training objective:**
```
Given (x, edge_index, edge_attr, u) → predict y_edge
i.e. given topology features → predict per-link latency under all load conditions
```

**Output:** `gnn_dataset.pt` — 5 PyG Data objects, ready to load in one line:
```python
data_list = torch.load("gnn_dataset.pt")
```

---

---

## SLIDE 11 — Validation

**Title:** Automated Validation — `validate_group11.py`

**Purpose:** 35-check test script that verifies every output file before handoff to the GNN team.

**6 check categories:**

| Check | What it verifies |
|---|---|
| ✅ Check 1 | `topology_index.json` exists and is valid JSON |
| ✅ Check 2 | Feature `.npy` files: correct shapes, all values in `[0,1]`, no NaN/Inf |
| ✅ Check 3 | Traffic JSONs: all 4 app types present, all 4 load conditions present |
| ✅ Check 4 | Enriched labels: all `edge_id` references are valid, at least 1 path entry |
| ✅ Check 5 | `gnn_dataset.pt`: loads, 5 graphs, correct tensor shapes, no NaN/Inf |
| ✅ Check 6 | `dataset_index.json`: dimension fields correct, graph IDs match |

**Result:**
```
============================================================
  ALL CHECKS PASSED — Phase 1 is complete.
============================================================
```
Exit code `0` — dataset certified clean.

---

---

## SLIDE 12 — Design Decisions

**Title:** Key Technical Decisions

**1. No Target Leakage**
Latency values never appear in input features (`edge_attr`). They only exist in `y_edge`. This is the most critical ML correctness constraint.

**2. Normalized Inputs, Raw Targets**
All features are `[0,1]` normalized for stable gradient flow. Targets (`y_edge`) are kept in raw milliseconds so the model learns the actual physical scale.

**3. Hop-Count Routing in Labels**
Shortest paths are computed by hop count, not by latency. The GNN must learn which paths are fast — we don't hand it the answer.

**4. Index-Driven Architecture**
No script ever hardcodes topology IDs or scans the filesystem. All scripts begin by reading `topology_index.json`. Adding new topologies only requires updating that one file — scales to 500+ topologies with zero code changes.

**5. Seeded Reproducibility**
Every topology uses a fixed `generation_seed`. Traffic uses `seed + 1000`. The entire dataset can be regenerated identically at any time.

---

---

## SLIDE 13 — What We Delivered

**Title:** Deliverables — Phase 1 Complete

**Scripts (6 files, ~700 lines of code):**
- `generate_mock_data.py` — synthetic topology + label generation
- `build_features.py` — feature engineering (3 arrays per topology)
- `simulate_traffic.py` — 16 traffic matrices + diurnal + flash crowd
- `enrich_labels.py` — all-pairs shortest path latencies + bottleneck IDs
- `assemble_dataset.py` — final PyG dataset assembly
- `validate_group11.py` — 35-check automated validation

**Data files produced:**
- 1,500 NumPy feature arrays (`500 topologies × 3 arrays`)
- 500 traffic JSON files (16 matrices each)
- 500 enriched label JSON files
- `gnn_dataset.pt` — final dataset, 500 PyG `Data` objects
- `dataset_index.json` — metadata index

**Validation:** All 35 checks pass across all 500 topologies. Exit code 0.

---

---

## SLIDE 14 — Dataset Summary

**Title:** Final Dataset Specifications

| Metric | Value |
|---|---|
| Total graph samples | **500** |
| Topology type | Random graphs |
| Node count range | 12 – 95+ nodes per graph |
| Edge count range | 15 – 421+ edges per graph |
| Domain count range | 2 – 5 domains per graph |
| Timesteps per topology | 12 |
| Node feature dimensions | 12 |
| Edge feature dimensions | 5 |
| Global feature dimensions | 11 |
| Target dimensions | 4 (low / medium / high / flash) |
| All features normalized to | `[0, 1]` |
| Targets in | raw ms (not normalized) |
| Format | PyTorch Geometric `Data` |
| File | `gnn_dataset.pt` |

**Ready for Phase 2 GNN training.**

---

---

## SLIDE 15 — What's Next

**Title:** Handoff and Phase 2 Readiness

**What the GNN team receives:**
- `gnn_dataset.pt` — one-line load, 500 ready-to-train graph samples
- `dataset_index.json` — metadata (feature dims, graph sizes, topology types)
- `group11_study_guide.md` — full documentation of every script and file

**Immediate next steps (Phase 2):**
- GNN model architecture selection (GCN / GAT / GraphSAGE / MPNN)
- Training loop: predict `y_edge` from `(x, edge_index, edge_attr, u)`
- Evaluation: MSE/MAE on per-link latency prediction across load conditions
- Dataset of 500 real Group 3 topologies is ready — no further data work needed

**Group 11's pipeline scaled automatically** — processing all 500 topologies required only updating `topology_index.json`. No code changes were needed.

---

---

## SLIDE 16 — Thank You / Q&A

**Title:** Group 11 — Summary

**One sentence:** We built a complete, validated, reproducible data engineering pipeline that transforms raw network topology files into a PyTorch Geometric dataset ready for GNN training.

**Questions welcome.**

---

> **Presentation tips:**
> - Total slides: 17
> - Suggested timing: ~15–20 minutes
> - Slides 4, 7, 9, 10 are the most technical — spend the most time here
> - Slide 11 (validation) is a strong closing point before the handoff slide
> - Have the actual pipeline diagram from Slide 4 ready to refer back to
