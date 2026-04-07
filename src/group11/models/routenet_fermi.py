"""
routenet_fermi.py — RouteNet-Fermi Model
=========================================
Full implementation of RouteNet-Fermi (Ferriol-Galmés et al., 2022,
arXiv:2212.12070, IEEE/ACM ToN 2023) adapted for per-edge latency
regression under four traffic load conditions.

Core idea
---------
RouteNet-Fermi operates on a **heterogeneous path-link bipartite graph**.
Unlike GCN/GAT/MPNN which only maintain node (router) hidden states,
RouteNet-Fermi maintains two sets of hidden states simultaneously:

  h_link  (M, hidden_dim)  — one state per network link (edge)
  h_path  (P, hidden_dim)  — one state per source-destination path

These states are updated for K rounds through alternating GRU-gated
scatter-mean message passing:

  Round k:
    Links ← Paths  :  aggregate path states over all paths that use each link
    Paths ← Links  :  aggregate link states over all links on each path

The path-link membership is encoded in path_link_index (2, E_pl), a COO
sparse tensor where row 0 = path indices and row 1 = edge (link) indices.

After K rounds, each link's hidden state h_link is decoded by a small MLP
to produce per-edge latency predictions under all 4 load conditions.

Architecture
------------
  1.  Link state init:  Linear(edge_dim + global_dim → hidden_dim) + tanh
  2.  Path state init:  Linear(path_dim → hidden_dim)              + tanh
  3.  K = 8 alternating steps (shared weights across steps):
        a. Links ← Paths:  mpl_p2l(h_path[path_ids]) scatter-mean → h_link(*)
           h_link = GRUCell(h_link(*), h_link)
        b. Paths ← Links:  mpl_l2p(h_link[link_ids]) scatter-mean → h_path(*)
           h_path = GRUCell(h_path(*), h_path)
        c. Dropout on both h_link and h_path
  4.  Edge decoder MLP:  [h_link || edge_attr] → hidden → hidden//2 → num_targets

Input (from RouteNetData / batched RouteNetData):
  edge_attr        (M, 5)      — link features
  u                (B, 11)     — global features (one row per graph in batch)
  path_link_index  (2, E_pl)   — path-link COO membership
  path_attr        (P, 2)      — path structural features
  num_paths        scalar/list — number of paths per graph (used for total P)
  batch            (N,)        — node-to-graph assignment (from DataLoader)

Note on batching
----------------
PyG's DataLoader automatically increments path_link_index correctly
(via RouteNetData.__inc__) when batch_size > 1, so h_link and h_path
concatenated across graphs in a batch remain correctly indexed.
However, the global feature broadcast requires mapping each edge to its
graph. We derive batch_e (edge-to-graph tensor) from batch and edge_index.
batch_size=1 is the recommended setting during training to keep the
path_batch logic simple, but batch_size>1 is also supported.

Reference
---------
Ferriol-Galmés et al. "RouteNet-Fermi: Network Modeling with Graph Neural
Networks." IEEE/ACM Transactions on Networking, 2023.
https://arxiv.org/abs/2212.12070
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class RouteNetFermiPredictor(nn.Module):
    """RouteNet-Fermi for per-edge latency regression.

    Args:
        edge_dim:    Number of input edge (link) features (default 5).
        path_dim:    Number of input path features (default 2).
        global_dim:  Number of global graph features (default 11).
        hidden_dim:  Hidden channel width for all layers (default 64).
        num_targets: Number of regression targets per edge (default 4).
        dropout:     Dropout probability after each MP step (default 0.1).
        steps:       Number of alternating message-passing rounds K (default 8).
    """

    def __init__(
        self,
        edge_dim:    int   = 5,
        path_dim:    int   = 4,
        global_dim:  int   = 11,
        hidden_dim:  int   = 64,
        num_targets: int   = 4,
        dropout:     float = 0.1,
        steps:       int   = 8,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.dropout     = dropout
        self.steps       = steps

        # ------------------------------------------------------------------ #
        # State initialisation networks
        # ------------------------------------------------------------------ #
        # Link state: edge features + global context (broadcast per edge)
        self.link_init = nn.Linear(edge_dim + global_dim, hidden_dim)

        # Path state: structural path features only
        self.path_init = nn.Linear(path_dim, hidden_dim)

        # ------------------------------------------------------------------ #
        # Message networks (shared across all K steps)
        # ------------------------------------------------------------------ #
        # Paths → Links: transform path state before aggregating onto link
        self.mpl_p2l = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Links → Paths: transform link state before aggregating onto path
        self.mpl_l2p = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------ #
        # GRU update cells (shared across all K steps)
        # ------------------------------------------------------------------ #
        self.gru_link = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_path = nn.GRUCell(hidden_dim, hidden_dim)

        # ------------------------------------------------------------------ #
        # Edge (link) decoder
        # ------------------------------------------------------------------ #
        # Input: [h_link || edge_attr_raw]
        decoder_in = hidden_dim + edge_dim  # 69

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #
    def forward(self, data):
        """
        Args:
            data: RouteNetData or batched RouteNetData from DataLoader.
                  Required fields: edge_index, edge_attr, u, path_link_index,
                                   path_attr, num_paths, batch (optional).

        Returns:
            (M, num_targets) predicted latency values in ms.
        """
        edge_index      = data.edge_index        # (2, M)
        edge_attr       = data.edge_attr         # (M, 5)
        u               = data.u                 # (B, 11) or (1, 11)
        path_link_index = data.path_link_index   # (2, E_pl)
        path_attr       = data.path_attr         # (P, 2)
        batch_node      = getattr(data, "batch", None)  # (N,) or None

        M = edge_attr.size(0)  # number of links
        P = path_attr.size(0)  # number of paths

        path_ids = path_link_index[0]  # (E_pl,)
        link_ids = path_link_index[1]  # (E_pl,)

        # ------------------------------------------------------------------
        # Derive batch_e: edge-to-graph assignment (for global broadcast)
        # ------------------------------------------------------------------
        if batch_node is None:
            # Single graph (no DataLoader batching)
            batch_e = torch.zeros(M, dtype=torch.long, device=edge_attr.device)
        else:
            # Use source node's graph assignment as the edge's graph assignment
            batch_e = batch_node[edge_index[0]]  # (M,)

        # Ensure u is 2-D: (B, global_dim)
        if u.dim() == 1:
            u = u.unsqueeze(0)

        # ------------------------------------------------------------------
        # Initialise hidden states
        # ------------------------------------------------------------------
        # Link state: concat edge features with per-edge global context
        u_edge  = u[batch_e]                                       # (M, 11)
        link_in = torch.cat([edge_attr, u_edge], dim=-1)           # (M, 16)
        h_link  = torch.tanh(self.link_init(link_in))              # (M, hidden)

        # Path state: structural path features only
        h_path  = torch.tanh(self.path_init(path_attr))            # (P, hidden)

        # ------------------------------------------------------------------
        # K alternating message-passing steps
        # ------------------------------------------------------------------
        # Pre-compute per-link path count for normalised sum aggregation.
        # Raw sum preserves the congestion signal (more paths = more load)
        # but with all-pairs routing the fan-in can be O(N²), causing GRU
        # saturation.  Dividing by sqrt(count) keeps the scale bounded while
        # still encoding relative load (a link with 4× paths gets 2× signal).
        path_count = scatter(
            torch.ones(link_ids.size(0), device=edge_attr.device),
            link_ids, dim=0, dim_size=M, reduce="sum"
        ).clamp(min=1.0)                                            # (M,)
        sqrt_count = path_count.sqrt().unsqueeze(-1)               # (M, 1)

        for _ in range(self.steps):
            # --- Links ← Paths ---
            # Normalised sum: preserves congestion signal, avoids saturation.
            msgs_p2l = self.mpl_p2l(h_path[path_ids])             # (E_pl, hidden)
            agg_link  = scatter(msgs_p2l, link_ids, dim=0,
                                dim_size=M, reduce="sum")           # (M, hidden)
            agg_link  = agg_link / sqrt_count                      # normalise
            h_link    = self.gru_link(agg_link, h_link)            # (M, hidden)

            # --- Paths ← Links ---
            # Each path aggregates (mean) transformed link states over all
            # links it traverses.
            msgs_l2p = self.mpl_l2p(h_link[link_ids])             # (E_pl, hidden)
            agg_path  = scatter(msgs_l2p, path_ids, dim=0,
                                dim_size=P, reduce="mean")          # (P, hidden)
            h_path    = self.gru_path(agg_path, h_path)            # (P, hidden)

            # Dropout after each full round
            h_link = F.dropout(h_link, p=self.dropout, training=self.training)
            h_path = F.dropout(h_path, p=self.dropout, training=self.training)

        # ------------------------------------------------------------------
        # Decode link states → per-edge latency predictions
        # ------------------------------------------------------------------
        dec_in = torch.cat([h_link, edge_attr], dim=-1)            # (M, hidden+5)
        return self.decoder(dec_in)                                 # (M, num_targets)
