"""
temporal_gnn.py — Spatial-Temporal GNN for Traffic Forecasting
================================================================
Predicts per-edge latency at the NEXT timestep given k=6 timesteps of
historical traffic observations (load, utilisation, queue, bottleneck status)
per link, plus static graph structure.

Architecture (ST-GNN)
---------------------
  For each history timestep s ∈ {0 .. k-1}:
    1. Build per-node input by concatenating:
         - Static node features x       (N, 12)
         - Static global features u     (N, 11)  broadcast
         - Temporal global features u_seq[s]  (N, 5)  broadcast
       → input_proj maps (28,) → hidden_dim with tanh
    2. Apply shared GCNConv for spatial message passing → (N, hidden_dim)
    → yields per-node embeddings h_s  (N, hidden_dim) for step s

  Stack h_0 .. h_{k-1} into node sequence (N, k, hidden_dim).

  Temporal encoder: nn.TransformerEncoderLayer over the k-step sequence
  with nhead=4 attention heads, applied independently to each of the N nodes.
  Take the LAST token's output → (N, hidden_dim).

  Edge decoder:
    [h_src || h_dst || edge_attr (static) || edge_seq[:,-1,:] (last step)]
    → Linear(hidden_dim*2 + edge_dim + edge_temporal_dim → hidden_dim)
    → ReLU → Dropout
    → Linear(hidden_dim → hidden_dim//2) → ReLU
    → Linear(hidden_dim//2 → num_targets=1)
    → (M, 1)  predicted latency_ms at t+1

Key design choices
------------------
- Shared spatial encoder across k steps: topology is fixed; only traffic changes.
  Sharing GCNConv weights reflects this inductive bias.
- Transformer over GRU for time: self-attention can directly attend to a flash
  event 5 steps ago regardless of position, without vanishing-gradient concerns.
- Static + temporal global features both present at every step: ensures the
  model sees the network's structural context (size, type) at every encoding step.
- Last-step edge temporal features in decoder: the most recent traffic state is
  directly visible to the decoder alongside the aggregated temporal representation.
- Forward signature: takes entire Data/batch object (like RouteNet-Fermi) → new
  isinstance dispatch branch in run_epoch().

Input fields (from TemporalData object)
-----------------------------------------
  x               (N, 12)      Static node features
  edge_index      (2, M)       COO edge connectivity
  edge_attr       (M, 5)       Static edge features
  u               (1, 11)      Static global features (or (B, 11) when batched)
  edge_seq        (M, k, 4)    Temporal edge features per history step
  u_seq           (k, 5)       Temporal global features per history step
  batch           (N,) or None Node-to-graph assignment

Output
------
  (M, 1) predicted latency_ms at the next timestep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SpatialTemporalGNN(nn.Module):
    """Spatial-Temporal GNN for next-timestep per-edge latency forecasting.

    Args:
        node_dim:           Static node feature dimension (default 12).
        edge_dim:           Static edge feature dimension (default 5).
        global_dim:         Static global feature dimension (default 11).
        edge_temporal_dim:  Temporal edge feature dimension per step (default 4).
        global_temporal_dim: Temporal global feature dimension per step (default 5).
        hidden_dim:         Hidden channel width (default 64).
        num_targets:        Regression targets per edge (default 1).
        dropout:            Dropout probability (default 0.1).
        steps:              Number of history timesteps k (default 6).
        nhead:              Transformer attention heads (default 4).
    """

    def __init__(
        self,
        node_dim: int = 12,
        edge_dim: int = 5,
        global_dim: int = 11,
        edge_temporal_dim: int = 4,
        global_temporal_dim: int = 5,
        hidden_dim: int = 64,
        num_targets: int = 1,
        dropout: float = 0.1,
        steps: int = 6,
        nhead: int = 4,
    ):
        super().__init__()
        self.dropout = dropout
        self.steps   = steps
        self.hidden_dim = hidden_dim

        # Input: [x || u_static || u_temporal_step]
        proj_in = node_dim + global_dim + global_temporal_dim  # 12 + 11 + 5 = 28
        self.input_proj = nn.Linear(proj_in, hidden_dim)

        # Shared spatial encoder (same weights applied at each timestep)
        self.spatial_conv = GCNConv(hidden_dim, hidden_dim)

        # Temporal encoder: Transformer over k-step node sequences
        # batch_first=True: input shape (batch=N, seq=k, dim=hidden_dim)
        self.temporal_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,  # 128
            dropout=dropout,
            batch_first=True,
        )

        # Edge decoder: [h_src || h_dst || edge_attr || edge_seq_last_step]
        decoder_in = hidden_dim * 2 + edge_dim + edge_temporal_dim  # 128 + 5 + 4 = 137
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def forward(self, data):
        """
        Args:
            data: TemporalData with fields:
                  x          (N, 12)
                  edge_index (2, M)
                  edge_attr  (M, 5)
                  u          (1, 11) or (B, 11) when batched
                  edge_seq   (M, k, 4)
                  u_seq      (k, 5)
                  batch      (N,) or None

        Returns:
            (M, num_targets) predicted latency values.
        """
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        u          = data.u
        edge_seq   = data.edge_seq    # (M, k, 4)
        u_seq      = data.u_seq       # (k, 5)
        batch      = getattr(data, "batch", None)

        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Ensure u is 2-D
        if u.dim() == 1:
            u = u.unsqueeze(0)

        # Broadcast static global features to every node
        if batch is None:
            u_node = u.expand(N, -1)   # (N, 11)
        else:
            u_node = u[batch]           # (N, 11)

        # -----------------------------------------------------------------
        # Spatial encoding: one GCN pass per history timestep (shared weights)
        # u_seq[s] is (5,) per sample — broadcast to (N, 5) for each step
        # -----------------------------------------------------------------
        h_steps = []
        for s in range(self.steps):
            # u_seq shape: (k, 5) — take row s and broadcast to (N, 5)
            u_t = u_seq[s].unsqueeze(0).expand(N, -1)  # (N, 5)

            node_in = torch.cat([x, u_node, u_t], dim=-1)  # (N, 28)
            h_proj  = torch.tanh(self.input_proj(node_in))  # (N, hidden_dim)

            h_s = F.relu(self.spatial_conv(h_proj, edge_index))  # (N, hidden_dim)
            h_s = F.dropout(h_s, p=self.dropout, training=self.training)
            h_steps.append(h_s)

        # Stack into sequence: (N, k, hidden_dim)
        h_all = torch.stack(h_steps, dim=1)

        # -----------------------------------------------------------------
        # Temporal encoding via Transformer (self-attention over k steps)
        # TransformerEncoderLayer with batch_first=True expects (B, seq, dim)
        # Here B=N (each node is an independent sequence)
        # -----------------------------------------------------------------
        h_temporal = self.temporal_encoder(h_all)     # (N, k, hidden_dim)
        h_final    = h_temporal[:, -1, :]             # (N, hidden_dim) — last step

        # -----------------------------------------------------------------
        # Edge decoder
        # Concatenate: [h_src || h_dst || static edge_attr || last-step edge_seq]
        # -----------------------------------------------------------------
        h_src         = h_final[src]                  # (M, hidden_dim)
        h_dst         = h_final[dst]                  # (M, hidden_dim)
        edge_seq_last = edge_seq[:, -1, :]            # (M, 4) — most recent traffic state

        e_in = torch.cat([h_src, h_dst, edge_attr, edge_seq_last], dim=-1)  # (M, 137)
        return self.decoder(e_in)                     # (M, num_targets)
