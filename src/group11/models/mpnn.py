"""
mpnn.py — Message Passing Neural Network Model
================================================
Edge-conditioned MPNN using NNConv, the most expressive of the three
architectures for this task. NNConv uses a small edge-feature network (ENN)
to compute message weights — directly incorporating link properties
(bandwidth, delay, reliability, cost, inter-domain flag) into the
message-passing kernel.

Architecture:
  1. Project node input (with global context) to hidden_dim
  2. Two NNConv (edge-conditioned MPNN) layers with GRU-style update
  3. Edge decoder: [h_src || h_dst || edge_attr] -> MLP -> 4 outputs

Input fields from PyG Data object:
  x          (N, 12)
  edge_index (2, M)
  edge_attr  (M, 5)
  u          (1, 11)
  y_edge     (M, 4)   — targets (not used in forward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class MPNNLatencyPredictor(nn.Module):
    """MPNN (NNConv) model for per-edge latency regression.

    Args:
        node_dim:    Number of input node features (default 12).
        edge_dim:    Number of input edge features (default 5).
        global_dim:  Number of global features (default 11).
        hidden_dim:  Hidden channel width (default 64).
        num_targets: Number of regression targets per edge (default 4).
        dropout:     Dropout probability (default 0.1).
        steps:       Number of MPNN steps (message-passing rounds) (default 3).
    """

    def __init__(
        self,
        node_dim: int = 12,
        edge_dim: int = 5,
        global_dim: int = 11,
        hidden_dim: int = 64,
        num_targets: int = 4,
        dropout: float = 0.1,
        steps: int = 3,
    ):
        super().__init__()
        self.dropout = dropout
        self.steps   = steps
        self.hidden_dim = hidden_dim

        in_channels = node_dim + global_dim  # 23

        # Input projection: raw features → hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # NNConv requires an edge-feature network (ENN) that maps
        # edge_attr -> (hidden_dim × hidden_dim) weight matrix
        # We use a 2-layer MLP per MPNN step (shared across steps)
        enn = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv = NNConv(hidden_dim, hidden_dim, enn, aggr="mean")

        # GRU cell for hidden state update (shared across steps)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Edge decoder
        decoder_in = hidden_dim * 2 + edge_dim  # 133

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def forward(self, x, edge_index, edge_attr, u, batch=None):
        """
        Args:
            x:          (N, node_dim)
            edge_index: (2, M)
            edge_attr:  (M, edge_dim)
            u:          (B, global_dim) or (1, global_dim)
            batch:      (N,) graph assignment for each node (None for single graph)

        Returns:
            (M, num_targets) predicted latency values in ms.
        """
        # --- Broadcast global features ---
        if batch is None:
            u_node = u.expand(x.size(0), -1)
        else:
            u_node = u[batch]

        h_in = torch.cat([x, u_node], dim=-1)      # (N, node_dim + global_dim)
        h    = torch.tanh(self.input_proj(h_in))    # (N, hidden_dim)

        # --- MPNN steps with GRU update ---
        for _ in range(self.steps):
            m = F.relu(self.conv(h, edge_index, edge_attr))   # (N, hidden_dim)
            m = F.dropout(m, p=self.dropout, training=self.training)
            h = self.gru(m, h)                                # (N, hidden_dim)

        # --- Edge decoder ---
        src, dst  = edge_index[0], edge_index[1]
        h_src     = h[src]
        h_dst     = h[dst]
        e_in      = torch.cat([h_src, h_dst, edge_attr], dim=-1)

        return self.decoder(e_in)  # (M, num_targets)
