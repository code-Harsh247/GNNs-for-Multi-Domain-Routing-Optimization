"""
gat.py — Graph Attention Network Model
========================================
Multi-head GAT followed by an edge-level MLP decoder.
GAT learns to weight neighbours by attention, which is well-suited for
heterogeneous network topologies where not all neighbours are equally relevant.

Architecture:
  1. Prepend global features u to every node embedding
  2. Two GATConv layers (multi-head, concat then average)
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
from torch_geometric.nn import GATConv


class GATLatencyPredictor(nn.Module):
    """GAT model for per-edge latency regression.

    Args:
        node_dim:    Number of input node features (default 12).
        edge_dim:    Number of input edge features (default 5).
        global_dim:  Number of global features (default 11).
        hidden_dim:  Per-head hidden channel width (default 32).
        heads:       Number of attention heads (default 4).
        num_targets: Number of regression targets per edge (default 4).
        dropout:     Dropout probability (default 0.1).
    """

    def __init__(
        self,
        node_dim: int = 12,
        edge_dim: int = 5,
        global_dim: int = 11,
        hidden_dim: int = 32,
        heads: int = 4,
        num_targets: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        in_channels = node_dim + global_dim  # 23

        # Layer 1: concat multi-head → output width = hidden_dim * heads
        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True, dropout=dropout)

        # Layer 2: average multi-head → output width = hidden_dim
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, dropout=dropout)

        # Edge decoder: [h_src || h_dst || edge_attr]
        decoder_in = hidden_dim * 2 + edge_dim  # 69

        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets),
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

        h = torch.cat([x, u_node], dim=-1)  # (N, node_dim + global_dim)

        # --- GAT layers ---
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # --- Edge decoder ---
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]
        h_dst = h[dst]
        e_in  = torch.cat([h_src, h_dst, edge_attr], dim=-1)

        return self.decoder(e_in)  # (M, num_targets)
