"""
gcn.py — GCN Baseline Model
=============================
2-layer Graph Convolutional Network followed by an edge-level MLP decoder.
Used as the baseline for comparison against GAT and MPNN.

Architecture:
  1. Prepend global features u to every node embedding (node_dim + global_dim)
  2. Two GCNConv layers with ReLU + Dropout
  3. Edge decoder: [h_src || h_dst || edge_attr] -> MLP -> 4 outputs
     (one per load condition: low, medium, high, flash)

Input fields from PyG Data object:
  x          (N, 12)   — node features
  edge_index (2, M)    — COO connectivity
  edge_attr  (M, 5)    — edge features
  u          (1, 11)   — global features (broadcast to all nodes)
  y_edge     (M, 4)    — targets (only used during training, not in forward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNLatencyPredictor(nn.Module):
    """GCN baseline for per-edge latency regression.

    Args:
        node_dim:    Number of input node features (default 12).
        edge_dim:    Number of input edge features (default 5).
        global_dim:  Number of global features (default 11).
        hidden_dim:  Hidden channel width for GCN and MLP layers (default 64).
        num_targets: Number of regression targets per edge (default 4).
        dropout:     Dropout probability applied after each GCN layer (default 0.1).
    """

    def __init__(
        self,
        node_dim: int = 12,
        edge_dim: int = 5,
        global_dim: int = 11,
        hidden_dim: int = 64,
        num_targets: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        # After concatenating global features the effective input width is:
        in_channels = node_dim + global_dim  # 23

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Edge decoder:  [h_src (hidden) || h_dst (hidden) || edge_attr (edge_dim)]
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
            u:          (B, global_dim) or (1, global_dim)  — one row per graph in batch
            batch:      (N,) graph assignment for each node (None for single graph)

        Returns:
            (M, num_targets) predicted latency values in ms.
        """
        # --- Broadcast global features to each node ---
        if batch is None:
            # Single graph: u is (1, global_dim)
            u_node = u.expand(x.size(0), -1)
        else:
            # Batched: u is (B, global_dim); index by graph assignment
            u_node = u[batch]  # (N, global_dim)

        h = torch.cat([x, u_node], dim=-1)  # (N, node_dim + global_dim)

        # --- GCN layers ---
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # --- Edge decoder ---
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]                          # (M, hidden_dim)
        h_dst = h[dst]                          # (M, hidden_dim)
        e_in  = torch.cat([h_src, h_dst, edge_attr], dim=-1)  # (M, decoder_in)

        return self.decoder(e_in)               # (M, num_targets)
