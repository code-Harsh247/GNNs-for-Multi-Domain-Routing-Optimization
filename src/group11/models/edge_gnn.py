"""
edge_gnn.py — Custom GNN with Edge-Weight Learning
====================================================
Introduces an Edge Weight Network (EWN) that learns a scalar gate
α_e ∈ (0, 1) per link from [edge_attr || u_edge] — a 16-dimensional
joint representation of link properties and global topology context.
The scalar modulates how much each source-node message is weighted
before scatter-mean aggregation, giving the model an interpretable,
per-link importance score.

Architecture:
  1. Broadcast global features u to every node and edge
  2. Project node input (with global context) → hidden_dim via tanh
  3. Compute per-edge scalar gate α_e via EWN (Sigmoid output)
  4. K=3 recurrent steps (shared weights):
       a. Compute messages from source nodes: m_e = ReLU(msg_net(h_src))
       b. Gate messages:  m_e_weighted = α_e * m_e
       c. Scatter-mean gated messages to destination nodes
       d. Update hidden state via GRUCell
  5. Edge decoder: [h_src || h_dst || edge_attr] → MLP → 4 outputs

Interpretability:
  After every forward() call, self.last_edge_weights holds the α_e
  tensor (M, 1) for the most recent graph, with values in (0, 1).
  High values indicate links the model considers structurally important
  for latency prediction in that topology.

Input fields from PyG Data object:
  x          (N, 12)
  edge_index (2, M)
  edge_attr  (M, 5)
  u          (1, 11)  — or (B, 11) when batched
  y_edge     (M, 4)   — targets (not used in forward)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter


class EdgeWeightGNN(nn.Module):
    """Custom GNN with context-aware per-edge scalar gating for latency regression.

    Args:
        node_dim:    Number of input node features (default 12).
        edge_dim:    Number of input edge features (default 5).
        global_dim:  Number of global features (default 11).
        hidden_dim:  Hidden channel width (default 64).
        num_targets: Number of regression targets per edge (default 4).
        dropout:     Dropout probability (default 0.1).
        steps:       Number of recurrent message-passing rounds (default 3).
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
        self.steps = steps

        in_channels = node_dim + global_dim          # 23

        # Node input projection → hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Edge Weight Network (EWN): [edge_attr || u_edge] → scalar gate α_e
        ewn_in = edge_dim + global_dim               # 16
        self.ewn = nn.Sequential(
            nn.Linear(ewn_in, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),                            # output in (0, 1)
        )

        # Message network: maps source hidden state → message (shared across K steps)
        self.msg_net = nn.Linear(hidden_dim, hidden_dim)

        # GRU cell for recurrent state update (shared across K steps)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Edge decoder: [h_src || h_dst || edge_attr] → num_targets
        decoder_in = hidden_dim * 2 + edge_dim       # 133
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_targets),
        )

        # Stores the most recently computed edge importance weights (M, 1)
        self.last_edge_weights: Optional[Tensor] = None

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
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # --- Broadcast global features to nodes and edges ---
        if batch is None:
            u_node = u.expand(N, -1)                           # (N, global_dim)
            u_edge = u.expand(edge_attr.size(0), -1)           # (M, global_dim)
        else:
            u_node = u[batch]                                   # (N, global_dim)
            u_edge = u[batch[src]]                              # (M, global_dim)

        # --- Node input projection ---
        h = torch.tanh(self.input_proj(torch.cat([x, u_node], dim=-1)))  # (N, hidden_dim)

        # --- Edge Weight Network: compute scalar α_e per link ---
        alpha = self.ewn(torch.cat([edge_attr, u_edge], dim=-1))  # (M, 1)
        self.last_edge_weights = alpha.detach()                    # store for interpretability

        # --- K recurrent message-passing steps (shared weights) ---
        for _ in range(self.steps):
            # Compute messages from source nodes
            m = F.relu(self.msg_net(h[src]))                   # (M, hidden_dim)

            # Gate messages by learned per-edge scalar
            m_weighted = alpha * m                             # (M, hidden_dim)

            # Aggregate gated messages at destination nodes (mean)
            agg = scatter(m_weighted, dst, dim=0, dim_size=N, reduce="mean")  # (N, hidden_dim)

            # GRU recurrent state update
            h = self.gru(agg, h)                               # (N, hidden_dim)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # --- Edge decoder ---
        e_in = torch.cat([h[src], h[dst], edge_attr], dim=-1)  # (M, 133)
        return self.decoder(e_in)                               # (M, num_targets)
