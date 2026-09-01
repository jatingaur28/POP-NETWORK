"""
Production Graph Attention Network (GAT) for Predictive Routing (The Oracle).
Learns dynamic attention weights across the physical router topology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class HardwareGAT(nn.Module):
    """
    Replaces static GCN with a dynamic Graph Attention Network.
    Computes attention coefficients to prioritize severely degraded links
    within the physical network topology.
    """
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, alpha: float = 0.2):
        super(HardwareGAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # Linear projection layers
        self.W = nn.Linear(in_features, hidden_dim, bias=False)
        self.a = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.fc_out = nn.Linear(hidden_dim, out_features)

        # Caching for blazing fast inference
        self.register_buffer('cached_adj', None)
        self.topology_hash: Optional[int] = None

    def _compute_adjacency_matrix(self, node_links_map: Dict[str, List[str]], link_to_idx: Dict[str, int], device: torch.device) -> torch.Tensor:
        """Builds the raw, unnormalized physical adjacency matrix."""
        n = len(link_to_idx)
        adj = torch.eye(n, device=device)
        for links in node_links_map.values():
            for src in links:
                for dst in links:
                    if src != dst and src in link_to_idx and dst in link_to_idx:
                        adj[link_to_idx[src], link_to_idx[dst]] = 1.0
        return adj

    def forward(self, x: torch.Tensor, node_links_map: Dict[str, List[str]], link_to_idx: Dict[str, int]) -> torch.Tensor:
        """
        Executes a localized attention pass over live telemetry tensor 'x'.
        """
        current_hash = hash(frozenset(link_to_idx.keys()))
        if self.cached_adj is None or self.topology_hash != current_hash:
            adj_matrix = self._compute_adjacency_matrix(node_links_map, link_to_idx, x.device)
            self.register_buffer('cached_adj', adj_matrix)
            self.topology_hash = current_hash

        N = x.size(0)
        h = self.W(x)  # [N, hidden_dim]

        # Broadcast capabilities for self-attention
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N, -1)
        e = F.leaky_relu(self.a(a_input).squeeze(2), self.alpha)

        # Masked attention (Only attend to physical neighbors dictated by the cache)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(self.cached_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=0.2, training=self.training)

        # Node feature update based on attention weights
        h_prime = torch.matmul(attention, h)
        return self.fc_out(F.elu(h_prime))