"""Flexible PyG GNN wrapper (GraphSAGE / GCN / GAT) and kNN jet graphs.

Ported from 2024 ``nn/gnn.py`` and ``nn/graph_dataset.py``. Changes from 2024:

* Graphs are built on (dy, dphi), not (p_y, phi), without self-loops by default,
  and in memory: the 2024 code wrote one ``data_{idx}.pt`` file per jet (1.1 GB).
* ``knn_edge_index`` is plain torch, so it can be tested without torch_geometric.
* ``JetGNNClassifier`` adds the missing graph-level readout and a single logit.

Spec: specs/11_deep_sets_and_gnn.md
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, Module


def knn_edge_index(coords: Tensor, k: int, loop: bool = False) -> Tensor:
    """Directed kNN edges (source=neighbour, target=centre) for one point cloud, shape (2, n*k')."""
    n = coords.shape[0]
    k_eff = min(k, n if loop else n - 1)
    if k_eff <= 0:
        return torch.empty(2, 0, dtype=torch.long)
    d = torch.cdist(coords, coords)
    if not loop:
        d.fill_diagonal_(float("inf"))
    nbr = d.topk(k_eff, largest=False).indices  # (n, k)
    tgt = torch.arange(n).repeat_interleave(k_eff)
    return torch.stack([nbr.reshape(-1), tgt])


def jet_to_graph(features: np.ndarray, coords: np.ndarray, label: int, k: int = 16):
    """One jet -> torch_geometric Data(x, edge_index, y)."""
    from torch_geometric.data import Data

    x = torch.as_tensor(features, dtype=torch.float32)
    ei = knn_edge_index(torch.as_tensor(coords, dtype=torch.float32), k)
    return Data(x=x, edge_index=ei, y=torch.tensor([float(label)]))


class GNN(Module):
    """A flexible GNN model with plain_last option (ported unchanged)."""

    def __init__(self, *, conv: str, output_dim: int, hidden_dim: int = 16, num_layers: int = 2, dropout: float = 0.0,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, batch_norm: bool = False, jk: str | None = None,
                 plain_last: bool = True, **conv_kwargs):
        super().__init__()
        from torch_geometric.nn import GAT, GCN, GraphSAGE

        model_kwargs = dict(in_channels=-1, hidden_channels=hidden_dim, out_channels=output_dim, num_layers=num_layers,
                            act=activation_fn, dropout=dropout, norm="batchnorm" if batch_norm else None, jk=jk)
        if conv == "sage":
            self.model = GraphSAGE(project=False, **model_kwargs, **conv_kwargs)
        elif conv == "gcn":
            self.model = GCN(**model_kwargs, **conv_kwargs)
        elif conv == "gat":
            self.model = GAT(**model_kwargs, **conv_kwargs)
        else:
            raise NotImplementedError(f"Unknown conv type: {conv}")
        self.dropout_fn = Dropout(dropout)
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.plain_last = plain_last
        if not plain_last and batch_norm:
            self.bn = BatchNorm1d(output_dim)

    def forward(self, x: Tensor, adj_t: Tensor) -> Tensor:
        x = self.model(x, adj_t)
        if not self.plain_last:
            x = self.bn(x) if self.batch_norm else x
            x = self.dropout_fn(x)
            x = self.activation_fn(x)
        return x

    def reset_parameters(self):
        self.model.reset_parameters()
        if hasattr(self, "bn"):
            self.bn.reset_parameters()


class JetGNNClassifier(Module):
    """GNN node encoder + mean readout + linear head -> one logit per jet."""

    def __init__(self, conv: str = "sage", hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = GNN(conv=conv, output_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                           dropout=dropout, plain_last=False)
        self.head = Linear(hidden_dim, 1)

    def forward(self, data) -> Tensor:
        from torch_geometric.nn import global_mean_pool

        h = self.encoder(data.x, data.edge_index)
        return self.head(global_mean_pool(h, data.batch)).squeeze(-1)
