"""Simplicial convolutional network (Ebli, Defferrard, Spreemann 2020 style).

Ported from 2024 ``scnn/scnn.py``, ``scnn/chebyshev.py`` and ``scnn/model.py``.
Changes from 2024:

* ``chebyshev_assemble`` and ``Coboundary`` are vectorized (no Python loops over
  batch and channel); tests check they equal the 2024 loop versions.
* ``SCNNClassifier`` takes real simplex features with configurable channel counts,
  processes a whole block-diagonal batch at once, mean-pools per jet and per
  dimension with an index vector, and returns ONE logit per jet for
  ``BCEWithLogitsLoss`` (2024 returned 2 logits against a BCE target).
* Jets with no triangles (or no edges) contribute a zero pooled vector for that
  dimension instead of being dropped by a try/except.

Spec: specs/13_scnn.md
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..topology.batching import ComplexBatch


def chebyshev_assemble(K: int, L: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Stack Chebyshev terms T_k(L) x for k < K. x: (B, C, M) -> (B, C, M, K)."""
    B, C, M = x.shape
    assert L.shape == (M, M) and K > 0
    x0 = x.reshape(B * C, M).T  # (M, BC)
    terms = [x0]
    if K > 1:
        terms.append(torch.sparse.mm(L, x0) if L.is_sparse else L @ x0)
    for _ in range(2, K):
        lt = torch.sparse.mm(L, terms[-1]) if L.is_sparse else L @ terms[-1]
        terms.append(2 * lt - terms[-2])
    X = torch.stack(terms, dim=-1)  # (M, BC, K)
    return X.reshape(M, B, C, K).permute(1, 2, 0, 3)


class SimplicialConvolution(nn.Module):
    """y = sum_k theta_k T_k(L) x + b (ported)."""

    def __init__(self, K: int, C_in: int, C_out: int, enable_bias: bool = True, variance: float = 1.0):
        super().__init__()
        assert C_in > 0 and C_out > 0 and K > 0
        self.K, self.C_in, self.C_out = K, C_in, C_out
        self.theta = nn.Parameter(variance * torch.randn(C_out, C_in, K))
        self.bias = nn.Parameter(torch.zeros(1, C_out, 1)) if enable_bias else None

    def forward(self, L: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.C_in
        X = chebyshev_assemble(self.K, L, x)
        y = torch.einsum("bimk,oik->bom", X, self.theta)
        return y if self.bias is None else y + self.bias


class Coboundary(nn.Module):
    """Linear map through a (co)boundary matrix D (N, M): (B, C_in, M) -> (B, C_out, N) (ported)."""

    def __init__(self, C_in: int, C_out: int, enable_bias: bool = True, variance: float = 1.0):
        super().__init__()
        self.C_in, self.C_out = C_in, C_out
        self.theta = nn.Parameter(variance * torch.randn(C_out, C_in))
        self.bias = nn.Parameter(torch.zeros(1, C_out, 1)) if enable_bias else None

    def forward(self, D: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, C, M = x.shape
        assert D.shape[1] == M and C == self.C_in
        x0 = x.reshape(B * C, M).T
        X = (torch.sparse.mm(D, x0) if D.is_sparse else D @ x0).T.reshape(B, C, D.shape[0])
        y = torch.einsum("oi,bin->bon", self.theta, X)
        return y if self.bias is None else y + self.bias


def scatter_mean(src: torch.Tensor, index: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Mean of rows of src (M, C) grouped by index (M,); empty groups give zeros."""
    out = torch.zeros(num_groups, src.shape[1], dtype=src.dtype, device=src.device)
    cnt = torch.zeros(num_groups, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    cnt.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


class SCNNClassifier(nn.Module):
    """Per-dimension Chebyshev stacks on nodes / edges / triangles, pooled and concatenated."""

    def __init__(
        self,
        in_channels: tuple[int, ...] = (5, 2, 3),
        hidden: tuple[int, ...] = (32, 64),
        K: int = 3,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
        variance: float = 0.1,
    ):
        super().__init__()
        self.n_dims = len(in_channels)
        self.stacks = nn.ModuleList()
        for c_in in in_channels:
            chans = (c_in, *hidden)
            self.stacks.append(
                nn.ModuleList(
                    SimplicialConvolution(K, a, b, enable_bias=True, variance=variance) for a, b in zip(chans[:-1], chans[1:])
                )
            )
        self.act = nn.LeakyReLU()
        self.head = nn.Sequential(
            nn.Linear(self.n_dims * hidden[-1], mlp_hidden), nn.LeakyReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1)
        )
        self.hidden_out = hidden[-1]

    def forward(self, batch: ComplexBatch) -> torch.Tensor:
        pooled = []
        for d in range(self.n_dims):
            x = batch.features[d]
            if x.shape[0] == 0:
                pooled.append(x.new_zeros(batch.num_graphs, self.hidden_out))
                continue
            h = x.T.unsqueeze(0)  # (1, C, M)
            for conv in self.stacks[d]:
                h = self.act(conv(batch.laplacians[d], h))
            pooled.append(scatter_mean(h[0].T, batch.index[d], batch.num_graphs))
        return self.head(torch.cat(pooled, dim=1)).squeeze(-1)
