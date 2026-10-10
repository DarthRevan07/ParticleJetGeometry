"""Particle Flow Network and Energy Flow Network (Komiske, Metodiev, Thaler 2019) in PyTorch.

STUB (new in 2026). Spec: specs/11_deep_sets_and_gnn.md

EnergyFlow's reference PFN/EFN are Keras models; these PyTorch versions let every
model share one trainer. Both take padded inputs with a mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PFN(nn.Module):
    """F( sum_i Phi(p_i) ) over real constituents. p_i = per-constituent features."""

    def __init__(self, in_dim: int = 7, phi_sizes: tuple[int, ...] = (100, 100, 128), f_sizes: tuple[int, ...] = (100, 100, 100)):
        super().__init__()
        raise NotImplementedError("PFN: see specs/11_deep_sets_and_gnn.md")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # x (B, P, in_dim), mask (B, P) -> (B,)
        raise NotImplementedError


class EFN(nn.Module):
    """F( sum_i z_i Phi(dy_i, dphi_i) ): IRC safe by construction."""

    def __init__(self, phi_sizes: tuple[int, ...] = (100, 100, 128), f_sizes: tuple[int, ...] = (100, 100, 100)):
        super().__init__()
        raise NotImplementedError("EFN: see specs/11_deep_sets_and_gnn.md")

    def forward(self, z: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # -> (B,)
        raise NotImplementedError
