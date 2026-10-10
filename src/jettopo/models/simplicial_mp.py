"""Energy-weighted message passing on graphs and on simplicial 2-complexes (research question Q4).

STUB (new in 2026). Spec: specs/14_q4_expressivity_probe.md

Three models with matched parameter count, all reading out sum_simplex (energy weight) * h:
  * EnergyWeightedEFN       : Deep Sets baseline (2-point at most after one layer)
  * EnergyWeightedPairMP    : messages along edges (Konar, Ngairangbam, Spannowsky 2021)
  * EnergyWeightedSimplicialMP : node <-> edge <-> triangle messages through |B_1|, |B_2|
"""
from __future__ import annotations

import torch.nn as nn

from ..topology.batching import ComplexBatch


class EnergyWeightedPairMP(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 1):
        super().__init__()
        raise NotImplementedError("EnergyWeightedPairMP: see specs/14_q4_expressivity_probe.md")

    def forward(self, batch: ComplexBatch, z):  # -> (num_graphs,)
        raise NotImplementedError


class EnergyWeightedSimplicialMP(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 1):
        super().__init__()
        raise NotImplementedError("EnergyWeightedSimplicialMP: see specs/14_q4_expressivity_probe.md")

    def forward(self, batch: ComplexBatch, z):  # -> (num_graphs,)
        raise NotImplementedError
