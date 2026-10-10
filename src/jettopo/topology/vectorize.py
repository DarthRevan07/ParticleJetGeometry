"""Fixed-length vectorizations of persistence diagrams.

STUB (new in 2026). Spec: specs/10_vectorization.md

Landscapes are 1-Lipschitz in bottleneck distance, so they preserve the IRC-safety
guarantee of dtm.py. Betti curves are not stable, which is why Q3 compares them.
"""
from __future__ import annotations

import numpy as np


def persistence_landscape(dgm: np.ndarray, n_layers: int = 5, n_bins: int = 50, t_min: float = 0.0, t_max: float = 0.8) -> np.ndarray:
    """lambda_k(t) for k = 1..n_layers on n_bins evenly spaced t in [t_min, t_max], float64[n_layers * n_bins]."""
    raise NotImplementedError("persistence_landscape: see specs/10_vectorization.md")


def betti_curve(dgm: np.ndarray, n_bins: int = 50, t_min: float = 0.0, t_max: float = 0.8) -> np.ndarray:
    """Number of bars alive at each of n_bins evenly spaced t (birth <= t < death), float64[n_bins]."""
    raise NotImplementedError("betti_curve: see specs/10_vectorization.md")


def persistence_image(dgm: np.ndarray, resolution: int = 20, sigma: float = 0.02, extent: tuple[float, float, float, float] = (0.0, 0.8, 0.0, 0.8)) -> np.ndarray:
    """Persistence image in (birth, persistence) coordinates with linear weight in persistence, float64[resolution**2]."""
    raise NotImplementedError("persistence_image: see specs/10_vectorization.md")


def summary_statistics(dgm: np.ndarray) -> np.ndarray:
    """[n_bars, total persistence, max persistence, persistence entropy], float64[4]."""
    raise NotImplementedError("summary_statistics: see specs/10_vectorization.md")
