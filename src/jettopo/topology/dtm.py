"""IRC-safe persistence: distance-to-measure (DTM) sublevel filtration on a fixed grid.

STUB (new in 2026; the core of research question Q3). Spec: specs/09_dtm_persistence.md

The jet is the probability measure mu = sum_i z_i delta_{x_i} on the (dy, dphi) disk
of radius R, with z_i = pT_i / sum pT. For a mass parameter m in (0, 1],

    d_{mu,m}(x)^2 = (1/m) * integral_0^m delta_{mu,t}(x)^2 dt,

where delta_{mu,t}(x) is the smallest radius r with mu(B(x, r)) >= t. For a weighted
point cloud this is: sort points by distance to x, accumulate weights until mass m
is reached, and average squared distances with those weights (the last point
contributing only the remaining fraction).

Guarantee to be tested: ||d_mu - d_nu||_inf <= m^{-1/2} W_2(mu, nu), hence the
sublevel-set persistence diagrams on a FIXED grid move by at most that much in
bottleneck distance.
"""
from __future__ import annotations

import numpy as np

Diagrams = dict[int, np.ndarray]


def make_grid(radius: float, n: int) -> np.ndarray:
    """Cell centres of an n x n grid covering [-radius, radius]^2, float64[n*n, 2] in row-major order."""
    raise NotImplementedError("make_grid: see specs/09_dtm_persistence.md")


def dtm_values(points: np.ndarray, weights: np.ndarray, queries: np.ndarray, m: float) -> np.ndarray:
    """DTM d_{mu,m} of the weighted point measure evaluated at ``queries``, float64[Q].

    ``weights`` are non-negative and are normalized to sum to 1 inside.
    """
    raise NotImplementedError("dtm_values: see specs/09_dtm_persistence.md")


def dtm_cubical_persistence(
    points: np.ndarray,
    weights: np.ndarray,
    m: float = 0.03,
    radius: float = 0.8,
    n_grid: int = 40,
    homology_dims: tuple[int, ...] = (0, 1),
) -> Diagrams:
    """Sublevel-set persistence of d_{mu,m} on the fixed grid (gudhi CubicalComplex).

    Returns {dim: float64[k, 2]}; the essential H0 bar's death is set to the grid
    maximum of d_{mu,m} so all values are finite.
    """
    raise NotImplementedError("dtm_cubical_persistence: see specs/09_dtm_persistence.md")
