"""Persistence diagrams of jets.

Ported from 2024 ``persistence/gudhilib/simplex_tree.py`` (``PersistenceModule``).
Changes from 2024:

* One function per jet returning diagrams, instead of a class that held every
  simplex tree for the dataset in memory.
* The 2024 multiparameter code fitted the codensity KDE on all columns but scored
  only ``event[:, :3]``; it now scores the same columns it was fitted on.
* No ``sparse=0.8`` Rips approximation by default.

Spec: specs/08_rips_persistence.md
"""
from __future__ import annotations

import numpy as np

Diagrams = dict[int, np.ndarray]


def rips_persistence(
    points: np.ndarray,
    max_edge_length: float = np.inf,
    homology_dims: tuple[int, ...] = (0, 1),
    inf_value: float | None = None,
) -> Diagrams:
    """Vietoris-Rips persistence of a point cloud.

    Returns {dim: float64[k, 2] of (birth, death)}. The essential H0 class has
    death = inf unless ``inf_value`` is given, in which case it is replaced.
    Bars of zero length are dropped.
    """
    import gudhi as gd

    points = np.asarray(points, dtype=np.float64)
    max_dim = max(homology_dims) + 1
    st = gd.RipsComplex(points=points, max_edge_length=max_edge_length).create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    out: Diagrams = {}
    for d in homology_dims:
        dgm = np.asarray(st.persistence_intervals_in_dimension(d), dtype=np.float64).reshape(-1, 2)
        if inf_value is not None:
            dgm[np.isinf(dgm[:, 1]), 1] = inf_value
        dgm = dgm[dgm[:, 1] > dgm[:, 0]]
        out[d] = dgm[np.lexsort((dgm[:, 1], dgm[:, 0]))]
    return out


def bottleneck(d1: np.ndarray, d2: np.ndarray, e: float = 0.0) -> float:
    """Bottleneck distance between two diagrams of the same dimension (gudhi)."""
    import gudhi as gd

    return float(gd.bottleneck_distance(np.asarray(d1), np.asarray(d2), e))


def multiparameter_rips_codensity(points: np.ndarray, bandwidth: float = 0.2, collapse: int = -2):
    """Two-parameter (Rips x codensity) filtration as a multipers SimplexTreeMulti.

    Port of 2024 ``compute_simplex_trees_multi``; requires the optional ``multipers``.
    Exploratory only, not part of the Q1-Q5 plan.
    """
    import gudhi as gd
    import multipers as mp
    from multipers.ml.convolutions import KDE

    points = np.asarray(points, dtype=np.float64)
    st = gd.RipsComplex(points=points).create_simplex_tree()
    st_multi = mp.SimplexTreeMulti(st, num_parameters=2)
    codensity = -KDE(bandwidth=bandwidth).fit(points).score_samples(points)
    st_multi.fill_lowerstar(codensity, parameter=1)
    st_multi.collapse_edges(collapse)
    st_multi.expansion(2)
    return st_multi
