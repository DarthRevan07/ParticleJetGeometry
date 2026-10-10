"""Vietoris-Rips complexes, boundary matrices and Hodge Laplacians for one jet.

Ported from 2024 ``scnn/build_matrices.py``, ``scnn/chebyshev.normalize`` and
``coordinates_extract.EventProcessor._compute_rips``. Changes from 2024:

* Built on (dy, dphi), with ``max_edge_length`` in units of angular distance
  (the 2024 code used (p_y [GeV], phi) and a cut of 25).
* No ``sparse=`` approximation by default: a sparse Rips complex is a different
  complex, so the Laplacians changed with the sparsity parameter.
* Degenerate jets (one constituent, no edges, no triangles) return empty-but-valid
  matrices instead of raising inside ``build_laplacians``.
* Simplices are oriented by increasing vertex index. Because constituents are
  stored pT-ordered, this orientation is canonical (see specs/07_complexes.md).

Spec: specs/07_complexes.md
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def extract_simplices(simplex_tree, max_dim: int | None = None) -> list[dict[frozenset, int]]:
    """List over dimension d of {frozenset(vertices): index} (ported)."""
    top = simplex_tree.dimension() if max_dim is None else min(max_dim, simplex_tree.dimension())
    simplices: list[dict[frozenset, int]] = [dict() for _ in range(max(top, 0) + 1)]
    for simplex, _ in simplex_tree.get_skeleton(max(top, 0)):
        k = len(simplex) - 1
        if k <= top:
            simplices[k][frozenset(simplex)] = len(simplices[k])
    return simplices


def build_boundaries(simplices: list[dict[frozenset, int]]) -> list[sp.coo_matrix]:
    """Signed boundary operators B_d : C_d -> C_{d-1}, d = 1..top (ported).

    B_d has shape (n_{d-1}, n_d); entry (face, simplex) = (-1)^i where the face
    omits the i-th vertex of the sorted simplex.
    """
    boundaries = []
    for d in range(1, len(simplices)):
        rows, cols, vals = [], [], []
        for simplex, j in simplices[d].items():
            for i, left_out in enumerate(sorted(simplex)):
                rows.append(simplices[d - 1][simplex - {left_out}])
                cols.append(j)
                vals.append((-1.0) ** i)
        boundaries.append(
            sp.coo_matrix((vals, (rows, cols)), dtype=np.float32, shape=(len(simplices[d - 1]), len(simplices[d])))
        )
    return boundaries


def build_laplacians(boundaries: list[sp.spmatrix], n_vertices: int | None = None) -> list[sp.coo_matrix]:
    """Hodge Laplacians L_0..L_top (ported; handles the no-edge case).

    L_0 = B_1 B_1^T, L_d = B_d^T B_d + B_{d+1} B_{d+1}^T, L_top = B_top^T B_top.
    """
    if not boundaries:
        if n_vertices is None:
            raise ValueError("n_vertices is required when there are no edges")
        return [sp.coo_matrix((n_vertices, n_vertices), dtype=np.float32)]
    lap = [sp.coo_matrix(boundaries[0] @ boundaries[0].T)]
    for d in range(len(boundaries) - 1):
        lap.append(sp.coo_matrix(boundaries[d].T @ boundaries[d] + boundaries[d + 1] @ boundaries[d + 1].T))
    lap.append(sp.coo_matrix(boundaries[-1].T @ boundaries[-1]))
    return lap


def normalize_laplacian(L: sp.spmatrix, half_interval: bool = False) -> sp.coo_matrix:
    """Rescale the spectrum to [-1, 1] (or [0, 1]) for Chebyshev filters (ported).

    Returns 2 L / lambda_max - I. A zero Laplacian (no edges) maps to -I, which is
    what the formula gives with lambda_max := 1 (all-zero spectrum sits at -1).
    """
    L = sp.csr_matrix(L, dtype=np.float64)
    m = L.shape[0]
    if m == 0:
        return sp.coo_matrix(L)
    if L.nnz == 0:
        top = 1.0
    elif m <= 2:
        top = float(np.max(np.abs(np.linalg.eigvalsh(L.toarray()))))
    else:
        top = float(spl.eigsh(L, k=1, which="LM", return_eigenvectors=False)[0])
    top = top if top > 1e-12 else 1.0
    if half_interval:
        return sp.coo_matrix(L / top)
    return sp.coo_matrix(2.0 * L / top - sp.identity(m, format="csr"))


@dataclass
class JetComplex:
    """A jet's clique complex up to ``max_dim``: simplices, boundaries, Laplacians."""

    simplices: list[dict[frozenset, int]]
    boundaries: list[sp.coo_matrix]
    laplacians: list[sp.coo_matrix]

    @property
    def counts(self) -> list[int]:
        return [len(s) for s in self.simplices]


def rips_complex(points: np.ndarray, max_edge_length: float, max_dim: int = 2) -> JetComplex:
    """Vietoris-Rips complex of a point cloud at a fixed scale (no filtration kept).

    ``points`` are (dy, dphi) of real constituents, in pT order.
    """
    import gudhi as gd

    points = np.asarray(points, dtype=np.float64)
    st = gd.RipsComplex(points=points, max_edge_length=max_edge_length).create_simplex_tree(max_dimension=max_dim)
    simplices = extract_simplices(st, max_dim)
    if len(simplices[0]) != len(points):
        raise RuntimeError("every constituent must appear as a vertex")
    boundaries = build_boundaries(simplices)
    laplacians = build_laplacians(boundaries, n_vertices=len(points))
    return JetComplex(simplices, boundaries, laplacians)


def coo2tensor(A: sp.spmatrix):
    """scipy sparse -> torch sparse COO float32 (ported)."""
    import torch

    A = sp.coo_matrix(A)
    idx = torch.as_tensor(np.vstack((A.row, A.col)), dtype=torch.long)
    vals = torch.as_tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, size=A.shape, check_invariants=False).coalesce()
