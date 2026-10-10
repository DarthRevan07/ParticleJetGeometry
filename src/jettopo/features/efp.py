"""Energy Flow Polynomials (Komiske, Metodiev, Thaler 2018) via the EnergyFlow package.

EFPs are indexed by multigraphs: a vertex is a factor z_i, an edge a factor
theta_ij^beta. They give (i) a linear IRC-safe baseline for Q1/Q2 and (ii) the
regression targets for the Q4 expressivity probe (e.g. the triangle graph is a
genuine 3-point correlator).

Spec: specs/06_efp_features.md
"""
from __future__ import annotations

import numpy as np

# Named multigraphs used as Q4 targets (edges between vertex labels).
PROBE_GRAPHS = {
    "dumbbell": [(0, 1)],                    # 2-point
    "open_path": [(0, 1), (1, 2)],           # 3-point, tree
    "triangle": [(0, 1), (1, 2), (2, 0)],    # 3-point, cycle
    "square": [(0, 1), (1, 2), (2, 3), (3, 0)],  # 4-point, cycle
}


def _to_ptyphi(p4: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    from ..data.preprocess import kinematics

    k = kinematics(p4, mask)
    return [np.stack([k["pt"][i][mask[i]], k["y"][i][mask[i]], k["phi"][i][mask[i]]], axis=1) for i in range(len(p4))]


def efp_set(p4: np.ndarray, mask: np.ndarray, dmax: int = 5, beta: float = 1.0, n_jobs: int = -1) -> np.ndarray:
    """All EFPs up to degree ``dmax`` (connected and disconnected), float[N, n_efps]."""
    import energyflow as ef

    efpset = ef.EFPSet(("d<=", dmax), measure="hadr", beta=beta, normed=True, coords="ptyphim")
    return np.asarray(efpset.batch_compute(_to_ptyphi(p4, mask), n_jobs=n_jobs))


def efp_graph(p4: np.ndarray, mask: np.ndarray, graph: str | list[tuple[int, int]], beta: float = 1.0) -> np.ndarray:
    """A single EFP for a named graph in PROBE_GRAPHS or an explicit edge list."""
    import energyflow as ef

    edges = PROBE_GRAPHS[graph] if isinstance(graph, str) else graph
    efp = ef.EFP(edges, measure="hadr", beta=beta, normed=True, coords="ptyphim")
    return np.asarray(efp.batch_compute(_to_ptyphi(p4, mask)))
