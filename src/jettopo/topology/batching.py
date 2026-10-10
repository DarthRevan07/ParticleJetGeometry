"""Turn jets into complexes and batch many complexes as one block-diagonal complex.

Replaces the 2024 ``prepare_inputs`` / ``TensorDataset`` pipeline, which effectively
ran with batch size 1 and stored dense per-jet Laplacians (5.5 GB for 30k test jets).
Batching rule: Laplacians and boundaries of the jets in a batch are placed on the
block diagonal, so one sparse matmul processes the whole batch and no messages
cross between jets. ``index[d][j]`` says which jet the j-th d-simplex belongs to.

Spec: specs/13_scnn.md
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..data.preprocess import constituent_features, relative_coords
from .complexes import JetComplex, normalize_laplacian, rips_complex
from .simplex_features import simplex_features


@dataclass
class ComplexItem:
    complex: JetComplex
    features: list[np.ndarray]
    label: int


def jet_to_complex_item(
    p4: np.ndarray, mask: np.ndarray, label: int, max_edge_length: float = 0.2, max_dim: int = 2
) -> ComplexItem:
    """Build the Rips complex of one jet in (dy, dphi) with physical simplex features.

    ``p4`` is float[P, 4] and ``mask`` bool[P] for a single jet.
    """
    rc = relative_coords(p4[None], mask[None])
    sel = mask
    coords = np.stack([rc["dy"][0][sel], rc["dphi"][0][sel]], axis=1)
    z = rc["z"][0][sel]
    node = constituent_features(p4[None], mask[None])[0][sel][:, 2:]  # drop dy, dphi: keep E(2) invariance
    cx = rips_complex(coords, max_edge_length=max_edge_length, max_dim=max_dim)
    return ComplexItem(cx, simplex_features(cx, coords, z, node_features=node), int(label))


@dataclass
class ComplexBatch:
    laplacians: list  # torch sparse (M_d, M_d), normalized, block-diagonal
    boundaries: list  # torch sparse (M_{d-1}, M_d), block-diagonal; boundaries[d-1] = B_d
    features: list  # torch float (M_d, C_d)
    index: list  # torch long (M_d,)
    num_graphs: int
    y: object  # torch float (num_graphs,)

    def to(self, device):
        mv = lambda xs: [x.to(device) for x in xs]
        return ComplexBatch(mv(self.laplacians), mv(self.boundaries), mv(self.features), mv(self.index),
                            self.num_graphs, self.y.to(device))


def _block_diag(blocks: list[sp.coo_matrix], shapes: list[tuple[int, int]]) -> sp.coo_matrix:
    rows, cols, vals = [], [], []
    r0 = c0 = 0
    for b, (nr, nc) in zip(blocks, shapes):
        if b is not None and b.nnz:
            b = sp.coo_matrix(b)
            rows.append(b.row + r0)
            cols.append(b.col + c0)
            vals.append(b.data)
        r0 += nr
        c0 += nc
    if rows:
        return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, c0))
    return sp.coo_matrix((r0, c0), dtype=np.float32)


def collate_complexes(items: list[ComplexItem], max_dim: int = 2, feature_dims: tuple[int, ...] = (5, 2, 3)) -> ComplexBatch:
    """Batch complexes; dimensions a jet lacks contribute zero rows."""
    import torch

    from .complexes import coo2tensor

    n_dims = max_dim + 1
    counts = [[(it.complex.counts[d] if d < len(it.complex.counts) else 0) for d in range(n_dims)] for it in items]
    laps, bnds, feats, index = [], [], [], []
    for d in range(n_dims):
        blocks = [normalize_laplacian(it.complex.laplacians[d]) if counts[g][d] else None for g, it in enumerate(items)]
        laps.append(coo2tensor(_block_diag(blocks, [(c[d], c[d]) for c in counts])))
        f = [it.features[d] for g, it in enumerate(items) if counts[g][d]]
        feats.append(torch.as_tensor(np.concatenate(f) if f else np.zeros((0, feature_dims[d]), np.float32)))
        index.append(torch.as_tensor(np.repeat(np.arange(len(items)), [c[d] for c in counts]), dtype=torch.long))
        if d >= 1:
            bb = [it.complex.boundaries[d - 1] if counts[g][d] else None for g, it in enumerate(items)]
            bnds.append(coo2tensor(_block_diag(bb, [(c[d - 1], c[d]) for c in counts])))
    y = torch.as_tensor([it.label for it in items], dtype=torch.float32)
    return ComplexBatch(laps, bnds, feats, index, len(items), y)
