"""Physical input features on nodes, edges and triangles of a jet complex.

Replaces the 2024 ``torch.rand`` inputs, which meant the network saw no physics.
All features are invariant to E(2) motions of the (dy, dphi) plane except the node
coordinates themselves, which are included only when ``with_coords=True``.

Spec: specs/07_complexes.md (section "Simplex features")
"""
from __future__ import annotations

import numpy as np

from .complexes import JetComplex

EPS = 1e-12


def _ordered(simplex_dict: dict[frozenset, int]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = [()] * len(simplex_dict)
    for s, j in simplex_dict.items():
        out[j] = tuple(sorted(s))
    return out


def simplex_features(
    cx: JetComplex,
    coords: np.ndarray,
    z: np.ndarray,
    node_features: np.ndarray | None = None,
    with_coords: bool = False,
) -> list[np.ndarray]:
    """Feature matrices [X_0 (n0, c0), X_1 (n1, 2), X_2 (n2, 3)] (as many as the complex has).

    * nodes: ``node_features`` (e.g. preprocess.constituent_features without dy, dphi)
      or, if None, [log z]; optionally prefixed by (dy, dphi).
    * edges: [length dR_ij, log(z_i z_j)]
    * triangles: [perimeter, area, log(z_i z_j z_k)]
    """
    coords = np.asarray(coords, np.float64)
    z = np.asarray(z, np.float64)
    if node_features is None:
        x0 = np.log(np.maximum(z, EPS))[:, None]
    else:
        x0 = np.asarray(node_features, np.float64)
    if with_coords:
        x0 = np.concatenate([coords, x0], axis=1)
    feats = [x0.astype(np.float32)]
    if len(cx.simplices) > 1:
        edges = np.array(_ordered(cx.simplices[1]), dtype=np.int64).reshape(-1, 2)
        length = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
        logzz = np.log(np.maximum(z[edges[:, 0]] * z[edges[:, 1]], EPS))
        feats.append(np.stack([length, logzz], axis=1).astype(np.float32))
    if len(cx.simplices) > 2:
        tri = np.array(_ordered(cx.simplices[2]), dtype=np.int64).reshape(-1, 3)
        a, b, c = coords[tri[:, 0]], coords[tri[:, 1]], coords[tri[:, 2]]
        per = np.linalg.norm(a - b, axis=1) + np.linalg.norm(b - c, axis=1) + np.linalg.norm(c - a, axis=1)
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        logzzz = np.log(np.maximum(z[tri[:, 0]] * z[tri[:, 1]] * z[tri[:, 2]], EPS))
        feats.append(np.stack([per, area, logzzz], axis=1).astype(np.float32))
    return feats
