"""specs/07_complexes.md: T7.x"""
import numpy as np
import pytest

from jettopo.topology.complexes import (build_boundaries, build_laplacians, coo2tensor, normalize_laplacian,
                                        rips_complex)
from jettopo.topology.simplex_features import simplex_features

TRI = np.array([[0.0, 0.0], [0.1, 0.0], [0.05, 0.08], [0.9, 0.9]])  # triangle + far isolated point


def test_T7_1_counts():
    cx = rips_complex(TRI, max_edge_length=0.2, max_dim=2)
    assert cx.counts == [4, 3, 1]


def test_T7_2_boundary_of_boundary_is_zero():
    cx = rips_complex(TRI, 0.2)
    assert np.allclose((cx.boundaries[0] @ cx.boundaries[1]).toarray(), 0)


def test_T7_3_L0_is_graph_laplacian():
    cx = rips_complex(TRI, 0.2)
    A = np.zeros((4, 4))
    for e in cx.simplices[1]:
        i, j = sorted(e)
        A[i, j] = A[j, i] = 1
    np.testing.assert_allclose(cx.laplacians[0].toarray(), np.diag(A.sum(1)) - A)


def test_T7_4_hodge_L1_kernel_counts_holes():
    # a 4-cycle without its diagonals has beta_1 = 1 -> dim ker L1 = 1
    sq = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]], float)
    cx = rips_complex(sq, max_edge_length=0.11, max_dim=2)
    ev = np.linalg.eigvalsh(cx.laplacians[1].toarray())
    assert int(np.sum(ev < 1e-9)) == 1


def test_T7_5_single_constituent_jet_is_valid():
    cx = rips_complex(np.array([[0.0, 0.0]]), 0.2)
    assert cx.counts == [1] and cx.laplacians[0].shape == (1, 1) and cx.boundaries == []


def test_T7_6_normalized_spectrum_in_unit_interval():
    cx = rips_complex(np.random.default_rng(0).uniform(-0.3, 0.3, (30, 2)), 0.15)
    for L in cx.laplacians:
        ev = np.linalg.eigvalsh(normalize_laplacian(L).toarray())
        assert ev.min() >= -1 - 1e-6 and ev.max() <= 1 + 1e-6


def test_T7_7_build_laplacians_without_edges_needs_size():
    with pytest.raises(ValueError):
        build_laplacians([])
    assert build_laplacians([], n_vertices=3)[0].shape == (3, 3)


def test_T7_8_orientation_follows_vertex_index():
    simplices = [{frozenset([0]): 0, frozenset([1]): 1}, {frozenset([0, 1]): 0}]
    B1 = build_boundaries(simplices)[0].toarray()
    np.testing.assert_array_equal(B1[:, 0], [-1, 1])  # d[v0, v1] = v1 - v0


def test_T7_9_simplex_feature_shapes_and_values():
    cx = rips_complex(TRI, 0.2)
    z = np.array([0.4, 0.3, 0.2, 0.1])
    X = simplex_features(cx, TRI, z)
    assert [x.shape for x in X] == [(4, 1), (3, 2), (1, 3)]
    d = np.hypot(0.05, 0.08)
    np.testing.assert_allclose(sorted(X[1][:, 0]), [d, d, 0.1], atol=1e-6)  # edge lengths
    assert np.isclose(X[2][0, 1], 0.5 * 0.1 * 0.08, atol=1e-7)  # triangle area


def test_T7_10_coo2tensor_roundtrip():
    torch = pytest.importorskip("torch")
    cx = rips_complex(TRI, 0.2)
    t = coo2tensor(cx.laplacians[0])
    assert torch.allclose(t.to_dense(), torch.tensor(cx.laplacians[0].toarray(), dtype=torch.float32))
