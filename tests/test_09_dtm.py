"""specs/09_dtm_persistence.md: T9.x (stub module: XFAIL until implemented)."""
import numpy as np
import pytest

from jettopo.topology.dtm import dtm_cubical_persistence, dtm_values, make_grid
from jettopo.topology.persistence import bottleneck
from conftest import circle

R = 0.8
D = 2 * np.sqrt(2) * R  # diameter of the grid box [-R, R]^2


def _jet(seed=0, n=25):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 0.12, (n, 2)).clip(-R, R)
    w = rng.exponential(1.0, n)
    return pts, w / w.sum()


def test_T9_1_grid():
    g = make_grid(R, 40)
    assert g.shape == (1600, 2)
    assert np.all(np.abs(g) < R)
    np.testing.assert_allclose(np.unique(g[:, 0])[:2], [-R + R / 40, -R + 3 * R / 40])
    np.testing.assert_allclose(g.mean(0), [0, 0], atol=1e-12)


def test_T9_2_single_point_is_euclidean_distance():
    q = np.random.default_rng(0).uniform(-R, R, (50, 2))
    for m in (0.01, 0.3, 1.0):
        np.testing.assert_allclose(dtm_values(np.array([[0.1, -0.2]]), np.array([1.0]), q, m),
                                   np.linalg.norm(q - [0.1, -0.2], axis=1), rtol=1e-10)


def test_T9_3_two_equal_points_small_m_is_nearest_distance():
    pts = np.array([[-0.3, 0.0], [0.3, 0.0]])
    q = np.random.default_rng(1).uniform(-R, R, (50, 2))
    near = np.min(np.linalg.norm(q[:, None] - pts[None], axis=-1), axis=1)
    np.testing.assert_allclose(dtm_values(pts, np.array([0.5, 0.5]), q, 0.4), near, rtol=1e-10)


def test_T9_4_m_equal_one_is_rms_distance():
    pts, w = _jet()
    q = np.array([[0.0, 0.0], [0.2, -0.1]])
    expected = np.sqrt((w[None] * np.sum((q[:, None] - pts[None]) ** 2, -1)).sum(1))
    np.testing.assert_allclose(dtm_values(pts, w, q, 1.0), expected, rtol=1e-10)


def test_T9_5_weights_are_normalized_inside():
    pts, w = _jet()
    q = make_grid(R, 10)
    np.testing.assert_allclose(dtm_values(pts, 7 * w, q, 0.1), dtm_values(pts, w, q, 0.1), rtol=1e-12)


@pytest.mark.parametrize("m", [0.01, 0.05, 0.2])
def test_T9_6_wasserstein_stability(m):
    pts, w = _jet(2)
    delta = 0.01
    moved = pts + np.random.default_rng(3).uniform(-1, 1, pts.shape) * delta / np.sqrt(2)  # each point moves <= delta
    q = make_grid(R, 30)
    diff = np.max(np.abs(dtm_values(pts, w, q, m) - dtm_values(moved, w, q, m)))
    assert diff <= delta / np.sqrt(m) + 1e-12


@pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6])
def test_T9_7_infrared_safety_bound(eps):
    m = 0.05
    pts, w = _jet(4)
    pts2 = np.vstack([pts, [[0.7, -0.7]]])
    w2 = np.r_[(1 - eps) * w, eps]
    q = make_grid(R, 30)
    diff = np.max(np.abs(dtm_values(pts, w, q, m) - dtm_values(pts2, w2, q, m)))
    assert diff <= np.sqrt(eps) * D / np.sqrt(m) + 1e-12


def test_T9_8_exact_collinear_split_is_identical():
    pts, w = _jet(5)
    pts2 = np.vstack([pts, pts[:1]])
    w2 = np.r_[w[0] * 0.3, w[1:], w[0] * 0.7]
    q = make_grid(R, 20)
    np.testing.assert_allclose(dtm_values(pts, w, q, 0.05), dtm_values(pts2, w2, q, 0.05), rtol=1e-12)


def test_T9_9_cubical_diagrams_finite_and_ir_stable():
    m, eps = 0.05, 1e-4
    pts, w = _jet(6)
    a = dtm_cubical_persistence(pts, w, m=m, radius=R, n_grid=40)
    pts2, w2 = np.vstack([pts, [[-0.6, 0.6]]]), np.r_[(1 - eps) * w, eps]
    b = dtm_cubical_persistence(pts2, w2, m=m, radius=R, n_grid=40)
    for dim in (0, 1):
        assert np.all(np.isfinite(a[dim])) and a[dim].shape[1] == 2
        assert bottleneck(a[dim], b[dim]) <= np.sqrt(eps) * D / np.sqrt(m) + 1e-9
    assert len(a[0]) >= 1


def test_T9_10_ring_has_a_persistent_loop():
    pts = circle(60, r=0.4)
    dg = dtm_cubical_persistence(pts, np.full(60, 1 / 60), m=0.02, radius=R, n_grid=48)
    pers = dg[1][:, 1] - dg[1][:, 0]
    assert pers.max() > 0.15


def test_T9_11_quarter_turn_invariance_on_square_grid():
    pts, w = _jet(7)
    rot = pts @ np.array([[0, -1], [1, 0]]).T
    a = dtm_cubical_persistence(pts, w, m=0.05, radius=R, n_grid=40)
    b = dtm_cubical_persistence(rot, w, m=0.05, radius=R, n_grid=40)
    assert all(bottleneck(a[d], b[d]) < 1e-9 for d in (0, 1))
