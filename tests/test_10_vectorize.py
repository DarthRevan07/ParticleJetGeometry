"""specs/10_vectorization.md: T10.x (stub module: XFAIL until implemented)."""
import numpy as np

from jettopo.topology.persistence import bottleneck
from jettopo.topology.vectorize import betti_curve, persistence_image, persistence_landscape, summary_statistics

T = np.linspace(0, 0.8, 50)


def test_T10_1_landscape_shape_and_empty():
    assert persistence_landscape(np.zeros((0, 2)), n_layers=3, n_bins=50).shape == (150,)
    assert np.all(persistence_landscape(np.zeros((0, 2)), n_layers=3, n_bins=50) == 0)


def test_T10_2_single_bar_tent():
    lam = persistence_landscape(np.array([[0.2, 0.6]]), n_layers=2, n_bins=50).reshape(2, 50)
    expected = np.maximum(0, np.minimum(T - 0.2, 0.6 - T))
    np.testing.assert_allclose(lam[0], expected, atol=1e-12)
    assert np.all(lam[1] == 0)


def test_T10_3_landscape_is_1_lipschitz_in_bottleneck():
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = rng.uniform(0, 0.4, (6, 1))
        d1 = np.hstack([b, b + rng.uniform(0, 0.4, (6, 1))])
        d2 = d1 + rng.uniform(-0.02, 0.02, d1.shape)
        d2[:, 1] = np.maximum(d2[:, 1], d2[:, 0])
        gap = np.max(np.abs(persistence_landscape(d1) - persistence_landscape(d2)))
        assert gap <= bottleneck(d1, d2) + 1e-9


def test_T10_4_betti_curve_counts():
    dg = np.array([[0.0, 0.3], [0.1, 0.5], [0.4, 0.45]])
    bc = betti_curve(dg, n_bins=50)
    for t, v in zip(T, bc):
        assert v == np.sum((dg[:, 0] <= t) & (t < dg[:, 1]))


def test_T10_5_betti_curve_is_not_stable():
    # a bar of length 1e-6 changes the curve by a full count somewhere: no Lipschitz bound
    dg = np.array([[0.0, 0.3]])
    tiny = np.vstack([dg, [[T[10], T[10] + 1e-6]]])
    assert np.max(np.abs(betti_curve(dg) - betti_curve(tiny))) == 1


def test_T10_6_persistence_image_shape_nonneg():
    img = persistence_image(np.array([[0.1, 0.4], [0.2, 0.25]]), resolution=20)
    assert img.shape == (400,) and np.all(img >= 0) and img.sum() > 0
    assert np.all(persistence_image(np.zeros((0, 2)), resolution=20) == 0)


def test_T10_7_summary_statistics():
    s = summary_statistics(np.array([[0.0, 0.5], [0.1, 0.6]]))
    np.testing.assert_allclose(s, [2, 1.0, 0.5, np.log(2)], atol=1e-12)
