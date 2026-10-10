"""specs/08_rips_persistence.md: T8.x (includes the symmetry lemma and Claim A of Q3)."""
import numpy as np

from jettopo.topology.persistence import bottleneck, rips_persistence
from conftest import circle


def test_T8_1_H0_deaths_on_a_line():
    pts = np.array([[0.0, 0], [1.0, 0], [3.0, 0]])
    d = rips_persistence(pts, homology_dims=(0,))[0]
    assert np.isinf(d[:, 1]).sum() == 1
    np.testing.assert_allclose(sorted(d[np.isfinite(d[:, 1]), 1]), [1.0, 2.0])


def test_T8_2_circle_has_one_long_H1_bar():
    d1 = rips_persistence(circle(40, r=0.3), homology_dims=(0, 1))[1]
    pers = d1[:, 1] - d1[:, 0]
    assert len(d1) >= 1 and pers.max() > 0.2 and np.sum(pers > 0.1) == 1


def test_T8_3_inf_value_replacement():
    d = rips_persistence(circle(10), inf_value=0.8)[0]
    assert np.all(np.isfinite(d)) and d[:, 1].max() == 0.8


def test_T8_4_invariant_to_E2_motion_and_permutation():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.4, 0.4, (25, 2))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    moved = (pts @ R.T + [0.13, -0.05])[rng.permutation(25)]
    a, b = rips_persistence(pts, inf_value=1.0), rips_persistence(moved, inf_value=1.0)
    for dim in (0, 1):
        assert bottleneck(a[dim], b[dim]) < 1e-9


def test_T8_5_stability_under_small_motion():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.4, 0.4, (30, 2))
    delta = 0.01
    noisy = pts + rng.uniform(-1, 1, pts.shape) * delta / np.sqrt(2)
    a, b = rips_persistence(pts, inf_value=1.0), rips_persistence(noisy, inf_value=1.0)
    for dim in (0, 1):
        assert bottleneck(a[dim], b[dim]) <= 2 * delta + 1e-9


def test_T8_6_claimA_collinear_duplicate_changes_nothing():
    pts = np.random.default_rng(2).uniform(-0.4, 0.4, (20, 2))
    dup = np.vstack([pts, pts[:1]])
    a, b = rips_persistence(pts, inf_value=1.0), rips_persistence(dup, inf_value=1.0)
    assert all(bottleneck(a[d], b[d]) < 1e-12 for d in (0, 1))


def test_T8_7_claimA_soft_far_emission_is_not_small():
    # unweighted Rips ignores pT, so an arbitrarily soft particle at distance 0.5 adds an H0 bar of length ~0.5
    pts = np.random.default_rng(3).uniform(-0.05, 0.05, (20, 2))
    far = np.vstack([pts, [[0.5, 0.5]]])
    a, b = rips_persistence(pts, inf_value=1.0), rips_persistence(far, inf_value=1.0)
    assert bottleneck(a[0], b[0]) > 0.2
