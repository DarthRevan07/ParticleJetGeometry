"""specs/15_q3_irc_and_generator_shift.md: T15.x (stub module: XFAIL until implemented)."""
import numpy as np

from jettopo.data.preprocess import jet_kinematics, relative_coords
from jettopo.experiments.irc import add_soft_emission, collinear_split, emd
from conftest import massless_p4


def _jet():
    p4 = np.zeros((8, 4))
    mask = np.zeros(8, bool)
    p4[:4] = massless_p4([300, 150, 100, 20], [0.1, 0.4, -0.2, 0.3], [0.0, 0.3, 0.5, -0.4])
    mask[:4] = True
    return p4, mask


def test_T15_1_soft_emission_adds_one_particle_where_asked():
    p4, mask = _jet()
    jet = jet_kinematics(p4[None], mask[None])
    q4, qm = add_soft_emission(p4, mask, 1e-3, dy=0.5, dphi=-0.3)
    assert qm.sum() == mask.sum() + 1 and np.array_equal(p4, _jet()[0])  # input untouched
    new = np.flatnonzero(qm & ~mask)[0]
    pt_new = np.hypot(q4[new, 1], q4[new, 2])
    np.testing.assert_allclose(pt_new, 1e-3 * jet["pt"][0], rtol=1e-6)
    y_new = 0.5 * np.log((q4[new, 0] + q4[new, 3]) / (q4[new, 0] - q4[new, 3]))
    np.testing.assert_allclose(y_new - jet["y"][0], 0.5, atol=1e-6)


def test_T15_2_exact_collinear_split_conserves_four_momentum():
    p4, mask = _jet()
    q4, qm = collinear_split(p4, mask, index=0, fraction=0.3)
    assert qm.sum() == mask.sum() + 1
    np.testing.assert_allclose(q4[qm].sum(0), p4[mask].sum(0), rtol=1e-9)
    rc = relative_coords(q4[None], qm[None])
    a, b = 0, np.flatnonzero(qm & ~mask)[0]
    np.testing.assert_allclose([rc["dy"][0, a], rc["dphi"][0, a]], [rc["dy"][0, b], rc["dphi"][0, b]], atol=1e-9)


def test_T15_3_split_with_angle_separates_daughters():
    p4, mask = _jet()
    q4, qm = collinear_split(p4, mask, index=1, fraction=0.5, angle=0.02)
    rc = relative_coords(q4[None], qm[None])
    b = np.flatnonzero(qm & ~mask)[0]
    sep = np.hypot(rc["dy"][0, 1] - rc["dy"][0, b], rc["dphi"][0, 1] - rc["dphi"][0, b])
    np.testing.assert_allclose(sep, 0.02, rtol=1e-3)


def test_T15_4_emd_properties():
    p4, mask = _jet()
    assert emd(p4, mask, p4, mask) < 1e-9
    q4, qm = add_soft_emission(p4, mask, 1e-4, dy=0.5, dphi=0.5)
    d = emd(p4, mask, q4, qm)
    assert 0 < d < 1e-3
    assert np.isclose(d, emd(q4, qm, p4, mask))
