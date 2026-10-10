"""specs/02_preprocessing.md: T2.x"""
import numpy as np

from jettopo.data.preprocess import (FEATURE_NAMES, constituent_features, jet_kinematics, kinematics,
                                     longitudinal_boost, ptyphi_to_p4, relative_coords, wrap_phi)
from conftest import massless_p4


def test_T2_1_rapidity_and_phi_of_massless_particle():
    p4 = massless_p4([50.0], [0.7], [-1.2])[None]
    k = kinematics(p4, np.array([[True]]))
    np.testing.assert_allclose([k["y"][0, 0], k["eta"][0, 0], k["phi"][0, 0], k["pt"][0, 0]], [0.7, 0.7, -1.2, 50.0], atol=1e-9)


def test_T2_2_dphi_wraps_across_pi():
    p4 = massless_p4([100.0, 100.0], [0.0, 0.0], [np.pi - 0.01, -np.pi + 0.01])[None]
    rc = relative_coords(p4, np.array([[True, True]]))
    assert np.all(np.abs(rc["dphi"]) < 0.02)
    assert np.all(wrap_phi(np.array([3 * np.pi, -3 * np.pi + 1e-9])) <= np.pi)


def test_T2_3_z_sums_to_one_and_masked_entries_zero(toy_jets):
    p4, mask, _, _ = toy_jets
    rc = relative_coords(p4, mask)
    np.testing.assert_allclose(rc["z"].sum(1), 1.0)
    for v in rc.values():
        assert np.all(v[~mask] == 0)


def test_T2_4_dy_invariant_under_longitudinal_boost(toy_jets):
    p4, mask, _, _ = toy_jets
    a = relative_coords(p4, mask)
    b = relative_coords(np.where(mask[..., None], longitudinal_boost(p4, 0.6), 0), mask)
    np.testing.assert_allclose(a["dy"], b["dy"], atol=1e-8)
    np.testing.assert_allclose(a["dphi"], b["dphi"], atol=1e-8)


def test_T2_5_jet_mass_of_two_massless_particles():
    # m^2 = 2 pt1 pt2 (cosh dy - cos dphi)
    p4 = massless_p4([100.0, 50.0], [0.1, -0.2], [0.0, 0.4])[None]
    m = jet_kinematics(p4, np.array([[True, True]]))["mass"][0]
    expected = np.sqrt(2 * 100 * 50 * (np.cosh(0.3) - np.cos(0.4)))
    np.testing.assert_allclose(m, expected, rtol=1e-9)


def test_T2_6_feature_stack_shape(toy_jets):
    p4, mask, _, _ = toy_jets
    f = constituent_features(p4, mask)
    assert f.shape == (2, 6, len(FEATURE_NAMES)) and f.dtype == np.float32
    assert np.all(f[~mask] == 0) and np.all(np.isfinite(f))


def test_T2_7_ptyphi_roundtrip():
    x = np.array([[[30.0, 0.2, 1.0, 22], [0, 0, 0, 0]]])
    p4, mask = ptyphi_to_p4(x)
    k = kinematics(p4, mask)
    np.testing.assert_allclose([k["pt"][0, 0], k["y"][0, 0], k["phi"][0, 0]], [30, 0.2, 1.0], rtol=1e-5)
    assert mask.tolist() == [[True, False]]
