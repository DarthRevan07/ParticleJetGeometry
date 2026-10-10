"""specs/05_classical_features.md: T5.x"""
import numpy as np

from jettopo.features.classical import classical_feature_table, girth, multiplicity, nsubjettiness, ptd
from conftest import massless_p4


def test_T5_1_multiplicity(toy_jets):
    _, mask, _, _ = toy_jets
    assert multiplicity(mask).tolist() == [5, 3]


def test_T5_2_girth_and_ptd_single_particle():
    p4 = massless_p4([100.0], [0.3], [0.2])[None]
    m = np.array([[True]])
    assert girth(p4, m)[0] == 0.0 and np.isclose(ptd(p4, m)[0], 1.0)


def test_T5_3_table_finite(toy_jets):
    p4, mask, _, _ = toy_jets
    t = classical_feature_table(p4, mask)
    assert set(t) == {"mass", "multiplicity", "girth", "ptd"}
    assert all(np.all(np.isfinite(v)) and v.shape == (2,) for v in t.values())


def test_T5_4_nsubjettiness_ratios_separate_prongs(toy_jets):
    p4, mask, _, _ = toy_jets
    tau = nsubjettiness(p4, mask, n_max=3)
    assert tau.shape == (2, 3)
    assert np.all(tau >= 0) and np.all(np.diff(tau, axis=1) <= 1e-9)  # tau_N non-increasing in N


def test_T5_5_nsubjettiness_single_particle_is_zero():
    p4 = massless_p4([100.0], [0.0], [0.0])[None]
    assert np.allclose(nsubjettiness(p4, np.array([[True]]), n_max=1), 0.0)
