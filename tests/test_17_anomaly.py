"""specs/17_q5_anomaly_symmetry.md: T17.x (stub module: XFAIL until implemented)."""
import numpy as np
from sklearn.metrics import roc_auc_score

from jettopo.experiments.anomaly import fit_density_score, symmetry_gap


def test_T17_1_symmetry_gap():
    assert np.isclose(symmetry_gap(0.9, 0.6), 0.3)
    assert symmetry_gap(0.6, 0.9) == symmetry_gap(0.9, 0.6)


def test_T17_2_density_score_detects_shifted_blob():
    rng = np.random.default_rng(0)
    bkg_train = rng.normal(0, 1, (3000, 4))
    bkg_test, sig = rng.normal(0, 1, (1000, 4)), rng.normal(3, 1, (1000, 4))
    score = fit_density_score(bkg_train, method="gmm", seed=0)
    s = score(np.vstack([bkg_test, sig]))
    assert roc_auc_score(np.r_[np.zeros(1000), np.ones(1000)], s) > 0.95


def test_T17_3_deterministic_given_seed():
    x = np.random.default_rng(1).normal(size=(500, 3))
    a = fit_density_score(x, seed=3)(x[:10])
    b = fit_density_score(x, seed=3)(x[:10])
    np.testing.assert_allclose(a, b)
