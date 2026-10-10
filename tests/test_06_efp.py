"""specs/06_efp_features.md: T6.x (skipped unless energyflow is installed)."""
import numpy as np
import pytest

ef = pytest.importorskip("energyflow")

from jettopo.data.preprocess import relative_coords  # noqa: E402
from jettopo.features.efp import efp_graph, efp_set  # noqa: E402


def _manual_dumbbell(p4, mask):
    rc = relative_coords(p4, mask)
    out = []
    for i in range(len(p4)):
        z, x = rc["z"][i][mask[i]], np.stack([rc["dy"][i][mask[i]], rc["dphi"][i][mask[i]]], 1)
        th = np.linalg.norm(x[:, None] - x[None], axis=-1)
        out.append(np.einsum("i,j,ij->", z, z, th))
    return np.array(out)


def test_T6_1_efp_set_shape(toy_jets):
    p4, mask, _, _ = toy_jets
    X = efp_set(p4, mask, dmax=3, n_jobs=1)
    assert X.shape[0] == 2 and X.shape[1] > 3 and np.all(np.isfinite(X))


def test_T6_2_dumbbell_matches_manual_sum(toy_jets):
    p4, mask, _, _ = toy_jets
    np.testing.assert_allclose(efp_graph(p4, mask, "dumbbell"), _manual_dumbbell(p4, mask), rtol=1e-3)


def test_T6_3_triangle_ir_safe(toy_jets):
    p4, mask, _, _ = toy_jets
    p4b, maskb = p4.copy(), mask.copy()
    p4b[1, 3] = [1e-9 * np.cosh(0.5), 1e-9, 0.0, 1e-9 * np.sinh(0.5)]
    maskb[1, 3] = True
    np.testing.assert_allclose(efp_graph(p4, mask, "triangle"), efp_graph(p4b, maskb, "triangle"), rtol=1e-6, atol=1e-12)
