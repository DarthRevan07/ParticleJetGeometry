"""specs/11_deep_sets_and_gnn.md: T11.x"""
import numpy as np
import pytest
import torch

from jettopo.models.deepsets import EFN, PFN
from jettopo.models.gnn import knn_edge_index
from jettopo.models.jumping_knowledge import JumpingKnowledge


def test_T11_1_knn_edges():
    pts = torch.tensor(np.random.default_rng(0).normal(size=(10, 2)), dtype=torch.float32)
    ei = knn_edge_index(pts, k=3)
    assert ei.shape == (2, 30)
    assert not torch.any(ei[0] == ei[1])
    assert torch.equal(torch.bincount(ei[1], minlength=10), torch.full((10,), 3))


def test_T11_2_knn_small_cloud():
    assert knn_edge_index(torch.zeros(1, 2), k=4).shape == (2, 0)
    assert knn_edge_index(torch.rand(3, 2), k=16).shape == (2, 6)


@pytest.mark.parametrize("mode,out", [("cat", 8 * 3), ("sum", 8), ("mean", 8), ("max", 8), ("wsum", 8), ("attn", 8)])
def test_T11_3_jumping_knowledge_shapes(mode, out):
    jk = JumpingKnowledge(mode, hidden_dim=8, num_heads=2)
    assert jk(torch.randn(5, 8, 3)).shape == (5, out)


def _batch(B=4, P=12, C=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, P, C, generator=g)
    mask = torch.zeros(B, P, dtype=torch.bool)
    for b in range(B):
        mask[b, : 3 + 2 * b] = True
    return x * mask[..., None], mask


def test_T11_4_pfn_shape_and_permutation_invariance():
    torch.manual_seed(0)
    m = PFN(in_dim=7).eval()
    x, mask = _batch()
    perm = torch.randperm(x.shape[1])
    out = m(x, mask)
    assert out.shape == (4,)
    torch.testing.assert_close(out, m(x[:, perm], mask[:, perm]), rtol=1e-5, atol=1e-5)


def test_T11_5_pfn_ignores_padding_values():
    m = PFN(in_dim=7).eval()
    x, mask = _batch()
    x2 = x.clone()
    x2[~mask] = 123.0
    torch.testing.assert_close(m(x, mask), m(x2, mask))


def test_T11_6_efn_irc_safe():
    torch.manual_seed(0)
    m = EFN().eval()
    B, P = 3, 10
    z = torch.rand(B, P)
    mask = torch.ones(B, P, dtype=torch.bool)
    mask[:, 8:] = False
    z = z * mask
    z = z / z.sum(1, keepdim=True)
    coords = torch.randn(B, P, 2) * 0.2
    base = m(z, coords, mask)
    # infrared: add a zero-energy particle
    mask_ir = mask.clone(); mask_ir[:, 8] = True
    torch.testing.assert_close(base, m(z, coords, mask_ir), rtol=1e-5, atol=1e-6)
    # collinear: split particle 0 into two at the same position
    z_c, c_c, m_c = z.clone(), coords.clone(), mask.clone()
    z_c[:, 9], z_c[:, 0] = 0.4 * z[:, 0], 0.6 * z[:, 0]
    c_c[:, 9] = coords[:, 0]
    m_c[:, 9] = True
    torch.testing.assert_close(base, m(z_c, c_c, m_c), rtol=1e-5, atol=1e-6)
