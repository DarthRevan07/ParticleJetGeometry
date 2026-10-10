"""specs/13_scnn.md: T13.x"""
import numpy as np
import torch

from jettopo.models.scnn import Coboundary, SCNNClassifier, SimplicialConvolution, chebyshev_assemble, scatter_mean
from jettopo.topology.batching import collate_complexes, jet_to_complex_item
from jettopo.topology.complexes import coo2tensor, normalize_laplacian, rips_complex


def _assemble_2024(K, L, x):
    """The 2024 loop implementation, kept verbatim as a reference."""
    B, C_in, M = x.shape
    X = []
    for b in range(B):
        X123 = []
        for c in range(C_in):
            X23 = [x[b, c, :].unsqueeze(1)]
            if K > 1:
                X23.append(L.mm(X23[0]))
            for k in range(2, K):
                X23.append(2 * (L.mm(X23[k - 1])) - X23[k - 2])
            X123.append(torch.cat(X23, 1).unsqueeze(0))
        X.append(torch.cat(X123, 0).unsqueeze(0))
    return torch.cat(X, 0)


def _L(n=15, seed=0):
    pts = np.random.default_rng(seed).uniform(-0.3, 0.3, (n, 2))
    return coo2tensor(normalize_laplacian(rips_complex(pts, 0.2).laplacians[0]))


def test_T13_1_vectorized_chebyshev_equals_2024_loop():
    L = _L()
    x = torch.randn(3, 4, L.shape[0])
    for K in (1, 2, 4):
        torch.testing.assert_close(chebyshev_assemble(K, L, x), _assemble_2024(K, L, x))


def test_T13_2_coboundary_equals_dense():
    torch.manual_seed(0)
    D = torch.randn(7, 5)
    cb = Coboundary(3, 2)
    x = torch.randn(2, 3, 5)
    expected = torch.einsum("oi,bin->bon", cb.theta, torch.einsum("nm,bim->bin", D, x)) + cb.bias
    torch.testing.assert_close(cb(D.to_sparse(), x), expected)


def test_T13_3_scatter_mean():
    src = torch.tensor([[1.0], [3.0], [5.0]])
    out = scatter_mean(src, torch.tensor([0, 0, 2]), 3)
    assert out.flatten().tolist() == [2.0, 0.0, 5.0]


def _items(n=4, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for g in range(n):
        P = 8 + 3 * g
        pt = rng.exponential(50, P)
        y, phi = rng.normal(0, 0.15, P), rng.normal(0, 0.15, P)
        p4 = np.stack([pt * np.cosh(y), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(y)], 1)
        order = np.argsort(-pt)
        items.append(jet_to_complex_item(p4[order], np.ones(P, bool), g % 2, max_edge_length=0.15))
    return items


def test_T13_4_batched_forward_equals_one_by_one():
    torch.manual_seed(0)
    model = SCNNClassifier().eval()
    items = _items()
    together = model(collate_complexes(items))
    alone = torch.cat([model(collate_complexes([it])) for it in items])
    assert together.shape == (4,)
    torch.testing.assert_close(together, alone, rtol=1e-4, atol=1e-5)


def test_T13_5_jet_without_triangles_or_edges():
    p4 = np.array([[100.0, 100.0, 0, 0]])
    it = jet_to_complex_item(p4, np.array([True]), 1)
    out = SCNNClassifier().eval()(collate_complexes([it] + _items(2)))
    assert out.shape == (3,) and torch.all(torch.isfinite(out))


def test_T13_6_gradients_reach_every_dimension():
    model = SCNNClassifier()
    batch = collate_complexes(_items())
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), batch.y)
    loss.backward()
    for stack in model.stacks:
        assert stack[0].theta.grad is not None and stack[0].theta.grad.abs().sum() > 0


def test_T13_7_convolution_shapes():
    conv = SimplicialConvolution(3, 5, 8)
    L = _L()
    assert conv(L, torch.randn(2, 5, L.shape[0])).shape == (2, 8, L.shape[0])
