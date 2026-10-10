"""specs/14_q4_expressivity_probe.md: T14.x (stub models: XFAIL until implemented)."""
import numpy as np
import torch

from jettopo.models.simplicial_mp import EnergyWeightedPairMP, EnergyWeightedSimplicialMP
from jettopo.topology.batching import collate_complexes, jet_to_complex_item


def _jets(n=6, seed=0):
    rng = np.random.default_rng(seed)
    items, zs = [], []
    for g in range(n):
        P = 6 + g
        pt = np.sort(rng.exponential(50, P))[::-1]
        y, phi = rng.normal(0, 0.1, P), rng.normal(0, 0.1, P)
        p4 = np.stack([pt * np.cosh(y), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(y)], 1)
        items.append(jet_to_complex_item(p4, np.ones(P, bool), 0, max_edge_length=0.3))
        zs.append(torch.tensor(pt / pt.sum(), dtype=torch.float32))
    return collate_complexes(items), torch.cat(zs)


def test_T14_1_output_shapes():
    batch, z = _jets()
    for cls in (EnergyWeightedPairMP, EnergyWeightedSimplicialMP):
        assert cls(hidden=16, layers=1)(batch, z).shape == (6,)


def test_T14_2_parameter_counts_matched_within_10_percent():
    a = sum(p.numel() for p in EnergyWeightedPairMP(hidden=64, layers=1).parameters())
    b = sum(p.numel() for p in EnergyWeightedSimplicialMP(hidden=64, layers=1).parameters())
    assert abs(a - b) / max(a, b) < 0.10


def test_T14_3_zero_energy_node_does_not_change_output():
    torch.manual_seed(0)
    model = EnergyWeightedSimplicialMP(hidden=16, layers=1).eval()
    rng = np.random.default_rng(1)
    P = 8
    pt = np.sort(rng.exponential(50, P))[::-1]
    y, phi = rng.normal(0, 0.1, P), rng.normal(0, 0.1, P)
    p4 = np.stack([pt * np.cosh(y), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(y)], 1)
    a = collate_complexes([jet_to_complex_item(p4, np.ones(P, bool), 0, 0.3)])
    za = torch.tensor(pt / pt.sum(), dtype=torch.float32)
    p4b = np.vstack([p4, [[1e-12, 1e-12, 0, 0]]])
    b = collate_complexes([jet_to_complex_item(p4b, np.ones(P + 1, bool), 0, 0.3)])
    zb = torch.cat([za, torch.zeros(1)])
    torch.testing.assert_close(model(a, za), model(b, zb), rtol=1e-4, atol=1e-5)
