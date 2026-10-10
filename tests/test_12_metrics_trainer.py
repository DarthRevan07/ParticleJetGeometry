"""specs/12_training_and_metrics.md: T12.x"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from jettopo.training.metrics import aggregate_seeds, background_rejection, benchmark_metrics, bootstrap_ci
from jettopo.training.trainer import TrainConfig, fit, predict


def test_T12_1_perfect_and_random_classifiers():
    y = np.r_[np.ones(500), np.zeros(500)]
    m = benchmark_metrics(y, y * 0.8 + 0.1)
    assert m["auc"] == 1.0 and m["acc"] == 1.0 and np.isinf(m["rej@30"])
    r = benchmark_metrics(y, np.random.default_rng(0).uniform(size=1000))
    assert abs(r["auc"] - 0.5) < 0.05 and 1.5 < r["rej@50"] < 2.6


def test_T12_2_rejection_on_known_distributions():
    # signal ~ U(0.5, 1.5), background ~ U(0, 1): at eps_S = 0.3 the cut is 1.2 -> eps_B = 0 ; at eps_S = 0.7 cut 0.8 -> eps_B = 0.2
    rng = np.random.default_rng(0)
    s, b = rng.uniform(0.5, 1.5, 200000), rng.uniform(0, 1, 200000)
    y, x = np.r_[np.ones_like(s), np.zeros_like(b)], np.r_[s, b]
    assert abs(background_rejection(y, x, 0.7) - 5.0) < 0.1


def test_T12_3_bootstrap_ci_brackets_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = y + rng.normal(0, 1, 2000)
    lo, hi = bootstrap_ci(y, s, n_boot=100)
    full = benchmark_metrics(y, 1 / (1 + np.exp(-s)))["auc"]
    assert lo < full < hi and hi - lo < 0.06


def test_T12_4_aggregate_seeds():
    agg = aggregate_seeds([{"auc": 0.90}, {"auc": 0.92}])
    assert np.isclose(agg["auc"][0], 0.91) and np.isclose(agg["auc"][1], np.std([0.9, 0.92], ddof=1))


def _toy_loaders(n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).float()
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=128, shuffle=True, generator=g), DataLoader(ds, batch_size=512)


class Lin(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(4, 1)

    def forward(self, x):
        return self.l(x).squeeze(-1)


def test_T12_5_fit_learns_separable_data():
    torch.manual_seed(0)
    tr, va = _toy_loaders()
    model = Lin()
    res = fit(model, tr, va, TrainConfig(epochs=5, lr=0.05, device="cpu"))
    y, p = predict(model, va, "cpu")
    assert res["best_val"] > 0.98 and benchmark_metrics(y, p)["auc"] > 0.98
    assert len(res["history"]) == 5


def test_T12_6_resume_from_checkpoint(tmp_path):
    torch.manual_seed(0)
    tr, va = _toy_loaders()
    fit(Lin(), tr, va, TrainConfig(epochs=2, lr=0.05, device="cpu", ckpt_dir=str(tmp_path)))
    res = fit(Lin(), tr, va, TrainConfig(epochs=4, lr=0.05, device="cpu", ckpt_dir=str(tmp_path)))
    assert [h["epoch"] for h in res["history"]] == [0, 1, 2, 3]
