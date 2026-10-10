"""Benchmark metrics as reported in the jet-tagging literature.

Spec: specs/12_training_and_metrics.md
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    """Accuracy of ``scores > threshold`` (scores are probabilities)."""
    return float(np.mean((np.asarray(scores) > threshold) == (np.asarray(labels) == 1)))


def background_rejection(labels: np.ndarray, scores: np.ndarray, signal_efficiency: float) -> float:
    """1 / eps_B at the working point where the signal efficiency is ``signal_efficiency``.

    eps_B is linearly interpolated on the ROC curve. Returns inf if eps_B == 0.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    eps_b = float(np.interp(signal_efficiency, tpr, fpr))
    return float("inf") if eps_b <= 0 else 1.0 / eps_b


def benchmark_metrics(labels: np.ndarray, scores: np.ndarray, efficiencies: tuple[float, ...] = (0.3, 0.5)) -> dict[str, float]:
    """{'acc', 'auc', 'rej@30', 'rej@50', ...} for probability-valued scores."""
    out = {"acc": accuracy(labels, scores), "auc": auc(labels, scores)}
    for e in efficiencies:
        out[f"rej@{int(round(100 * e))}"] = background_rejection(labels, scores, e)
    return out


def bootstrap_ci(labels: np.ndarray, scores: np.ndarray, metric=auc, n_boot: int = 200, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of a metric over the test set."""
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(labels), np.asarray(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        vals.append(metric(labels[idx], scores[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def aggregate_seeds(results: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """{metric: (mean, std over seeds)}; std uses ddof=1 (0.0 for a single seed)."""
    keys = results[0].keys()
    return {
        k: (float(np.mean([r[k] for r in results])), float(np.std([r[k] for r in results], ddof=1)) if len(results) > 1 else 0.0)
        for k in keys
    }
