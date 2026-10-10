"""Background-only anomaly scores and the direction-symmetry gap (research question Q5).

STUB (new in 2026). Spec: specs/17_q5_anomaly_symmetry.md
"""
from __future__ import annotations

import numpy as np


def fit_density_score(train_features: np.ndarray, method: str = "gmm", seed: int = 0):
    """Fit a density model on background-only features; return a callable features -> anomaly score (higher = more anomalous)."""
    raise NotImplementedError("fit_density_score: see specs/17_q5_anomaly_symmetry.md")


def symmetry_gap(auc_a_to_b: float, auc_b_to_a: float) -> float:
    """|AUC(trained on A, detect B) - AUC(trained on B, detect A)|."""
    raise NotImplementedError("symmetry_gap: see specs/17_q5_anomaly_symmetry.md")
