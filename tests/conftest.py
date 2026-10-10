"""Shared fixtures and the stub convention.

A test that hits a stub (NotImplementedError) is reported as XFAIL ("not implemented
yet: see specs/..."), so the suite is green on a fresh port. Once a coding agent
implements the module, the same test must PASS; a wrong implementation shows up
as a normal FAIL. Never weaken a test to make it pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except NotImplementedError as exc:
        pytest.xfail(f"not implemented yet: {exc}")


def massless_p4(pt, y, phi):
    pt, y, phi = map(np.asarray, (pt, y, phi))
    return np.stack([pt * np.cosh(y), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(y)], axis=-1)


@pytest.fixture
def toy_jets():
    """Two padded jets (P = 6): a 3-prong 'top-like' jet and a 1-prong 'QCD-like' jet."""
    rng = np.random.default_rng(0)
    P = 6
    p4 = np.zeros((2, P, 4))
    mask = np.zeros((2, P), bool)
    j0 = massless_p4([300, 150, 100, 20, 5], [0.10, 0.45, -0.25, 0.30, -0.6], [0.0, 0.35, 0.5, -0.4, 0.2])
    j1 = massless_p4([500, 40, 10], [0.0, 0.05, -0.08], [1.0, 1.03, 0.95])
    p4[0, :5], mask[0, :5] = j0, True
    p4[1, :3], mask[1, :3] = j1, True
    return p4.astype(np.float64), mask, np.array([1, 0]), rng


@pytest.fixture
def toy_frame():
    """Two-jet DataFrame in the official HDF5 column layout (ported 2024 unittest fixture)."""
    return pd.DataFrame({
        "PX_0": [1, 9], "PY_0": [3, 0], "PZ_0": [5, 13], "E_0": [7, 15],
        "PX_1": [0, 10], "PY_1": [4, 12], "PZ_1": [6, 14], "E_1": [8, 0],
        "is_signal_new": [1, 0], "ttv": [0, 1],
    })


def circle(n=40, r=0.3, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1) + noise * rng.normal(size=(n, 2))
