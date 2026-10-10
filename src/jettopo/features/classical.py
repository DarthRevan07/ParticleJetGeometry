"""Classical jet observables for the bottom rung of the baseline ladder.

Spec: specs/05_classical_features.md. N-subjettiness is a stub to implement.
"""
from __future__ import annotations

import numpy as np

from ..data.preprocess import jet_kinematics, relative_coords


def jet_mass(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return jet_kinematics(p4, mask)["mass"]


def multiplicity(mask: np.ndarray) -> np.ndarray:
    """Number of constituents. IRC unsafe: a soft or collinear emission changes it."""
    return mask.sum(axis=-1).astype(np.int32)


def girth(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """pT-weighted width sum_i z_i dR_i (IRC safe)."""
    rc = relative_coords(p4, mask)
    return (rc["z"] * rc["dR"]).sum(axis=-1)


def ptd(p4: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """p_T^D = sqrt(sum_i z_i^2) (IR safe, collinear unsafe)."""
    rc = relative_coords(p4, mask)
    return np.sqrt((rc["z"] ** 2).sum(axis=-1))


def nsubjettiness(p4: np.ndarray, mask: np.ndarray, n_max: int = 3, beta: float = 1.0, R: float = 0.8) -> np.ndarray:
    """tau_N for N = 1..n_max with exclusive-kT axes, returns float[N_jets, n_max].

    STUB: see specs/05_classical_features.md (use the scikit-hep ``fastjet`` package).
    """
    raise NotImplementedError("nsubjettiness: see specs/05_classical_features.md")


def classical_feature_table(p4: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """All implemented classical features as a dict of 1-D arrays."""
    return {
        "mass": jet_mass(p4, mask),
        "multiplicity": multiplicity(mask).astype(np.float64),
        "girth": girth(p4, mask),
        "ptd": ptd(p4, mask),
    }
