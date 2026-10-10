"""Synthetic infrared and collinear perturbations of jets (research question Q3).

STUB (new in 2026). Spec: specs/15_q3_irc_and_generator_shift.md

All functions act on ONE jet given as float[P, 4] (E, px, py, pz) and bool[P] mask,
and return a new (p4, mask) without modifying the inputs. Padding slots are used
for added particles; raise ValueError if none is free.
"""
from __future__ import annotations

import numpy as np


def add_soft_emission(p4: np.ndarray, mask: np.ndarray, energy_fraction: float, dy: float, dphi: float) -> tuple[np.ndarray, np.ndarray]:
    """Add one massless particle at (jet_y + dy, jet_phi + dphi) with pT = energy_fraction * jet pT."""
    raise NotImplementedError("add_soft_emission: see specs/15_q3_irc_and_generator_shift.md")


def collinear_split(p4: np.ndarray, mask: np.ndarray, index: int, fraction: float, angle: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Split constituent ``index`` into two with momentum fractions (fraction, 1 - fraction).

    ``angle`` = 0 gives an exactly collinear split (the two four-momenta sum to the
    original); angle > 0 separates the daughters by that angular distance in (y, phi)
    symmetrically about the parent direction.
    """
    raise NotImplementedError("collinear_split: see specs/15_q3_irc_and_generator_shift.md")


def emd(p4_a: np.ndarray, mask_a: np.ndarray, p4_b: np.ndarray, mask_b: np.ndarray, R: float = 0.8) -> float:
    """Energy mover's distance between two jets (normalized pT measures, beta = 1), via EnergyFlow or POT."""
    raise NotImplementedError("emd: see specs/15_q3_irc_and_generator_shift.md")
