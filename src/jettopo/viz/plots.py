"""Jet and diagram plots.

Ported from 2024 ``persistence/plotters/{y_phi,pt_y_phi,cartesian}.py``. Changes:
axes are labelled (dy, dphi) (the 2024 labels said "y (Cartesian coordinate)"),
marker size/colour encode z = pT fraction, and the three near-identical
interactive selectors are merged into ``interactive_selector``.
"""
from __future__ import annotations

import numpy as np


def plot_jet(dy: np.ndarray, dphi: np.ndarray, z: np.ndarray, ax=None, radius: float | None = 0.8, title: str | None = None):
    """Scatter one jet in the (dy, dphi) plane; area proportional to z."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(5, 5))[1]
    sc = ax.scatter(dy, dphi, s=2000 * np.asarray(z), c=z, cmap="viridis", alpha=0.8, edgecolors="none")
    if radius:
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, ls="--", lw=0.8, color="grey"))
        ax.set_xlim(-radius * 1.1, radius * 1.1)
        ax.set_ylim(-radius * 1.1, radius * 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta y$")
    ax.set_ylabel(r"$\Delta\phi$")
    if title:
        ax.set_title(title)
    plt.colorbar(sc, ax=ax, label=r"$z = p_T / \sum p_T$")
    return ax


def plot_jet_3d(dy: np.ndarray, dphi: np.ndarray, log_pt: np.ndarray, title: str | None = None):
    """3-D scatter (dy, dphi, log pT): port of the 2024 pt_y_phi plot with fixed axes."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(dy, dphi, log_pt, c=log_pt, cmap="viridis")
    ax.set_xlabel(r"$\Delta y$")
    ax.set_ylabel(r"$\Delta\phi$")
    ax.set_zlabel(r"$\log p_T$")
    if title:
        ax.set_title(title)
    return ax


def plot_diagram(dgms: dict[int, np.ndarray], ax=None, title: str | None = None):
    """Persistence diagram, one colour per homology dimension; infinite deaths drawn at the top."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
    finite = [d[np.isfinite(d[:, 1])] for d in dgms.values() if len(d)]
    top = max([f[:, 1].max() for f in finite if len(f)] + [1e-3]) * 1.1
    for dim, d in sorted(dgms.items()):
        dd = d.copy()
        dd[~np.isfinite(dd[:, 1]), 1] = top
        ax.scatter(dd[:, 0], dd[:, 1], s=12, label=f"H{dim}")
    ax.plot([0, top], [0, top], color="grey", lw=0.8)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


def interactive_selector(jets: dict, plot_fn):
    """Dropdown over jet ids calling ``plot_fn(jets[id])`` (needs ipywidgets)."""
    import ipywidgets as widgets
    from IPython.display import display

    ids = list(jets.keys())
    sel = widgets.Dropdown(options=ids, value=ids[0] if ids else None, description="Jet:")
    sel.observe(lambda ch: plot_fn(jets[ch.new]), names="value")
    display(sel)
    return sel
