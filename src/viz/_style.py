"""Shared plotting style and a save helper. Source-agnostic — no project globals."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colour-blind-safe sequential / diverging maps used across figures.
SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def apply_style() -> None:
    """Apply a consistent, dense, readable Matplotlib style."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "font.size": 9.5,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save(fig: plt.Figure, path) -> Path:
    """Save ``fig`` to ``path`` (creating parents) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
