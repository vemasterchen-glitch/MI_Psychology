"""
Shared matplotlib configuration for all experiment scripts.
Call setup_matplotlib() before any plt calls.
"""

import matplotlib
import matplotlib.pyplot as plt


def setup_matplotlib():
    """Set consistent font and style for all figures."""
    matplotlib.rcParams.update({
        # Font: prefer PingFang HK for CJK, fall back to STHeiti, then sans-serif
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang HK", "STHeiti", "Arial Unicode MS", "DejaVu Sans"],
        # Fix minus sign rendering (without this, '−' renders as a box)
        "axes.unicode_minus": False,
        # General style
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
