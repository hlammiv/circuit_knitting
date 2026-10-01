"""Shared Matplotlib style for the paper's numerical-result figures."""

from __future__ import annotations

import matplotlib as mpl


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
BLACK = "#000000"
YELLOW = "#F0E442"
SERIES_MARKERS = ("o", "s", "^")
FIGSIZE = (7.2, 4.6)


def apply_paper_style(*, large_text: bool = False) -> None:
    """Apply the manuscript's LaTeX font and shared style to Figures 5--7."""
    text_sizes = (
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
        if large_text
        else {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    mpl.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman"],
            "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 5.5,
            "errorbar.capsize": 2.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            **text_sizes,
        }
    )


def finish_axes(
    ax,
    *,
    title: str | None = None,
    xlabel: str = "evolution time",
    ylabel: str = "mean fermion number",
) -> None:
    """Apply the common labels, grid, and optional title."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.15)


def add_trotter_regions(
    ax,
    *,
    ylim: tuple[float, float] | None = None,
    label_y: float = 0.95,
    left: float = -0.10,
) -> None:
    """Mark and label the zero-, one-, and two-step time regions."""
    _, right = ax.get_xlim()
    right = max(right, 1.70)
    ax.set_xlim(left, right)
    if ylim is None:
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper + 0.14 * (upper - lower))
    else:
        ax.set_ylim(*ylim)
    for boundary in (0.1, 0.9):
        ax.axvline(
            boundary,
            color="0.55",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )
    regions = (
        ((left + 0.1) / 2.0, 0),
        (0.5, 1),
        ((0.9 + right) / 2.0, 2),
    )
    for center, steps in regions:
        ax.text(
            center,
            label_y,
            rf"$N_t={steps}$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
        )


def add_parameter_box(ax, text: str, *, x: float = 0.98, y: float = 0.97) -> None:
    """Place a compact Okabe--Ito yellow parameter box at upper right."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": YELLOW,
            "edgecolor": "#E69F00",
            "linewidth": 0.8,
            "alpha": 0.55,
        },
        zorder=10,
    )
