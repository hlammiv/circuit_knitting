#!/usr/bin/env python3
"""Reproduce paper Figure 6 (fixed-step versus variable-step evolution)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    from .paper_figure_style import BLUE, FIGSIZE, ORANGE, SERIES_MARKERS, add_parameter_box, add_trotter_regions, apply_paper_style, finish_axes
except ImportError:  # Direct execution: python scripts/plot_fig6.py
    from paper_figure_style import BLUE, FIGSIZE, ORANGE, SERIES_MARKERS, add_parameter_box, add_trotter_regions, apply_paper_style, finish_axes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "mass_scan_plot_data"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "trotter_vs_exact.pdf"


def truncated_cosine(t, amplitude, frequency, offset):
    return amplitude * np.cos(frequency * t) + offset


def load_array(data_dir: Path, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(np.load(data_dir / f"{name}.npy"), dtype=float)
    if result.ndim != 1 or (length is not None and len(result) < length) or not np.all(np.isfinite(result)):
        raise ValueError(f"invalid Figure 6 array: {name}")
    return result if length is None else result[:length]


def fit_series(t: np.ndarray, values: np.ndarray, errors: np.ndarray) -> np.ndarray:
    parameters, _ = curve_fit(
        truncated_cosine,
        t,
        values,
        sigma=errors,
        p0=[max(float(np.ptp(values)) / 2, 0.01), 2.0, float(np.mean(values))],
        bounds=([-1, 0, -1], [1, np.pi / 0.2, 1]),
        absolute_sigma=True,
        maxfev=50000,
    )
    return parameters


def make_figure(data_dir: Path, output: Path) -> None:
    apply_paper_style(large_text=True)
    t = np.linspace(0.0, 1.6, 9)
    dense_t = np.linspace(t[0], t[-1], 400)
    fixed = load_array(data_dir, "ferm_num_m1p125_e0p005", 9)
    fixed_error = load_array(data_dir, "bs_error_m1p125_e0p005", 9)
    variable = load_array(data_dir, "ferm_num_m1p125_trot", 9)
    variable_error = load_array(data_dir, "bs_error_m1p125_trot", 9)
    if np.any(fixed_error <= 0) or np.any(variable_error <= 0):
        raise ValueError("Figure 6 uncertainties must be positive")
    fixed_fit = fit_series(t, fixed, fixed_error)
    variable_fit = fit_series(t, variable, variable_error)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(
        t,
        fixed,
        yerr=fixed_error,
        fmt=SERIES_MARKERS[0],
        color=BLUE,
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=7.5,
        linewidth=0.7,
        label=r"fixed $\epsilon=0.05$",
        zorder=3,
    )
    ax.plot(dense_t, truncated_cosine(dense_t, *fixed_fit), color=BLUE)
    for index, (time, value, error) in enumerate(zip(t, variable, variable_error)):
        ax.errorbar(
            time,
            value,
            yerr=error,
            fmt=SERIES_MARKERS[1],
            color=ORANGE,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7.5,
            linewidth=0.7,
            label=r"variable $\epsilon$" if index == 0 else None,
            zorder=4,
        )
    ax.plot(dense_t, truncated_cosine(dense_t, *variable_fit), color=ORANGE)
    finish_axes(
        ax,
        xlabel=r"$t$",
        ylabel=r"$\langle N_f\rangle$",
    )
    add_trotter_regions(ax, ylim=(0.497, 0.552), label_y=0.98)
    add_parameter_box(ax, r"$m=1.125$", y=0.87)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    make_figure(args.data_dir, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
