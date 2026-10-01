#!/usr/bin/env python3
"""Reproduce paper Figure 5 (the long-time mass scan)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    from .paper_figure_style import BLUE, FIGSIZE, GREEN, ORANGE, SERIES_MARKERS, add_parameter_box, apply_paper_style, finish_axes
except ImportError:  # Direct execution: python scripts/plot_fig5.py
    from paper_figure_style import BLUE, FIGSIZE, GREEN, ORANGE, SERIES_MARKERS, add_parameter_box, apply_paper_style, finish_axes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "mass_scan_plot_data"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "mass_scan.pdf"


def damped_cosine(t, amplitude, damping, frequency, offset, slope):
    return amplitude * np.exp(-damping * t) * np.cos(frequency * t) + offset + slope * t


def load_series(data_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(np.load(data_dir / f"ferm_num_{stem}.npy"), dtype=float)
    errors = np.asarray(np.load(data_dir / f"bs_error_{stem}.npy"), dtype=float)
    if values.shape != (33,) or errors.shape != values.shape:
        raise ValueError(f"{stem} must contain 33 values and 33 errors")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError(f"{stem} contains invalid values or errors")
    return values, errors


def fit_series(t: np.ndarray, values: np.ndarray, errors: np.ndarray) -> np.ndarray:
    best = None
    center = float(np.mean(values))
    scale = max(float(np.ptp(values)) / 2, 0.01)
    for frequency in np.linspace(0.5, 4.0, 15):
        for damping in (0.0, 0.05, 0.2, 0.5):
            start = [scale, damping, frequency, center, 0.0]
            try:
                parameters, _ = curve_fit(
                    damped_cosine,
                    t,
                    values,
                    sigma=errors,
                    p0=start,
                    bounds=([-1, 0, 0, -1, -1], [1, 10, np.pi / 0.2, 1, 1]),
                    absolute_sigma=True,
                    maxfev=50000,
                )
            except (RuntimeError, ValueError):
                continue
            chi2 = float(np.sum(((values - damped_cosine(t, *parameters)) / errors) ** 2))
            if best is None or chi2 < best[0]:
                best = chi2, parameters
    if best is None:
        raise RuntimeError("all Figure 5 fit starts failed")
    return best[1]


def make_figure(data_dir: Path, output: Path) -> None:
    apply_paper_style(large_text=True)
    t = np.arange(33, dtype=float) * 0.2
    dense_t = np.linspace(t[0], t[-1], 600)
    series = (
        ("m1p0_e0p005", r"$m=1$", BLUE),
        ("m1p125_e0p005", r"$m=1.125$", ORANGE),
        ("m1p25_e0p005", r"$m=1.25$", GREEN),
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for (stem, label, color), marker in zip(series, SERIES_MARKERS):
        values, errors = load_series(data_dir, stem)
        parameters = fit_series(t, values, errors)
        ax.errorbar(
            t,
            values,
            yerr=errors,
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=7.5,
            linewidth=0.7,
            label=label,
            zorder=3,
        )
        ax.plot(dense_t, damped_cosine(dense_t, *parameters), color=color)
    finish_axes(
        ax,
        xlabel=r"$t$",
        ylabel=r"$\langle N_f\rangle$",
    )
    add_parameter_box(ax, r"$\epsilon=0.05$")
    ax.legend(ncol=3)
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
