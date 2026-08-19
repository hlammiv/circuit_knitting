#!/usr/bin/env python3
"""Fit and plot the knitted and non-knitted data used in paper Fig. 7."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_distribution


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_FIGURE = ROOT / "paper" / "figures" / "fermion_number_with_vs_without_knitting_scatter.pdf"
DEFAULT_JSON = ROOT / "paper" / "generated" / "fig7_fit_results.json"
DEFAULT_TEX = ROOT / "paper" / "generated" / "fig7_fit_results.tex"


def truncated_cosine(t: np.ndarray, amplitude: float, frequency: float, offset: float) -> np.ndarray:
    return amplitude * np.cos(frequency * t) + offset


def damped_cosine(
    t: np.ndarray,
    amplitude: float,
    damping: float,
    frequency: float,
    offset: float,
    slope: float,
) -> np.ndarray:
    return amplitude * np.exp(-damping * t) * np.cos(frequency * t) + offset + slope * t


@dataclass(frozen=True)
class Model:
    name: str
    function: Callable[..., np.ndarray]
    parameter_names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    frequency_index: int


@dataclass
class FitResult:
    model: Model
    parameters: np.ndarray
    covariance: np.ndarray
    chi2: float
    dof: int
    p_value: float
    aicc: float


def models_for_spacing(dt: float) -> dict[str, Model]:
    nyquist = math.pi / dt
    return {
        "truncated": Model(
            "truncated",
            truncated_cosine,
            ("amplitude", "frequency", "offset"),
            np.array([-1.0, 0.0, -1.0]),
            np.array([1.0, nyquist, 1.0]),
            1,
        ),
        "damped": Model(
            "damped",
            damped_cosine,
            ("amplitude", "damping", "frequency", "offset", "slope"),
            np.array([-1.0, 0.0, 0.0, -1.0, -1.0]),
            np.array([1.0, 10.0, nyquist, 1.0, 1.0]),
            2,
        ),
    }


def _starting_points(model: Model, y: np.ndarray) -> list[np.ndarray]:
    center = float(np.mean(y))
    scale = max(float(np.ptp(y)) / 2.0, 0.01)
    starts: list[np.ndarray] = []
    for frequency in np.linspace(0.4, 8.0, 16):
        if model.name == "truncated":
            starts.extend(
                [np.array([scale, frequency, center]), np.array([-scale, frequency, center])]
            )
        else:
            for damping in (0.0, 0.2, 1.0):
                starts.extend(
                    [
                        np.array([scale, damping, frequency, center, 0.0]),
                        np.array([-scale, damping, frequency, center, 0.0]),
                    ]
                )
    return starts


def _validate_series(t: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> None:
    if t.ndim != 1 or y.shape != t.shape or sigma.shape != t.shape:
        raise ValueError("time, central values, and uncertainties must be equal-length 1D arrays")
    if len(t) < 6:
        raise ValueError("at least six time points are required for model comparison")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(sigma)):
        raise ValueError("fit inputs must be finite")
    if np.any(sigma <= 0):
        raise ValueError("all fit uncertainties must be positive")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time points must be strictly increasing")


def fit_series(
    model: Model,
    t: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    initial: np.ndarray | None = None,
) -> FitResult:
    _validate_series(t, y, sigma)
    starts = [initial] if initial is not None else _starting_points(model, y)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for start in starts:
        if start is None:
            continue
        try:
            parameters, covariance = curve_fit(
                model.function,
                t,
                y,
                sigma=sigma,
                p0=np.clip(start, model.lower + 1e-10, model.upper - 1e-10),
                bounds=(model.lower, model.upper),
                absolute_sigma=True,
                maxfev=50000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        residual = (y - model.function(t, *parameters)) / sigma
        chi2 = float(residual @ residual)
        if best is None or chi2 < best[0]:
            best = chi2, parameters, covariance
    if best is None:
        raise RuntimeError(f"all {model.name} fit starts failed")
    chi2, parameters, covariance = best
    k = len(parameters)
    n = len(t)
    dof = n - k
    if n <= k + 1:
        raise ValueError("AICc is undefined when n <= k + 1")
    aicc = chi2 + 2 * k + 2 * k * (k + 1) / (n - k - 1)
    return FitResult(
        model=model,
        parameters=parameters,
        covariance=covariance,
        chi2=chi2,
        dof=dof,
        p_value=float(chi2_distribution.sf(chi2, dof)),
        aicc=float(aicc),
    )


def bootstrap_fit(
    fit: FitResult,
    t: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float]:
    draws: list[np.ndarray] = []
    failures = 0
    for _ in range(samples):
        y_draw = rng.normal(y, sigma)
        try:
            result = fit_series(fit.model, t, y_draw, sigma, initial=fit.parameters)
        except RuntimeError:
            try:
                result = fit_series(fit.model, t, y_draw, sigma)
            except RuntimeError:
                failures += 1
                continue
        draws.append(result.parameters)
    array = np.asarray(draws, dtype=float)
    if not len(array):
        raise RuntimeError(f"all bootstrap fits failed for {fit.model.name}")
    tolerance = 1e-4 * (fit.model.upper - fit.model.lower)
    near_lower = np.any(array <= fit.model.lower + tolerance, axis=1)
    near_upper = np.any(array >= fit.model.upper - tolerance, axis=1)
    boundary_fraction = float(np.mean(near_lower | near_upper))
    return array, failures, boundary_fraction


def load_data(data_dir: Path) -> dict[str, np.ndarray]:
    names = {
        "exact": "ferm_num_exact_no_noise_no_knit.npy",
        "exact_sigma": "bs_error_exact_no_noise_no_knit.npy",
        "no_knit": "no_knit_noisy_resampled_fermion_number.npy",
        "no_knit_sigma": "no_knit_noisy_resampled_bootstrap_error.npy",
        "knitted": "knitted_noisy_resampled_fermion_number.npy",
        "knitted_sigma": "knitted_noisy_resampled_bootstrap_error.npy",
    }
    result = {key: np.asarray(np.load(data_dir / filename), dtype=float) for key, filename in names.items()}
    n = len(result["no_knit"])
    result["t"] = np.linspace(0.0, 0.2 * (n - 1), n)
    result["exact"] = result["exact"][:n]
    result["exact_sigma"] = result["exact_sigma"][:n]
    _validate_series(result["t"], result["no_knit"], result["no_knit_sigma"])
    _validate_series(result["t"], result["knitted"], result["knitted_sigma"])
    if result["exact"].shape != result["t"].shape:
        raise ValueError("exact reference does not cover all Fig. 7 time points")
    return result


def _fit_to_dict(fit: FitResult, bootstrap: np.ndarray, failures: int, boundary_fraction: float) -> dict:
    frequency = bootstrap[:, fit.model.frequency_index]
    return {
        "model": fit.model.name,
        "parameters": dict(zip(fit.model.parameter_names, map(float, fit.parameters))),
        "chi2": fit.chi2,
        "dof": fit.dof,
        "chi2_per_dof": fit.chi2 / fit.dof,
        "p_value": fit.p_value,
        "aicc": fit.aicc,
        "bootstrap_successes": int(len(bootstrap)),
        "bootstrap_failures": int(failures),
        "bootstrap_boundary_fraction": boundary_fraction,
        "frequency_median": float(np.median(frequency)),
        "frequency_interval_68": [float(x) for x in np.percentile(frequency, [16, 84])],
    }


def _stable(summary: dict, requested: int) -> bool:
    return (
        summary["bootstrap_failures"] / requested <= 0.05
        and summary["bootstrap_boundary_fraction"] <= 0.05
    )


def analyze(data: dict[str, np.ndarray], bootstrap_samples: int, seed: int) -> tuple[dict, dict]:
    t = data["t"]
    model_map = models_for_spacing(float(np.min(np.diff(t))))
    rng = np.random.default_rng(seed)
    details: dict[str, dict[str, dict]] = {}
    bootstrap_arrays: dict[tuple[str, str], np.ndarray] = {}
    fits: dict[tuple[str, str], FitResult] = {}
    for series in ("no_knit", "knitted"):
        details[series] = {}
        for model_name, model in model_map.items():
            fit = fit_series(model, t, data[series], data[f"{series}_sigma"])
            bootstrap, failures, boundary_fraction = bootstrap_fit(
                fit, t, data[series], data[f"{series}_sigma"], bootstrap_samples, rng
            )
            fits[(series, model_name)] = fit
            bootstrap_arrays[(series, model_name)] = bootstrap
            details[series][model_name] = _fit_to_dict(fit, bootstrap, failures, boundary_fraction)

    damped_preferred = all(
        details[series]["damped"]["aicc"] <= details[series]["truncated"]["aicc"] - 6.0
        and _stable(details[series]["damped"], bootstrap_samples)
        for series in ("no_knit", "knitted")
    )
    primary = "damped" if damped_preferred else "truncated"
    no_knit_frequency = bootstrap_arrays[("no_knit", primary)][:, model_map[primary].frequency_index]
    knitted_frequency = bootstrap_arrays[("knitted", primary)][:, model_map[primary].frequency_index]
    count = min(len(no_knit_frequency), len(knitted_frequency))
    delta = knitted_frequency[:count] - no_knit_frequency[:count]
    summary = {
        "bootstrap_samples_requested": bootstrap_samples,
        "random_seed": seed,
        "primary_model": primary,
        "primary_rule": (
            "Use damped only if its AICc is at least 6 lower for both series and its "
            "bootstrap failure and boundary fractions are each <= 5%; otherwise use truncated."
        ),
        "series": details,
        "frequency_difference_knitted_minus_no_knit": {
            "median": float(np.median(delta)),
            "interval_68": [float(x) for x in np.percentile(delta, [16, 84])],
            "interval_95": [float(x) for x in np.percentile(delta, [2.5, 97.5])],
            "probability_positive": float(np.mean(delta > 0)),
        },
    }
    internals = {"fits": fits, "bootstrap": bootstrap_arrays, "models": model_map}
    return summary, internals


def _raw_samples(data_dir: Path, filename: str) -> list[tuple[np.ndarray, np.ndarray]]:
    path = data_dir / filename
    if not path.exists():
        return []
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    samples = []
    for value in payload.values():
        samples.append(
            (np.asarray(value["fermion_number"], dtype=float), np.asarray(value["bootstrap_error"], dtype=float))
        )
    return samples


def make_figure(
    data: dict[str, np.ndarray],
    data_dir: Path,
    summary: dict,
    internals: dict,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    t = data["t"]
    dense_t = np.linspace(t[0], t[-1], 400)
    colors = {"no_knit": "#0072B2", "knitted": "#D55E00"}
    labels = {"no_knit": "without knitting", "knitted": "with knitting"}
    raw_files = {
        "no_knit": "noisy_no_knit_trotter_evolution.pkl",
        "knitted": "noisy_knitted_trotter_evolution.pkl",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(t, data["exact"], color="black", linewidth=1.7, label="exact, noiseless")
    primary = summary["primary_model"]
    for series in ("no_knit", "knitted"):
        color = colors[series]
        for raw_y, raw_sigma in _raw_samples(data_dir, raw_files[series]):
            ax.errorbar(t, raw_y, yerr=raw_sigma, fmt=".", color=color, alpha=0.14, linewidth=0.5)
        for index, (time, value, error) in enumerate(zip(t, data[series], data[f"{series}_sigma"])):
            marker = "o" if index == 0 else ("s" if index <= 4 else "^")
            ax.errorbar(
                time,
                value,
                yerr=error,
                fmt=marker,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2.5,
                zorder=4,
            )
        fit = internals["fits"][(series, primary)]
        model = internals["models"][primary]
        boot = internals["bootstrap"][(series, primary)]
        predictions = np.asarray([model.function(dense_t, *parameters) for parameters in boot])
        lower, upper = np.percentile(predictions, [16, 84], axis=0)
        ax.fill_between(dense_t, lower, upper, color=color, alpha=0.16, linewidth=0)
        ax.plot(dense_t, model.function(dense_t, *fit.parameters), color=color, linewidth=2.0, label=labels[series])
    ax.axvline(0.1, color="0.4", linewidth=0.8)
    ax.axvline(0.9, color="0.4", linewidth=0.8)
    ax.set_xlabel("evolution time")
    ax.set_ylabel("mean fermion number")
    ax.set_title("Time Evolution of Mean Fermion Number")
    marker_handles = [
        Line2D([], [], marker="o", color="0.25", linestyle="none", label="0 Trotter steps"),
        Line2D([], [], marker="s", color="0.25", linestyle="none", label="1 Trotter step"),
        Line2D([], [], marker="^", color="0.25", linestyle="none", label="2 Trotter steps"),
    ]
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles + marker_handles, legend_labels + [h.get_label() for h in marker_handles], ncol=2, fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _format_interval(summary: dict) -> str:
    median = summary["frequency_median"]
    low, high = summary["frequency_interval_68"]
    return f"${median:.3f}^{{+{high - median:.3f}}}_{{-{median - low:.3f}}}$"


def write_tex(summary: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    primary = summary["primary_model"]
    no_knit = summary["series"]["no_knit"][primary]
    knitted = summary["series"]["knitted"][primary]
    delta = summary["frequency_difference_knitted_minus_no_knit"]
    delta_low, delta_high = delta["interval_68"]
    delta_median = delta["median"]
    model_text = "truncated cosine" if primary == "truncated" else "damped cosine"
    content = f"""% Generated by scripts/fit_fig7.py; do not edit by hand.
\\begin{{table}}[t!]
\\centering
\\begin{{tabular}}{{lccc}}\\hline\\hline
data & $C$ & $\\chi^2/\\mathrm{{dof}}$ & AICc \\\\ \\hline
without knitting & {_format_interval(no_knit)} & {no_knit['chi2']:.2f}/{no_knit['dof']} & {no_knit['aicc']:.2f} \\\\
with knitting & {_format_interval(knitted)} & {knitted['chi2']:.2f}/{knitted['dof']} & {knitted['aicc']:.2f} \\\\ \\hline\\hline
\\end{{tabular}}
\\caption{{Weighted {model_text} fits to the nine noisy time points in Fig.~\\ref{{fig:knitting-time-evolution}}. Uncertainties are 68\\% parametric-bootstrap intervals.}}
\\label{{tab:fig7-fits}}
\\end{{table}}

Using the model-selection rule described in the text, the {model_text} is the
primary model.  The fitted knitted-minus-un-knitted frequency difference is
$\\Delta C={delta_median:.3f}^{{+{delta_high - delta_median:.3f}}}_{{-{delta_median - delta_low:.3f}}}$
(68\\% parametric-bootstrap interval).
"""
    output.write_text(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    data = load_data(args.data_dir)
    summary, internals = analyze(data, args.bootstrap_samples, args.seed)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_tex(summary, args.tex)
    make_figure(data, args.data_dir, summary, internals, args.figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
