#!/usr/bin/env python3
"""Fit and plot the knitted and non-knitted data used in paper Fig. 7."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_distribution

try:
    from .paper_figure_style import (
        BLACK,
        BLUE,
        FIGSIZE,
        ORANGE,
        SERIES_MARKERS,
        add_trotter_regions,
        apply_paper_style,
        finish_axes,
    )
except ImportError:  # Direct execution: python scripts/fit_fig7.py
    from paper_figure_style import (
        BLACK,
        BLUE,
        FIGSIZE,
        ORANGE,
        SERIES_MARKERS,
        add_trotter_regions,
        apply_paper_style,
        finish_axes,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_FIGURE = ROOT / "paper" / "figures" / "fermion_number_with_vs_without_knitting_scatter.pdf"
DEFAULT_JSON = ROOT / "paper" / "generated" / "fig7_fit_results.json"
DEFAULT_TEX = ROOT / "paper" / "generated" / "fig7_fit_results.tex"
DEFAULT_DATA_TEX = ROOT / "paper" / "generated" / "fig7_data_table.tex"
DEFAULT_CACHE = ROOT / "paper" / "generated" / "fig7_bootstrap.npz"


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
    workers: int = 1,
) -> tuple[np.ndarray, int, float]:
    # Generate every pseudo-dataset in the parent process so a fixed seed gives
    # identical draws and ordering regardless of the number of workers.
    pseudo_data = rng.normal(y, sigma, size=(samples, len(y)))
    tasks = ((fit.model, t, y_draw, sigma, fit.parameters) for y_draw in pseudo_data)
    if workers == 1:
        results = map(_fit_bootstrap_draw, tasks)
        draws_or_none = list(results)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            draws_or_none = list(executor.map(_fit_bootstrap_draw, tasks, chunksize=32))
    draws = [draw for draw in draws_or_none if draw is not None]
    failures = len(draws_or_none) - len(draws)
    array = np.asarray(draws, dtype=float)
    if not len(array):
        raise RuntimeError(f"all bootstrap fits failed for {fit.model.name}")
    tolerance = 1e-4 * (fit.model.upper - fit.model.lower)
    near_lower = np.any(array <= fit.model.lower + tolerance, axis=1)
    near_upper = np.any(array >= fit.model.upper - tolerance, axis=1)
    boundary_fraction = float(np.mean(near_lower | near_upper))
    return array, failures, boundary_fraction


def _fit_bootstrap_draw(
    task: tuple[Model, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray | None:
    """Fit one pre-generated pseudo-dataset; suitable for process workers."""
    model, t, y_draw, sigma, initial = task
    try:
        return fit_series(model, t, y_draw, sigma, initial=initial).parameters
    except RuntimeError:
        try:
            return fit_series(model, t, y_draw, sigma).parameters
        except RuntimeError:
            return None


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
    medians = {}
    intervals = {}
    for index, name in enumerate(fit.model.parameter_names):
        values = bootstrap[:, index]
        medians[name] = float(np.median(values))
        intervals[name] = [float(x) for x in np.percentile(values, [16, 84])]
    return {
        "model": fit.model.name,
        "parameters": dict(zip(fit.model.parameter_names, map(float, fit.parameters))),
        "parameter_medians": medians,
        "parameter_intervals_68": intervals,
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


def analyze(
    data: dict[str, np.ndarray], bootstrap_samples: int, seed: int, workers: int = 1
) -> tuple[dict, dict]:
    t = data["t"]
    model_map = models_for_spacing(float(np.min(np.diff(t))))
    rng = np.random.default_rng(seed)
    details: dict[str, dict[str, dict]] = {}
    bootstrap_arrays: dict[tuple[str, str], np.ndarray] = {}
    fits: dict[tuple[str, str], FitResult] = {}
    for series in ("exact", "no_knit", "knitted"):
        details[series] = {}
        for model_name, model in model_map.items():
            fit = fit_series(model, t, data[series], data[f"{series}_sigma"])
            bootstrap, failures, boundary_fraction = bootstrap_fit(
                fit,
                t,
                data[series],
                data[f"{series}_sigma"],
                bootstrap_samples,
                rng,
                workers,
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
    exact_frequency = bootstrap_arrays[("exact", primary)][:, model_map[primary].frequency_index]
    count = min(len(no_knit_frequency), len(knitted_frequency))
    delta = knitted_frequency[:count] - no_knit_frequency[:count]
    comparisons = {}
    for series, frequency in (("no_knit", no_knit_frequency), ("knitted", knitted_frequency)):
        comparison_count = min(len(frequency), len(exact_frequency))
        difference = frequency[:comparison_count] - exact_frequency[:comparison_count]
        comparisons[series] = {
            "median": float(np.median(difference)),
            "interval_68": [float(x) for x in np.percentile(difference, [16, 84])],
            "interval_95": [float(x) for x in np.percentile(difference, [2.5, 97.5])],
            "probability_positive": float(np.mean(difference > 0)),
        }
    summary = {
        "bootstrap_samples_requested": bootstrap_samples,
        "random_seed": seed,
        "bootstrap_workers": workers,
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
        "frequency_difference_from_exact": comparisons,
    }
    summary["bootstrap_convergence"] = bootstrap_convergence(
        bootstrap_arrays, model_map, primary, seed
    )
    internals = {"fits": fits, "bootstrap": bootstrap_arrays, "models": model_map}
    return summary, internals


def bootstrap_convergence(
    arrays: dict[tuple[str, str], np.ndarray],
    models: dict[str, Model],
    primary: str,
    seed: int,
) -> dict[str, dict]:
    """Estimate Monte Carlo error and interval stability of bootstrap quantiles."""
    diagnostics = {}
    frequency_index = models[primary].frequency_index
    for offset, series in enumerate(("exact", "no_knit", "knitted")):
        values = arrays[(series, primary)][:, frequency_index]
        n = len(values)
        full = np.percentile(values, [16, 50, 84])
        half = np.percentile(values[: n // 2], [16, 50, 84])
        width = float(full[2] - full[0])
        blocks = min(10, n // 100)
        half_shift = np.abs(half[[0, 2]] - full[[0, 2]])
        half_shift_fraction = half_shift / width
        if blocks >= 2:
            rng = np.random.default_rng(seed + 1000 + offset)
            shuffled = rng.permutation(values)[: blocks * (n // blocks)].reshape(blocks, -1)
            block_endpoints = np.asarray(
                [np.percentile(block, [16, 84]) for block in shuffled]
            )
            endpoint_mcse = np.std(block_endpoints, axis=0, ddof=1) / np.sqrt(blocks)
            mcse_fraction = endpoint_mcse / width
            endpoint_mcse_json = [float(x) for x in endpoint_mcse]
            mcse_fraction_json = [float(x) for x in mcse_fraction]
            sufficient = bool(
                np.all(mcse_fraction <= 0.01) and np.all(half_shift_fraction <= 0.02)
            )
        else:
            endpoint_mcse_json = [None, None]
            mcse_fraction_json = [None, None]
            sufficient = False
        diagnostics[series] = {
            "samples": n,
            "interval_68": [float(full[0]), float(full[2])],
            "median": float(full[1]),
            "endpoint_mcse": endpoint_mcse_json,
            "endpoint_mcse_fraction_of_interval_width": mcse_fraction_json,
            "half_sample_endpoint_shift": [float(x) for x in half_shift],
            "half_sample_shift_fraction_of_interval_width": [float(x) for x in half_shift_fraction],
            "sufficient": sufficient,
        }
    return diagnostics


def save_analysis_cache(
    path: Path,
    data: dict[str, np.ndarray],
    summary: dict,
    internals: dict,
) -> None:
    """Save fitted parameters and bootstrap draws needed to replot without refitting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": data["t"],
        "samples": np.asarray(summary["bootstrap_samples_requested"], dtype=np.int64),
        "seed": np.asarray(summary["random_seed"], dtype=np.int64),
        "primary_model": np.asarray(summary["primary_model"]),
    }
    for (series, model), fit in internals["fits"].items():
        payload[f"fit__{series}__{model}"] = fit.parameters
        payload[f"bootstrap__{series}__{model}"] = internals["bootstrap"][(series, model)]
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_analysis_cache(
    path: Path,
    data: dict[str, np.ndarray],
    summary: dict,
) -> dict:
    """Load and validate cached fit objects used by ``make_figure``."""
    model_map = models_for_spacing(float(np.min(np.diff(data["t"]))))
    fits = {}
    bootstraps = {}
    with np.load(path, allow_pickle=False) as payload:
        if not np.array_equal(payload["time"], data["t"]):
            raise ValueError("Figure 7 cache time points do not match the input data")
        if int(payload["samples"]) != summary["bootstrap_samples_requested"]:
            raise ValueError("Figure 7 cache sample count does not match the JSON summary")
        if int(payload["seed"]) != summary["random_seed"]:
            raise ValueError("Figure 7 cache seed does not match the JSON summary")
        if str(payload["primary_model"]) != summary["primary_model"]:
            raise ValueError("Figure 7 cache primary model does not match the JSON summary")
        for series in ("exact", "no_knit", "knitted"):
            for model_name, model in model_map.items():
                parameters = np.asarray(payload[f"fit__{series}__{model_name}"], dtype=float)
                result = summary["series"][series][model_name]
                fits[(series, model_name)] = FitResult(
                    model=model,
                    parameters=parameters,
                    covariance=np.empty((len(parameters), len(parameters))),
                    chi2=result["chi2"],
                    dof=result["dof"],
                    p_value=result["p_value"],
                    aicc=result["aicc"],
                )
                bootstraps[(series, model_name)] = np.asarray(
                    payload[f"bootstrap__{series}__{model_name}"], dtype=float
                )
    return {"fits": fits, "bootstrap": bootstraps, "models": model_map}


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
    apply_paper_style()
    output.parent.mkdir(parents=True, exist_ok=True)
    t = data["t"]
    dense_t = np.linspace(t[0], t[-1], 400)
    colors = {"no_knit": BLUE, "knitted": ORANGE}
    labels = {"no_knit": "without knitting", "knitted": "with knitting"}
    raw_files = {
        "no_knit": "noisy_no_knit_trotter_evolution.pkl",
        "knitted": "noisy_knitted_trotter_evolution.pkl",
    }
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        t,
        data["exact"],
        color=BLACK,
        linewidth=1.7,
        label=r"noiseless, $\epsilon=0.05$",
    )
    primary = summary["primary_model"]
    for series in ("no_knit", "knitted"):
        color = colors[series]
        for raw_y, raw_sigma in _raw_samples(data_dir, raw_files[series]):
            ax.errorbar(t, raw_y, yerr=raw_sigma, fmt=".", color=color, alpha=0.14, linewidth=0.5)
        marker = SERIES_MARKERS[0] if series == "no_knit" else SERIES_MARKERS[1]
        for index, (time, value, error) in enumerate(
            zip(t, data[series], data[f"{series}_sigma"])
        ):
            ax.errorbar(
                time,
                value,
                yerr=error,
                fmt=marker,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2.5,
                label=labels[series] if index == 0 else None,
                zorder=4,
            )
        fit = internals["fits"][(series, primary)]
        model = internals["models"][primary]
        boot = internals["bootstrap"][(series, primary)]
        predictions = np.asarray([model.function(dense_t, *parameters) for parameters in boot])
        lower, upper = np.percentile(predictions, [16, 84], axis=0)
        ax.fill_between(dense_t, lower, upper, color=color, alpha=0.16, linewidth=0)
        ax.plot(
            dense_t,
            model.function(dense_t, *fit.parameters),
            color=color,
            linewidth=2.0,
            linestyle="-.",
            alpha=0.80,
        )
    finish_axes(
        ax,
        xlabel=r"$t$",
        ylabel=r"$\langle N_f\rangle$",
    )
    add_trotter_regions(ax, ylim=(0.475, 0.585), left=-0.10)
    ax.legend(ncol=3, fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _format_interval(summary: dict) -> str:
    return _format_parameter(summary, "frequency")


def _rounded_error(error: float) -> tuple[float, int]:
    """Round an uncertainty to two digits only when its first digit is one."""
    exponent = math.floor(math.log10(error))
    leading_digit = int(error / 10**exponent)
    significant_digits = 2 if leading_digit == 1 else 1
    decimals = max(0, -exponent + significant_digits - 1)
    rounded = round(error, decimals)
    rounded_exponent = math.floor(math.log10(rounded))
    decimals = max(0, -rounded_exponent + significant_digits - 1)
    return rounded, decimals


def _format_measurement(value: float, error: float) -> str:
    """Format a symmetric value and uncertainty using the paper's convention."""
    if not np.isfinite(error) or error <= 0:
        return f"${value:.4g}$"
    rounded, decimals = _rounded_error(error)
    uncertainty_digits = int(round(rounded * 10**decimals))
    return f"${value:.{decimals}f}({uncertainty_digits})$"


def _format_parameter(summary: dict, name: str) -> str:
    """Format a bootstrap interval using the paper's uncertainty-digit rule."""
    median = summary["parameter_medians"][name]
    low, high = summary["parameter_intervals_68"][name]
    lower_error = max(median - low, 0.0)
    upper_error = max(high - median, 0.0)
    if not np.isfinite(lower_error) or not np.isfinite(upper_error):
        return f"${median:.4g}$"
    if lower_error <= 0 or upper_error <= 0:
        return f"${median:.4g}$"

    lower_error, lower_decimals = _rounded_error(lower_error)
    upper_error, upper_decimals = _rounded_error(upper_error)
    central_decimals = max(lower_decimals, upper_decimals)
    return (
        f"${median:.{central_decimals}f}"
        f"^{{+{upper_error:.{upper_decimals}f}}}"
        f"_{{-{lower_error:.{lower_decimals}f}}}$"
    )


def add_parameter_intervals(summary: dict, internals: dict) -> None:
    """Populate an existing summary from the parameter draws in its cache."""
    for (series, model_name), bootstrap in internals["bootstrap"].items():
        result = summary["series"][series][model_name]
        model = internals["models"][model_name]
        result["parameter_medians"] = {}
        result["parameter_intervals_68"] = {}
        for index, name in enumerate(model.parameter_names):
            values = bootstrap[:, index]
            result["parameter_medians"][name] = float(np.median(values))
            result["parameter_intervals_68"][name] = [
                float(x) for x in np.percentile(values, [16, 84])
            ]


def write_tex(summary: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    primary = summary["primary_model"]
    exact = summary["series"]["exact"][primary]
    no_knit = summary["series"]["no_knit"][primary]
    knitted = summary["series"]["knitted"][primary]
    delta = summary["frequency_difference_knitted_minus_no_knit"]
    delta_low, delta_high = delta["interval_68"]
    delta_median = delta["median"]
    model_text = "truncated cosine" if primary == "truncated" else "damped cosine"

    content = f"""% Generated by scripts/fit_fig7.py; do not edit by hand.
\\begin{{table*}}[t!]
\\centering
\\begin{{tabular}}{{lccccc}}\\hline\\hline
data & $A$ & $C$ & $D$ & $\\chi^2/\\mathrm{{dof}}$ & AICc \\\\ \\hline
noiseless, $\\epsilon=0.05$ & {_format_parameter(exact, 'amplitude')} & {_format_interval(exact)} & {_format_parameter(exact, 'offset')} & {exact['chi2']:.2f}/{exact['dof']} & {exact['aicc']:.2f} \\\\
noisy, without knitting & {_format_parameter(no_knit, 'amplitude')} & {_format_interval(no_knit)} & {_format_parameter(no_knit, 'offset')} & {no_knit['chi2']:.2f}/{no_knit['dof']} & {no_knit['aicc']:.2f} \\\\
noisy, with knitting & {_format_parameter(knitted, 'amplitude')} & {_format_interval(knitted)} & {_format_parameter(knitted, 'offset')} & {knitted['chi2']:.2f}/{knitted['dof']} & {knitted['aicc']:.2f} \\\\ \\hline\\hline
\\end{{tabular}}
\\caption{{Weighted {model_text} fits to the same nine time points shown in Fig.~\\ref{{fig:knitting-time-evolution}}. Here $A$, $C$, and $D$ are the amplitude, frequency, and offset in Eq.~\\eqref{{eq:trunc_fit}}. Each entry gives the bootstrap median and central 68\\% parametric-bootstrap interval.}}
\\label{{tab:fig7-fits}}
\\end{{table*}}

Using the model-selection rule described in the text, the {model_text} is the
primary model.  The fitted knitted-minus-un-knitted frequency difference is
$\\Delta C={delta_median:.3f}^{{+{delta_high - delta_median:.3f}}}_{{-{delta_median - delta_low:.3f}}}$
(68\\% parametric-bootstrap interval).
"""
    output.write_text(content)


def write_data_table(data: dict[str, np.ndarray], output: Path) -> None:
    """Write the Fig. 7 point values and their measured sampling costs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    group_starts = {
        0: ("0", "0", "1", "$2$", "$2$"),
        1: (r"\multirow{4}{*}{1}", r"\multirow{4}{*}{2}",
            r"\multirow{4}{*}{36}", r"\multirow{4}{*}{$41\text{--}50$}",
            r"\multirow{4}{*}{$39$}"),
        5: (r"\multirow{4}{*}{2}", r"\multirow{4}{*}{4}",
            r"\multirow{4}{*}{1{,}296}", r"\multirow{4}{*}{$98\text{--}116$}",
            r"\multirow{4}{*}{$67\text{--}85$}"),
    }
    for index, time in enumerate(data["t"]):
        if index in group_starts:
            trotter, cuts, terms, cnots_no_knit, cnots_knit = group_starts[index]
        else:
            trotter = cuts = terms = cnots_no_knit = cnots_knit = ""
        rows.append(
            " & ".join(
                [
                    f"${time:.1f}$",
                    trotter,
                    cuts,
                    terms,
                    cnots_no_knit,
                    cnots_knit,
                    _format_measurement(data["no_knit"][index], data["no_knit_sigma"][index]),
                    _format_measurement(data["knitted"][index], data["knitted_sigma"][index]),
                ]
            )
            + r" \\"
        )
        if index in (0, 4):
            rows[-1] += r" \hline"

    content = r"""% Generated by scripts/fit_fig7.py; do not edit by hand.
\begin{table}[t!]
\centering
\small
\begin{tabular}{c|c|c|c|c|c|c|c}\hline\hline
\multirow{2}{*}{$t$} & \multirow{2}{*}{$N_t$} & \multirow{2}{*}{$k$}
& terms
& \multicolumn{2}{c|}{$N_{\mathrm{CNOT}}$}
& \multicolumn{2}{c}{$\langle N_f\rangle$} \\ \cline{5-8}
& & & $(6^k)$ & no knit & knit & no knit & knit \\ \hline
""" + "\n".join(rows) + r"""
\hline\hline
\end{tabular}
\caption{Noisy data plotted in Fig.~\ref{fig:knitting-time-evolution} and the
associated sampling cost.  Each direct circuit and each knitted decomposition
term used $2^{14}=16{,}384$ shots for each of ten independent seeds.  The
aggregate count for each knitted data point is therefore
$N_{\mathrm{shots}}=(6^k)\mathbin{\times}10\mathbin{\times}2^{14}$; each
no-knit point has $N_{\mathrm{shots}}=10\mathbin{\times}2^{14}$.  Here
$k$ counts only the CNOTs replaced by circuit cuts, not all CNOTs in the
transpiled circuits.  The $N_{\mathrm{CNOT}}$ entries give compiled CNOT
counts observed in a compilation-only reconstruction with
\texttt{FakeWashingtonV2}, optimization level one, and eleven transpiler
seeds.  The two-layer knitted range samples six representative decomposition
terms per seed and is not retained metadata from the original runs.
Uncertainties are the one-standard-deviation resampling
uncertainties used in Fig.~\ref{fig:knitting-time-evolution}.}
\label{tab:fig7-data}
\end{table}
"""
    output.write_text(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--data-table", type=Path, default=DEFAULT_DATA_TEX)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes used to compute bootstrap fits (results remain seed-deterministic)",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="write fit summaries and reusable bootstrap cache without plotting",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate JSON and TeX summaries from the existing bootstrap cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    data = load_data(args.data_dir)
    if args.report_only:
        summary = json.loads(args.json.read_text())
        internals = load_analysis_cache(args.cache, data, summary)
        add_parameter_intervals(summary, internals)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        write_tex(summary, args.tex)
        write_data_table(data, args.data_table)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    summary, internals = analyze(data, args.bootstrap_samples, args.seed, args.workers)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_tex(summary, args.tex)
    write_data_table(data, args.data_table)
    save_analysis_cache(args.cache, data, summary, internals)
    if not args.analysis_only:
        make_figure(data, args.data_dir, summary, internals, args.figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
