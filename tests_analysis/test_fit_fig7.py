from __future__ import annotations

import numpy as np
import pytest

from scripts.fit_fig7 import (
    analyze,
    fit_series,
    load_analysis_cache,
    models_for_spacing,
    save_analysis_cache,
    truncated_cosine,
    write_data_table,
)


def test_truncated_fit_recovers_synthetic_frequency() -> None:
    t = np.linspace(0.0, 1.6, 9)
    sigma = np.full_like(t, 0.002)
    y = truncated_cosine(t, 0.04, 1.9, 0.50)
    model = models_for_spacing(0.2)["truncated"]
    result = fit_series(model, t, y, sigma)
    assert result.parameters[model.frequency_index] == pytest.approx(1.9, abs=1e-5)
    assert result.chi2 < 1e-8


def test_analysis_is_deterministic_and_defaults_to_truncated() -> None:
    t = np.linspace(0.0, 1.6, 9)
    sigma = np.full_like(t, 0.004)
    no_knit = truncated_cosine(t, 0.04, 1.8, 0.50)
    knitted = truncated_cosine(t, 0.038, 1.85, 0.501)
    data = {
        "t": t,
        "exact": truncated_cosine(t, 0.04, 1.9, 0.50),
        "exact_sigma": np.full_like(t, 0.001),
        "no_knit": no_knit,
        "no_knit_sigma": sigma,
        "knitted": knitted,
        "knitted_sigma": sigma,
    }
    first, _ = analyze(data, bootstrap_samples=100, seed=17)
    second, _ = analyze(data, bootstrap_samples=100, seed=17)
    assert first == second
    assert first["primary_model"] == "truncated"


def test_rejects_nonpositive_uncertainty() -> None:
    t = np.linspace(0.0, 1.6, 9)
    y = truncated_cosine(t, 0.04, 1.9, 0.50)
    sigma = np.full_like(t, 0.002)
    sigma[3] = 0.0
    model = models_for_spacing(0.2)["truncated"]
    with pytest.raises(ValueError, match="positive"):
        fit_series(model, t, y, sigma)


def test_analysis_cache_round_trip(tmp_path) -> None:
    t = np.linspace(0.0, 1.6, 9)
    sigma = np.full_like(t, 0.004)
    data = {
        "t": t,
        "exact": truncated_cosine(t, 0.04, 1.9, 0.50),
        "exact_sigma": np.full_like(t, 0.001),
        "no_knit": truncated_cosine(t, 0.04, 1.8, 0.50),
        "no_knit_sigma": sigma,
        "knitted": truncated_cosine(t, 0.038, 1.85, 0.501),
        "knitted_sigma": sigma,
    }
    summary, internals = analyze(data, bootstrap_samples=100, seed=19)
    cache = tmp_path / "bootstrap.npz"
    save_analysis_cache(cache, data, summary, internals)
    restored = load_analysis_cache(cache, data, summary)
    for key, values in internals["bootstrap"].items():
        np.testing.assert_array_equal(restored["bootstrap"][key], values)
    assert set(summary["bootstrap_convergence"]) == {"exact", "no_knit", "knitted"}


def test_data_table_records_knitting_costs(tmp_path) -> None:
    t = np.linspace(0.0, 1.6, 9)
    data = {
        "t": t,
        "no_knit": np.full_like(t, 0.5),
        "no_knit_sigma": np.full_like(t, 0.006),
        "knitted": np.full_like(t, 0.51),
        "knitted_sigma": np.full_like(t, 0.01),
    }
    output = tmp_path / "fig7_data_table.tex"
    write_data_table(data, output)
    table = output.read_text()
    assert r"\multirow{4}{*}{36}" in table
    assert r"\multirow{4}{*}{1{,}296}" in table
    assert r"N_{\mathrm{shots}}=(6^k)\mathbin{\times}10\mathbin{\times}2^{14}" in table
    assert r"\begin{table}[t!]" in table
    assert r"$0.500(6)$" in table
