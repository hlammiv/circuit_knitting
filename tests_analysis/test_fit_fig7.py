from __future__ import annotations

import numpy as np
import pytest

from scripts.fit_fig7 import analyze, fit_series, models_for_spacing, truncated_cosine


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
