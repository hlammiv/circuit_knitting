from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.plot_fig5 import load_series, make_figure as make_fig5
from scripts.plot_fig6 import load_array, make_figure as make_fig6


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "mass_scan_plot_data"


def test_figure5_source_data_are_complete() -> None:
    for stem in ("m1p0_e0p005", "m1p125_e0p005", "m1p25_e0p005"):
        values, errors = load_series(DATA, stem)
        assert values.shape == (33,)
        assert errors.shape == values.shape
        assert np.all(errors > 0)


def test_figure6_source_data_are_complete() -> None:
    for name in (
        "ferm_num_m1p125_e0p005",
        "bs_error_m1p125_e0p005",
        "ferm_num_m1p125_trot",
        "bs_error_m1p125_trot",
    ):
        assert load_array(DATA, name, 9).shape == (9,)


def test_figure_loaders_reject_bad_shapes(tmp_path: Path) -> None:
    np.save(tmp_path / "ferm_num_bad.npy", np.zeros(4))
    np.save(tmp_path / "bs_error_bad.npy", np.ones(4))
    with pytest.raises(ValueError, match="33"):
        load_series(tmp_path, "bad")


def test_figures_generate_pdf_files(tmp_path: Path) -> None:
    fig5 = tmp_path / "fig5.pdf"
    fig6 = tmp_path / "fig6.pdf"
    make_fig5(DATA, fig5)
    make_fig6(DATA, fig6)
    assert fig5.read_bytes().startswith(b"%PDF")
    assert fig6.read_bytes().startswith(b"%PDF")
