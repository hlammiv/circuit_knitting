#!/usr/bin/env python3
"""Three panels: raw; readout correction + charge post-selection; the same plus
the depth-matched Clifford rescaling.  Reads data/calibration_line/rescaled.json."""
import json, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paper_figure_style import BLACK, BLUE, ORANGE, FIGSIZE, SERIES_MARKERS, add_trotter_regions, apply_paper_style, finish_axes
d = json.load(open(ROOT / "data" / "calibration_line" / "rescaled.json")); t = np.array(d["t"]); ideal = np.array(d["ideal"])
fixed = np.load(ROOT / "data" / "ferm_num_exact_no_noise_no_knit.npy")[:9]
panels = (("raw", {"no_knit": "raw", "knitted": "raw"}),
          ("readout correction + charge post-selection", {"no_knit": "postsel", "knitted": "full"}),
          ("+ Clifford-calibrated rescaling", {"no_knit": "postsel_rescaled", "knitted": "full_rescaled"}))
apply_paper_style()
fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 2.6, FIGSIZE[1]), sharey=True)
for ax, (title, keys) in zip(axes, panels):
    ax.plot(t, fixed, color=BLACK, linewidth=1.7, label=r"noiseless, $\epsilon=0.05$")
    ax.plot(t, ideal, color=BLACK, linewidth=1.2, linestyle="--", marker="x", markersize=5, label="noiseless, same Trotter steps")
    for series, color, marker, label in (("no_knit", BLUE, SERIES_MARKERS[0], "without knitting"), ("knitted", ORANGE, SERIES_MARKERS[1], "with knitting")):
        y = np.array(d[series][keys[series]]["mean"], float); e = np.array(d[series][keys[series]]["sem"], float)
        ok = np.isfinite(y)
        ax.errorbar(t[ok], y[ok], yerr=e[ok], fmt=marker, color=color, markeredgecolor="white", markeredgewidth=0.5, capsize=2.5, label=label, zorder=4)
    finish_axes(ax, xlabel=r"$t$", ylabel=r"$\langle N_f\rangle$" if ax is axes[0] else "")
    add_trotter_regions(ax, left=-0.10); ax.set_title(title, fontsize=9)
axes[0].legend(fontsize=8, loc="lower left")
fig.tight_layout(); out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper" / "figures" / "mitigation_comparison.pdf")
fig.savefig(out, bbox_inches="tight", dpi=200); print("saved", out)
