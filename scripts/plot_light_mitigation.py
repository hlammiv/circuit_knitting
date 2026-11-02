#!/usr/bin/env python3
"""Raw vs readout correction + charge post-selection (no calibration circuits):
values (left) and deviation from the same-step noiseless curve (right)."""
import json, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paper_figure_style import BLACK, BLUE, ORANGE, FIGSIZE, add_trotter_regions, apply_paper_style, finish_axes
src = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "data" / "mitigation_combined_level.json")
d = json.load(open(src)); t = np.array(d["t"]); ideal = np.array(d["ideal"])
series = (("no_knit", "raw", BLUE, "o", "without knitting, raw", dict(mfc="white")),
          ("knitted", "raw", ORANGE, "s", "with knitting, raw", dict(mfc="white")),
          ("no_knit", "both", BLUE, "o", "without knitting, readout + post-sel.", {}),
          ("knitted", "both_full", ORANGE, "s", "with knitting, readout + post-sel.", {}))
apply_paper_style()
fig, (a, b) = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2.1, FIGSIZE[1]))
a.plot(t, ideal, color=BLACK, lw=1.2, ls="--", marker="x", ms=5, label="noiseless, same Trotter steps")
b.axhline(0, color=BLACK, lw=1.0, ls="--")
for k, (s, key, c, m, lab, kw) in enumerate(series):
    y = np.array(d[s][key]["mean"], float); e = np.array(d[s][key]["sem"], float); dx = (k - 1.5) * 0.012
    a.errorbar(t + dx, y, yerr=e, fmt=m, color=c, capsize=2, label=lab, zorder=4, **kw)
    b.errorbar(t + dx, y - ideal, yerr=e, fmt=m, color=c, capsize=2, zorder=4, **kw)
    if t.size == 9:
        print(f"{lab:42s} two-step mean |bias| = {np.mean(np.abs(y - ideal)[5:]):.4f}")
finish_axes(a, xlabel=r"$t$", ylabel=r"$\langle N_f\rangle$"); finish_axes(b, xlabel=r"$t$", ylabel=r"deviation from noiseless")
for ax in (a, b): add_trotter_regions(ax, left=-0.10)
a.legend(fontsize=7, loc="lower left")
fig.tight_layout(); out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper" / "figures" / "light_mitigation_comparison.png")
fig.savefig(out, bbox_inches="tight", dpi=200); print("saved", out)
