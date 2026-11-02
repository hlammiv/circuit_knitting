#!/usr/bin/env python3
"""Zero-noise extrapolation of both line series from the runs at noise scale
1 (nominal), 1.5 and 2 (zne_knitted.py, zne_direct.py).

Per replica and time slot the value of <n_6> at each scale is extrapolated to
scale 0: linear through 1 and 1.5, exponential decay toward 1/2 through 1 and
1.5, and quadratic (Richardson) through all three.  Done for the raw counts, the
charge-post-selected counts, and readout-corrected + post-selected counts
(readout matrices of the run's own scale).  Writes data/zne_line.json.
"""
import json, pickle, sys, warnings
from itertools import product
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
import rerun_sec5 as r
import calibration_drift as cd
from qiskit_aer import AerSimulator

r.LAYOUT = "line"
N = 12; MID = r.MID
SCALES = (1.0, 1.5, 2.0)
SLOTS = [(1, e) for e in (0.2, 0.4, 0.6, 0.8)] + [(2, e) for e in (0.5, 0.6, 0.7, 0.8)]
VARIANTS = ("raw", "postsel", "full")
idx = np.arange(2 ** N)
charge = sum(((idx >> q) & 1) for q in (0, 2, 8, 10)) + (1 - ((idx >> 4) & 1)) + (1 - ((idx >> 6) & 1))
sector = charge == 3
bit6 = ((idx >> (MID + 2)) & 1).astype(bool)
ideal = np.load(ROOT / "mass_scan_plot_data" / "ferm_num_m1p125_trot.npy")


def extrapolate(y):
    """y: values at SCALES (missing scales as nan) -> dict of zero-noise estimates."""
    x = np.array(SCALES); out = {"none": y[0]}
    if not np.isnan(y[1]):
        out["linear"] = y[0] - (y[1] - y[0]) / (x[1] - x[0]) * x[0]
        ratio = (y[0] - 0.5) / (y[1] - 0.5)
        out["exponential"] = 0.5 + (y[0] - 0.5) * ratio ** (x[0] / (x[1] - x[0])) if ratio > 0 else np.nan
    if not np.isnan(y[1:]).any():
        out["quadratic"] = np.polyval(np.polyfit(x, y, 2), 0.0)
    return out


# ---- knitted: per-term signed weights, combined with the knitting prefactors
def knitted_value(four_sum, variant):
    v = four_sum[1 if variant == "full" else 0]       # [0] raw counts, [1] readout-corrected
    return v[0] / v[1] if variant == "raw" else v[2] / v[3]


def knitted_series():
    """values[variant][seed index, slot, scale]"""
    vals = {k: np.full((len(r.KNIT_SEEDS), len(SLOTS), len(SCALES)), np.nan) for k in VARIANTS}
    for si, (steps, eps) in enumerate(SLOTS):
        num_cx = 2 * steps
        pref = np.array([float(np.prod(e)) for e in product(*[[1/2, 1/2, -1/2, 1/2, -1/2, 1/2]] * num_cx)])
        for ri, seed in enumerate(r.KNIT_SEEDS):
            f = r.DATA / "knitted_terms" / f"step{steps}_eps{eps:g}_seed{seed}.npz"
            if f.exists():
                s = np.einsum("j,jvk->vk", pref, np.load(f)["values"])
                for k in VARIANTS: vals[k][ri, si, 0] = knitted_value(s, k)
        for ci, scale in enumerate(SCALES[1:], 1):
            d = r.DATA / "zne_knitted" / f"s{scale:g}" / f"step{steps}_eps{eps:g}"
            files = [d / f"term{j:04d}.npz" for j in range(6 ** num_cx)]
            if not all(f.exists() for f in files):
                continue
            reps = np.array([np.load(f)["replicas"] for f in files])          # terms x seeds x variant x 4
            s = np.einsum("j,jrvk->rvk", pref, reps)
            for k in VARIANTS:
                for ri in range(len(r.KNIT_SEEDS)): vals[k][ri, si, ci] = knitted_value(s[ri], k)
    return vals


# ---- un-knitted: counts per replica and slot
def inverses(scale):
    nm = AerSimulator.from_backend(cd.scaled_washington(scale)()).options.noise_model
    return [np.linalg.inv(np.asarray(nm._local_readout_errors[(q,)].probabilities, float)) for q in r.line_layout()]


def readout(v, mats):
    p = v.reshape((2,) * N)
    for q, m in enumerate(mats):
        axis = N - 1 - q
        p = np.moveaxis(np.tensordot(m.T, p, axes=([1], [axis])), 0, axis)
    return p.reshape(-1)


def direct_series():
    vals = {k: np.full((len(r.NOISY_SEEDS), len(SLOTS), len(SCALES)), np.nan) for k in VARIANTS}
    for ci, scale in enumerate(SCALES):
        path = (r.DATA / "noisy_no_knit_trotter_evolution_line_aer.pkl" if scale == 1.0 else
                r.DATA / "zne_direct" / f"noisy_no_knit_trotter_evolution_line_aer_s{scale:g}.pkl")
        if not path.exists():
            continue
        runs = pickle.load(open(path, "rb")); mats = inverses(scale)
        for ri, seed in enumerate(r.NOISY_SEEDS):
            for si in range(len(SLOTS)):
                v = np.zeros(2 ** N)
                for s, c in runs[seed]["counts"][si + 1].items(): v[int(s, 2)] = c
                w = readout(v, mats)
                vals["raw"][ri, si, ci] = v[bit6].sum() / v.sum()
                vals["postsel"][ri, si, ci] = v[bit6 & sector].sum() / v[sector].sum()
                vals["full"][ri, si, ci] = w[bit6 & sector].sum() / w[sector].sum()
    return vals


out = {"t": [steps * e for steps, e in SLOTS], "ideal": [float(ideal[i + 1]) for i in range(len(SLOTS))]}
target = np.array(out["ideal"])
for name, series in (("no_knit", direct_series()), ("knitted", knitted_series())):
    out[name] = {}
    for variant in VARIANTS:
        a = series[variant]
        per = [[extrapolate(a[ri, si]) for si in range(len(SLOTS))] for ri in range(a.shape[0])]
        for method in ("none", "linear", "exponential", "quadratic"):
            m = np.array([[p[si].get(method, np.nan) for si in range(len(SLOTS))] for p in per])
            ok = ~np.isnan(m).any(1)
            if not ok.any():
                continue
            m = m[ok]
            mean = m.mean(0); sem = m.std(0, ddof=1) / np.sqrt(len(m)) if len(m) > 1 else np.full(len(SLOTS), np.nan)
            out[name][f"{variant}_{method}"] = {"mean": mean.tolist(), "sem": sem.tolist(), "replicas": int(ok.sum())}
            dev = mean - target
            print(f"{name:8s} {variant:8s} {method:12s} n={ok.sum():2d}  one-step mean|dev| {np.abs(dev[:4]).mean():.4f}"
                  f"  two-step mean|dev| {np.abs(dev[4:]).mean():.4f}  signed {np.round(dev[4:], 4)}  sem {np.nanmean(sem[4:]):.4f}")
json.dump(out, open(r.DATA / "zne_line.json", "w"), indent=1)
