#!/usr/bin/env python3
"""Shot-noise-free comparison of zero-noise extrapolation, knitted vs un-knitted.

Uses the exact distributions of zne_knitted.py and zne_direct_exact.py at noise
scales 1, 1.5, 2, so what remains is the error of the extrapolation itself.  The
shot-noise column is the spread of the 10 sampled replicas at these same
settings (16384 shots per term knitted, 16384 per run un-knitted), taken from
the replica arrays of zne_knitted.py and from zne_analysis.py.  Un-knitted values
are averaged over the 10 transpiler seeds; the spread between seeds is reported.
Writes data/zne_exact_compare.json.
"""
import json, sys, warnings
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
SCALES = (1.0, 1.5, 2.0); LABEL = ("1", "1.5", "2")
SLOTS = [(1, e) for e in (0.2, 0.4, 0.6, 0.8)] + [(2, e) for e in (0.5, 0.6, 0.7, 0.8)]
VARIANTS = ("raw", "postsel", "full"); METHODS = ("none", "linear", "exponential", "quadratic")
idx = np.arange(2 ** N)
charge = sum(((idx >> q) & 1) for q in (0, 2, 8, 10)) + (1 - ((idx >> 4) & 1)) + (1 - ((idx >> 6) & 1))
sector = charge == 3
bit6 = ((idx >> (MID + 2)) & 1).astype(bool)
ideal = np.load(ROOT / "mass_scan_plot_data" / "ferm_num_m1p125_trot.npy")[1:9]


def extrapolate(y):
    x = np.array(SCALES)
    ratio = (y[0] - 0.5) / (y[1] - 0.5)
    return (y[0], y[0] - (y[1] - y[0]) * 2,
            0.5 + (y[0] - 0.5) * ratio ** 2 if ratio > 0 else np.nan,
            np.polyval(np.polyfit(x, y, 2), 0.0))


def knitted_value(v, variant):
    v = v[1 if variant == "full" else 0]
    return v[0] / v[1] if variant == "raw" else v[2] / v[3]


def knitted():
    """exact[variant, slot, method] and replicas[variant, replica, slot, method]"""
    exact = np.full((3, 8, 4), np.nan); reps = np.full((3, 10, 8, 4), np.nan)
    for si, (steps, eps) in enumerate(SLOTS):
        n = 2 * steps
        pref = np.array([float(np.prod(e)) for e in product(*[[1/2, 1/2, -1/2, 1/2, -1/2, 1/2]] * n)])
        ex, rp = [], []
        for label in LABEL:
            d = r.DATA / "zne_knitted" / f"s{label}" / f"step{steps}_eps{eps:g}"
            loaded = [np.load(d / f"term{j:04d}.npz") for j in range(6 ** n)]
            ex.append(np.einsum("j,jvk->vk", pref, np.array([f["exact"] for f in loaded])))
            rp.append(np.einsum("j,jrvk->rvk", pref, np.array([f["replicas"] for f in loaded])))
        for vi, variant in enumerate(VARIANTS):
            exact[vi, si] = extrapolate([knitted_value(e, variant) for e in ex])
            for ri in range(10):
                reps[vi, ri, si] = extrapolate([knitted_value(p[ri], variant) for p in rp])
    return exact, reps


def direct():
    """exact[variant, seed, slot, method]; the readout inverse is of the qubit actually read."""
    out = np.full((3, len(r.NOISY_SEEDS), 8, 4), np.nan); moved = 0
    y = np.full((3, len(r.NOISY_SEEDS), 8, 3), np.nan)
    line = r.line_layout()
    for ci, (scale, label) in enumerate(zip(SCALES, LABEL)):
        nm = AerSimulator.from_backend(cd.scaled_washington(scale)()).options.noise_model
        inv = {q: np.linalg.inv(np.asarray(nm._local_readout_errors[(q,)].probabilities, float)) for q in line}
        for ri, seed in enumerate(r.NOISY_SEEDS):
            for si in range(8):
                f = np.load(r.DATA / "zne_direct_exact" / f"s{label}" / f"seed{seed}_slot{si}.npz")
                v = f["p"]; physical = [int(q) for q in f["physical"]]
                moved += physical != list(line)
                p = v.reshape((2,) * N)
                for q, m in enumerate(physical):
                    axis = N - 1 - q
                    p = np.moveaxis(np.tensordot(inv[m].T, p, axes=([1], [axis])), 0, axis)
                w = p.reshape(-1)
                y[0, ri, si, ci] = v[bit6].sum() / v.sum()
                y[1, ri, si, ci] = v[bit6 & sector].sum() / v[sector].sum()
                y[2, ri, si, ci] = w[bit6 & sector].sum() / w[sector].sum()
    for vi in range(3):
        for ri in range(len(r.NOISY_SEEDS)):
            for si in range(8):
                out[vi, ri, si] = extrapolate(y[vi, ri, si])
    print(f"un-knitted runs whose final qubits are permuted from the line order: {moved} of 240")
    return out


kx, kr = knitted(); dx = direct()
sampled = json.load(open(r.DATA / "zne_line.json"))
res = {}
for vi, variant in enumerate(VARIANTS):
    print(f"\n{variant}: two-step, deviation from exact (shot-noise-free); sem = 10-replica shot error")
    for mi, method in enumerate(METHODS):
        kd = kx[vi, :, mi] - ideal
        dd = np.nanmean(dx[vi, :, :, mi], 0) - ideal
        seeds = np.nanstd(dx[vi, :, :, mi], 0, ddof=1)
        ksem = np.nanstd(kr[vi, :, :, mi], 0, ddof=1) / np.sqrt(10)
        key = f"{variant}_{method}"
        dsem = np.array(sampled["no_knit"][key]["sem"])
        res[key] = {"knitted_dev": kd.tolist(), "direct_dev": dd.tolist(), "direct_seed_spread": seeds.tolist(),
                    "knitted_sem": ksem.tolist(), "direct_sem": dsem.tolist()}
        print(f"  {method:12s} knitted |dev| {np.abs(kd[4:]).mean():.4f} sem {ksem[4:].mean():.4f} signed {np.round(kd[4:], 4)}"
              f" | un-knitted |dev| {np.abs(dd[4:]).mean():.4f} sem {dsem[4:].mean():.4f} seed-spread {seeds[4:].mean():.4f} signed {np.round(dd[4:], 4)}")
    print(f"  one-step |dev| knitted {np.round(np.abs(kx[vi, :4] - ideal[:4, None]).mean(0), 4)} un-knitted {np.round(np.abs(np.nanmean(dx[vi, :, :4], 0) - ideal[:4, None]).mean(0), 4)}")
json.dump(res, open(r.DATA / "zne_exact_compare.json", "w"), indent=1)
