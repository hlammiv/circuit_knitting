#!/usr/bin/env python3
"""Term-class rescaling of the knitted line runs (data/knitted_terms/*.npz from
knitted_term_values.py).

The knitted estimate of <n_6> is 1/2 + sum_j c_j D_j / sum_j c_j N_j with, per
term j, D_j = A_j - N_j/2 (signed weight of the occupied strings minus half the
signed total) and N_j the signed total.  Noise shrinks D_j toward 0 and, for
terms with mid-circuit measurements, also N_j.  A single factor f per step
count (global Clifford rescaling) assumes every term shrinks alike.  Here the
terms are grouped into classes and each class sum of D (and of N) is divided by
its own factor, taken from the Clifford calibration run's same class:
  f_class = sum_class c D_cal(noisy) / sum_class c D_cal(exact).
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "data" / "knitted_terms"
ideal_curve = np.load(ROOT / "mass_scan_plot_data" / "ferm_num_m1p125_trot.npy")
EPS = (0.5, 0.6, 0.7, 0.8)
SLOT = {e: 5 + i for i, e in enumerate(EPS)}
drift = sys.argv[1] if len(sys.argv) > 1 else ""        # e.g. "_drift_s0.9"


def load(name):
    d = np.load(T / name); v = d["values"]
    if v.ndim == 2:                                  # exact values: same for raw and readout-corrected
        v = np.stack([v, v], 1)
    return v, d["mids"], d["prefactors"]


def classes(mids, scheme, num_cx):
    t = np.arange(len(mids))
    digits = np.array([[(j // 6 ** (num_cx - 1 - p)) % 6 for p in range(num_cx)] for j in t])
    kind = np.select([digits < 2, digits < 4], [0, 1], 2)            # rotation / measure target / measure control
    if scheme == "global": return np.zeros(len(mids), int)
    if scheme == "mids": return mids
    if scheme == "pattern": return (kind * 3 ** np.arange(num_cx)).sum(1)
    if scheme == "term": return t


def class_sums(values, prefactors, cls, variant, post):
    """per class: (sum c D, sum c N) for the chosen variant (0 raw, 1 readout) and post-selection."""
    A = values[:, variant, 2 if post else 0]; N = values[:, variant, 3 if post else 1]
    D = A - N / 2
    out = {}
    for k in np.unique(cls):
        m = cls == k
        out[k] = np.array([(prefactors[m] * D[m]).sum(), (prefactors[m] * N[m]).sum()])
    return out


def estimate(phys, cal_noisy, cal_ideal, floor=1e-3):
    """1/2 + sum_k D_k/f_k / sum_k N_k/g_k; classes whose exact calibration sum is ~0 keep factor 1."""
    num = den = 0.0
    for k, (D, N) in phys.items():
        Dc, Nc = cal_noisy[k]; Di, Ni = cal_ideal[k]
        f = Dc / Di if abs(Di) > floor else 1.0
        g = Nc / Ni if abs(Ni) > floor else 1.0
        num += D / f; den += N / g
    return 0.5 + num / den


cal_noisy_v, mids, pref = load(f"cal{drift}_step2.npz")
cal_ideal_v, mids_i, _ = load("ideal_cal_step2.npz")
assert (mids == mids_i).all()
results = {}
for scheme in ("global", "mids", "pattern"):
    cls = classes(mids, scheme, 4)
    for variant, post, label in ((0, False, "raw"), (0, True, "postsel"), (1, True, "full")):
        cn = class_sums(cal_noisy_v, pref, cls, variant, post); ci = class_sums(cal_ideal_v, pref, cls, 0, post)
        per_eps = {}
        for e in EPS:
            vals = []
            for f in sorted(T.glob(f"step2_eps{e:g}_seed*.npz")):
                v, m, p = load(f.name)
                vals.append(estimate(class_sums(v, p, cls, variant, post), cn, ci))
            per_eps[e] = vals
        results[(scheme, label)] = per_eps
        dev = [np.mean(per_eps[e]) - ideal_curve[SLOT[e]] for e in EPS if per_eps[e]]
        n = [len(per_eps[e]) for e in EPS]
        print(f"{scheme:8s} {label:8s} replicas {n}  signed dev {np.round(dev, 4)}  mean|dev| {np.mean(np.abs(dev)):.4f}")

# diagnostic: per-class contraction of D in the calibration vs in the physics (exact noiseless physics known)
print("\nclass (mid-circuit measurements): contraction of sum c D, calibration vs physics eps=0.8 (raw counts)")
cls = classes(mids, "mids", 4)
cn = class_sums(cal_noisy_v, pref, cls, 0, False); ci = class_sums(cal_ideal_v, pref, cls, 0, False)
pi = class_sums(load("ideal_step2_eps0.8.npz")[0], pref, cls, 0, False)
phys = [class_sums(load(f.name)[0], pref, cls, 0, False) for f in sorted(T.glob("step2_eps0.8_seed*.npz"))]
for k in cn:
    pn = np.mean([p[k] for p in phys], 0) if phys else [np.nan, np.nan]
    print(f"  k={k}: cal D {ci[k][0]:+.4f} -> {cn[k][0]:+.4f} (f {cn[k][0]/ci[k][0] if abs(ci[k][0])>1e-9 else np.nan:.3f}),"
          f" N {ci[k][1]:+.4f} -> {cn[k][1]:+.4f} | phys D {pi[k][0]:+.4f} -> {pn[0]:+.4f} (f {pn[0]/pi[k][0] if abs(pi[k][0])>1e-9 else np.nan:.3f}),"
          f" N {pi[k][1]:+.4f} -> {pn[1]:+.4f}")
