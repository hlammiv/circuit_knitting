#!/usr/bin/env python3
"""How much of the knitted shot budget does the two-step estimate need?

Two options, both evaluated from the per-term arrays of zne_knitted.py:

1. Optimal shot allocation among all terms.  A term gets shots in proportion to
   |prefactor| x (per-shot standard deviation of its contribution to the
   estimate), the standard minimum-variance rule.  The gain over equal shots is
   computed from the variance of the term contributions over the replicas.
2. Keeping only a subset of terms: those with no mid-circuit measurement or few
   of them (a structural rule), or the largest ideal signal (needs classical
   simulation, as an upper limit).  This adds a bias from the dropped terms,
   which the exact arrays measure, at every noise scale and after extrapolation.

Writes data/knit_shot_budget.json.
"""
import json, sys, warnings
from itertools import product
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np

DATA = ROOT / "data"
LABEL = ("1", "1.5", "2"); X = np.array([1.0, 1.5, 2.0])
SLOTS = [(2, e) for e in (0.5, 0.6, 0.7, 0.8)]
VARIANTS = ("raw", "postsel", "full"); METHODS = ("none", "linear", "exponential", "quadratic")
ideal = np.load(ROOT / "mass_scan_plot_data" / "ferm_num_m1p125_trot.npy")[5:9]
idx = {"raw": (0, 0, 1), "postsel": (0, 2, 3), "full": (1, 2, 3)}


def extrapolate(y):
    ratio = (y[0] - 0.5) / (y[1] - 0.5)
    return np.array([y[0], y[0] - (y[1] - y[0]) * 2, 0.5 + (y[0] - 0.5) * ratio ** 2 if ratio > 0 else np.nan,
                     np.polyval(np.polyfit(X, y, 2), 0.0)])


def value(s, variant):
    v, a, n = idx[variant]
    return s[..., v, a] / s[..., v, n]


loaded = {}
for steps, eps in SLOTS:
    n = 2 * steps
    pref = np.array([float(np.prod(e)) for e in product(*[[1/2, 1/2, -1/2, 1/2, -1/2, 1/2]] * n)])
    ex, rp = [], []
    for label in LABEL:
        d = DATA / "zne_knitted" / f"s{label}" / f"step{steps}_eps{eps:g}"
        files = [np.load(d / f"term{j:04d}.npz") for j in range(6 ** n)]
        ex.append(np.array([f["exact"] for f in files])); rp.append(np.array([f["replicas"] for f in files]))
        mids = np.array([int(f["mids"]) for f in files])
    ide = np.load(DATA / "knitted_terms" / f"ideal_step{steps}_eps{eps:g}.npz")["values"]
    loaded[eps] = dict(pref=pref, ex=np.array(ex), rp=np.array(rp), mids=mids, ideal=ide)


def subsets(d):
    # ideal signal of each term in the raw estimate: prefactor x (A - N/2)
    sig = np.abs(d["pref"] * (d["ideal"][:, 0] - d["ideal"][:, 1] / 2)) if d["ideal"].ndim == 2 else None
    out = {"all": np.ones(len(d["pref"]), bool)}
    for k in (0, 1, 2, 3):
        out[f"mids<={k}"] = d["mids"] <= k
    for k in (16, 64, 256):
        keep = np.zeros(len(d["pref"]), bool); keep[np.argsort(-sig)[:k]] = True
        out[f"top{k}"] = keep
    return out


res = {"terms_by_mids": {int(k): int((loaded[0.8]["mids"] == k).sum()) for k in np.unique(loaded[0.8]["mids"])}}
print("terms by number of mid-circuit measurements:", res["terms_by_mids"])
names = list(subsets(loaded[0.8]))
for variant in VARIANTS:
    print(f"\n{variant}: two-step mean over slots.  |bias| from exact arrays; sem = 10-replica shot error at 16384 shots per kept term")
    res[variant] = {}
    for name in names:
        bias, sem = [], []
        for ti, (steps, eps) in enumerate(SLOTS):
            d = loaded[eps]; keep = subsets(d)[name]; p = d["pref"][keep]
            y = [value(np.einsum("j,jvk->vk", p, d["ex"][c][keep]), variant) for c in range(3)]
            bias.append(extrapolate(y) - ideal[ti])
            reps = np.array([extrapolate([value(np.einsum("j,jvk->vk", p, d["rp"][c][keep, ri]), variant) for c in range(3)]) for ri in range(10)])
            sem.append(np.nanstd(reps, 0, ddof=1) / np.sqrt(10))
        bias = np.array(bias); sem = np.array(sem); kept = int(subsets(loaded[0.8])[name].sum())
        res[variant][name] = {"terms": kept, "bias": bias.tolist(), "sem": sem.tolist()}
        print(f"  {name:8s} terms {kept:5d}  " + "  ".join(f"{m[:3]} {np.abs(bias[:, i]).mean():.4f}/{np.nanmean(sem[:, i]):.4f}" for i, m in enumerate(METHODS)))

# optimal allocation among all terms: variance of each term's contribution to the estimate
print("\noptimal shot allocation vs equal shots (variance ratio at the same total shots, scale 1)")
res["allocation"] = {}
for variant in VARIANTS:
    gains = []
    for steps, eps in SLOTS:
        d = loaded[eps]; v, a, n = idx[variant]
        r = d["rp"][0]                                          # terms x replicas x variant x 4
        total = d["pref"] @ d["ex"][0][:, v, n]; val = (d["pref"] @ d["ex"][0][:, v, a]) / total
        contrib = d["pref"][:, None] * (r[:, :, v, a] - val * r[:, :, v, n]) / total
        sd = contrib.std(1, ddof=1)
        by_mid = np.array([np.sqrt((sd[d["mids"] == m] ** 2).mean()) for m in np.unique(d["mids"])])     # pooled per class
        pooled = by_mid[np.searchsorted(np.unique(d["mids"]), d["mids"])]
        gains.append((pooled ** 2).sum() * len(sd) / pooled.sum() ** 2)
    res["allocation"][variant] = gains
    print(f"  {variant:8s} equal/optimal variance {np.round(gains, 3)}")
json.dump(res, open(DATA / "knit_shot_budget.json", "w"), indent=1)

# ---- rotation terms only (no mid-circuit measurement) vs the un-knitted runs, every time slot, at equal total shots
print("\nrotation terms vs un-knitted, equal total shots per point: 2^(cut gates) x 16384 (one replica)")
ideal_all = np.load(ROOT / "mass_scan_plot_data" / "ferm_num_m1p125_trot.npy")[1:9]
direct = json.load(open(DATA / "zne_exact_compare.json"))
ALL = [(1, e) for e in (0.2, 0.4, 0.6, 0.8)] + SLOTS
res["rotation_vs_direct"] = {}
for variant in VARIANTS:
    for mi, method in enumerate(METHODS[1:], 1):
        rows = []
        for si, (steps, eps) in enumerate(ALL):
            n = 2 * steps
            terms = [j for j in range(6 ** n) if not any((j // 6 ** d) % 6 > 1 for d in range(n))]
            files = [[np.load(DATA / "zne_knitted" / f"s{l}" / f"step{steps}_eps{eps:g}" / f"term{j:04d}.npz") for j in terms] for l in LABEL]
            p = 0.5 ** n
            y = [value(p * np.array([f["exact"] for f in fc]).sum(0), variant) for fc in files]
            reps = np.array([extrapolate([value(p * np.array([f["replicas"][ri] for f in fc]).sum(0), variant) for fc in files]) for ri in range(10)])
            d = direct[f"{variant}_{method}"]
            rows.append((extrapolate(y)[mi] - ideal_all[si], np.nanstd(reps[:, mi], ddof=1), d["direct_dev"][si],
                         d["direct_sem"][si] * np.sqrt(10) / np.sqrt(len(terms))))   # one rotation-set run; un-knitted with the same shots
        rows = np.array(rows)
        res["rotation_vs_direct"][f"{variant}_{method}"] = rows.tolist()
        for name, part in (("one-step", rows[:4]), ("two-step", rows[4:])):
            print(f"  {variant:8s}{method:12s}{name}: rotation |bias| {np.abs(part[:, 0]).mean():.4f} shot {part[:, 1].mean():.4f}"
                  f" | un-knitted |bias| {np.abs(part[:, 2]).mean():.4f} shot {part[:, 3].mean():.4f}")
json.dump(res, open(DATA / "knit_shot_budget.json", "w"), indent=1)
