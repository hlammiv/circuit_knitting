#!/usr/bin/env python3
"""Zero-noise extrapolation of observables at and away from the cut, knitted vs
un-knitted, from the exact distributions at noise scales 1, 1.5, 2.

Knitted: data/zne_joint (joint[final, mid-circuit] per term, from zne_knitted.py),
summed with the knitting prefactors; readout correction inverts the qubit that
was actually read (term structure rebuilt from the cached term circuit).
Un-knitted: data/zne_direct_exact, averaged over the 10 transpiler seeds.
Shot error: 10 replicas resampled at 16384 shots per term (knitted) and one
16384-shot sample per transpiler seed (un-knitted).
Check: the knitted n6 must match the exact values stored in data/zne_knitted.
Writes data/zne_cut_observables.json.
"""
import hashlib, importlib.metadata, json, pickle, sys, warnings
from itertools import product
from multiprocessing import Pool
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
import rerun_sec5 as r
import calibration_drift as cd
from improved.exact_sampling import analyze_term
from qiskit_aer import AerSimulator

r.LAYOUT = "line"
N = 12; MID = r.MID; SHOTS = r.NOISY_SHOTS
SCALES = (1.0, 1.5, 2.0); LABEL = ("1", "1.5", "2")
SLOTS = list(r.KNIT_RUNS)
VARIANTS = ("raw", "postsel", "full"); METHODS = ("none", "linear", "exponential", "quadratic")
idx = np.arange(2 ** N)
bit = lambda q: ((idx >> q) & 1).astype(float)
charge = sum(((idx >> q) & 1) for q in (0, 2, 8, 10)) + (1 - ((idx >> 4) & 1)) + (1 - ((idx >> 6) & 1))
sector = (charge == 3).astype(float)
SITES = {"n0": 0, "n10": 10, "n2": 2, "n8": 8, "n4": 4, "n6": 6}
PAIRS = {"C(0,10)": (0, 10), "C(0,2)": (0, 2), "C(8,10)": (8, 10), "C(4,6)": (4, 6)}
NAMES = list(SITES) + list(PAIRS)
# moments: one per site, one per pair product; the last column is the normalisation
MOM = np.array([bit(q) for q in SITES.values()] + [bit(a) * bit(b) for a, b in PAIRS.values()] + [np.ones(2 ** N)])
NS = len(SITES)
line = r.line_layout()
INV = []
for s in SCALES:
    nm = AerSimulator.from_backend(cd.scaled_washington(s)()).options.noise_model
    INV.append({q: np.linalg.inv(np.asarray(nm._local_readout_errors[(q,)].probabilities, float))
                for q in line if (q,) in nm._local_readout_errors})


def observables(m):
    """m[moment] -> observable values; m already restricted to the variant's selection."""
    e = m[:-1] / m[-1]
    occ = e[:NS]
    site = {q: occ[i] for i, q in enumerate(SITES.values())}
    corr = [e[NS + i] - site[a] * site[b] for i, (a, b) in enumerate(PAIRS.values())]
    return np.concatenate([occ, corr])


ASYM = {"raw": observables(MOM @ np.ones(2 ** N)), "sel": observables(MOM @ sector)}


def extrapolate(y, asym):
    x = np.array(SCALES); y = np.asarray(y)
    out = np.full((4, y.shape[1]), np.nan)
    out[0] = y[0]; out[1] = y[0] - 2 * (y[1] - y[0])
    ratio = (y[0] - asym) / (y[1] - asym)
    ok = ratio > 0
    out[2, ok] = (asym + (y[0] - asym) * ratio ** 2)[ok]
    out[3] = np.polyfit(x, y, 2)[-1]
    return out


def correct(p, physical_of_axis_final, physical_of_internal, inv):
    """Invert readout on every final and mid-circuit bit; p[final, internal]."""
    k = len(physical_of_internal)
    p = p.reshape((2,) * N + (2,) * k)
    for i, q in enumerate(physical_of_axis_final):
        m = inv.get(q)
        if m is not None:
            axis = N - 1 - i
            p = np.moveaxis(np.tensordot(m.T, p, axes=([1], [axis])), 0, axis)
    for j, q in enumerate(physical_of_internal):
        m = inv.get(q)
        if m is not None:
            axis = N + (k - 1 - j)
            p = np.moveaxis(np.tensordot(m.T, p, axes=([1], [axis])), 0, axis)
    return p.reshape(2 ** N, 2 ** k)


def knit_moments(joint, final_phys, internal_phys, inv):
    """[variant][moment] of the signed distribution; postsel and full restrict to the sector."""
    k = len(internal_phys)
    sign = np.array([(-1) ** bin(j).count("1") for j in range(2 ** k)], float)
    w = joint @ sign
    wc = correct(joint, final_phys, internal_phys, inv) @ sign
    return np.array([MOM @ w, MOM @ (w * sector), MOM @ (wc * sector)])


versions = {p: importlib.metadata.version(p) for p in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")}
conq, tarq = r.CUT


def slot_terms(steps, eps):
    fn = r.load_notebook_functions()
    circuit = r.circuit_for(fn, steps, eps, r.MASS)
    ic = circuit.copy(name="knit-input"); ic._base_name = "knit-input"
    comp = hashlib.sha256(pickle.dumps((ic, conq, tarq, True, 3, r.KNIT_TRANSPILER_SEED, versions, tuple(line)),
                                       protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    return ROOT / ".cache" / "circuit_knitting" / comp


def one_term(job):
    """moments[scale, replica(0 = exact, 1..10 sampled), variant, moment] for one term."""
    steps, eps, j, cache = job
    num_cx = 2 * steps
    term = pickle.load(open(cache / f"batch-{j:05d}.pkl", "rb"))[0]
    st = analyze_term(term, num_cx, N)
    final_phys = [st.physical[st.final[num_cx + i]] for i in range(N)]
    internal_phys = [st.physical[q] for q, _ in st.internal]
    out = np.zeros((3, 11, 3, MOM.shape[0]))
    for ci, label in enumerate(LABEL):
        joint = np.load(r.DATA / "zne_joint" / f"s{label}" / f"step{steps}_eps{eps:g}" / f"term{j:04d}.npz")["joint"].astype(float)
        joint = np.clip(joint, 0, None)
        out[ci, 0] = knit_moments(joint, final_phys, internal_phys, INV[ci])
        prob = (joint / joint.sum()).reshape(-1)
        for ri, seed in enumerate(r.KNIT_SEEDS):
            rng = np.random.default_rng([seed, 7, ci, steps, int(round(eps * 10)), j])
            draw = rng.multinomial(SHOTS, prob).reshape(joint.shape) / SHOTS
            out[ci, 1 + ri] = knit_moments(draw, final_phys, internal_phys, INV[ci])
    return out


def knitted():
    """vals[slot, replica(0 exact), variant, method, observable], n6 check"""
    vals = np.full((8, 11, 3, 4, len(NAMES)), np.nan); check = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for si, (steps, eps) in enumerate(SLOTS):
            n = 2 * steps
            pref = np.array([float(np.prod(e)) for e in product(*[[1/2, 1/2, -1/2, 1/2, -1/2, 1/2]] * n)])
            cache = slot_terms(steps, eps)
            m = sum(p * t for p, t in zip(pref, pool.imap(one_term, [(steps, eps, j, cache) for j in range(6 ** n)], chunksize=8)))
            obs = np.array([[[observables(m[ci, ri, vi]) for vi in range(3)] for ri in range(11)] for ci in range(3)])
            for ri in range(11):
                for vi, v in enumerate(VARIANTS):
                    vals[si, ri, vi] = extrapolate(obs[:, ri, vi], ASYM["raw" if v == "raw" else "sel"])
            # check against zne_knitted's exact (bit6 = n6), scale 1
            ref = []
            for label in LABEL:
                d = r.DATA / "zne_knitted" / f"s{label}" / f"step{steps}_eps{eps:g}"
                e = sum(p * np.load(d / f"term{j:04d}.npz")["exact"] for j, p in enumerate(pref))
                ref.append([e[0][0] / e[0][1], e[0][2] / e[0][3], e[1][2] / e[1][3]])
            mine = obs[:, 0, :, NAMES.index("n6")]
            check.append(float(np.abs(np.array(ref) - mine).max()))
            print(f"  knitted step{steps} eps{eps:g}: n6 vs zne_knitted max diff {check[-1]:.1e}", flush=True)
    return vals, check


def direct():
    """vals[seed, slot, replica(0 exact, 1 sampled), variant, method, observable]"""
    vals = np.full((len(r.NOISY_SEEDS), 8, 2, 3, 4, len(NAMES)), np.nan)
    for ri, seed in enumerate(r.NOISY_SEEDS):
        for si in range(8):
            obs = np.zeros((3, 2, 3, len(NAMES)))
            for ci, label in enumerate(LABEL):
                f = np.load(r.DATA / "zne_direct_exact" / f"s{label}" / f"seed{seed}_slot{si}.npz")
                p = np.clip(f["p"], 0, None); physical = [int(q) for q in f["physical"]]
                rng = np.random.default_rng([seed, 11, ci, si])
                for k, v in enumerate((p, rng.multinomial(SHOTS, p / p.sum()) / SHOTS)):
                    w = correct(v.reshape(-1, 1), physical, [], INV[ci]).reshape(-1)
                    obs[ci, k] = [observables(MOM @ v), observables(MOM @ (v * sector)), observables(MOM @ (w * sector))]
            for k in range(2):
                for vi, var in enumerate(VARIANTS):
                    vals[ri, si, k, vi] = extrapolate(obs[:, k, vi], ASYM["raw" if var == "raw" else "sel"])
    return vals


exact = np.array([[[observables(MOM @ np.load(r.DATA / "cut_observables" / f"step{s}_eps{e:g}.npz")["exact"])]] for s, e in SLOTS])[:, 0, 0]
print("un-knitted ...", flush=True); dv = direct()
print("knitted ...", flush=True); kv, check = knitted()
res = {"names": NAMES, "exact": exact.tolist(), "n6_check_max_diff": check}
for vi, var in enumerate(VARIANTS):
    print(f"\n{var}: mean |deviation| from exact over the two-step slots (shot-free) +- shot sem")
    print(f"  {'':9s}" + "".join(f"{m:>26s}" for m in METHODS))
    res[var] = {}
    for ni, name in enumerate(NAMES):
        row = f"  {name:9s}"
        res[var][name] = {}
        for mi, meth in enumerate(METHODS):
            kd = kv[4:, 0, vi, mi, ni] - exact[4:, ni]
            dd = np.nanmean(dv[:, 4:, 0, vi, mi, ni], 0) - exact[4:, ni]
            ks = np.nanstd(np.nanmean(kv[4:, 1:, vi, mi, ni], 0), ddof=1) / np.sqrt(10)
            ds = np.nanstd(np.nanmean(dv[:, 4:, 1, vi, mi, ni], 1), ddof=1) / np.sqrt(10)
            row += f"  {np.abs(kd).mean():.3f}±{ks:.3f} | {np.abs(dd).mean():.3f}±{ds:.3f}"
            res[var][name][meth] = {"knitted_dev": kd.tolist(), "direct_dev": dd.tolist(),
                                    "knitted_sem": float(ks), "direct_sem": float(ds)}
        print(row)
    print("  (each cell: knitted | un-knitted)")
json.dump(res, open(r.DATA / "zne_cut_observables.json", "w"), indent=1)
