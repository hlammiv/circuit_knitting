#!/usr/bin/env python3
"""Observables at and away from the cut, from the saved nominal-noise replicas of
both line series (no extrapolation): occupation bits, and connected correlators
across the cut bond.  Raw, charge-post-selected, and readout-corrected +
post-selected; deviation from the exact value of data/cut_observables.
Writes data/cut_observables_noisy.json.
"""
import json, pickle, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
import rerun_sec5 as r
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2

r.LAYOUT = "line"; r.SUFFIX = "_line"; r.KNIT_DIR = r.DATA / "noisy_knitted_trotter_evol_line_exact"; r.DIRECT_SIM = "aer"
fn = r.load_notebook_functions()
N = 12
idx = np.arange(2 ** N)
bit = lambda q: ((idx >> q) & 1).astype(float)
charge = sum(((idx >> q) & 1) for q in (0, 2, 8, 10)) + (1 - ((idx >> 4) & 1)) + (1 - ((idx >> 6) & 1))
sector = charge == 3
nm = NoiseModel.from_backend(FakeWashingtonV2())
Ainv = {q: np.linalg.inv(np.asarray(nm._local_readout_errors[(q,)].probabilities, float)) for q in r.line_layout()}
SITES = {"n0": (0,), "n10": (10,), "n2": (2,), "n8": (8,), "n4": (4,), "n6": (6,)}
PAIRS = {"C(0,10)": (0, 10), "C(0,2)": (0, 2), "C(8,10)": (8, 10), "C(4,6)": (4, 6)}
NAMES = list(SITES) + list(PAIRS)


def vec(counts):
    v = np.zeros(2 ** N)
    for s, c in counts.items(): v[int(s, 2)] = c
    return v


def invert(v, physical):
    p = v.reshape((2,) * N)
    for q, m in enumerate(physical):
        axis = N - 1 - q
        p = np.moveaxis(np.tensordot(Ainv[m].T, p, axes=([1], [axis])), 0, axis)
    return p.reshape(-1)


def observables(v):
    v = v / v.sum()
    out = [v @ bit(q[0]) for q in SITES.values()]
    out += [v @ (bit(a) * bit(b)) - (v @ bit(a)) * (v @ bit(b)) for a, b in PAIRS.values()]
    return np.array(out)


VARIANTS = ("raw", "postsel", "full")
SLOTS = r.KNIT_RUNS
exact = np.array([observables(np.load(r.DATA / "cut_observables" / f"step{s}_eps{e:g}.npz")["exact"]) for s, e in SLOTS])
nk = pickle.load(open(r.DATA / "noisy_no_knit_trotter_evolution_line_aer.pkl", "rb"))
direct = np.full((len(nk), 3, 8, len(NAMES)), np.nan)
for ri, (seed, run) in enumerate(nk.items()):
    ts = r.direct_transpiler_seeds(seed)
    for si, (s, e) in enumerate(SLOTS):
        v = vec(run["counts"][si + 1])
        w = invert(v, r.final_physical(r.circuit_for(fn, s, e, r.MASS), ts[si + 1]))
        direct[ri, :, si] = observables(v), observables(v * sector), observables(w * sector)
knit = np.full((len(r.KNIT_SEEDS), 3, 8, len(NAMES)), np.nan)
for ri, seed in enumerate(r.KNIT_SEEDS):
    for si, (s, e) in enumerate(SLOTS):
        v = vec(pickle.load(open(r.knit_path(s, e, seed), "rb")))
        w = vec(pickle.load(open(r.DATA / "noisy_knitted_trotter_evol_line_exact_readout" / r.knit_path(s, e, seed).name, "rb")))
        knit[ri, :, si] = observables(v), observables(v * sector), observables(w * sector)
res = {"names": NAMES, "exact": exact.tolist()}
for vi, variant in enumerate(VARIANTS):
    print(f"\n{variant}: signed deviation from exact +- sem (10 replicas)   un-knitted 16384 shots | knitted 16384 per term")
    for label, part in (("one-step", slice(0, 4)), ("two-step", slice(4, 8))):
        print(f" {label}: mean over slots; exact value in brackets")
        for ni, name in enumerate(NAMES):
            dd = direct[:, vi, part, ni] - exact[part, ni]; kd = knit[:, vi, part, ni] - exact[part, ni]
            ds = dd.mean(1).std(ddof=1) / np.sqrt(10); ks = kd.mean(1).std(ddof=1) / np.sqrt(10)
            print(f"   {name:8s} [{exact[part, ni].mean():+.3f}]  un-knitted {dd.mean():+.4f} +- {ds:.4f}  |dev| {np.abs(dd.mean(0)).mean():.4f}   knitted {kd.mean():+.4f} +- {ks:.4f}  |dev| {np.abs(kd.mean(0)).mean():.4f}")
    res[variant] = {"direct": direct[:, vi].tolist(), "knit": knit[:, vi].tolist()}
json.dump(res, open(r.DATA / "cut_observables_noisy.json", "w"))
