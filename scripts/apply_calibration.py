#!/usr/bin/env python3
"""Rescale the physics points by the calibration contraction factors.

For every replica and time slot, value -> 1/2 + (value - 1/2) / f, where f is
the surviving-signal fraction of the depth-matched Clifford calibration
circuit (scripts/calibration_runs.py): un-knitted, the calibration run with the
same transpiler seed as that physics run; knitted, the single calibration run
per step count (all knitted replicas share one transpiler seed).  Done for the
raw counts and for the charge-post-selected (and, knitted, readout-corrected)
counts, with the calibration post-selected the same way.  Writes
data/calibration_line/rescaled.json.
"""
import json, pickle, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
import rerun_sec5 as r
from qiskit.quantum_info import Statevector
from clifford_calibration import calibration_circuit
from calibration_runs import ANGLES, SCHEDULE
r.LAYOUT = "line"
fn = r.load_notebook_functions()
N = 12; MID = r.MID
idx = np.arange(2 ** N)
charge = sum(((idx >> q) & 1) for q in (0, 2, 8, 10)) + (1 - ((idx >> 4) & 1)) + (1 - ((idx >> 6) & 1))
sector = charge == 3
bit6 = ((idx >> (MID + 2)) & 1).astype(bool)

def vec(counts):
    v = np.zeros(2 ** N)
    for s, c in counts.items(): v[int(s, 2)] = c
    return v
def value(v, post):
    if post: v = v * sector
    return float(v[bit6].sum() / v.sum())
def shot_error(v, post, shots=r.NOISY_SHOTS):
    # binomial-type error of the observable from `shots` draws of |v|-weighted strings
    w = np.abs(v) * (sector if post else 1); w = w / w.sum()
    p = float(w[bit6].sum()); return np.sqrt(p * (1 - p) / (shots * w.sum()))

# calibration circuits conserve the charge too (checked exactly)
for steps in (1, 2):
    bare = calibration_circuit(fn, steps, *ANGLES).remove_final_measurements(inplace=False)
    probs = Statevector(bare).probabilities()
    assert abs(probs[sector].sum() - 1) < 1e-9, "calibration circuit leaves the charge sector"
print("calibration circuits stay in the charge sector exactly")

cal = pickle.load(open(r.DATA / "calibration_line" / "direct.pkl", "rb"))
def f_direct(seed, slot, post):
    run = cal["runs"][(seed, slot)]; v = vec(run["counts"]); e = cal["exact"][run["steps"]]
    val = value(v, post); err = shot_error(v, post)
    return (0.5 - val) / (0.5 - e), err / (0.5 - e)
knit_cal = {}
for steps in (1, 2):
    paths = sorted((r.DATA / "calibration_line").glob(f"knitted_step{steps}_seed*.pkl"))
    if paths:
        d = pickle.load(open(paths[0], "rb")); v = vec(d["results"])
        knit_cal[steps] = {post: ((0.5 - value(v, post)) / (0.5 - d["exact"]), shot_error(v, post) / (0.5 - d["exact"])) for post in (False, True)}
print("knitted calibration f:", {s: {("post" if p else "raw"): f"{x[0]:.3f}({x[1]:.3f})" for p, x in c.items()} for s, c in knit_cal.items()})

def rescale(val, err, f, ferr):
    out = 0.5 + (val - 0.5) / f
    return out, np.sqrt((err / f) ** 2 + ((val - 0.5) / f ** 2 * ferr) ** 2)

out = {"t": [s for s in np.arange(9) * 0.2], "ideal": np.load(ROOT / "mass_scan_plot_data" / "ferm_num_m1p125_trot.npy").tolist()}
# un-knitted: counts per replica/slot are in the line pickle
nk = pickle.load(open(r.DATA / "noisy_no_knit_trotter_evolution_line.pkl", "rb"))
table = {k: [] for k in ("raw", "postsel", "raw_rescaled", "postsel_rescaled")}
for seed, run in nk.items():
    rows = {k: [] for k in table}
    for slot, counts in enumerate(run["counts"]):
        v = vec(counts); steps = SCHEDULE[slot][0]
        for post, key in ((False, "raw"), (True, "postsel")):
            val, err = value(v, post), shot_error(v, post)
            rows[key].append((val, err))
            if steps == 0: rows[key + "_rescaled"].append((val, err)); continue
            f, ferr = f_direct(seed, slot, post)
            rows[key + "_rescaled"].append(rescale(val, err, f, ferr))
    for k in table: table[k].append(rows[k])
def summarize(table):
    res = {}
    for k, reps in table.items():
        a = np.array(reps)                      # replicas x 9 x (val, err)
        vals = a[:, :, 0]; errs = a[:, :, 1]
        res[k] = {"mean": vals.mean(0).tolist(), "sem": (vals.std(0, ddof=1) / np.sqrt(len(vals))).tolist(),
                  "mean_shot_err": (np.sqrt((errs ** 2).sum(0)) / len(vals)).tolist()}
    return res
out["no_knit"] = summarize(table)
# knitted: raw results per replica, readout-corrected results when present, t=0 direct per replica
table = {k: [] for k in ("raw", "postsel", "raw_rescaled", "postsel_rescaled", "full", "full_rescaled")}
for seed in r.KNIT_SEEDS:
    np.random.seed(seed); np.random.seed(np.random.randint(1024 ** 2))
    res0 = r.run_direct(fn, r.circuit_for(fn, 0, 0.0, r.MASS), r.NOISY_SHOTS,
                        simulator_seed=np.random.randint(1024 ** 2), transpiler_seed=np.random.randint(1024 ** 2))
    raw = [res0] + [pickle.load(open(r.DATA / "noisy_knitted_trotter_evol_line_exact" / r.knit_path(s, e, seed).name, "rb")) for s, e in r.KNIT_RUNS]
    full = [res0] + [pickle.load(open(r.DATA / "noisy_knitted_trotter_evol_line_exact_readout" / r.knit_path(s, e, seed).name, "rb")) for s, e in r.KNIT_RUNS]
    rows = {k: [] for k in table}
    for slot in range(9):
        steps = SCHEDULE[slot][0]
        for src, post, key in ((raw, False, "raw"), (raw, True, "postsel"), (full, True, "full")):
            v = vec(src[slot]); val, err = value(v, post), shot_error(v, post)
            rows[key].append((val, err))
            if steps == 0 or steps not in knit_cal: rows[key + "_rescaled"].append((val, err) if steps == 0 else (np.nan, np.nan)); continue
            f, ferr = knit_cal[steps][post]
            rows[key + "_rescaled"].append(rescale(val, err, f, ferr))
    for k in table: table[k].append(rows[k])
out["knitted"] = summarize(table)
json.dump(out, open(r.DATA / "calibration_line" / "rescaled.json", "w"), indent=1)
ideal = np.array(out["ideal"])
print("\n t    ideal | un-knitted: raw  resc.  postsel  ps+resc | knitted: raw  resc.  postsel+readout  +resc")
for i in range(9):
    g = lambda s, k: out[s][k]["mean"][i]
    print(f"{i*0.2:.1f}  {ideal[i]:.3f} |  {g('no_knit','raw'):.3f}  {g('no_knit','raw_rescaled'):.3f}  {g('no_knit','postsel'):.3f}   {g('no_knit','postsel_rescaled'):.3f}  |  {g('knitted','raw'):.3f}  {g('knitted','raw_rescaled'):.3f}  {g('knitted','full'):.3f}           {g('knitted','full_rescaled'):.3f}")
for s in ("no_knit", "knitted"):
    for k in out[s]:
        m = np.array(out[s][k]["mean"]); dev = np.abs(m - ideal)
        print(f"{s:8s} {k:17s} |dev| one-step {np.nanmean(dev[1:5]):.3f} two-step {np.nanmean(dev[5:9]):.3f}   sem two-step {np.nanmean(out[s][k]['sem'][5:9]):.4f}")
