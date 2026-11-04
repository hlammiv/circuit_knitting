"""Readout mitigation (final 12 bits) and charge post-selection (three occupied
sites) applied to the saved counts of both line series, no new simulation."""
import sys, pickle, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import argparse
import numpy as np
import rerun_sec5 as r
parser = argparse.ArgumentParser()
parser.add_argument("--direct-sim", choices=("fake", "aer"), default="fake",
                    help="un-knitted data set (see rerun_sec5.DIRECT_SIM); aer = same simulator as the knitter")
args = parser.parse_args()
TAG = "" if args.direct_sim == "fake" else "_aer"
r.DIRECT_SIM = args.direct_sim
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
r.LAYOUT = "line"; r.SUFFIX = "_line"; r.KNIT_DIR = r.DATA / "noisy_knitted_trotter_evol_line_exact"
fn = r.load_notebook_functions(); line = r.line_layout()
nm = NoiseModel.from_backend(FakeWashingtonV2())
A = {q: np.asarray(nm._local_readout_errors[(q,)].probabilities, float) for q in line}   # A[q][true, recorded]
Ainv = {q: np.linalg.inv(a) for q, a in A.items()}
print("readout P(flip) per line qubit:", [f"{a[0,1]:.3f}/{a[1,0]:.3f}" for a in A.values()])
N = 12; MID = r.MID
KNIT_FINAL = [32, 31, 51, 36, 50, 49, 68, 55, 69, 70, 74, 89]   # physical qubit per final clbit, the same in every knitted term
def to_vec(counts):
    v = np.zeros(2 ** N)
    for s, c in counts.items(): v[int(s, 2)] = c      # int(s,2): leftmost char = clbit 11 = bit 11
    return v
def apply(v, mats, physical):
    p = v.reshape((2,) * N)                            # axis a holds clbit N-1-a
    for q, m in enumerate(physical):                   # physical[q] was read into clbit q
        axis = N - 1 - q
        p = np.moveaxis(np.tensordot(mats[m].T, p, axes=([1], [axis])), 0, axis)   # p'[r] = sum_t m[t,r] p[t]
    return p.reshape(-1)
idx = np.arange(2 ** N)
# conserved charge: site occupation before the meson operator X4 Z5 X6 that precedes
# readout, i.e. with the measured bits of qubits 4 and 6 un-flipped (exactly 3 noiselessly)
site_bits = sum(((idx >> q) & 1) for q in (0, 2, 8, 10)) + (1 - ((idx >> 4) & 1)) + (1 - ((idx >> 6) & 1))
sector = site_bits == 3
bit6 = ((idx >> (MID + 2)) & 1).astype(bool)          # observable: clbit 6 = site qubit 6 (string index mid+1)
def observable(v, post):
    if post: v = v * sector
    return float(v[bit6].sum() / v.sum())
def variants(counts, physical):
    v = to_vec(counts)
    return {"raw": observable(v, False), "readout": observable(apply(v, Ainv, physical), False),
            "postsel": observable(v, True), "both": observable(apply(v, Ainv, physical), True),
            "sector_fraction": float((v * sector).sum() / v.sum())}
# un-knitted line
nk = pickle.load(open(f"data/noisy_no_knit_trotter_evolution_line{TAG}.pkl", "rb"))
res = {"no_knit": {}, "knitted": {}}
for seed, run in nk.items():
    res["no_knit"][seed] = [variants(c, r.final_physical(r.circuit_for(fn, s, e, r.MASS), t))
                            for c, (s, e), t in zip(run["counts"], [(0, 0.0)] + r.KNIT_RUNS, r.direct_transpiler_seeds(seed))]
# knitted line_exact: t=0 direct per replica, then the eight knitted runs (combined signed counts)
for seed in r.KNIT_SEEDS:
    np.random.seed(seed); np.random.seed(np.random.randint(1024 ** 2))
    sim0, trans0 = np.random.randint(1024 ** 2), np.random.randint(1024 ** 2)
    res0 = r.run_direct(fn, r.circuit_for(fn, 0, 0.0, r.MASS), r.NOISY_SHOTS, simulator_seed=sim0, transpiler_seed=trans0)
    physical0 = r.final_physical(r.circuit_for(fn, 0, 0.0, r.MASS), trans0)
    # knitted combined counts come from many terms, all with this final order (see below)
    physical_k = KNIT_FINAL
    series = [res0] + [pickle.load(open(r.knit_path(s, e, seed), "rb")) for s, e in r.KNIT_RUNS]
    res["knitted"][seed] = [variants(c, physical0 if i == 0 else physical_k) for i, c in enumerate(series)]
    full = [r.DATA / "noisy_knitted_trotter_evol_line_exact_readout" / r.knit_path(s, e, seed).name for s, e in r.KNIT_RUNS]
    if all(f.exists() for f in full):
        # t=0 is a direct run: final-bit inversion only, same as "readout"
        v0 = apply(to_vec(res0), Ainv, physical0)
        full_series = [v0] + [to_vec(pickle.load(open(f, "rb"))) for f in full]
        for i, v in enumerate(full_series):
            res["knitted"][seed][i]["readout_full"] = observable(v, False)
            res["knitted"][seed][i]["both_full"] = observable(v, True)
    print("knitted seed", seed, "done", flush=True)
ideal = np.load("mass_scan_plot_data/ferm_num_m1p125_trot.npy")
out = {"ideal": ideal.tolist(), "t": (np.arange(9) * 0.2).tolist()}
for series in ("no_knit", "knitted"):
    out[series] = {}
    keys = ["raw", "readout", "postsel", "both", "sector_fraction"]
    if series == "knitted":
        keys += [k for k in ("readout_full", "both_full") if all(k in res[series][seed][0] for seed in res[series])]
    for key in keys:
        arr = np.array([[res[series][seed][i][key] for i in range(9)] for seed in res[series]])
        out[series][key] = {"mean": arr.mean(0).tolist(), "sem": (arr.std(0, ddof=1) / np.sqrt(len(arr))).tolist()}
json.dump(out, open(f"data/mitigation_combined_level{TAG}.json", "w"), indent=1)
print("\n t     ideal | no-knit: raw  readout  postsel  both | knitted: raw  readout  postsel  both | sector frac nk / k")
for i in range(9):
    g = lambda s, k: out[s][k]["mean"][i]
    print(f"{i*0.2:.1f}  {ideal[i]:.3f} |  {g('no_knit','raw'):.3f}  {g('no_knit','readout'):.3f}   {g('no_knit','postsel'):.3f}  {g('no_knit','both'):.3f} |  {g('knitted','raw'):.3f}  {g('knitted','readout'):.3f}   {g('knitted','postsel'):.3f}  {g('knitted','both'):.3f} |  {g('no_knit','sector_fraction'):.2f} / {g('knitted','sector_fraction'):.2f}")
for series in ("no_knit", "knitted"):
    for key in [k for k in ("raw", "readout", "postsel", "both", "readout_full", "both_full") if k in out[series]]:
        dev = np.abs(np.array(out[series][key]["mean"]) - ideal)
        print(f"{series:8s} {key:8s} mean|dev| one-step {dev[1:5].mean():.3f}  two-step {dev[5:9].mean():.3f}  sem(two-step) {np.mean(out[series][key]['sem'][5:9]):.4f}")
