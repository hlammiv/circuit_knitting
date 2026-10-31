"""Free-layout knitted series for the points that exist (t<=1.0, ten seeds),
assembled exactly as stage_knitted would, plus the matching un-knitted free points."""
import sys, pickle, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import rerun_sec5 as r
fn = r.load_notebook_functions()
runs = {}
for seed in r.KNIT_SEEDS:
    np.random.seed(seed); np.random.seed(np.random.randint(1024 ** 2))
    res0 = r.run_direct(fn, r.circuit_for(fn, 0, 0.0, r.MASS), r.NOISY_SHOTS,
                        simulator_seed=np.random.randint(1024 ** 2), transpiler_seed=np.random.randint(1024 ** 2))
    series = [res0] + [pickle.load(open(r.knit_path(s, e, seed), "rb")) for s, e in r.KNIT_RUNS[:5]]
    runs[seed] = {"fermion_number": np.array([fn["fermion_number"](x, r.MID) for x in series]),
                  "bootstrap_error": np.array([fn["bootstrap_error"](x, r.MID, r.NOISY_SHOTS) for x in series])}
    print(seed, np.round(runs[seed]["fermion_number"], 4), flush=True)
np.random.seed(0)
mean, err = r.resample(runs, n_points=6)
print("knitted free (t=0..1.0):", np.round(mean, 4)); print("err:", np.round(err, 4))
nk = np.load("data/no_knit_noisy_resampled_fermion_number.npy")[:6]; nke = np.load("data/no_knit_noisy_resampled_bootstrap_error.npy")[:6]
print("no-knit free       :", np.round(nk, 4)); print("err:", np.round(nke, 4))
ideal = np.load("mass_scan_plot_data/ferm_num_m1p125_trot.npy")[:6]
print("ideal              :", np.round(ideal, 4))
json.dump({"t": [0, .2, .4, .6, .8, 1.0], "knitted_free": mean.tolist(), "knitted_free_err": err.tolist(),
           "no_knit_free": nk.tolist(), "no_knit_free_err": nke.tolist(), "ideal": ideal.tolist(),
           "seeds": r.KNIT_SEEDS, "note": "free-layout knitted runs exist only for t<=1.0 (option-3 decision 2026-10-28)"},
          open("data/knitted_free_layout_partial.json", "w"), indent=1)
