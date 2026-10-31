"""Compiled CNOT counts of the un-knitted circuits pinned to the line, for the
exact transpiler seeds used by stage_noisy (replicated seed chain), plus the
knitted-term ranges from the cache scan.  Writes data/cx_counts_line.json."""
import sys, json, warnings, collections
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import rerun_sec5 as r
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
r.LAYOUT = "line"
fn = r.load_notebook_functions()
backend = FakeWashingtonV2(); line = r.line_layout()
schedule = [(0, 0.0)] + [(1, e) for e in (0.2, 0.4, 0.6, 0.8)] + [(2, e) for e in (0.5, 0.6, 0.7, 0.8)]
counts = collections.defaultdict(list)   # (steps, eps) -> [cx per seed]
for seed in r.NOISY_SEEDS:
    my_seed = seed
    for steps, eps in schedule:
        np.random.seed(my_seed); my_seed = np.random.randint(1024 ** 2)
        _sim = np.random.randint(1024 ** 2); tseed = np.random.randint(1024 ** 2)
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=tseed, initial_layout=line)
        tc = pm.run(r.circuit_for(fn, steps, eps, r.MASS))
        ops = tc.count_ops(); counts[(steps, eps)].append(int(ops.get("cx", 0)))
    print("seed", seed, [counts[k][-1] for k in schedule], flush=True)
by_steps = collections.defaultdict(list)
for (steps, eps), v in counts.items(): by_steps[steps] += v
scan = {}
for f in (sys.argv[1], sys.argv[2]):
    scan.update(json.load(open(f)))
knit = {}
for v in scan.values():
    steps = 1 if v["terms"] == 36 else 2
    k = knit.setdefault(steps, {"cx_min": 10**9, "cx_max": 0, "terms": v["terms"], "k_hist": v["k_hist"]})
    k["cx_min"] = min(k["cx_min"], v["cx_min"]); k["cx_max"] = max(k["cx_max"], v["cx_max"])
out = {
    "layout": line, "optimization_level": 3, "backend": backend.name,
    "direct": {str(s): {"cx_min": min(v), "cx_max": max(v), "cx_mean": float(np.mean(v)), "per_t": {str(e): c for (st, e), c in counts.items() if st == s}} for s, v in by_steps.items()},
    "knitted_terms": {str(s): v for s, v in knit.items()},
    "note": "direct: FakeWashingtonV2, level 3, initial_layout=line, the ten stage_noisy transpiler seeds; knitted: all cached transpiled terms of the line_exact runs (KNIT_TRANSPILER_SEED)",
}
json.dump(out, open("data/cx_counts_line.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("direct", "knitted_terms")}, indent=1))
