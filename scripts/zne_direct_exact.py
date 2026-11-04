#!/usr/bin/env python3
"""Exact (shot-noise-free) outcome distributions of the un-knitted line replicas
at scaled noise, the counterpart of the "exact" arrays of zne_knitted.py.

Same seeds and transpiled circuits as zne_direct.py; the distribution comes from
improved.exact_sampling with no mid-circuit measurements.  Runs are split over
workers (index mod --nworkers); finished runs are skipped.

  python scripts/zne_direct_exact.py --worker 0 --nworkers 10 [--scales 1 1.5 2]
writes data/zne_direct_exact/s<scale>/seed<seed>_slot<i>.npz: p[4096], physical[12]
(the physical qubit read into each clbit)
"""
import argparse, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
import rerun_sec5 as r
import calibration_drift as cd
from improved.exact_sampling import term_distribution
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator

parser = argparse.ArgumentParser()
parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0])
parser.add_argument("--worker", type=int, default=0)
parser.add_argument("--nworkers", type=int, default=1)
parser.add_argument("--threads", type=int, default=1)
args = parser.parse_args()
r.LAYOUT = "line"
fn = r.load_notebook_functions()
SCHEDULE = [(1, e) for e in (0.2, 0.4, 0.6, 0.8)] + [(2, e) for e in (0.5, 0.6, 0.7, 0.8)]

jobs = []
for seed in r.NOISY_SEEDS:
    my_seed = seed
    for slot, (steps, eps) in enumerate([(0, 0.0)] + SCHEDULE):     # same seed chain as stage_noisy
        np.random.seed(my_seed)
        my_seed = np.random.randint(1024 ** 2)
        np.random.randint(1024 ** 2)                                # simulator seed
        transpiler_seed = np.random.randint(1024 ** 2)
        if slot:
            for scale in args.scales:
                jobs.append((scale, seed, slot - 1, steps, eps, transpiler_seed))
for n, (scale, seed, slot, steps, eps, transpiler_seed) in enumerate(jobs):
    if n % args.nworkers != args.worker:
        continue
    out = r.DATA / "zne_direct_exact" / f"s{scale:g}" / f"seed{seed}_slot{slot}.npz"
    if out.exists():
        continue
    started = time.perf_counter()
    backend = cd.scaled_washington(scale)()
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                                seed_transpiler=transpiler_seed, initial_layout=r.line_layout())
    transpiled = pass_manager.run(r.circuit_for(fn, steps, eps, r.MASS))
    noise_model = AerSimulator.from_backend(backend).options.noise_model
    dist = term_distribution(transpiled, 0, 12, noise_model, threads=args.threads)
    physical = [dist.structure.physical[dist.structure.final[c]] for c in range(12)]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, p=dist.joint.reshape(-1), physical=physical)
    tmp.rename(out)
    print(f"  s{scale:g} seed {seed} slot {slot}: {time.perf_counter() - started:.1f} s, "
          f"{dist.structure.num_local_qubits} qubits", flush=True)
