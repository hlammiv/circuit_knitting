#!/usr/bin/env python3
"""Knitted line runs at scaled noise, for zero-noise extrapolation.

Every gate error and readout error of the Washington model is multiplied by
--scale (T1, T2 divided by it; calibration_drift.scaled_washington).  Uniform
scaling leaves the transpilation unchanged, so the nominal transpiled terms of
each knitted run (the knitter's cache) are reused.  For each term the exact
noisy outcome distribution is computed once (improved.exact_sampling) and the
replicas are drawn from it, 16384 shots each, as the knitter would.  Stored per
term: the signed per-shot weights A, N, Aps, Nps (see knitted_term_values.py),
raw and readout-corrected (with the scaled assignment matrices), exactly and for
every replica.

Terms are split over workers (term index mod --nworkers), so several processes
on several machines can share one scale; finished terms are skipped.

  python scripts/zne_knitted.py --scale 1.5 --worker 0 --nworkers 10
writes data/zne_knitted/s<scale>/step<N>_eps<e>/term<j>.npz
"""
import argparse, hashlib, importlib.metadata, pickle, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
import rerun_sec5 as r
import calibration_drift as cd
from improved.exact_sampling import term_distribution, sample_counts
from qiskit_aer import AerSimulator

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float, required=True)
parser.add_argument("--worker", type=int, default=0)
parser.add_argument("--nworkers", type=int, default=1)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--steps", type=int, nargs="+", default=[1, 2])
parser.add_argument("--reverse", action="store_true",
                    help="take this worker's terms from the end, to help a slower machine that works from the start")
args = parser.parse_args()
r.LAYOUT = "line"
fn = r.load_notebook_functions(); line = r.line_layout(); conq, tarq = r.CUT
NQ = 12; MID = r.MID; SHOTS = r.NOISY_SHOTS
noise_model = AerSimulator.from_backend(cd.scaled_washington(args.scale)()).options.noise_model
inverse = {q: np.linalg.inv(np.asarray(noise_model._local_readout_errors[(q,)].probabilities, float))
           for q in line + [p for p in range(127)] if (q,) in noise_model._local_readout_errors}
versions = {p: importlib.metadata.version(p) for p in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")}
idx = np.arange(2 ** NQ)
charge = sum(((idx >> q) & 1) for q in (0, 2, 8, 10)) + (1 - ((idx >> 4) & 1)) + (1 - ((idx >> 6) & 1))
sector = (charge == 3).astype(float)
bit6 = ((idx >> (MID + 2)) & 1).astype(float)
OUT = r.DATA / "zne_knitted" / f"s{args.scale:g}"


def four(w):
    return np.array([w @ bit6, w.sum(), w @ (bit6 * sector), w @ sector])


def both(joint, structure, num_cx):
    """four() of the signed final distribution, raw and readout-corrected; joint[final, internal j-bits]."""
    k = len(structure.internal)
    sign = np.array([(-1) ** bin(j).count("1") for j in range(2 ** k)], float)
    raw = four(joint @ sign)
    p = joint.copy()                                    # p[final, internal]; final bit i = data qubit i
    p = p.reshape((2,) * NQ + (2 ** k,))
    for i in range(NQ):
        m = inverse.get(line[i])
        if m is not None:
            axis = NQ - 1 - i
            p = np.moveaxis(np.tensordot(m.T, p, axes=([1], [axis])), 0, axis)
    p = p.reshape(2 ** NQ, 2 ** k).reshape(2 ** NQ, *(2,) * k) if k else p.reshape(2 ** NQ, 1)
    for j, (local_q, _) in enumerate(structure.internal):
        m = inverse.get(structure.physical[local_q])
        if m is not None:
            axis = 1 + (k - 1 - j)                      # internal index bit j <-> axis
            p = np.moveaxis(np.tensordot(m.T, p, axes=([1], [axis])), 0, axis)
    return raw, four(p.reshape(2 ** NQ, 2 ** k) @ sign)


jobs = []
for steps, eps in r.KNIT_RUNS:
    if steps not in args.steps:
        continue
    circuit = r.circuit_for(fn, steps, eps, r.MASS)
    ic = circuit.copy(name="knit-input"); ic._base_name = "knit-input"
    comp = hashlib.sha256(pickle.dumps((ic, conq, tarq, True, 3, r.KNIT_TRANSPILER_SEED, versions, tuple(line)),
                                       protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    cache = ROOT / ".cache" / "circuit_knitting" / comp
    for j in range(6 ** (2 * steps)):
        jobs.append((steps, eps, j, cache / f"batch-{j:05d}.pkl"))
mine = [job for n, job in enumerate(jobs) if n % args.nworkers == args.worker]
if args.reverse:
    mine = mine[::-1]
print(f"scale {args.scale}: worker {args.worker}/{args.nworkers}, {len(mine)} of {len(jobs)} terms", flush=True)
for steps, eps, j, cache_file in mine:
    out = OUT / f"step{steps}_eps{eps:g}" / f"term{j:04d}.npz"
    if out.exists():
        continue
    started = time.perf_counter()
    num_cx = 2 * steps
    term = pickle.load(open(cache_file, "rb"))[0]
    dist = term_distribution(term, num_cx, NQ, noise_model, threads=args.threads)
    exact = both(dist.joint, dist.structure, num_cx)
    reps = []
    for seed in r.KNIT_SEEDS:
        rng = np.random.default_rng([seed, int(round(args.scale * 1000)), steps, int(round(eps * 10)), j])
        draw = rng.multinomial(SHOTS, (dist.joint / dist.joint.sum()).reshape(-1)).reshape(dist.joint.shape) / SHOTS
        reps.append(both(draw, dist.structure, num_cx))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, exact=np.array(exact), replicas=np.array(reps), mids=len(dist.structure.internal))
    tmp.rename(out)
    print(f"  step{steps} eps{eps:g} term {j}: {time.perf_counter() - started:.1f} s", flush=True)
