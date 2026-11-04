#!/usr/bin/env python3
"""Readout correction of the knitted Clifford calibration runs, term by term,
exactly as knitted_readout_mitigation.py does for the physics runs, so that the
readout-corrected physics points can be rescaled by a calibration processed the
same way.  Reads the checkpoint batches and transpile cache on this machine.

  python scripts/knitted_calibration_readout.py [--drift S]
writes <calibration dir>/knitted_step{N}_seed{seed}_readout.pkl
"""
import argparse, glob, hashlib, importlib.metadata, pickle, sys, warnings
from itertools import product
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
import rerun_sec5 as r
from improved.exact_sampling import analyze_term
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from clifford_calibration import calibration_circuit
from calibration_runs import ANGLES

parser = argparse.ArgumentParser()
parser.add_argument("--drift", type=float, default=None)
parser.add_argument("--seed", type=int, default=20261101)
args = parser.parse_args()
r.LAYOUT = "line"
fn = r.load_notebook_functions(); line = r.line_layout(); conq, tarq = r.CUT
NQ = 12
# The drifted runs saw scaled readout errors, so they are corrected with the
# scaled matrices (the calibration is meant to be processed like the physics
# taken under its own conditions).
if args.drift is None:
    backend = FakeWashingtonV2(); CAL = r.DATA / "calibration_line"; ckroot = ROOT / "checkpoints" / "circuit_knitting"
else:
    import calibration_drift as cd
    backend = cd.scaled_washington(args.drift)(); CAL = r.DATA / "calibration_line" / f"drift_s{args.drift:g}"
    ckroot = ROOT / "checkpoints" / "circuit_knitting" / f"drift_s{args.drift:g}"
noise_model = NoiseModel.from_backend(backend)
inverse = {q: np.linalg.inv(np.asarray(noise_model._local_readout_errors[(q,)].probabilities, float))
           for q in range(backend.num_qubits) if (q,) in noise_model._local_readout_errors}
versions = {p: importlib.metadata.version(p) for p in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")}


def identities(circuit, sim_seed):
    ic = circuit.copy(name="knit-input"); ic._base_name = "knit-input"
    comp = hashlib.sha256(pickle.dumps((ic, conq, tarq, True, 3, r.KNIT_TRANSPILER_SEED, versions, tuple(line)),
                                       protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    run = hashlib.sha256(pickle.dumps((comp, r.NOISY_SHOTS, sim_seed, "exact_sampling", 1),
                                      protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    return comp, run


def mitigate_term(counts, term_circuit, num_cx):
    nbits = NQ + num_cx
    structure = analyze_term(term_circuit, num_cx, NQ)
    physical_of_clbit = {num_cx + i: structure.physical[structure.final[num_cx + i]] for i in range(NQ)}
    for local_q, clbit in structure.internal:
        physical_of_clbit[clbit] = structure.physical[local_q]
    v = np.zeros(2 ** nbits)
    for s, c in counts.items():
        v[int(s, 2)] = c
    p = v.reshape((2,) * nbits)
    for clbit, q in physical_of_clbit.items():
        m = inverse.get(q)
        if m is None:
            continue
        axis = nbits - 1 - clbit
        p = np.moveaxis(np.tensordot(m.T, p, axes=([1], [axis])), 0, axis)
    return p.reshape(-1)


for steps in (1, 2):
    circuit = calibration_circuit(fn, steps, *ANGLES)
    comp, run = identities(circuit, args.seed)
    batches = sorted(glob.glob(str(ckroot / run / "batch-*.pkl")))
    cache = ROOT / ".cache" / "circuit_knitting" / comp
    num_cx = 2 * steps; expected = 6 ** num_cx
    if not batches or not cache.exists():
        print(f"  step {steps}: checkpoints not on this machine ({ckroot / run})"); continue
    prefactors = [float(np.prod(e)) for e in product(*[[1/2, 1/2, -1/2, 1/2, -1/2, 1/2]] * num_cx)]
    signs = np.array([(-1) ** bin(j).count("1") for j in range(2 ** num_cx)], float)
    combined = np.zeros(2 ** NQ); term = 0
    for batch in batches:
        for counts, term_circuit in zip(pickle.load(open(batch, "rb")), pickle.load(open(cache / Path(batch).name, "rb"))):
            combined += prefactors[term] * (mitigate_term(counts, term_circuit, num_cx).reshape(2 ** NQ, 2 ** num_cx) @ signs)
            term += 1
    if term != expected:
        print(f"  step {steps}: {term} of {expected} terms on disk, skipped"); continue
    result = {format(int(i), f"0{NQ}b"): float(combined[i]) for i in np.nonzero(np.abs(combined) > 1e-12)[0]}
    out = CAL / f"knitted_step{steps}_seed{args.seed}_readout.pkl"
    pickle.dump(result, open(out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  step {steps}: wrote {out.name}, <n_6> = {fn['fermion_number'](result, r.MID):.4f}", flush=True)
