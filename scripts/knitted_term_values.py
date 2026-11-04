#!/usr/bin/env python3
"""Per-term values of the knitted line runs, for term-class rescaling.

A knitted estimate is sum_j c_j (signed counts of term j).  For every term j of
every run found on this machine this stores, per shot, the signed weight of
  A    strings with the measured site occupied,   N    all strings,
  Aps  the same inside the three-particle sector, Nps  the sector,
from the raw counts and from the readout-corrected counts (same inversion as
knitted_readout_mitigation.py), plus the number of mid-circuit measurements of
the term.  With --ideal it stores the same four numbers for the noiseless term
circuits (exact, statevector branch enumeration of the transpiled terms).

  python scripts/knitted_term_values.py [--drift S] [--ideal]
writes data/knitted_terms/*.npz
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
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from clifford_calibration import calibration_circuit
from calibration_runs import ANGLES

parser = argparse.ArgumentParser()
parser.add_argument("--drift", type=float, default=None, help="calibration runs of data/calibration_line/drift_s<S> only")
parser.add_argument("--ideal", action="store_true", help="noiseless per-term values instead of the noisy runs")
parser.add_argument("--cal-seed", type=int, default=20261101)
args = parser.parse_args()
r.LAYOUT = "line"
fn = r.load_notebook_functions(); line = r.line_layout(); conq, tarq = r.CUT
NQ = 12; MID = r.MID
OUT = r.DATA / "knitted_terms"; OUT.mkdir(exist_ok=True)
versions = {p: importlib.metadata.version(p) for p in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")}
idx = np.arange(2 ** NQ)
charge = sum(((idx >> q) & 1) for q in (0, 2, 8, 10)) + (1 - ((idx >> 4) & 1)) + (1 - ((idx >> 6) & 1))
sector = (charge == 3).astype(float)
bit6 = ((idx >> (MID + 2)) & 1).astype(float)
if args.drift is None:
    backend = FakeWashingtonV2(); ckroot = ROOT / "checkpoints" / "circuit_knitting"
else:
    import calibration_drift as cd
    backend = cd.scaled_washington(args.drift)(); ckroot = ROOT / "checkpoints" / "circuit_knitting" / f"drift_s{args.drift:g}"
noise_model = NoiseModel.from_backend(backend)
inverse = {q: np.linalg.inv(np.asarray(noise_model._local_readout_errors[(q,)].probabilities, float))
           for q in range(backend.num_qubits) if (q,) in noise_model._local_readout_errors}


def identities(circuit, sim_seed):
    ic = circuit.copy(name="knit-input"); ic._base_name = "knit-input"
    comp = hashlib.sha256(pickle.dumps((ic, conq, tarq, True, 3, r.KNIT_TRANSPILER_SEED, versions, tuple(line)),
                                       protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    run = hashlib.sha256(pickle.dumps((comp, r.NOISY_SHOTS, sim_seed, "exact_sampling", 1),
                                      protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    return comp, run


def four(w):
    """A, N, Aps, Nps of a signed distribution over the twelve final bits."""
    return np.array([w @ bit6, w.sum(), w @ (bit6 * sector), w @ sector])


def signed(joint, num_cx):
    """joint[final_index * 2^num_cx + mid_bits] -> signed final distribution."""
    signs = np.array([(-1) ** bin(j).count("1") for j in range(2 ** num_cx)], float)
    return joint.reshape(2 ** NQ, 2 ** num_cx) @ signs


def readout_inverse(v, term_circuit, num_cx):
    nbits = NQ + num_cx
    structure = analyze_term(term_circuit, num_cx, NQ)
    physical_of_clbit = {num_cx + i: structure.physical[structure.final[num_cx + i]] for i in range(NQ)}
    for local_q, clbit in structure.internal:
        physical_of_clbit[clbit] = structure.physical[local_q]
    p = v.reshape((2,) * nbits)
    for clbit, q in physical_of_clbit.items():
        m = inverse.get(q)
        if m is None:
            continue
        axis = nbits - 1 - clbit
        p = np.moveaxis(np.tensordot(m.T, p, axes=([1], [axis])), 0, axis)
    return p.reshape(-1)


def noisy_terms(batches, cache, num_cx):
    values, mids = [], []
    for batch in batches:
        for counts, term_circuit in zip(pickle.load(open(batch, "rb")), pickle.load(open(cache / Path(batch).name, "rb"))):
            v = np.zeros(2 ** (NQ + num_cx))
            for s, c in counts.items():
                v[int(s, 2)] = c
            shots = v.sum()
            values.append([four(signed(v, num_cx)) / shots,
                           four(signed(readout_inverse(v, term_circuit, num_cx), num_cx)) / shots])
            mids.append(len(analyze_term(term_circuit, num_cx, NQ).internal))
    return np.array(values), np.array(mids)


def ideal_term(term_circuit, num_cx):
    """Exact noiseless joint distribution (final bits, mid-circuit bits) of one term."""
    st = analyze_term(term_circuit, num_cx, NQ)
    n = st.num_local_qubits
    final_q = [st.final[num_cx + i] for i in range(NQ)]
    joint = np.zeros((2 ** NQ, 2 ** num_cx))
    segments, current = [], QuantumCircuit(n)
    for op, arg in st.ops:
        if op == "MEAS":
            segments.append(current); current = QuantumCircuit(n)
        else:
            current.append(op, arg)
    segments.append(current)

    def branch(state, seg, bits):
        state = state.evolve(segments[seg])
        if seg == len(segments) - 1:
            p = state.probabilities(final_q)             # bit i of the index = data qubit i
            mid = sum(b << st.internal[j][1] for j, b in enumerate(bits))
            joint[:, mid] += p
            return
        q = st.internal[seg][0]
        data = state.data.reshape((2,) * n)               # axis a holds qubit n-1-a
        for b in (0, 1):
            proj = np.zeros_like(data); sl = [slice(None)] * n; sl[n - 1 - q] = b
            proj[tuple(sl)] = data[tuple(sl)]
            if np.vdot(proj, proj).real > 1e-14:
                branch(Statevector(proj.reshape(-1)), seg + 1, bits + (b,))
    branch(Statevector.from_label("0" * n), 0, ())
    assert abs(joint.sum() - 1) < 1e-9
    return four(signed(joint.reshape(-1), num_cx)), len(st.internal)


def save(name, cache, batches, num_cx):
    out = OUT / name
    if out.exists():
        return
    if args.ideal:
        files = sorted(cache.glob("batch-*.pkl"))
        res = [ideal_term(c, num_cx) for f in files for c in pickle.load(open(f, "rb"))]
        values = np.array([x for x, _ in res]); mids = np.array([k for _, k in res])
    else:
        values, mids = noisy_terms(batches, cache, num_cx)
    if len(values) != 6 ** num_cx:
        print(f"  {name}: {len(values)} terms, skipped"); return
    prefactors = np.array([float(np.prod(e)) for e in product(*[[1/2, 1/2, -1/2, 1/2, -1/2, 1/2]] * num_cx)])
    np.savez(out, values=values, mids=mids, prefactors=prefactors)
    combined = (prefactors[:, None] * values.reshape(len(values), -1)).sum(0).reshape(values.shape[1:])
    print(f"  {name}: combined <n6> raw {combined[..., 0] / combined[..., 1]}", flush=True)


def run_set(label, circuit, sim_seed, num_cx):
    comp, run = identities(circuit, sim_seed)
    cache = ROOT / ".cache" / "circuit_knitting" / comp
    batches = sorted(glob.glob(str(ckroot / run / "batch-*.pkl")))
    if not cache.exists() or (not args.ideal and len(batches) == 0):
        return
    save(("ideal_" if args.ideal else "") + label + ".npz", cache, batches, num_cx)


tag = "" if args.drift is None else f"_drift_s{args.drift:g}"
for steps in (1, 2):
    run_set(f"cal{'' if args.ideal else tag}_step{steps}", calibration_circuit(fn, steps, *ANGLES), args.cal_seed, 2 * steps)
if args.drift is None:
    done = set()
    for seed in r.KNIT_SEEDS:
        np.random.seed(seed)
        for steps, eps in r.KNIT_RUNS:
            sim_seed, _ = np.random.randint(1024 ** 2), np.random.randint(1024 ** 2)
            if args.ideal and (steps, eps) in done:
                continue
            label = f"step{steps}_eps{eps:g}" + ("" if args.ideal else f"_seed{seed}")
            run_set(label, r.circuit_for(fn, steps, eps, r.MASS), sim_seed, 2 * steps)
            done.add((steps, eps))
