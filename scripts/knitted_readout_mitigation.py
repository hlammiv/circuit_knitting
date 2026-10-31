#!/usr/bin/env python3
"""Readout-error mitigation of the knitted line runs from their saved per-term
counts: invert the calibrated assignment matrix on every classical bit of
every term (the twelve final measurements and the k mid-circuit knitting
measurements), then combine the terms exactly as circuit_knitter does.

Reads the checkpoint batches (counts) and the transpile cache (which physical
qubit each mid-circuit bit measured) of each run found on this machine and
writes data/noisy_knitted_trotter_evol_line_exact_readout/<same name>.pkl
holding the mitigated combined distribution (same format as the raw results).
"""
import glob, hashlib, importlib.metadata, pickle, sys, warnings
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

r.LAYOUT = "line"
fn = r.load_notebook_functions(); line = r.line_layout(); conq, tarq = r.CUT
OUT = r.DATA / "noisy_knitted_trotter_evol_line_exact_readout"; OUT.mkdir(exist_ok=True)
noise_model = NoiseModel.from_backend(FakeWashingtonV2())
inverse = {q: np.linalg.inv(np.asarray(noise_model._local_readout_errors[(q,)].probabilities, float))
           for q in range(FakeWashingtonV2().num_qubits) if (q,) in noise_model._local_readout_errors}
versions = {p: importlib.metadata.version(p) for p in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")}
NQ = 12

def identities(steps, epsilon, sim_seed):
    circuit = r.circuit_for(fn, steps, epsilon, r.MASS)
    ic = circuit.copy(name="knit-input"); ic._base_name = "knit-input"
    comp = hashlib.sha256(pickle.dumps((ic, conq, tarq, True, 3, r.KNIT_TRANSPILER_SEED, versions, tuple(line)),
                                       protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    run = hashlib.sha256(pickle.dumps((comp, r.NOISY_SHOTS, sim_seed, "exact_sampling", 1),
                                      protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    return comp, run

def mitigate_term(counts, term_circuit, num_cx):
    nbits = NQ + num_cx
    structure = analyze_term(term_circuit, num_cx, NQ)
    physical_of_clbit = {num_cx + i: line[i] for i in range(NQ)}      # final bit i <-> circuit qubit i
    for local_q, clbit in structure.internal:
        physical_of_clbit[clbit] = structure.physical[local_q]
    v = np.zeros(2 ** nbits)
    for s, c in counts.items():
        v[int(s, 2)] = c                                              # leftmost char = highest clbit
    p = v.reshape((2,) * nbits)                                       # axis a holds clbit nbits-1-a
    for clbit, q in physical_of_clbit.items():
        m = inverse.get(q)
        if m is None:
            continue
        axis = nbits - 1 - clbit
        p = np.moveaxis(np.tensordot(m.T, p, axes=([1], [axis])), 0, axis)   # p_true = A^{-1} p_recorded
    return p.reshape(-1)

done = skipped = 0
for seed in r.KNIT_SEEDS:
    np.random.seed(seed)
    for index, (steps, epsilon) in enumerate(r.KNIT_RUNS):
        sim_seed, _ = np.random.randint(1024 ** 2), np.random.randint(1024 ** 2)
        out_path = OUT / r.knit_path(steps, epsilon, seed).name
        if out_path.exists():
            done += 1; continue
        comp, run = identities(steps, epsilon, sim_seed)
        ckpt = ROOT / "checkpoints" / "circuit_knitting" / run
        cache = ROOT / ".cache" / "circuit_knitting" / comp
        batches = sorted(glob.glob(str(ckpt / "batch-*.pkl")))
        expected = 6 ** (2 * steps)
        if len(batches) != expected or not cache.exists():
            skipped += 1; continue
        num_cx = 2 * steps
        prefactors = [float(np.prod(e)) for e in product(*[[1/2, 1/2, -1/2, 1/2, -1/2, 1/2]] * num_cx)]
        # comb_measure: every term's count enters with sign (-1)^(number of
        # mid-circuit bits equal to 1), whatever the final bits; the mid-circuit
        # bits are the lowest num_cx bits of the index.
        signs = np.array([(-1) ** bin(j).count("1") for j in range(2 ** num_cx)], float)
        combined = np.zeros(2 ** NQ)
        term = 0
        for batch in batches:
            raw_terms = pickle.load(open(batch, "rb"))
            circuits = pickle.load(open(cache / Path(batch).name, "rb"))
            for counts, circuit in zip(raw_terms, circuits):
                mitigated = mitigate_term(counts, circuit, num_cx)
                combined += prefactors[term] * (mitigated.reshape(2 ** NQ, 2 ** num_cx) @ signs)
                term += 1
        if term != expected:
            raise RuntimeError(f"{out_path.name}: {term} terms, expected {expected}")
        result = {format(int(i), f"0{NQ}b"): float(combined[i]) for i in np.nonzero(np.abs(combined) > 1e-12)[0]}
        pickle.dump(result, open(out_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        done += 1
        print(f"  {out_path.name}: value {fn['fermion_number'](result, r.MID):.4f}", flush=True)
print(f"mitigated runs on this machine: {done}, not available here: {skipped}")
