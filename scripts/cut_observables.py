#!/usr/bin/env python3
"""Noiseless test of the rotation-only estimate on observables at the cut.

The cut gates sit on qubits 10 and 0, the bond that closes the ring.  For every
time slot this compares, without noise: the exact state; the full knitted sum;
the sum over only the rotation terms (no mid-circuit measurement).  Stored: the
distribution over the twelve measured bits for all three, and for exact and
rotation-only the off-diagonal Paulis on that bond, which need other measurement
bases and so cannot come from the distributions.

  python scripts/cut_observables.py [--no-full]
writes data/cut_observables/step<N>_eps<e>.npz
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
from qiskit.quantum_info import Pauli, Statevector

parser = argparse.ArgumentParser()
parser.add_argument("--no-full", action="store_true", help="skip the full knitted sum")
args = parser.parse_args()
r.LAYOUT = "line"
fn = r.load_notebook_functions(); line = r.line_layout(); conq, tarq = r.CUT
NQ = 12
OUT = r.DATA / "cut_observables"; OUT.mkdir(exist_ok=True)
versions = {p: importlib.metadata.version(p) for p in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")}
# Pauli strings on qubits 10, 11, 0 of the closing bond, written q0 ... q11
BOND = {"X10 Z11 X0": {10: "X", 11: "Z", 0: "X"}, "Y10 Z11 Y0": {10: "Y", 11: "Z", 0: "Y"},
        "X10 Z11 Y0": {10: "X", 11: "Z", 0: "Y"}, "Y10 Z11 X0": {10: "Y", 11: "Z", 0: "X"},
        "X0 Z1 X2": {0: "X", 1: "Z", 2: "X"}, "Y0 Z1 Y2": {0: "Y", 1: "Z", 2: "Y"},
        "X8 Z9 X10": {8: "X", 9: "Z", 10: "X"}, "Y8 Z9 Y10": {8: "Y", 9: "Z", 10: "Y"}}


def pauli_values(state, position):
    """position[i] = state qubit holding circuit qubit i"""
    out = []
    for ops in BOND.values():
        label = ["I"] * NQ
        for q, p in ops.items():
            label[NQ - 1 - position[q]] = p
        out.append(state.expectation_value(Pauli("".join(label))).real)
    return np.array(out)


def branches(st, num_cx):
    """Statevector branches of a term: (mid-bit signs applied) signed final distribution and, without measurements, the state."""
    n = st.num_local_qubits
    final_q = [st.final[num_cx + i] for i in range(NQ)]
    segments, current = [], QuantumCircuit(n)
    for op, arg in st.ops:
        if op == "MEAS":
            segments.append(current); current = QuantumCircuit(n)
        else:
            current.append(op, arg)
    segments.append(current)
    signed = np.zeros(2 ** NQ)

    def go(state, seg, sign):
        state = state.evolve(segments[seg])
        if seg == len(segments) - 1:
            signed[:] += sign * state.probabilities(final_q)
            return state
        q = st.internal[seg][0]
        data = state.data.reshape((2,) * n)
        for b in (0, 1):
            proj = np.zeros_like(data); sl = [slice(None)] * n; sl[n - 1 - q] = b
            proj[tuple(sl)] = data[tuple(sl)]
            if np.vdot(proj, proj).real > 1e-14:
                go(Statevector(proj.reshape(-1)), seg + 1, sign * (-1) ** b)
    last = go(Statevector.from_label("0" * n), 0, 1.0)
    return signed, last, final_q


for steps, eps in r.KNIT_RUNS:
    out = OUT / f"step{steps}_eps{eps:g}.npz"
    if out.exists():
        continue
    num_cx = 2 * steps
    circuit = r.circuit_for(fn, steps, eps, r.MASS)
    ic = circuit.copy(name="knit-input"); ic._base_name = "knit-input"
    comp = hashlib.sha256(pickle.dumps((ic, conq, tarq, True, 3, r.KNIT_TRANSPILER_SEED, versions, tuple(line)),
                                       protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    cache = ROOT / ".cache" / "circuit_knitting" / comp
    exact_state = Statevector(circuit.remove_final_measurements(inplace=False))
    exact = exact_state.probabilities()
    exact_pauli = pauli_values(exact_state, list(range(NQ)))
    prefactors = [float(np.prod(e)) for e in product(*[[1/2, 1/2, -1/2, 1/2, -1/2, 1/2]] * num_cx)]
    rotation, rot_pauli, full = np.zeros(2 ** NQ), np.zeros(len(BOND)), np.zeros(2 ** NQ)
    by_mids = np.zeros((num_cx + 1, 2 ** NQ))           # the full sum split by number of mid-circuit measurements
    for j, c in enumerate(prefactors):
        is_rotation = not any((j // 6 ** d) % 6 > 1 for d in range(num_cx))
        if args.no_full and not is_rotation:
            continue
        term = pickle.load(open(cache / f"batch-{j:05d}.pkl", "rb"))[0]
        st = analyze_term(term, num_cx, NQ)
        signed, state, final_q = branches(st, num_cx)
        full += c * signed
        by_mids[len(st.internal)] += c * signed
        if is_rotation:
            assert not st.internal
            rotation += c * signed
            rot_pauli += c * pauli_values(state, final_q)
    if args.no_full:
        full[:] = np.nan
    np.savez(out, exact=exact, rotation=rotation, full=full, exact_pauli=exact_pauli, rotation_pauli=rot_pauli, pauli_names=list(BOND), by_mids=by_mids)
    print(f"step{steps} eps{eps:g}: |rotation-exact| {np.abs(rotation - exact).sum():.4f}  |full-exact| {np.abs(full - exact).sum():.2e}", flush=True)
