#!/usr/bin/env python3
"""Depth-matched Clifford calibration circuits for the noise-contraction factor.

The physics circuit is state preparation, N_t Trotter steps, the meson operator
and a computational-basis measurement.  The calibration circuit keeps every gate
of that skeleton and only changes the three rotation angles (mass, gauge
kinetic, hopping) to odd multiples of pi/2, so it is a stabilizer circuit with
an exactly known outcome, compiles to the same two-qubit structure, and sees the
same noise.  The factor by which noise shrinks its central-site signal,
f = (1/2 - <n>_noisy) / (1/2 - <n>_exact), is then divided out of the physics
points with the same number of Trotter steps (Clifford data regression with one
training circuit per depth).
"""
import sys, warnings
from itertools import product
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import rerun_sec5 as r

N = r.N_SITES


def calibration_circuit(fn, steps, mass_angle, kinetic_angle, hopping_angle):
    """trotter_stepper's skeleton with the three rotation angles set directly.

    fermion_mass(eps, m, eta) rotates by -eps*m*eta/2, gauge_kinetic(eps) by
    eps/2 and fermion_hopping(eps, eta) by -eps/4 (twice); the eps arguments
    below make those rotations equal to the requested angles (up to the
    staggered sign eta of the mass term, which alternates as in trotter1).
    """
    qc = QuantumCircuit(2 * N, 0)
    qc.append(fn["superposition_state_prep"](N), range(2 * N))
    body = QuantumCircuit(2 * N)
    for _ in range(steps):
        for n in range(0, 2 * N, 2):
            eta = (-1) ** (n / 2 + 1)
            body.append(fn["fermion_mass"](-2 * mass_angle, 1.0, eta), [n])     # angle = mass_angle * eta
        for l in range(1, 2 * N, 2):
            body.append(fn["gauge_kinetic"](2 * kinetic_angle), [l])
        for n in range(0, 2 * N, 4):
            body.append(fn["fermion_hopping"](-4 * hopping_angle, 1), [n, n + 1, (n + 2) % (2 * N)])
        for n in range(2, 2 * N, 4):
            body.append(fn["fermion_hopping"](-4 * hopping_angle, 1), [n, n + 1, (n + 2) % (2 * N)])
    qc.append(body.to_gate(label="U_cal"), range(2 * N))
    qc.append(fn["meson_operator"](), [r.MID, r.MID + 1, r.MID + 2])
    circuit = qc.decompose().decompose()
    circuit.measure_all()
    return circuit


def exact_occupation(circuit):
    """<n_6> of the measured site, exactly (12 qubits, statevector)."""
    bare = circuit.remove_final_measurements(inplace=False)
    probabilities = Statevector(bare).probabilities([r.MID + 2])
    return float(probabilities[1])


def choose_angles(fn, steps, candidates=(np.pi / 2, -np.pi / 2, 3 * np.pi / 2)):
    """Clifford angles whose calibration observable is farthest from 1/2."""
    best = None
    for angles in product(candidates, repeat=3):
        value = exact_occupation(calibration_circuit(fn, steps, *angles))
        if best is None or abs(value - 0.5) > abs(best[1] - 0.5):
            best = (angles, value)
    return best


if __name__ == "__main__":
    fn = r.load_notebook_functions()
    for steps in (1, 2):
        physics = r.circuit_for(fn, steps, 0.5, r.MASS)
        angles, value = choose_angles(fn, steps)
        cal = calibration_circuit(fn, steps, *angles)
        print(f"steps {steps}: angles {np.round(np.array(angles) / np.pi, 2)} pi -> exact <n_6> = {value:.4f};"
              f" logical cx physics {physics.count_ops().get('cx')} calibration {cal.count_ops().get('cx')}")
