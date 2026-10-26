"""Regression test for the Sec. V initial state.

`superposition_state_prep` in functions.ipynb must prepare
    |phi> = (|Omega> + |P>)/sqrt(2),
    |Omega> = prod_{n odd} X_n  prod_links H_{n,n+1} |0...0>,
    |P>     = X_m sigma^z_{m,m+1} X_{m+1} |Omega>,   m = N_s/2 - 1,
which lies in the physical Gauss-law sector G_n = (-1)^n with
G_n = sigma^x_{n-1,n} Z_n sigma^x_{n,n+1}.

Until 2026-10-26 the circuit carried an extra X on the link sigma_{m,m+1}.
That prepared Z_{m,m+1} X_{m+1}|phi>, orthogonal to |phi> and carrying a
static charge at site m, so every Sec. V observable was computed in the wrong
sector.  Run with:  python -m pytest -q test_state_prep.py
"""
import json
import pathlib

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

ROOT = pathlib.Path(__file__).resolve().parent


def load_notebook_function(notebook, name):
    namespace = {"QuantumCircuit": QuantumCircuit, "np": np}
    for cell in json.load(open(ROOT / notebook))["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and f"def {name}(" in source:
            exec(source, namespace)
            return namespace[name]
    raise KeyError(f"{name} not found in {notebook}")


def pauli_on(n_qubits, ops):
    label = ["I"] * n_qubits
    for qubit, op in ops:
        label[n_qubits - 1 - qubit] = op
    return Pauli("".join(label))


def target_state(n_sites):
    """(|Omega>+|P>)/sqrt2 built directly from the paper's definition (PBC)."""
    n_qubits = 2 * n_sites
    m = n_sites // 2 - 1
    circuit = QuantumCircuit(n_qubits)
    for link in range(1, n_qubits, 2):
        circuit.h(link)
    for site in range(1, n_sites, 2):
        circuit.x(2 * site)
    omega = Statevector(circuit)
    excited = omega.evolve(pauli_on(n_qubits, [(2 * m, "X"), (2 * m + 1, "Z"), (2 * m + 2, "X")]))
    return Statevector((omega.data + excited.data) / np.sqrt(2))


def gauss_law(state, n_sites):
    n_qubits = 2 * n_sites
    return [
        round(state.expectation_value(
            pauli_on(n_qubits, [((2 * k - 1) % n_qubits, "X"), (2 * k, "Z"), (2 * k + 1, "X")])
        ).real, 6)
        for k in range(n_sites)
    ]


@pytest.mark.parametrize("notebook", ["functions.ipynb", "functions_coupling_to_0.ipynb"])
def test_prepared_state_is_omega_plus_p(notebook):
    prep = load_notebook_function(notebook, "superposition_state_prep")
    n_sites = 6  # the lattice used throughout Sec. V
    prepared = Statevector(prep(n_sites))
    overlap = abs(np.vdot(target_state(n_sites).data, prepared.data)) ** 2
    assert overlap == pytest.approx(1.0, abs=1e-12)


def test_prepared_state_is_in_physical_sector():
    prep = load_notebook_function("functions.ipynb", "superposition_state_prep")
    n_sites = 6
    assert gauss_law(Statevector(prep(n_sites)), n_sites) == [(-1) ** k for k in range(n_sites)]


def test_meson_operator_expectation_is_one_at_t0():
    prep = load_notebook_function("functions.ipynb", "superposition_state_prep")
    n_sites, m = 6, 2
    state = Statevector(prep(n_sites))
    meson = pauli_on(2 * n_sites, [(2 * m, "X"), (2 * m + 1, "Z"), (2 * m + 2, "X")])
    assert state.expectation_value(meson).real == pytest.approx(1.0, abs=1e-12)
