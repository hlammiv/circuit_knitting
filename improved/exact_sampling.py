"""Exact outcome distributions of knitted term circuits by branch enumeration.

Why this exists
---------------
A knitted term with a mid-circuit measurement cannot use Aer's end-of-circuit
measurement sampling.  With a noise model that contains Kraus (thermal
relaxation) errors Aer's ``automatic`` method then runs every shot as a
separate noisy statevector trajectory (about 10 ms per shot for the 12-qubit
Trotter terms), and the ``density_matrix`` method loops the shots at ~16 s
each.  Either way 16384 shots cost minutes per term.

What it does
------------
The transpiled term circuit is restricted to the physical qubits it touches
(with the noise model restricted the same way, so every physical gate keeps its
own error) and evolved once as a density matrix up to its first mid-circuit
measurement.  Each of the 2^k outcome branches of the k mid-circuit
measurements then continues from that saved state with the measurement
replaced by the projector |b><b| (applied as a one-operator, non-trace-
preserving ``kraus`` instruction, so the trace of the branch is its
probability).  A branch whose following segment is long is itself saved and
split again, so a long segment is never replayed per outcome.  The final
``save_probabilities`` on the data qubits, with the readout-assignment matrices
of the noise model applied classically to the final and the mid-circuit bits,
gives the exact joint distribution over all classical bits that Aer would
sample from.  Shots are then drawn multinomially from it, which is exactly
what Aer does itself when it can sample at the end of a circuit.

The result is a counts dict in the ``get_counts()`` format of the term circuit
(leftmost character = highest clbit), so the knitting combination code is
unchanged.
"""
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

_PROJECTORS = (
    np.array([[1, 0], [0, 0]], dtype=complex),
    np.array([[0, 0], [0, 1]], dtype=complex),
)


@dataclass
class TermStructure:
    """A transpiled term circuit restricted to the physical qubits it uses."""
    # (operation, local qubits) or ("MEAS", j) at the position of mid-circuit measure j
    ops: List[Tuple]
    # (local qubit, clbit) of mid-circuit measure j, in circuit order
    internal: List[Tuple[int, int]]
    # clbit -> local qubit of the final measurements
    final: Dict[int, int]
    # physical qubit of each local qubit
    physical: List[int]
    num_internal_clbits: int
    num_data_qubits: int

    @property
    def num_local_qubits(self) -> int:
        return len(self.physical)


def analyze_term(transpiled: QuantumCircuit, num_cx: int, num_qubits: int) -> TermStructure:
    """Split a transpiled term into gate list, mid-circuit measures and final measures.

    Clbits ``0 .. num_cx-1`` hold the mid-circuit (knitting) measurements and
    clbit ``num_cx + i`` the final measurement of circuit qubit ``i``, as
    assembled by ``circuit_knitter``.
    """
    ops, internal, final = [], [], {}
    for inst in transpiled.data:
        name = inst.operation.name
        qubits = [transpiled.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            clbit = transpiled.find_bit(inst.clbits[0]).index
            if clbit < num_cx:
                ops.append(("MEAS", len(internal)))
                internal.append((qubits[0], clbit))
            else:
                if clbit in final:
                    raise NotImplementedError("a final clbit is measured twice")
                final[clbit] = qubits[0]
            continue
        if inst.clbits or getattr(inst.operation, "condition", None) is not None:
            raise NotImplementedError(f"classically controlled '{name}' is not supported")
        ops.append((inst.operation, qubits))
    if sorted(final) != list(range(num_cx, num_cx + num_qubits)):
        raise ValueError("final measurements do not cover clbits num_cx .. num_cx+num_qubits-1")
    used = sorted(
        {q for op, qs in ops if op != "MEAS" for q in qs}
        | {q for q, _ in internal}
        | set(final.values())
    )
    local = {p: i for i, p in enumerate(used)}
    return TermStructure(
        ops=[(op, [local[q] for q in qs]) if op != "MEAS" else (op, qs) for op, qs in ops],
        internal=[(local[q], clbit) for q, clbit in internal],
        final={clbit: local[q] for clbit, q in final.items()},
        physical=used,
        num_internal_clbits=num_cx,
        num_data_qubits=num_qubits,
    )


def local_noise_model(noise_model: Optional[NoiseModel], physical: List[int]) -> Optional[NoiseModel]:
    """Restrict ``noise_model`` to the physical qubits ``physical`` (local index = position)."""
    if noise_model is None:
        return None
    local = {p: i for i, p in enumerate(physical)}
    restricted = NoiseModel(basis_gates=noise_model.basis_gates)
    quantum_errors = getattr(noise_model, "_local_quantum_errors", {})
    if "measure" in quantum_errors or "measure" in getattr(noise_model, "_default_quantum_errors", {}):
        # Aer applies such an error before the measurement; the projector
        # replacement below does not, so refuse rather than be subtly wrong.
        raise NotImplementedError("quantum errors attached to 'measure' are not supported")
    for name, table in quantum_errors.items():
        for qubits, error in table.items():
            if all(q in local for q in qubits):
                restricted.add_quantum_error(error, name, [local[q] for q in qubits], warnings=False)
    for name, error in getattr(noise_model, "_default_quantum_errors", {}).items():
        restricted.add_all_qubit_quantum_error(error, name, warnings=False)
    for qubits, error in getattr(noise_model, "_local_readout_errors", {}).items():
        if all(q in local for q in qubits):
            restricted.add_readout_error(error, [local[q] for q in qubits], warnings=False)
    default_readout = getattr(noise_model, "_default_readout_error", None)
    if default_readout is not None:
        restricted.add_all_qubit_readout_error(default_readout, warnings=False)
    return restricted


def _assignment_matrix(noise_model: Optional[NoiseModel], local_qubit: int) -> np.ndarray:
    """2x2 matrix A[true, recorded] = P(recorded | true) for one local qubit."""
    if noise_model is not None:
        local = getattr(noise_model, "_local_readout_errors", {})
        if (local_qubit,) in local:
            return np.asarray(local[(local_qubit,)].probabilities, dtype=float)
        default = getattr(noise_model, "_default_readout_error", None)
        if default is not None:
            return np.asarray(default.probabilities, dtype=float)
    return np.eye(2)


def _apply_readout(p: np.ndarray, assignment: List[np.ndarray]) -> np.ndarray:
    """p[index] with index bit i <-> assignment[i]; returns the distribution of recorded bits."""
    n = len(assignment)
    p = p.reshape((2,) * n)
    for i, matrix in enumerate(assignment):
        axis = n - 1 - i  # axis a of the reshaped array holds index bit n-1-a
        # p'[r] = sum_t P(r | t) p[t]
        p = np.moveaxis(np.tensordot(matrix.T, p, axes=([1], [axis])), 0, axis)
    return p.reshape(-1)


@dataclass
class TermDistribution:
    """Joint distribution ``joint[f, r]`` of the final index ``f`` (bit i <-> clbit
    num_cx + i) and the recorded mid-circuit index ``r`` (bit j <-> mid-circuit
    measure j), plus bookkeeping."""
    joint: np.ndarray
    structure: TermStructure
    branch_probabilities: List[float] = field(default_factory=list)
    split_positions: List[int] = field(default_factory=list)
    num_runs: int = 0
    aer_seconds: float = 0.0
    wall_seconds: float = 0.0


def _split_points(structure: TermStructure, min_split_ops: int) -> List[bool]:
    """Which mid-circuit measures end a saved segment.

    A measure splits when at least ``min_split_ops`` gates have been applied
    since the previous split (or the start), i.e. when replaying them per
    branch would cost more than saving and restoring the state once.
    """
    splits, last = [], 0
    for position, (op, j) in enumerate(structure.ops):
        if op != "MEAS":
            continue
        if position - last >= min_split_ops:
            splits.append(True)
            last = position + 1
        else:
            splits.append(False)
    return splits


def term_distribution(
    transpiled: QuantumCircuit,
    num_cx: int,
    num_qubits: int,
    noise_model: Optional[NoiseModel],
    threads: int = 0,
    min_split_ops: int = 32,
) -> TermDistribution:
    """Exact joint outcome distribution of one transpiled knitting term."""
    started = time.perf_counter()
    structure = analyze_term(transpiled, num_cx, num_qubits)
    restricted = local_noise_model(noise_model, structure.physical)
    simulator = AerSimulator(
        method="density_matrix",
        noise_model=restricted,
        max_parallel_threads=threads,
        max_parallel_experiments=1,
    )
    n = structure.num_local_qubits
    k = len(structure.internal)
    splits = _split_points(structure, min_split_ops)
    final_qubits = [structure.final[num_cx + i] for i in range(num_qubits)]
    final_assignment = [_assignment_matrix(restricted, q) for q in final_qubits]
    internal_assignment = [_assignment_matrix(restricted, q) for q, _ in structure.internal]
    result = TermDistribution(joint=np.zeros((2 ** num_qubits, 2 ** k)), structure=structure,
                              split_positions=[i for i, s in enumerate(splits) if s])

    def run(circuit: QuantumCircuit, label: str):
        aer_result = simulator.run(circuit).result()
        result.num_runs += 1
        result.aer_seconds += float(aer_result.results[0].time_taken)
        return aer_result.data(0)[label]

    def evaluate(rho: Optional[np.ndarray], start: int, outcomes: Tuple[int, ...]):
        circuit = QuantumCircuit(n)
        if rho is not None:
            # qc.set_density_matrix(DensityMatrix(rho)) would run DensityMatrix.is_valid,
            # an O(4^n * 2^n) eigendecomposition per branch; the state came out of
            # Aer and needs no check.
            circuit.append(Instruction("set_density_matrix", n, 0, [rho]), range(n))
        position = start
        while position < len(structure.ops):
            op, arg = structure.ops[position]
            if op == "MEAS":
                j = arg
                if j < len(outcomes):
                    qubit = structure.internal[j][0]
                    circuit.append(Instruction("kraus", 1, 0, [_PROJECTORS[outcomes[j]]]), [qubit])
                elif splits[j]:
                    break
                else:
                    # Short segment since the last saved state: replay it per outcome.
                    for outcome in (0, 1):
                        evaluate(rho, start, outcomes + (outcome,))
                    return
            else:
                circuit.append(op, arg)
            position += 1
        if position < len(structure.ops):
            # Stopped at a splitting measure: save the state, branch from it.
            circuit.save_density_matrix(label="rho")
            saved = np.asarray(run(circuit, "rho"))
            j = structure.ops[position][1]
            for outcome in (0, 1):
                evaluate(saved, position, outcomes + (outcome,))
            return
        circuit.save_probabilities(final_qubits, label="p")
        p = np.asarray(run(circuit, "p"), dtype=float)
        result.branch_probabilities.append(float(p.sum()))
        p = _apply_readout(p, final_assignment)
        weights = np.ones(2 ** k)
        for r in range(2 ** k):
            for j in range(k):
                weights[r] *= internal_assignment[j][outcomes[j], (r >> j) & 1]
        result.joint += np.outer(p, weights)

    evaluate(None, 0, ())
    total = result.joint.sum()
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError(f"branch probabilities sum to {total}, not 1")
    result.wall_seconds = time.perf_counter() - started
    return result


def counts_key(final_index: int, internal_index: int, structure: TermStructure) -> str:
    """``get_counts()`` key: clbit num_cx+n-1 leftmost ... clbit 0 rightmost."""
    num_cx = structure.num_internal_clbits
    internal_bits = ["0"] * num_cx
    for j, (_, clbit) in enumerate(structure.internal):
        internal_bits[clbit] = str((internal_index >> j) & 1)
    final_bits = "".join(
        str((final_index >> i) & 1) for i in range(structure.num_data_qubits - 1, -1, -1)
    )
    return final_bits + "".join(internal_bits[::-1])


def sample_counts(distribution: TermDistribution, shots: int, rng: np.random.Generator) -> Dict[str, int]:
    """Multinomial draw of ``shots`` from the exact distribution, as a counts dict."""
    flat = distribution.joint.reshape(-1)
    draw = rng.multinomial(shots, flat / flat.sum())
    k = len(distribution.structure.internal)
    counts = {}
    for index in np.nonzero(draw)[0]:
        final_index, internal_index = divmod(int(index), 2 ** k)
        counts[counts_key(final_index, internal_index, distribution.structure)] = int(draw[index])
    return counts


def exact_term_counts(
    transpiled: QuantumCircuit,
    num_cx: int,
    num_qubits: int,
    noise_model: Optional[NoiseModel],
    shots: int,
    seed,
    threads: int = 0,
    min_split_ops: int = 32,
) -> Tuple[Dict[str, int], Dict]:
    """Counts dict for one term (same format as the sampler) and a timing/info dict."""
    distribution = term_distribution(transpiled, num_cx, num_qubits, noise_model,
                                     threads=threads, min_split_ops=min_split_ops)
    counts = sample_counts(distribution, shots, np.random.default_rng(seed))
    info = {
        "mid_circuit_measurements": len(distribution.structure.internal),
        "local_qubits": distribution.structure.num_local_qubits,
        "split_measurements": distribution.split_positions,
        "num_runs": distribution.num_runs,
        "aer_seconds": distribution.aer_seconds,
        "wall_seconds": distribution.wall_seconds,
        "branch_probabilities": distribution.branch_probabilities,
    }
    return counts, info
