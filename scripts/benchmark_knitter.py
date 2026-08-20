#!/usr/bin/env python3
"""Low-shot benchmark comparing the original and reused execution setup.

This uses one synthetic cut and temporary files.  It does not produce paper
data.  Multi-circuit FakeWashington execution is deliberately not enabled on
the 16-GB reference workstation because the 127-qubit target exceeded memory.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qiskit import QuantumCircuit

from improved.config import ExperimentConfig
from improved.experiment import circuit_knitter, flattener, knit_lister, my_measure


def representative_circuit(cuts: int = 1) -> QuantumCircuit:
    if cuts <= 0:
        raise ValueError("cuts must be positive")
    circuit = QuantumCircuit(12)
    circuit.h(0)
    for index in range(cuts):
        circuit.cx(0, 10)
        circuit.rz(0.2 * (index + 1), 0)
    return circuit


def legacy_time(circuit: QuantumCircuit, shots: int, noise: bool) -> float:
    nested = []
    for instruction in circuit.data:
        if instruction.operation.name == "cx":
            nested.append(knit_lister(circuit, 0, 10, 0, 1))
        else:
            nested.append([instruction])
    terms = [flattener(term) for term in product(*nested)]
    started = time.perf_counter()
    for index, term in enumerate(terms):
        my_measure(
            term, 0, 10, circuit.num_qubits, 1, shots,
            20260819 + index, 20260819 + index, noise,
        )
    return time.perf_counter() - started


def optimized_time(
    circuit: QuantumCircuit,
    shots: int,
    noise: bool,
    root: Path,
    execution_batch_size: int = 1,
) -> float:
    config = ExperimentConfig(
        noise=noise,
        num_shots=shots,
        batch_size=execution_batch_size,
        execution_batch_size=execution_batch_size,
        cache_dir=str(root / "cache"),
        checkpoint_dir=str(root / "checkpoints"),
        resume=False,
    )
    started = time.perf_counter()
    circuit_knitter(
        circuit, 0, 10, shots, config,
        simulator_seed=20260819, transpiler_seed=20260819,
    )
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--optimized-only", action="store_true")
    parser.add_argument("--execution-batch-size", type=int, default=1)
    parser.add_argument("--cuts", type=int, default=1)
    args = parser.parse_args()
    circuit = representative_circuit(args.cuts)
    with tempfile.TemporaryDirectory(prefix="circuit-knitting-benchmark-") as path:
        optimized = optimized_time(
            circuit,
            args.shots,
            args.noise,
            Path(path),
            execution_batch_size=args.execution_batch_size,
        )
    print(f"reused setup: {optimized:.3f}s")
    if not args.optimized_only:
        legacy = legacy_time(circuit, args.shots, args.noise)
        print(f"original per-term setup: {legacy:.3f}s")
        print(f"speedup: {legacy / optimized:.2f}x")


if __name__ == "__main__":
    main()
