"""
Execution helpers for the circuit knitter.

This module contains low-level functions for circuit execution, measurement,
and result processing that support the main knitting algorithm.
"""
import numpy as np
from typing import Dict, List, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_ibm_runtime import SamplerV2
from collections import defaultdict


def run_circuit_experiment(
    circuit: QuantumCircuit,
    config: 'ExperimentConfig',
    simulator_seed: Optional[int] = None,
    transpiler_seed: Optional[int] = None
) -> Dict[str, int]:
    """
    Run a single circuit experiment and return measurement counts.
    
    Args:
        circuit: Quantum circuit to execute
        config: Experiment configuration
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
        
    Returns:
        Dictionary of measurement counts
    """
    try:
        from config import ExperimentConfig
    except ImportError:
        from .config import ExperimentConfig
    
    # Set up backend based on noise configuration
    backend = FakeWashingtonV2() if config.noise else AerSimulator()
    
    # Set up transpiler
    pass_manager = generate_preset_pass_manager(
        optimization_level=config.optimization_level,
        backend=backend,
        seed_transpiler=transpiler_seed or np.random.randint(1024**2)
    )
    
    # Transpile circuit
    transpiled_circuit = pass_manager.run(circuit)
    
    # Set up sampler with options
    options = {
        "simulator": {
            "seed_simulator": simulator_seed or np.random.randint(1024**2)
        }
    }
    sampler = SamplerV2(backend, options=options)
    
    # Run job and get results
    job = sampler.run([transpiled_circuit], shots=config.num_shots)
    result_dict = job.result()[0].data.c.get_counts()
    
    return result_dict


def my_measure(
    circuit_data: List,
    conq: int,
    tarq: int,
    num_qubits: int,
    num_cx: int,
    num_shots: int,
    simulator_seed: int,
    transpiler_seed: int,
    noise: bool
) -> Dict[str, int]:
    """
    Assembles a quantum circuit from a list of circuit components, runs and measures
    the circuit and returns the outcomes.
    
    Args:
        circuit_data: List of circuit components
        conq: Index of the control qubit
        tarq: Index of the target qubit
        num_qubits: Number of qubits in the original circuit
        num_cx: Number of CNOTs in the original circuit
        num_shots: Number of shots
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
        noise: Whether to use noisy backend
        
    Returns:
        Dictionary of measurement counts (includes internal measurements)
    """
    # Initialize circuit and assemble it from components in circuit_data
    my_circ = QuantumCircuit(num_qubits, num_cx+num_qubits)
    for item in circuit_data:
        my_circ.append(item)
    my_circ.measure([*range(num_qubits)], [*range(-num_qubits, 0)])

    # Select noisy or ideal backend
    if noise:
        backend = FakeWashingtonV2()
    else:
        backend = AerSimulator()
    
    # Transpiled circuit
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=transpiler_seed)
    transpiled_circuit = pass_manager.run(my_circ)
    
    # Initialize sampler
    options = {"simulator": {"seed_simulator": simulator_seed}}
    sampler = SamplerV2(backend, options=options)
    
    # Run job and get results
    job = sampler.run([transpiled_circuit], shots=num_shots)
    result_dict = job.result()[0].data.c.get_counts()
    
    return result_dict


def comb_measure(result_in: Dict[str, int], conq: int, tarq: int, num_cx: int) -> Dict[str, int]:
    """
    Combines quiskit results with circuit internal measurements.
    
    Args:
        result_in: Dictionary of measurement outcomes with internal measurements
        conq: Index of the control qubit
        tarq: Index of the target qubit
        num_cx: Number of CNOTs in the original circuit
        
    Returns:
        Dictionary of combined measurement outcomes
    """
    result_dict = defaultdict(int)
    for item in result_in:
        end_meas = item[:-num_cx]
        int_meas = item[-num_cx:]
        if end_meas[-conq-1]+end_meas[-tarq-1] == '00':
            if int_meas.count('1') % 2 == 0:
                result_dict[end_meas] += result_in[str(item)]
            else:
                result_dict[end_meas] -= result_in[str(item)]
        elif end_meas[-conq-1]+end_meas[-tarq-1] == '10':
            if int_meas.count('1') % 2 == 0:
                result_dict[end_meas] += result_in[str(item)]
            else:
                result_dict[end_meas] -= result_in[str(item)]
        elif end_meas[-conq-1]+end_meas[-tarq-1] == '01':
            if int_meas.count('1') % 2 == 0:
                result_dict[end_meas] += result_in[str(item)]
            else:
                result_dict[end_meas] -= result_in[str(item)]
        else:
            if int_meas.count('1') % 2 == 0:
                result_dict[end_meas] += result_in[str(item)]
            else:
                result_dict[end_meas] -= result_in[str(item)]
    return dict(result_dict)


def knit_lister(circuit: QuantumCircuit, conq: int, tarq: int, meas: int, num_cx: int) -> List:
    """
    Assembles the list of circuit components in the knitting decomposition.
    
    Args:
        circuit: The circuit to be knitted
        conq: Index of the control qubit
        tarq: Index of the target qubit
        meas: Which CNOT is being knitted
        num_cx: Total number of CNOTs in the circuit
        
    Returns:
        List of circuit components for the knitting decomposition
    """
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    circuit_list =[]
    qc.rz(np.pi/2, tarq)
    qc.rx(np.pi/2, conq)
    circuit_list.append(qc[:2])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.rz(-np.pi/2, tarq)

    qc.rz(np.pi, tarq)
    qc.h(conq)
    qc.measure(conq, meas)
    qc.h(conq)
    circuit_list.append(qc[:4])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.h(conq)
    qc.measure(conq, meas)
    qc.h(conq)
    circuit_list.append(qc[:3])
    del(qc)
    return circuit_list


def flattener(nested_list: List) -> List:
    """
    Flatten a nested list structure.
    
    Args:
        nested_list: Nested list to flatten
        
    Returns:
        Flattened list
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flattener(item))
        else:
            flat_list.append(item)
    return flat_list

# Compatibility exports from the canonical six-term implementation.
try:
    from ..experiment import comb_measure, flattener, knit_lister, run_circuit_experiment
except ImportError:
    from experiment import comb_measure, flattener, knit_lister, run_circuit_experiment

__all__ = [
    "comb_measure", "flattener", "knit_lister", "run_circuit_experiment",
]
