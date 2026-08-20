"""
Experiment execution and management functions.
"""
import hashlib
import json
import importlib.metadata
import os
import pickle
import time
from pathlib import Path

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
from qiskit_ibm_runtime import SamplerV2
from itertools import product
from collections import defaultdict

try:
    from .config import ExperimentConfig
except ImportError:
    from config import ExperimentConfig


def _make_backend(noise: bool):
    """Create a local simulator, using Washington calibration data when noisy."""
    if not noise:
        return AerSimulator()
    return AerSimulator.from_backend(FakeWashingtonV2())


def run_circuit_experiment(
    circuit: QuantumCircuit,
    config: ExperimentConfig,
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
    # Set up backend based on noise configuration
    backend = _make_backend(config.noise)
    
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
    
    # Run experiment
    job = sampler.run([transpiled_circuit], shots=config.num_shots)
    result = job.result()[0].data.c.get_counts()
    
    # Return sorted results
    return {key: result[key] for key in sorted(result)}


def knit_lister(circuit: QuantumCircuit, conloc: int, tarloc: int, meas: int, num_cx: int) -> List:
    """Assembles the list of circuit components in the knitting decomposition

    Parameters:
        circuit (QuantumCircuit): the circuit to be knitted (to match the number of qubits)
        conloc (int): index of the control qubit 
        tarloc (int): index of the target qubit 
        meas (int): which of the num_cx CNOTs is being knitted (this is the classical bit to which the measurement outputs)
        num_cx (int): the number of CNOTs in the original circuit (num_cx+the number of qubits is the number of classical bits necessary)
    Returns:
        circuit list (list): a list of length 6 containing the relevant terms in the knitting decomposition
    """
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    circuit_list =[]
    qc.rz(np.pi/2, tarloc)
    qc.rx(np.pi/2, conloc)
    circuit_list.append(qc[:2])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.rz(-np.pi/2, tarloc)
    qc.rx(-np.pi/2, conloc)
    circuit_list.append(qc[:2])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.measure(tarloc, meas)
    qc.rx(np.pi, conloc)
    circuit_list.append(qc[:2])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.measure(tarloc, meas)
    circuit_list.append(qc[:])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.rz(np.pi, tarloc)
    qc.h(conloc)
    qc.measure(conloc, meas)
    qc.h(conloc)
    circuit_list.append(qc[:4])
    del(qc)
    qc = QuantumCircuit(circuit.num_qubits, num_cx+circuit.num_qubits)
    qc.h(conloc)
    qc.measure(conloc, meas)
    qc.h(conloc)
    circuit_list.append(qc[:3])
    del(qc)
    return circuit_list


def flattener(nested_list: List) -> List:
    """Flatten a nested list structure."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flattener(item))
        else:
            flat_list.append(item)
    return flat_list


def my_measure(circuit_data: List, conq: int, tarq: int, num_qubits: int, num_cx: int, 
              num_shots: int, simulator_seed: int, transpiler_seed: int, noise: bool) -> Dict[str, int]:
    """Assembles a quantum circuit from a list of circuit components, runs and measures the circuit and returns the outcomes

    Parameters:
        circuit_data (list): a list of circuit components
        conq (int): index of the control qubit 
        tarq (int): index of the target qubit
        num_qubits (int): the number of qubits in the original circuit 
        num_cx (int): the number of CNOTs in the original circuit (num_cx+num_qubits is the number of classical bits necessary)
        num_shots (int): the number of shots
        noise (bool): whether to use an ideal or noisy (based on real hardware) backend
    Returns:
        result_dict (dict): a dictionary the keys of which are the measurement outcomes and the items are the counts (contains results
                            of circuit internal measurements, i.e., length of keys is (# of cx) + (# of qubits))
    """
    # initialize circuit and assemble it from components in circuit_data
    my_circ = QuantumCircuit(num_qubits, num_cx+num_qubits)
    for item in circuit_data:
        my_circ.append(item)
    my_circ.measure([*range(num_qubits)], [*range(-num_qubits, 0)])

    # select noisy or ideal backend
    backend = _make_backend(noise)
    
    # transpiled circuit 
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=transpiler_seed)
    transpiled_circuit = pass_manager.run(my_circ)
    
    # initialize sampler
    options = {"simulator": {"seed_simulator": simulator_seed}}
    sampler = SamplerV2(backend, options=options)
    
    # run job and get results
    job = sampler.run([transpiled_circuit], shots=num_shots)
    result_dict = job.result()[0].data.c.get_counts()
    
    return result_dict


def comb_measure(result_in: Dict[str, int], conq: int, tarq: int, num_cx: int) -> Dict[str, int]:
    """Combines quiskit results with circuit internal measurements, e.g., {'0000': 4, '0001': 3} -> {'000': 1}

    Parameters:
        result_in: a dictionary of measurement outcomes with circuit internal measurements
        conq (int): index of the control qubit 
        tarq (int): index of the target qubit 
        num_cx (int): the number of CNOTs in the original circuit (circuit internal measurements are on final num_cx classical bits)
    Returns:
        result_dict (dict): a dictionary the keys of which are the measurement outcomes and the items are the counts (contains results
                            of circuit internal measurements)
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


def circuit_knitter(
    circuit: QuantumCircuit,
    start_qubit: int,
    end_qubit: int,
    num_shots: int,
    config: ExperimentConfig,
    simulator_seed: Optional[int] = None,
    transpiler_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform circuit knitting on a given circuit using the full knitting algorithm.
    
    Args:
        circuit: Input quantum circuit
        start_qubit: Starting qubit index (control qubit)
        end_qubit: Ending qubit index (target qubit)
        num_shots: Number of shots for measurement
        config: Experiment configuration
        simulator_seed: Random seed for simulator
        transpiler_seed: Random seed for transpiler
        
    Returns:
        Dictionary containing knitting results
    """
    conq = start_qubit
    tarq = end_qubit
    
    # List of positions in circuit of CNOT gates acting on control and target qubits
    cx_list = [i for i, gate in enumerate(circuit.data) if gate.operation.name == 'cx' and \
               ((circuit.data[i].qubits[0]._index, circuit.data[i].qubits[1]._index) == (tarq, conq) or \
               (circuit.data[i].qubits[0]._index, circuit.data[i].qubits[1]._index) == (conq, tarq))]
    num_cx = len(cx_list)
    
    # List of 6 numerical prefactors of knitting terms -> list of 6**n prefactors for n CNOTS 
    prefac_list = [1/2, 1/2, -1/2, 1/2, -1/2, 1/2]
    prefac_list = [float(np.prod(elem)) for elem in [*product(*[prefac_list for i in range(num_cx)])]]
    
    # Creates a list corresponding to the circuit to be knitted with each target CNOT replaced by a list of the knitting terms
    nest_list = []
    current_cx = 0
    for i, gate in enumerate(circuit.data):
        if gate.operation.name == 'cx':
            if (gate.qubits[0]._index, gate.qubits[1]._index) == (tarq, conq):
                nest_list.append(knit_lister(circuit, conq, tarq, current_cx, num_cx))
                current_cx += 1
            elif (gate.qubits[0]._index, gate.qubits[1]._index) == (conq, tarq):
                nest_list.append(knit_lister(circuit, tarq, conq, current_cx, num_cx))
                current_cx += 1            
            else:
                nest_list.append([gate])
        elif gate.operation.name not in ("measure", "barrier"):
            nest_list.append([gate])
        else:
            None
    
    # A list of length 6**n containing the instructions to assemble the terms of the decomposition
    circuits = [*product(*nest_list)]
    circuits = [flattener(item) for item in circuits]
    
    if len(circuits) != 6**num_cx or len(circuits) != len(prefac_list):
        raise RuntimeError("Incomplete six-term circuit-knitting expansion")

    # Assemble every term, but construct the backend, pass manager, and sampler only once.
    assembled = []
    for term in circuits:
        term_circuit = QuantumCircuit(circuit.num_qubits, num_cx + circuit.num_qubits)
        for instruction in term:
            term_circuit.append(instruction)
        term_circuit.measure(
            range(circuit.num_qubits),
            range(num_cx, num_cx + circuit.num_qubits),
        )
        assembled.append(term_circuit)

    simulator_seed = simulator_seed if simulator_seed is not None else config.simulator_seed
    transpiler_seed = transpiler_seed if transpiler_seed is not None else config.transpiler_seed
    simulator_seed = 1 if simulator_seed is None else simulator_seed
    transpiler_seed = 1 if transpiler_seed is None else transpiler_seed
    batch_size = config.batch_size
    if batch_size <= 0 or config.execution_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    backend = _make_backend(config.noise)
    if hasattr(backend, "set_options"):
        try:
            backend.set_options(
                max_parallel_experiments=config.max_parallel_experiments,
                max_parallel_threads=config.max_parallel_threads,
            )
        except (AttributeError, TypeError):
            pass
    pass_manager = generate_preset_pass_manager(
        optimization_level=config.optimization_level,
        backend=backend,
        seed_transpiler=transpiler_seed,
    )
    sampler = SamplerV2(
        backend,
        options={"simulator": {"seed_simulator": simulator_seed}},
    )

    versions = {
        package: importlib.metadata.version(package)
        for package in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime")
    }
    compilation_identity = hashlib.sha256(
        pickle.dumps(
            (
                circuit,
                conq,
                tarq,
                config.noise,
                config.optimization_level,
                transpiler_seed,
                versions,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    ).hexdigest()
    run_identity = hashlib.sha256(
        pickle.dumps(
            (compilation_identity, num_shots, simulator_seed),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    ).hexdigest()
    cache_dir = Path(config.cache_dir) / compilation_identity
    checkpoint_dir = Path(config.checkpoint_dir) / run_identity
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_results = [None] * len(assembled)
    transpile_seconds = 0.0
    execute_seconds = 0.0
    for batch_number, start in enumerate(range(0, len(assembled), batch_size)):
        stop = min(start + batch_size, len(assembled))
        checkpoint_path = checkpoint_dir / f"batch-{batch_number:05d}.pkl"
        if config.resume and checkpoint_path.exists():
            with checkpoint_path.open("rb") as stream:
                batch_results = pickle.load(stream)
        else:
            cache_path = cache_dir / f"batch-{batch_number:05d}.pkl"
            if cache_path.exists():
                with cache_path.open("rb") as stream:
                    transpiled = pickle.load(stream)
            else:
                started = time.perf_counter()
                transpiled = pass_manager.run(assembled[start:stop])
                transpile_seconds += time.perf_counter() - started
                temporary = cache_path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    pickle.dump(transpiled, stream, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary, cache_path)

            batch_results = []
            for execution_start in range(0, len(transpiled), config.execution_batch_size):
                execution_stop = execution_start + config.execution_batch_size
                started = time.perf_counter()
                primitive_results = sampler.run(
                    transpiled[execution_start:execution_stop], shots=num_shots
                ).result()
                execute_seconds += time.perf_counter() - started
                batch_results.extend(
                    result.data.c.get_counts() for result in primitive_results
                )
            temporary = checkpoint_path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(batch_results, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, checkpoint_path)

        if len(batch_results) != stop - start:
            raise RuntimeError(f"Incomplete knitting batch {batch_number}")
        raw_results[start:stop] = batch_results

    cum_tot = defaultdict(float)
    for prefactor, internal_counts in zip(prefac_list, raw_results):
        temp_res = comb_measure(internal_counts, conq, tarq, num_cx)
        for bitstring, count in temp_res.items():
            cum_tot[bitstring] += count * prefactor
    res_dict = {key: cum_tot[key] for key in sorted(cum_tot)}

    return {
        'circuit': circuit,
        'results': res_dict,
        'config': config.__dict__,
        'timestamp': datetime.now().isoformat(),
        'qubits': {'start_qubit': start_qubit, 'end_qubit': end_qubit},
        'num_cnots': num_cx,
        'manifest': {
            'run_id': run_identity,
            'compilation_id': compilation_identity,
            'versions': versions,
            'num_terms': len(circuits),
            'batch_size': batch_size,
            'simulator_seed': simulator_seed,
            'transpiler_seed': transpiler_seed,
            'transpile_seconds': transpile_seconds,
            'execution_batch_size': config.execution_batch_size,
            'execute_seconds': execute_seconds,
        },
    }


def save_experiment_results(
    results: Dict[str, Any],
    filename: str,
    config: ExperimentConfig
) -> str:
    """
    Save experiment results to a file.
    
    Args:
        results: Experiment results to save
        filename: Base filename for results
        config: Experiment configuration
        
    Returns:
        Full path to saved file
    """
    result_dir = Path(config.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    full_path = result_dir / f"{filename}.pkl"
    temporary = full_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, full_path)

    if "manifest" in results:
        manifest_path = full_path.with_suffix(".manifest.json")
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(results["manifest"], indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_manifest, manifest_path)

    return str(full_path)


def load_experiment_results(filename: str, config: ExperimentConfig) -> Dict[str, Any]:
    """
    Load experiment results from a file.
    
    Args:
        filename: Base filename of results
        config: Experiment configuration
        
    Returns:
        Loaded experiment results
    """
    full_path = f"{config.results_dir}/{filename}.pkl"
    
    with open(full_path, 'rb') as file:
        return pickle.load(file)