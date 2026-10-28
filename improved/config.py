"""
Configuration management for circuit knitting experiments.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for circuit knitting experiments."""
    data_dir: str = "data"
    results_dir: str = "results"
    noise: bool = False
    num_shots: int = 1024
    epsilon_values: list = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    step_values: list = field(default_factory=lambda: [1, 2])
    simulator_seed: Optional[int] = None
    transpiler_seed: Optional[int] = None
    optimization_level: int = 1
    # Physical qubit for each circuit qubit (passed to the transpiler as
    # initial_layout).  None lets the layout pass choose.
    initial_layout: Optional[list] = None
    batch_size: int = 1
    execution_batch_size: int = 1
    max_parallel_experiments: int = 1
    max_parallel_threads: int = 0
    cache_dir: str = ".cache/circuit_knitting"
    checkpoint_dir: str = "checkpoints/circuit_knitting"
    resume: bool = True
    # Knitting terms with a mid-circuit measurement defeat Aer's end-of-circuit
    # shot sampling, so with a noise model every shot becomes a separate
    # statevector trajectory.  With exact_sampling the exact outcome
    # distribution of every term is computed instead by density-matrix branch
    # enumeration (improved/exact_sampling.py) and the shots are drawn from it
    # multinomially: same statistics as Aer's own sampling, a different random
    # stream.  Off by default; checkpoints of the two paths are kept apart.
    exact_sampling: bool = False
    # A mid-circuit measurement splits the branch tree only when at least this
    # many gates have run since the previous split; shorter segments are
    # replayed per outcome instead of saving and restoring the density matrix.
    # Performance only, the result does not depend on it.
    exact_sampling_min_split_ops: int = 32


def get_config() -> ExperimentConfig:
    """Get the default configuration."""
    return ExperimentConfig()


def ensure_directories(config: ExperimentConfig):
    """Ensure required directories exist."""
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
