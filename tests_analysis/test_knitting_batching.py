from __future__ import annotations

from types import SimpleNamespace

import pytest

qiskit = pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from improved.config import ExperimentConfig
from improved import experiment


class _Counts:
    def __init__(self, shots: int):
        self.shots = shots

    def get_counts(self):
        return {"0000": self.shots}


class _Job:
    def __init__(self, circuits, shots):
        self.circuits = circuits
        self.shots = shots

    def result(self):
        return [
            SimpleNamespace(data=SimpleNamespace(c=_Counts(self.shots)))
            for _ in self.circuits
        ]


class _Sampler:
    run_calls = 0

    def __init__(self, backend, options):
        self.backend = backend
        self.options = options

    def run(self, circuits, shots):
        type(self).run_calls += 1
        return _Job(circuits, shots)


class _Backend:
    def set_options(self, **options):
        self.options = options


class _PassManager:
    run_calls = 0

    def run(self, circuits):
        type(self).run_calls += 1
        return circuits


def test_six_term_expansion_batches_and_resumes(monkeypatch, tmp_path):
    _Sampler.run_calls = 0
    _PassManager.run_calls = 0
    monkeypatch.setattr(experiment, "AerSimulator", _Backend)
    monkeypatch.setattr(experiment, "SamplerV2", _Sampler)
    monkeypatch.setattr(
        experiment,
        "generate_preset_pass_manager",
        lambda **kwargs: _PassManager(),
    )

    circuit = qiskit.QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.rz(0.2, 0)
    circuit.cx(0, 1)

    config = ExperimentConfig(
        noise=False,
        num_shots=4,
        execution_batch_size=8,
        batch_size=8,
        cache_dir=str(tmp_path / "cache"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    first = experiment.circuit_knitter(
        circuit, 0, 1, 4, config, simulator_seed=11, transpiler_seed=17
    )

    assert first["num_cnots"] == 2
    assert first["manifest"]["num_terms"] == 36
    assert _Sampler.run_calls == 5
    assert _PassManager.run_calls == 5
    assert first["results"] == {"00": pytest.approx(4.0)}

    second = experiment.circuit_knitter(
        circuit, 0, 1, 4, config, simulator_seed=11, transpiler_seed=17
    )
    assert second["results"] == first["results"]
    assert _Sampler.run_calls == 5
    assert _PassManager.run_calls == 5


def test_invalid_batch_size_fails_before_execution(monkeypatch, tmp_path):
    circuit = qiskit.QuantumCircuit(2)
    circuit.cx(0, 1)
    config = ExperimentConfig(
        batch_size=0,
        cache_dir=str(tmp_path / "cache"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    with pytest.raises(ValueError, match="batch"):
        experiment.circuit_knitter(circuit, 0, 1, 4, config)
