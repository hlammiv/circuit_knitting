# Circuit-knitting execution performance

The production Fig. 7 workload cuts two CNOTs per Trotter layer.  The six-term
identity therefore produces 36 circuits for one layer and 1,296 circuits for
two layers.  Across four time points of each type and ten seeds, this is 53,280
circuit executions and approximately 873 million shots at 16,384 shots per
term.

The canonical engine is `improved.experiment.circuit_knitter`.  It preserves
the six-term coefficients and final pickle result structure while:

- fixing the transpiler seed across statistical replicas;
- constructing the backend, pass manager, and sampler once per experiment;
- transpiling and submitting circuits in configurable batches;
- caching transpiled batches;
- checkpointing raw counts after every completed batch; and
- recording seeds, term counts, batch size, and timings in a manifest.

Use `ExperimentConfig(batch_size=8, execution_batch_size=1,
max_parallel_experiments=1)` as the safe 16-GB CPU default.  The fake
Washington target has 127 physical qubits, and executing several transpiled
circuits concurrently exceeded this workstation's memory during the low-shot
benchmark.  The default therefore batches transpilation and caching while
executing one circuit at a time with a reused sampler.  Setting `resume=True`
reuses compatible checkpoints.  Cache and checkpoint identities include the
circuit, cut, shots, noise selection, optimization level, and both seeds.

Run `python scripts/benchmark_knitter.py` for the low-shot engineering
benchmark.  It uses temporary files and synthetic circuits and is not suitable
for paper results.  The full production workload should not be started until
## Lenore CPU profile

Lenore was tested through SSH on port 60022 with 32 CPU threads and 125 GiB
RAM.  The NVIDIA driver/library versions currently disagree, so these results
use CPU Aer only.  All tests used synthetic circuits and at most 36 terms.

| cut terms | shots/term | execution batch | engine time | peak RSS |
|---:|---:|---:|---:|---:|
| 6 | 16 | 1 | 14.20 s | 540 MiB |
| 6 | 16 | 2 | 10.79 s | 526 MiB |
| 6 | 16 | 4 | 8.44 s | 507 MiB |
| 6 | 16 | 6 | 5.24 s | 488 MiB |
| 36 | 8 | 36 | 6.48 s | 496 MiB |

For Lenore, start engineering tests with
`ExperimentConfig(batch_size=36, execution_batch_size=36,
max_parallel_experiments=1)`.  This is a host-specific profile, not the
portable default.  Before any production run, benchmark the four-cut expansion
at one shot per term and confirm memory and checkpoint behavior.

the low-shot benchmark and memory usage have been reviewed.
