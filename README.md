# Circuit knitting for periodic-boundary lattice simulations

This repository contains the circuit-knitting simulations and manuscript for
*Periodic Boundary Conditions for Lattice Field Theories with Circuit
Knitting*.

## Layout

- `improved/` — refactored circuit-knitting implementation and tests
- top-level notebooks — original simulation and analysis workflows, retained
  with their historical outputs
- `data/` — compact derived arrays and simulation samples used by the plots
- `scripts/fit_fig7.py` — reproducible fits and publication plot for Fig. 7
- `paper/` — LaTeX manuscript imported from the 2026-08-19 draft snapshot

## Fig. 7 analysis

Create an isolated environment, install the analysis dependencies, and run:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/fit_fig7.py --bootstrap-samples 10000 --seed 20260819
```

This regenerates the paper figure and writes machine-readable and LaTeX fit
summaries under `paper/generated/`. The script compares the truncated cosine
used in the paper with the older five-parameter damped model. It uses the
truncated form unless the damped model is clearly preferred for both knitted
and non-knitted data and remains stable under bootstrap resampling.

## Tests

```bash
python -m pytest -q tests_analysis
python improved/tests/run_all_tests.py
```

The full simulation stack has additional Qiskit dependencies listed in
`improved/requirements.txt`.
