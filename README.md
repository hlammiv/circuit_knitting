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

## Sec. V data (state-preparation fix of 2026-10-26)

`superposition_state_prep` in `functions.ipynb` used to apply an X to the link
qubit $\sigma_{m,m+1}$ as well as to the site $m+1$.  The resulting state is
$Z_{m,m+1}X_{m+1}\ket{\phi}$: orthogonal to the intended
$(\ket{\Omega}+\ket{P})/\sqrt2$ and in the Gauss-law sector with a static charge
at site $m$.  Every Sec. V number and figure (`mass_scan.pdf`,
`trotter_vs_exact.pdf`, Fig. 7) produced before that date comes from the wrong
sector and must be regenerated:

```bash
python -m pytest -q test_state_prep.py          # checks the prepared state
python scripts/rerun_sec5.py --benchmark        # time one run of each kind
python scripts/rerun_sec5.py --stage noiseless  # Figs. 5, 6 and Table II inputs
python scripts/rerun_sec5.py --stage noisy      # un-knitted noisy replicas
python scripts/rerun_sec5.py --stage knitted    # knitted replicas (slow)
python scripts/rerun_sec5.py --stage figures    # redraw Figs. 5-7
```

The driver reproduces the notebook workflows with fixed seeds and skips any
output that already exists (`--force` recomputes).  The pre-fix data remain in
the git history.

## Tests

```bash
python -m pytest -q tests_analysis
python improved/tests/run_all_tests.py
```

The full simulation stack has additional Qiskit dependencies listed in
`improved/requirements.txt`.
