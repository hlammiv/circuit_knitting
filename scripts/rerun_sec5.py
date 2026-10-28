#!/usr/bin/env python3
"""Regenerate every Sec. V data product after the 2026-10-26 state-preparation fix.

The circuit in ``superposition_state_prep`` (functions.ipynb) carried a spurious
X on the link sigma_{m,m+1}.  All Sec. V data were produced with it, so all of it
is regenerated here, reproducing the notebook workflows
(mass_epsilon_scan.ipynb, plot_data.ipynb, main.py) with fixed seeds.

Stages (run in this order; each is skipped if its outputs exist unless --force):

  noiseless   mass_scan_plot_data/ferm_num_*.npy, bs_error_*.npy and
              data/{ferm_num,bs_error}_exact_no_noise_no_knit.npy
              (Figs. 5 and 6, Table II)
  noisy       data/noisy_no_knit_trotter_evolution.pkl and the resampled
              no_knit_noisy_*.npy                                   (Fig. 7)
  knitted     data/noisy_knitted_trotter_evol/*.pkl, the seeds list,
              data/noisy_knitted_trotter_evolution.pkl and the resampled
              knitted_noisy_*.npy                                   (Fig. 7)
  figures     scripts/plot_fig5.py, plot_fig6.py, fit_fig7.py

Usage (from the repository root):
    python scripts/rerun_sec5.py --stage noiseless
    python scripts/rerun_sec5.py --stage noisy
    python scripts/rerun_sec5.py --stage knitted          # the slow one
    python scripts/rerun_sec5.py --stage knitted --seeds 20261126 --runs 6 7 --threads 2   # one worker
    python scripts/rerun_sec5.py --stage figures
    python scripts/rerun_sec5.py --benchmark              # time one run of each kind

Layouts (--layout, default free):

  free   the transpiler places the 12 qubits itself (level 3 un-knitted, as in
         functions.ipynb do_run; level 1 knitted, the knitter default).  On the
         heavy-hex lattice it folds the ring so that closing it costs ~6 SWAPs.
  line   both circuits are pinned to the same straight line of 12 physical
         qubits (the line with the lowest mean CNOT error) at level 3, so the
         periodic hopping term has to be routed across the whole line.  This
         models hardware that cannot fold the ring.  Outputs carry a "_line"
         suffix (data/*_line.*, data/noisy_knitted_trotter_evol_line/).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KNIT_DIR = DATA / "noisy_knitted_trotter_evol"
LAYOUT = "free"            # or "line"; see module docstring and main()
SUFFIX = ""                # "_line" for the pinned layout
SCAN_DIR = ROOT / "mass_scan_plot_data"

# Lattice and run parameters shared by every Sec. V result (params.ipynb).
N_SITES = 6
MID = N_SITES - 2          # qubit of the center site m = N_s/2 - 1
MASS = 1.125
MASSES = {"1p0": 1.0, "1p125": 1.125, "1p25": 1.25}
NOISY_SHOTS = 16384
NOISELESS_SHOTS = 1024 ** 2
# Qubits (site 0, site 5) between which the periodic hopping term is cut.
CUT = (0, 10)
# Seeds: the originals were wall-clock timestamps; these are fixed so the rerun
# is reproducible.  Ten statistical replicas, as in the paper.
NOISY_SEEDS = [20261026 + i for i in range(10)]
KNIT_SEEDS = [20261126 + i for i in range(10)]
# The batched knitting engine fixes compilation across replicas (main.py).
KNIT_TRANSPILER_SEED = 20260819
# Aer thread cap for this process (0 = all cores); set by --threads so several
# workers can share a machine.  Also exported as OMP_NUM_THREADS before Aer loads.
THREADS = 0
# --exact-sampling: per-term exact outcome distributions by density-matrix
# branch enumeration instead of Aer's per-shot trajectories (same statistics,
# different random stream; see ExperimentConfig.exact_sampling).  Outputs go
# to a separate directory so the shot-sampled data are never overwritten.
EXACT_SAMPLING = False

# Time grid of Figs. 6 and 7: t = 0, 0.2, ..., 1.6 with 0, 1, 1, 1, 1, 2, 2, 2, 2
# Trotter steps, i.e. epsilon = t for one step and t/2 for two.
def variable_step_schedule():
    for i in range(9):
        if i == 0:
            yield 0, 1.0
        elif i <= 4:
            yield 1, round(0.2 * i, 1)
        else:
            yield 2, round(0.1 * i, 1)


def load_notebook_functions():
    """Pull the simulation helpers out of functions.ipynb without import_ipynb."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2

    namespace = {
        "QuantumCircuit": QuantumCircuit, "np": np,
        "generate_preset_pass_manager": generate_preset_pass_manager,
        "AerSimulator": AerSimulator, "SamplerV2": SamplerV2,
        "FakeWashingtonV2": FakeWashingtonV2,
    }
    wanted = ["superposition_state_prep", "meson_operator", "trotter1", "fermion_mass",
              "gauge_kinetic", "fermion_hopping", "trotter_stepper", "fermion_number",
              "do_run", "bootstrap_error"]
    for cell in json.load(open(ROOT / "functions.ipynb"))["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and any(f"def {name}(" in source for name in wanted):
            exec(source, namespace)
    missing = [name for name in wanted if name not in namespace]
    if missing:
        raise RuntimeError(f"functions.ipynb is missing {missing}")
    namespace["bootstrap_error"] = bootstrap_error
    return namespace


def bootstrap_error(counts, mid, shots, seed=1, resamples=100):
    """Vectorised version of the notebook's bootstrap_error.

    The notebook draws ``shots`` bitstrings 100 times in a Python loop, which
    costs about a minute per point at 1024**2 shots.  Only the signed
    fraction of strings with bit ``mid+1`` set enters the estimate, so the
    resample reduces to a multinomial over three categories: positive-count
    strings with the bit set, negative-count strings with the bit set, and
    the rest.  Same statistics, no Python loop.
    """
    rng = np.random.default_rng(seed)
    weights = np.array([abs(c) for c in counts.values()], dtype=float)
    weights /= weights.sum()
    bit = np.array([s[mid + 1] == "1" for s in counts])
    negative = np.array([c < 0 for c in counts.values()])
    p = [weights[bit & ~negative].sum(), weights[bit & negative].sum()]
    p.append(max(0.0, 1.0 - sum(p)))
    draws = rng.multinomial(shots, p, size=resamples)
    return float(np.std((draws[:, 0] - draws[:, 1]) / shots))


def knitter():
    sys.path.insert(0, str(ROOT))
    from improved.config import ExperimentConfig
    from improved.experiment import circuit_knitter

    def make_config():
        if LAYOUT == "line":
            return ExperimentConfig(noise=True, num_shots=NOISY_SHOTS, max_parallel_threads=THREADS,
                                    optimization_level=3, initial_layout=line_layout(),
                                    exact_sampling=EXACT_SAMPLING)
        return ExperimentConfig(noise=True, num_shots=NOISY_SHOTS, max_parallel_threads=THREADS,
                                exact_sampling=EXACT_SAMPLING)
    return make_config, circuit_knitter


_LINE = None


def line_layout():
    """The straight 12-qubit line on FakeWashingtonV2 with the lowest mean CNOT error.

    A line is a shortest path between two qubits at graph distance 11.  Circuit
    qubit i is pinned to the i-th qubit of the line, so the cut hopping term
    (qubits 0 and 10) spans ten hops.  Also written to data/line_layout.json.
    """
    global _LINE
    if _LINE is not None:
        return _LINE
    import rustworkx as rx
    from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
    backend = FakeWashingtonV2()
    graph = backend.coupling_map.graph.to_undirected()
    lengths = rx.all_pairs_dijkstra_path_lengths(graph, lambda _: 1.0)

    def cx_error(a, b):
        key = (a, b) if (a, b) in backend.target["cx"] else (b, a)
        return backend.target["cx"][key].error

    lines = []
    for a in range(backend.num_qubits):
        for b, d in lengths[a].items():
            if a < b and d == 2 * N_SITES - 1:
                path = list(rx.dijkstra_shortest_paths(graph, a, b)[b])
                lines.append((np.mean([cx_error(path[i], path[i + 1]) for i in range(len(path) - 1)]), path))
    error, _LINE = min(lines, key=lambda item: (item[0], item[1]))
    _LINE = [int(q) for q in _LINE]
    json.dump({"backend": backend.name, "line": _LINE, "mean_cx_error": float(error)},
              open(DATA / "line_layout.json", "w"), indent=1)
    return _LINE


def run_direct(fn, circuit, num_shots, simulator_seed, transpiler_seed):
    """Noisy un-knitted run: the notebook's do_run (free layout, level 3) or the
    same thing pinned to the straight line."""
    if LAYOUT == "free":
        return fn["do_run"](circuit, num_shots, noise=True,
                            simulator_seed=simulator_seed, transpiler_seed=transpiler_seed)
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
    backend = FakeWashingtonV2()
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                                seed_transpiler=transpiler_seed, initial_layout=line_layout())
    sampler = SamplerV2(backend, options={"simulator": {"seed_simulator": simulator_seed}})
    result = sampler.run([pass_manager.run(circuit)], shots=num_shots).result()[0].data.meas.get_counts()
    return {key: result[key] for key in sorted(result)}


def circuit_for(fn, steps, epsilon, mass):
    circuit = fn["trotter_stepper"](steps, N_SITES, epsilon, mass, MID).decompose().decompose()
    circuit.measure_all()
    return circuit


def outputs_exist(paths):
    return all(Path(p).exists() for p in paths)


# ----------------------------------------------------------------------------
def stage_noiseless(fn, force):
    """mass_epsilon_scan.ipynb: exact (eps=0.05, 0.025) and Trotterized evolution."""
    SCAN_DIR.mkdir(exist_ok=True)

    def run(tag, schedule, mass):
        out = (SCAN_DIR / f"ferm_num_{tag}.npy", SCAN_DIR / f"bs_error_{tag}.npy")
        if outputs_exist(out) and not force:
            print(f"  {tag}: exists, skipping"); return
        values, errors = [], []
        for steps, epsilon in schedule:
            res = fn["do_run"](circuit_for(fn, steps, epsilon, mass), NOISELESS_SHOTS)
            values.append(fn["fermion_number"](res, MID))
            errors.append(fn["bootstrap_error"](res, MID, NOISELESS_SHOTS))
        np.save(out[0], np.array(values)); np.save(out[1], np.array(errors))
        print(f"  {tag}: {len(values)} points written")

    for key, mass in MASSES.items():
        run(f"m{key}_e0p005", [(4 * i, 0.05) for i in range(33)], mass)
        run(f"m{key}_e0p0025", [(8 * i, 0.025) for i in range(9)], mass)
        run(f"m{key}_trot", list(variable_step_schedule()), mass)
    # plot_data.ipynb cell 2 is the m=1.125, eps=0.05 series under another name.
    for stem in ("ferm_num", "bs_error"):
        np.save(DATA / f"{stem}_exact_no_noise_no_knit.npy", np.load(SCAN_DIR / f"{stem}_m1p125_e0p005.npy"))
    print("  data/*_exact_no_noise_no_knit.npy written")


def resample(runs, n_points=9):
    """plot_data.ipynb cells 12/21: pool Gaussian resamples of every replica."""
    means, errors = [], []
    for i in range(n_points):
        pooled = np.concatenate([
            np.random.normal(r["fermion_number"][i], r["bootstrap_error"][i], 1000) for r in runs.values()
        ])
        means.append(pooled.mean()); errors.append(pooled.std())
    return np.array(means), np.array(errors)


def stage_noisy(fn, force):
    """plot_data.ipynb make_runs: FakeWashingtonV2, no knitting, ten replicas."""
    out = DATA / f"noisy_no_knit_trotter_evolution{SUFFIX}.pkl"
    if out.exists() and not force:
        runs = pickle.load(open(out, "rb")); print("  replicas exist, resampling only")
    else:
        runs = {}
        for seed in NOISY_SEEDS:
            values, errors, counts = [], [], []
            my_seed = seed
            for steps, epsilon in [(0, 0.0)] + [(1, e) for e in (0.2, 0.4, 0.6, 0.8)] + [(2, e) for e in (0.5, 0.6, 0.7, 0.8)]:
                np.random.seed(my_seed)
                my_seed = np.random.randint(1024 ** 2)
                res = run_direct(fn, circuit_for(fn, steps, epsilon, MASS), NOISY_SHOTS,
                                 simulator_seed=np.random.randint(1024 ** 2),
                                 transpiler_seed=np.random.randint(1024 ** 2))
                values.append(fn["fermion_number"](res, MID))
                errors.append(fn["bootstrap_error"](res, MID, NOISY_SHOTS))
                counts.append(res)
            # counts are kept so other sites can be analysed without rerunning
            runs[seed] = {"fermion_number": np.array(values), "bootstrap_error": np.array(errors), "counts": counts}
            print(f"  seed {seed}: {np.round(values, 3)}", flush=True)
        pickle.dump(runs, open(out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    np.random.seed(0)
    mean, err = resample(runs)
    np.save(DATA / f"no_knit_noisy_resampled_fermion_number{SUFFIX}.npy", mean)
    np.save(DATA / f"no_knit_noisy_resampled_bootstrap_error{SUFFIX}.npy", err)
    print("  resampled:", np.round(mean, 4))


def knit_path(steps, epsilon, seed):
    return KNIT_DIR / f"step{steps}_epsilon0{str(epsilon)[-1]}_count{NOISY_SHOTS}_with_noise_knitted_seed{seed}.pkl"


KNIT_RUNS = [(1, e) for e in (0.2, 0.4, 0.6, 0.8)] + [(2, e) for e in (0.5, 0.6, 0.7, 0.8)]


def stage_knitted(fn, force, seeds, runs=None):
    """main.py + plot_data.ipynb cell 17: knitted noisy runs, ten replicas.

    ``runs`` (indices into KNIT_RUNS) lets two workers share one replica; the
    simulator seeds are drawn in the same order whichever runs a worker does.
    """
    make_config, circuit_knitter = knitter()
    KNIT_DIR.mkdir(exist_ok=True)
    config = make_config()
    for seed in seeds:
        np.random.seed(seed)
        for index, (steps, epsilon) in enumerate(KNIT_RUNS):
            sim_seed, _ = np.random.randint(1024 ** 2), np.random.randint(1024 ** 2)
            if runs is not None and index not in runs:
                continue
            path = knit_path(steps, epsilon, seed)
            if path.exists() and not force:
                print(f"  {path.name}: exists"); continue
            t0 = time.time()
            res = circuit_knitter(circuit_for(fn, steps, epsilon, MASS), *CUT, NOISY_SHOTS, config,
                                  simulator_seed=sim_seed, transpiler_seed=KNIT_TRANSPILER_SEED)
            pickle.dump(res["results"], open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  {path.name}: value {fn['fermion_number'](res['results'], MID):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    with open(KNIT_DIR / "noisy_knitted_seeds_list.txt", "w") as stream:
        stream.writelines(f"{s}\n" for s in KNIT_SEEDS)
    # Replicas may be split over several workers/machines (--seeds); assemble
    # only once every replica's eight runs are on disk.
    missing = [knit_path(s, e, seed).name for seed in KNIT_SEEDS for s, e in KNIT_RUNS
               if not knit_path(s, e, seed).exists()]
    if missing:
        print(f"  {len(missing)} of {8 * len(KNIT_SEEDS)} knitted runs still missing; assembly deferred "
              f"(rerun --stage knitted once all replicas are done)"); return

    # Assemble per-replica series; the t=0 point is an un-knitted noisy run.
    runs = {}
    for seed in KNIT_SEEDS:
        np.random.seed(seed)
        np.random.seed(np.random.randint(1024 ** 2))
        res0 = run_direct(fn, circuit_for(fn, 0, 0.0, MASS), NOISY_SHOTS,
                          simulator_seed=np.random.randint(1024 ** 2), transpiler_seed=np.random.randint(1024 ** 2))
        series = [res0] + [pickle.load(open(knit_path(s, e, seed), "rb")) for s, e in KNIT_RUNS]
        runs[seed] = {"fermion_number": np.array([fn["fermion_number"](r, MID) for r in series]),
                      "bootstrap_error": np.array([fn["bootstrap_error"](r, MID, NOISY_SHOTS) for r in series])}
    pickle.dump(runs, open(DATA / f"noisy_knitted_trotter_evolution{SUFFIX}.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    np.random.seed(0)
    mean, err = resample(runs)
    np.save(DATA / f"knitted_noisy_resampled_fermion_number{SUFFIX}.npy", mean)
    np.save(DATA / f"knitted_noisy_resampled_bootstrap_error{SUFFIX}.npy", err)
    print("  resampled:", np.round(mean, 4))


def stage_figures():
    for cmd in (["scripts/plot_fig5.py"], ["scripts/plot_fig6.py"],
                ["scripts/fit_fig7.py", "--bootstrap-samples", "10000", "--seed", "20260819"]):
        print("  running", " ".join(cmd), flush=True)
        subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def benchmark(fn):
    make_config, circuit_knitter = knitter()
    c = circuit_for(fn, 1, 0.4, MASS)
    t0 = time.time(); fn["do_run"](c, NOISELESS_SHOTS); print(f"  noiseless, {NOISELESS_SHOTS} shots: {time.time() - t0:.1f} s")
    t0 = time.time(); fn["do_run"](c, NOISY_SHOTS, noise=True, simulator_seed=1, transpiler_seed=1); print(f"  noisy no-knit: {time.time() - t0:.1f} s")
    t0 = time.time(); r = circuit_knitter(c, *CUT, NOISY_SHOTS, make_config(),
                                           simulator_seed=1, transpiler_seed=KNIT_TRANSPILER_SEED)
    print(f"  noisy knitted, 1 step ({r.get('num_cnots')} cut CNOTs): {time.time() - t0:.1f} s")
    print("  per stage: noiseless = 9 series x (33+9+9)/3 pts; noisy = 90 runs; knitted = 40 one-step + 40 two-step runs,\n"
          "  a two-step run has 6^4/6^2 = 36x the circuits of a one-step run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["noiseless", "noisy", "knitted", "figures", "all"])
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    parser.add_argument("--seeds", type=int, nargs="*", help="knitted stage only: subset of replicas to run")
    parser.add_argument("--runs", type=int, nargs="*", choices=range(8),
                        help="knitted stage only: which of the eight (steps, epsilon) runs per replica "
                             "(0-3 one-step, 4-7 two-step; default all)")
    parser.add_argument("--layout", choices=["free", "line"], default="free",
                        help="free: transpiler chooses the qubits (default); line: both circuits pinned "
                             "to the same straight 12-qubit line at level 3 (outputs get a _line suffix)")
    parser.add_argument("--exact-sampling", action="store_true",
                        help="knitted stage: exact per-term distributions by branch enumeration "
                             "(much faster; writes to data/noisy_knitted_trotter_evol*_exact)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Aer threads for this process (default 0 = all cores); use with --seeds to run "
                             "several workers side by side")
    args = parser.parse_args()
    global THREADS
    THREADS = args.threads
    global LAYOUT, SUFFIX, KNIT_DIR
    LAYOUT = args.layout
    if LAYOUT == "line":
        SUFFIX = "_line"; KNIT_DIR = DATA / "noisy_knitted_trotter_evol_line"
        print(f"  pinned line layout: {line_layout()}", flush=True)
    global EXACT_SAMPLING
    EXACT_SAMPLING = args.exact_sampling
    if EXACT_SAMPLING:
        SUFFIX += "_exact"; KNIT_DIR = KNIT_DIR.with_name(KNIT_DIR.name + "_exact")
    if THREADS:
        os.environ["OMP_NUM_THREADS"] = str(THREADS)
    if not args.stage and not args.benchmark:
        parser.error("give --stage or --benchmark")
    fn = load_notebook_functions()
    if args.benchmark:
        benchmark(fn); return
    stages = ["noiseless", "noisy", "knitted", "figures"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"== {stage}", flush=True)
        if stage == "noiseless": stage_noiseless(fn, args.force)
        elif stage == "noisy": stage_noisy(fn, args.force)
        elif stage == "knitted": stage_knitted(fn, args.force, args.seeds or KNIT_SEEDS, args.runs)
        else: stage_figures()


if __name__ == "__main__":
    main()
