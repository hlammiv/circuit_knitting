#!/usr/bin/env python3
"""Noisy runs of the Clifford calibration circuits on the pinned line.

--stage direct : un-knitted, every (replica, time slot) with the transpiler seed
                 of the corresponding physics run and a fresh simulator seed;
                 writes data/calibration_line/direct.pkl
--stage knitted: knitted with the production knitter (line layout, level 3,
                 exact sampling), one replica per step count; writes
                 data/calibration_line/knitted_step{N}.pkl
"""
import argparse, pickle, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
import rerun_sec5 as r
from clifford_calibration import calibration_circuit, exact_occupation

ANGLES = (np.pi / 2, np.pi / 2, np.pi / 2)
OUT = r.DATA / "calibration_line"; OUT.mkdir(exist_ok=True)
SCHEDULE = [(0, 0.0)] + [(1, e) for e in (0.2, 0.4, 0.6, 0.8)] + [(2, e) for e in (0.5, 0.6, 0.7, 0.8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("direct", "knitted"), required=True)
    parser.add_argument("--steps", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261101, help="knitted: replica seed")
    parser.add_argument("--direct-sim", choices=("fake", "aer"), default="fake",
                        help="direct stage: simulator for the un-knitted runs (see rerun_sec5.DIRECT_SIM)")
    args = parser.parse_args()
    r.LAYOUT = "line"; r.EXACT_SAMPLING = True; r.THREADS = args.threads; r.DIRECT_SIM = args.direct_sim
    fn = r.load_notebook_functions()
    circuits = {s: calibration_circuit(fn, s, *ANGLES) for s in (1, 2)}
    exact = {s: exact_occupation(circuits[s]) for s in (1, 2)}
    print("exact calibration <n_6>:", exact, flush=True)
    if args.stage == "direct":
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
        backend = FakeWashingtonV2(); line = r.line_layout()
        results = {"exact": exact, "runs": {}}
        for seed in r.NOISY_SEEDS:
            my_seed = seed
            for slot, (steps, epsilon) in enumerate(SCHEDULE):
                np.random.seed(my_seed); my_seed = np.random.randint(1024 ** 2)
                _sim = np.random.randint(1024 ** 2); transpiler_seed = np.random.randint(1024 ** 2)
                if steps == 0:
                    continue
                pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                                  seed_transpiler=transpiler_seed, initial_layout=line)
                transpiled = pm.run(circuits[steps])
                counts = r.run_direct(fn, circuits[steps], r.NOISY_SHOTS,
                                      simulator_seed=int(np.random.default_rng([seed, slot, 7]).integers(1024 ** 2)),
                                      transpiler_seed=transpiler_seed)
                value = fn["fermion_number"](counts, r.MID)
                results["runs"][(seed, slot)] = {"steps": steps, "epsilon": epsilon, "cx": transpiled.count_ops().get("cx", 0),
                                                 "value": value, "counts": counts}
                print(f"  seed {seed} slot {slot} steps {steps}: cx {results['runs'][(seed, slot)]['cx']}  <n_6> {value:.4f}"
                      f"  f = {(0.5 - value) / (0.5 - exact[steps]):.3f}", flush=True)
        pickle.dump(results, open(OUT / ("direct.pkl" if args.direct_sim == "fake" else "direct_aer.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        make_config, circuit_knitter = r.knitter()
        config = make_config()
        for steps in args.steps:
            path = OUT / f"knitted_step{steps}_seed{args.seed}.pkl"
            if path.exists():
                print(f"  {path.name}: exists"); continue
            t0 = time.time()
            res = circuit_knitter(circuits[steps], *r.CUT, r.NOISY_SHOTS, config,
                                  simulator_seed=args.seed, transpiler_seed=r.KNIT_TRANSPILER_SEED)
            value = fn["fermion_number"](res["results"], r.MID)
            pickle.dump({"exact": exact[steps], "results": res["results"], "num_cnots": res["num_cnots"]},
                        open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  {path.name}: <n_6> {value:.4f}  f = {(0.5 - value) / (0.5 - exact[steps]):.3f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
