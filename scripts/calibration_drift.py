#!/usr/bin/env python3
"""Calibration drift test: rerun the Clifford calibrations with every Washington
error rate scaled by a common factor s (gate and readout errors times s, T1 and
T2 divided by s), while the physics data stay at the nominal noise.  This mimics
a calibration taken when the device was slightly better or worse than during the
physics run.  Uniform scaling keeps the ordering of error rates, so the transpiled
circuits are unchanged and only the simulated noise differs.

--stage direct : all (replica, slot) calibrations, as in calibration_runs.py
--stage knitted: knitted calibrations for the requested step counts
Writes data/calibration_line/drift_s{s}/{direct,knitted_step{N}}.pkl
"""
import argparse, pickle, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import numpy as np
import qiskit_ibm_runtime.fake_provider as fake_provider
from qiskit.providers import QubitProperties

_Nominal = fake_provider.FakeWashingtonV2


def scaled_washington(scale):
    def factory(*args, **kwargs):
        backend = _Nominal(*args, **kwargs)
        target = backend.target
        for name in target.operation_names:
            for qargs, props in (target[name] or {}).items():
                if props is not None and props.error is not None:
                    props.error = min(props.error * scale, 0.5)
        # qubit_properties returns copies, so the list is rebuilt and reassigned
        target.qubit_properties = [
            None if p is None else QubitProperties(
                t1=None if p.t1 is None else p.t1 / scale,
                t2=None if p.t2 is None else p.t2 / scale,
                frequency=p.frequency)
            for p in target.qubit_properties or []]
        backend._setup_sim()                 # the noise model was built from the unscaled target
        return backend
    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--stage", choices=("direct", "knitted"), required=True)
    parser.add_argument("--steps", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()

    factory = scaled_washington(args.scale)
    fake_provider.FakeWashingtonV2 = factory          # used by rerun_sec5.run_direct
    import improved.experiment as experiment
    experiment.FakeWashingtonV2 = factory              # used by the knitter's backend
    import rerun_sec5 as r
    import calibration_runs as cr
    from clifford_calibration import calibration_circuit, exact_occupation

    out = r.DATA / "calibration_line" / f"drift_s{args.scale:g}"; out.mkdir(parents=True, exist_ok=True)
    r.LAYOUT = "line"; r.EXACT_SAMPLING = True; r.THREADS = args.threads; r.DIRECT_SIM = "aer"
    fn = r.load_notebook_functions()
    circuits = {s: calibration_circuit(fn, s, *cr.ANGLES) for s in (1, 2)}
    exact = {s: exact_occupation(circuits[s]) for s in (1, 2)}
    if args.stage == "direct":
        results = {"exact": exact, "scale": args.scale, "runs": {}}
        for seed in r.NOISY_SEEDS:
            my_seed = seed
            for slot, (steps, epsilon) in enumerate(cr.SCHEDULE):
                np.random.seed(my_seed); my_seed = np.random.randint(1024 ** 2)
                _sim = np.random.randint(1024 ** 2); transpiler_seed = np.random.randint(1024 ** 2)
                if steps == 0:
                    continue
                counts = r.run_direct(fn, circuits[steps], r.NOISY_SHOTS,
                                      simulator_seed=int(np.random.default_rng([seed, slot, 7]).integers(1024 ** 2)),
                                      transpiler_seed=transpiler_seed)
                value = fn["fermion_number"](counts, r.MID)
                results["runs"][(seed, slot)] = {"steps": steps, "epsilon": epsilon, "value": value, "counts": counts}
                print(f"  s={args.scale:g} seed {seed} slot {slot}: f = {(0.5 - value) / (0.5 - exact[steps]):.3f}", flush=True)
        pickle.dump(results, open(out / "direct.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        make_config, circuit_knitter = r.knitter()
        config = make_config()
        # Checkpoints are keyed by circuit, shots and seed but not by noise:
        # give every drift value its own checkpoint directory.
        config.checkpoint_dir = str(Path(config.checkpoint_dir) / f"drift_s{args.scale:g}")
        for steps in args.steps:
            path = out / f"knitted_step{steps}_seed{args.seed}.pkl"
            if path.exists():
                print(f"  {path}: exists"); continue
            t0 = time.time()
            res = circuit_knitter(circuits[steps], *r.CUT, r.NOISY_SHOTS, config,
                                  simulator_seed=args.seed, transpiler_seed=r.KNIT_TRANSPILER_SEED)
            value = fn["fermion_number"](res["results"], r.MID)
            pickle.dump({"exact": exact[steps], "scale": args.scale, "results": res["results"], "num_cnots": res["num_cnots"]},
                        open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  s={args.scale:g} {path.name}: f = {(0.5 - value) / (0.5 - exact[steps]):.3f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
