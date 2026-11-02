#!/usr/bin/env python3
"""Un-knitted line runs at scaled noise, for zero-noise extrapolation: the
--direct-sim aer physics replicas of rerun_sec5 (same seeds, same transpiled
circuits) with every Washington error rate scaled by --scale
(calibration_drift.scaled_washington).

  python scripts/zne_direct.py --scale 1.5
writes data/zne_direct/noisy_no_knit_trotter_evolution_line_aer_s<scale>.pkl
"""
import argparse, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import qiskit_ibm_runtime.fake_provider as fake_provider
import calibration_drift as cd

parser = argparse.ArgumentParser()
parser.add_argument("--scale", type=float, required=True)
args = parser.parse_args()
fake_provider.FakeWashingtonV2 = cd.scaled_washington(args.scale)    # used by rerun_sec5.run_direct
import rerun_sec5 as r
r.LAYOUT = "line"; r.DIRECT_SIM = "aer"
fn = r.load_notebook_functions()
r.DATA = r.DATA / "zne_direct"; r.DATA.mkdir(exist_ok=True)
r.SUFFIX = f"_line_aer_s{args.scale:g}"
r.stage_noisy(fn, force=False)
