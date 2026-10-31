"""Two-qubit gates in the causal cone of the measured centre site (logical 6)
for the circuits actually run on the line: direct (first replica's transpiler
seed) and every cached knitted term.  Survival = prod(1 - cx error) over the
cone, using the FakeWashingtonV2 target errors of the physical pairs."""
import sys, glob, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import rerun_sec5 as r
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2
r.LAYOUT = "line"
fn = r.load_notebook_functions(); backend = FakeWashingtonV2(); line = r.line_layout()
def cone(tc, clbit):
    tc = RemoveBarriers()(tc); dag = circuit_to_dag(tc); anc = set()
    for n in dag.op_nodes():
        if n.op.name == "measure" and tc.find_bit(n.cargs[0]).index == clbit: anc |= dag.ancestors(n)
    two = [n for n in anc if getattr(n, "op", None) is not None and n.op.num_qubits == 2]
    surv = float(np.prod([1 - backend.target[n.op.name][tuple(tc.find_bit(q).index for q in n.qargs)].error for n in two] or [1.0]))
    total = sum(1 for inst in tc.data if inst.operation.num_qubits == 2)
    return total, len(two), surv
out = {}
# direct: replicate the first replica's transpiler seed chain
seed = r.NOISY_SEEDS[0]; my_seed = seed
for steps, eps in [(0, 0.0)] + [(1, e) for e in (0.2, 0.4, 0.6, 0.8)] + [(2, e) for e in (0.5, 0.6, 0.7, 0.8)]:
    np.random.seed(my_seed); my_seed = np.random.randint(1024 ** 2); np.random.randint(1024 ** 2); tseed = np.random.randint(1024 ** 2)
    if eps not in (0.2, 0.5): continue
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=tseed, initial_layout=line)
    tc = pm.run(r.circuit_for(fn, steps, eps, r.MASS))
    total, lc, surv = cone(tc, r.MID + 2)   # measured site is qubit MID+2? check below
    out[f"direct_{steps}step"] = dict(total_cx=total, cone_cx=lc, survival=surv)
    print(f"direct {steps} step: total {total} cone {lc} survival {surv:.3f}", flush=True)
# knitted terms: one cache dir per step count (identical compile for all t)
seen = set()
for d in sorted(glob.glob(".cache/circuit_knitting/*/")):
    files = sorted(glob.glob(d + "batch-*.pkl"))
    if len(files) not in (36, 1296) or len(files) in seen: continue
    terms = [tc for f in files for tc in pickle.load(open(f, "rb"))]
    phys = {tc.find_bit(q).index for tc in terms[:5] for inst in tc.data for q in inst.qubits}
    if not phys <= set(line): continue
    seen.add(len(files)); num_cx = terms[0].num_clbits - 12
    stats = np.array([cone(tc, num_cx + r.MID + 2) for tc in terms])
    steps = 1 if len(files) == 36 else 2
    out[f"knitted_{steps}step"] = dict(terms=len(terms), total_cx_mean=float(stats[:,0].mean()), cone_cx_min=int(stats[:,1].min()), cone_cx_max=int(stats[:,1].max()), cone_cx_mean=float(stats[:,1].mean()), survival_min=float(stats[:,2].min()), survival_max=float(stats[:,2].max()), survival_mean=float(stats[:,2].mean()))
    print(f"knitted {steps} step: {out[f'knitted_{steps}step']}", flush=True)
json.dump(out, open("data/lightcone_line.json", "w"), indent=1)
