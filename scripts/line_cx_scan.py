"""CNOT counts of cached transpiled knitted terms, per cache dir (line layout only)."""
import glob, json, pickle, sys, os, collections
line = set(json.load(open("data/line_layout.json"))["line"])
out = {}
for d in sorted(glob.glob(".cache/circuit_knitting/*/")):
    files = sorted(glob.glob(d + "batch-*.pkl"))
    if len(files) not in (36, 1296): continue
    cx, phys, k = [], set(), collections.Counter()
    for f in files:
        for tc in pickle.load(open(f, "rb")):
            ops = tc.count_ops(); cx.append(ops.get("cx", 0) + ops.get("ecr", 0))
            meas = 0
            for inst in tc.data:
                if inst.operation.name == "measure" and tc.find_bit(inst.clbits[0]).index < tc.num_clbits - 12: meas += 1
                for q in inst.qubits: phys.add(tc.find_bit(q).index)
            k[meas] += 1
    if not phys <= line: continue
    out[os.path.basename(d.rstrip("/"))[:12]] = dict(terms=len(cx), cx_min=min(cx), cx_max=max(cx), cx_mean=sum(cx)/len(cx), k_hist=dict(sorted(k.items())))
for key, v in out.items(): print(key, v)
json.dump(out, open(sys.argv[1], "w"), indent=1)
