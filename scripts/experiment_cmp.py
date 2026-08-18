"""Differential-testing experiment, comparison step.

Reads the canonical Coq-extracted logits (coq_out.txt), the numpy float32
logits (expbatch/numpy_logits.json), and the configuration of each sample
(expbatch/configs.json), and reports, per configuration, how far numpy's
reduction order lands from the verified IEEE-754 reference: mean/max absolute
logit divergence, and the fraction of samples whose final-position next-token
argmax disagrees with the reference.
"""
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_coq(path):
    out = {}
    cur = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ID "):
                if cur is not None:
                    out[cur] = rows
                cur = line[3:].strip()
                rows = []
            elif line.startswith("FAIL"):
                pass
            elif line:
                rows.append([float(x) for x in line.split()])
    if cur is not None:
        out[cur] = rows
    return out

def argmax(xs):
    bi, bv = 0, xs[0]
    for i, v in enumerate(xs):
        if v > bv:
            bi, bv = i, v
    return bi

def main():
    coq = parse_coq(os.path.join(ROOT, "coq_out.txt"))
    npl = json.load(open(os.path.join(ROOT, "expbatch", "numpy_logits.json")))
    cfgs = json.load(open(os.path.join(ROOT, "expbatch", "configs.json")))

    order = []
    groups = defaultdict(lambda: {"n": 0, "sum_abs": 0.0, "entries": 0,
                                  "max_abs": 0.0, "flips": 0, "nan": 0})
    for sid, crows in coq.items():
        c = cfgs[sid]
        key = (c["layers"], c["d"], c["h"], c["t"], c["vocab"])
        if key not in groups:
            order.append(key)
        d = groups[key]
        d["n"] += 1
        nrows = npl.get(sid)
        if nrows is None or len(crows) != len(nrows):
            d["nan"] += 1
            continue
        bad = False
        smax = 0.0
        for cr, nr in zip(crows, nrows):
            for cx, nx in zip(cr, nr):
                if cx != cx or nx != nx:
                    bad = True
                    continue
                a = abs(cx - nx)
                d["sum_abs"] += a
                d["entries"] += 1
                smax = max(smax, a)
        if bad:
            d["nan"] += 1
            continue
        d["max_abs"] = max(d["max_abs"], smax)
        if argmax(crows[-1]) != argmax(nrows[-1]):
            d["flips"] += 1

    lines = ["# Differential test: numpy float32 vs verified IEEE-754 reference",
             "",
             "Each row is a model configuration. The reference is the extracted",
             "verified forward; numpy runs the same elementwise math with its own",
             "reduction order. `abs-err` is over every logit of every sample.",
             "",
             "| layers | d_model | heads | seq | vocab | samples | mean abs-err | max abs-err | next-token flips | nan |",
             "|--------|---------|-------|-----|-------|---------|--------------|-------------|------------------|-----|"]
    for key in sorted(order):
        L, D, H, T, V = key
        d = groups[key]
        ok = d["n"] - d["nan"]
        mean = d["sum_abs"] / max(d["entries"], 1)
        lines.append(f"| {L} | {D} | {H} | {T} | {V} | {d['n']} | {mean:.3e} | "
                     f"{d['max_abs']:.3e} | {d['flips']}/{ok} | {d['nan']} |")
    report = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "RESULTS.md"), "w", newline="\n",
              encoding="utf-8") as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    main()
