"""Per-input verified bounds against the actual error, over the GPT-2 sweep.

Reads bound_out/<id>.txt (runners/gpt2_bound_ref: one row per position, each
entry value:bound) and evaluates, for the same weights and tokens, the network
the bound is stated against: exact real arithmetic, with every exponential the
true exponential of its argument saturated to [-88, 88] and every program
constant (the GELU coefficients, the layer-norm epsilon) the binary32 value
the code holds. The evaluation runs in float64; --mp re-runs one sample per
configuration in mpmath at 50 digits and reports the largest distance between
the two evaluations, which bounds what the float64 evaluation itself
contributes to the measured error.

For each configuration it reports how many logits carry a finite bound, the
largest finite bound, the largest actual error |binary32 - reference| among
those logits, the median ratio of bound to actual error, and the number of
logits whose actual error exceeds its bound.

  python bound_cmp.py [--mp] [--write]
"""
import argparse
import json
import math
import os
import statistics
from collections import defaultdict

import numpy as np
from safetensors.numpy import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f32 = np.float32

# The program constants, as the binary32 values the code holds.
EPS = float(f32(1) / f32(100000))
C1 = float(f32(7978845608) / f32(10000000000))
C2 = float(f32(44715) / f32(1000000))


class F64:
    exp = staticmethod(np.exp)
    sqrt = staticmethod(np.sqrt)

    @staticmethod
    def arr(a):
        return np.asarray(a, dtype=np.float64)


class MP:
    def __init__(self):
        import mpmath
        mpmath.mp.dps = 50
        self.mp = mpmath
        self.exp = np.vectorize(mpmath.exp, otypes=[object])
        self.sqrt = np.vectorize(mpmath.sqrt, otypes=[object])

    def arr(self, a):
        a = np.asarray(a, dtype=np.float64)
        return np.vectorize(lambda v: self.mp.mpf(float(v)), otypes=[object])(a)


def sat(x):
    return np.minimum(np.maximum(x, -88), 88)


def reference(B, w, n_layer, toks, d, h):
    hd = d // h
    wte = B.arr(w["wte.weight"]); wpe = B.arr(w["wpe.weight"])
    T = len(toks)
    eps = B.arr(EPS)

    def layernorm(v, g, b):
        n = v.shape[-1]
        mean = v.sum(axis=-1, keepdims=True) / n
        u = v - mean
        var = (u * u).sum(axis=-1, keepdims=True) / n
        return B.arr(g) * (u / B.sqrt(var + eps)) + B.arr(b)

    def sigmoid(z):
        return 1 / (1 + B.exp(sat(-z)))

    def gelu(x):
        inner = C1 * (x + C2 * (x * (x * x)))
        tanh = 2 * sigmoid(2 * inner) - 1
        return (0.5 * x) * (1 + tanh)

    hidden = wte[np.array(toks)] + wpe[:T]
    scale = 1 / B.sqrt(B.arr(float(hd)))
    for i in range(n_layer):
        p = f"h.{i}."
        ln1 = layernorm(hidden, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        qkv = ln1 @ B.arr(w[p + "attn.c_attn.weight"]) + B.arr(w[p + "attn.c_attn.bias"])
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        attn = np.empty((T, d), dtype=q.dtype)
        for hh in range(h):
            sl = slice(hh * hd, (hh + 1) * hd)
            for t in range(T):
                scores = (k[:t + 1, sl] @ q[t, sl]) * scale
                e = B.exp(sat(scores - scores.max()))
                attn[t, sl] = (e / e.sum()) @ v[:t + 1, sl]
        proj = attn @ B.arr(w[p + "attn.c_proj.weight"]) + B.arr(w[p + "attn.c_proj.bias"])
        hidden = hidden + proj
        ln2 = layernorm(hidden, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        fc = gelu(ln2 @ B.arr(w[p + "mlp.c_fc.weight"]) + B.arr(w[p + "mlp.c_fc.bias"]))
        hidden = hidden + (fc @ B.arr(w[p + "mlp.c_proj.weight"]) + B.arr(w[p + "mlp.c_proj.bias"]))
    out = layernorm(hidden, w["ln_f.weight"], w["ln_f.bias"])
    return out @ wte.T


def parse(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("values") or line.startswith("FAIL"):
                continue
            rows.append([tuple(float(x) for x in tok.split(":")) for tok in line.split()])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    manifest = [l.split() for l in open(os.path.join(ROOT, "expbatch", "manifest.txt")) if l.strip()]
    groups = defaultdict(lambda: {"n": 0, "logits": 0, "finite": 0, "maxb": 0.0,
                                  "maxe": 0.0, "ratios": [], "viol": 0, "mp": 0.0})
    order = []
    mp_done = set()
    for sid, nl, toks, d, h, ff, vocab, npos in manifest:
        path = os.path.join(ROOT, "bound_out", sid + ".txt")
        if not os.path.exists(path):
            continue
        nl, d, h, vocab = int(nl), int(d), int(h), int(vocab)
        toks = [int(t) for t in toks.split(",")]
        key = (nl, d, h, len(toks), vocab)
        if key not in groups:
            order.append(key)
        g = groups[key]
        rows = parse(path)
        w = load_file(os.path.join(ROOT, "expbatch", sid + ".safetensors"))
        ref = reference(F64, w, nl, toks, d, h)
        if args.mp and key not in mp_done:
            mref = reference(MP(), w, nl, toks, d, h)
            g["mp"] = max(float(abs(float(a) - float(b))) for a, b in zip(mref.ravel(), ref.ravel()))
            mp_done.add(key)
        g["n"] += 1
        for row, rrow in zip(rows, ref):
            for (val, bnd), r in zip(row, rrow):
                g["logits"] += 1
                if not math.isfinite(bnd):
                    continue
                g["finite"] += 1
                e = abs(val - float(r))
                g["maxb"] = max(g["maxb"], bnd)
                g["maxe"] = max(g["maxe"], e)
                if e > bnd:
                    g["viol"] += 1
                if e > 0:
                    g["ratios"].append(bnd / e)
    lines = ["", "## Verified bound against the actual error (GPT-2 path)", "",
             "Bounds from `runners/gpt2_bound_ref`, the inductive extraction of the annotated",
             "forward pass `gpt2_logits_bounded` is stated about. The actual error is the",
             "distance from the binary32 logit to a float64 evaluation of the reference",
             "network: exact arithmetic, true exponential of the saturated argument, program",
             "constants as stored.", "",
             "| layers | d_model | heads | seq | vocab | samples | logits bounded | max bound | max actual error | median bound/error | exceeded |",
             "|--------|---------|-------|-----|-------|---------|----------------|-----------|------------------|--------------------|----------|"]
    for key in sorted(order):
        L, D, H, T, V = key
        g = groups[key]
        med = statistics.median(g["ratios"]) if g["ratios"] else float("nan")
        maxb = f"{g['maxb']:.3e}" if g["finite"] else "-"
        maxe = f"{g['maxe']:.3e}" if g["finite"] else "-"
        medr = f"{med:.3g}" if g["ratios"] else "-"
        lines.append(f"| {L} | {D} | {H} | {T} | {V} | {g['n']} | {g['finite']}/{g['logits']} | "
                     f"{maxb} | {maxe} | {medr} | {g['viol']} |")
        if args.mp:
            lines[-1] += f" mp {g['mp']:.2e}"
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        with open(os.path.join(ROOT, "RESULTS.md"), "a", newline="\n", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
