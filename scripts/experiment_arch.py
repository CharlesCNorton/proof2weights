"""Differential test of the Llama and Qwen3.5 paths.

Builds small random models of each architecture, writes their weights as raw
binary32 bit patterns so the reference and the numpy mirror see bit-identical
values, runs both, and reports how far numpy's reduction order lands from the
verified reference.

The reference is the inductive extraction (runners/llama_ref, runners/qwen_ref),
where binary32 is Flocq's binary_float and no floating-point boundary is
trusted. The mirror is scripts/arch_ref.py, which performs the same elementwise
math with natural numpy reductions.

  python experiment_arch.py [outdir]

Set P2W_BIN to the directory holding the built llama_ref and qwen_ref binaries.
"""
import os, sys, json, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_ref import f32, llama_forward, qwen_forward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("P2W_BIN", ROOT)
K = 16

LLAMA = [
    {"tag": "", "sid": 0, "d": 8, "nh": 2, "nkv": 1, "ff": 32, "vocab": 16, "nl": 1, "seq": 8},
    {"tag": "", "sid": 0, "d": 8, "nh": 2, "nkv": 1, "ff": 32, "vocab": 16, "nl": 2, "seq": 8},
    {"tag": "", "sid": 0, "d": 8, "nh": 2, "nkv": 1, "ff": 32, "vocab": 16, "nl": 4, "seq": 8},
    {"tag": "d16", "sid": 1, "d": 16, "nh": 4, "nkv": 2, "ff": 64, "vocab": 16, "nl": 2, "seq": 8},
    {"tag": "d32", "sid": 2, "d": 32, "nh": 8, "nkv": 4, "ff": 128, "vocab": 16, "nl": 2, "seq": 8},
    {"tag": "t16", "sid": 3, "d": 8, "nh": 2, "nkv": 1, "ff": 32, "vocab": 16, "nl": 2, "seq": 16},
    {"tag": "v64", "sid": 4, "d": 8, "nh": 2, "nkv": 1, "ff": 32, "vocab": 64, "nl": 2, "seq": 8},
]

QWEN = [
    {"tag": "", "sid": 0, "d": 8, "nh": 2, "nkv": 1, "hd": 4, "rd": 2, "ff": 32,
     "vocab": 16, "nl": 1, "lnh": 1, "lhd": 4, "ck": 2, "seq": 8, "kinds": ["delta"]},
    {"tag": "attn", "sid": 1, "d": 8, "nh": 2, "nkv": 1, "hd": 4, "rd": 2, "ff": 32,
     "vocab": 16, "nl": 1, "lnh": 1, "lhd": 4, "ck": 2, "seq": 8, "kinds": ["attn"]},
    {"tag": "mixed", "sid": 2, "d": 8, "nh": 2, "nkv": 1, "hd": 4, "rd": 2, "ff": 32,
     "vocab": 16, "nl": 4, "lnh": 1, "lhd": 4, "ck": 2, "seq": 8,
     "kinds": ["delta", "delta", "delta", "attn"]},
    {"tag": "t16", "sid": 3, "d": 8, "nh": 2, "nkv": 1, "hd": 4, "rd": 2, "ff": 32,
     "vocab": 16, "nl": 1, "lnh": 1, "lhd": 4, "ck": 2, "seq": 16, "kinds": ["delta"]},
    {"tag": "d16", "sid": 4, "d": 16, "nh": 2, "nkv": 1, "hd": 8, "rd": 4, "ff": 64,
     "vocab": 16, "nl": 2, "lnh": 2, "lhd": 4, "ck": 3, "seq": 8,
     "kinds": ["delta", "attn"]},
    {"tag": "d32", "sid": 5, "d": 32, "nh": 4, "nkv": 2, "hd": 8, "rd": 4, "ff": 128,
     "vocab": 16, "nl": 2, "lnh": 4, "lhd": 4, "ck": 3, "seq": 8,
     "kinds": ["delta", "attn"]},
]


def bits(a):
    return np.asarray(a, dtype=f32).view(np.uint32).reshape(-1)


def dump(tensors, path):
    with open(path, "w", newline="\n") as fh:
        for name, arr in tensors.items():
            a = np.atleast_2d(np.asarray(arr, dtype=f32))
            r, c = a.shape
            fh.write(f"{name} {r} {c} " + " ".join(str(int(x)) for x in bits(a)) + "\n")


def llama_weights(rng, c):
    d, nh, nkv, ff, vocab = c["d"], c["nh"], c["nkv"], c["ff"], c["vocab"]
    hd = d // nh
    t = lambda *s: (rng.standard_normal(s) * 0.2).astype(f32)
    W = {"emb": t(vocab, d), "norm": t(d),
         "cos": t(c["seq"], hd), "sin": t(c["seq"], hd)}
    for i in range(c["nl"]):
        p = f"L{i}."
        W[p + "ln1"] = t(d); W[p + "ln2"] = t(d)
        W[p + "q"] = t(nh * hd, d); W[p + "k"] = t(nkv * hd, d)
        W[p + "v"] = t(nkv * hd, d); W[p + "o"] = t(d, nh * hd)
        W[p + "gate"] = t(ff, d); W[p + "up"] = t(ff, d); W[p + "down"] = t(d, ff)
    return W, hd


def qwen_weights(rng, c):
    d, nh, nkv, hd, ff, vocab = c["d"], c["nh"], c["nkv"], c["hd"], c["ff"], c["vocab"]
    lnh, lhd, ck = c["lnh"], c["lhd"], c["ck"]
    ldim = lnh * lhd
    t = lambda *s: (rng.standard_normal(s) * 0.2).astype(f32)
    W = {"emb": t(vocab, d), "norm": t(d),
         "cos": t(c["seq"], hd), "sin": t(c["seq"], hd)}
    for i in range(c["nl"]):
        p = f"L{i}."
        W[p + "ln1"] = t(d); W[p + "ln2"] = t(d)
        W[p + "gate"] = t(ff, d); W[p + "up"] = t(ff, d); W[p + "down"] = t(d, ff)
        if c["kinds"][i] == "attn":
            W[p + "q"] = t(nh * hd * 2, d); W[p + "k"] = t(nkv * hd, d)
            W[p + "v"] = t(nkv * hd, d); W[p + "o"] = t(d, nh * hd)
            W[p + "q_norm"] = t(hd); W[p + "k_norm"] = t(hd)
        else:
            W[p + "in_qkv"] = t(3 * ldim, d); W[p + "in_z"] = t(ldim, d)
            W[p + "in_a"] = t(lnh, d); W[p + "in_b"] = t(lnh, d)
            W[p + "conv_w"] = t(3 * ldim, ck)
            W[p + "a_log"] = t(lnh); W[p + "dt_bias"] = t(lnh)
            W[p + "norm_w"] = t(lhd); W[p + "out"] = t(d, ldim)
    return W


def run(binary, args):
    out = subprocess.run([os.path.join(BIN, binary)] + args,
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None
    rows = [[float(x) for x in line.split()]
            for line in out.stdout.strip().split("\n") if line.strip()]
    return rows


def compare(ref, mir):
    if ref is None or len(ref) != len(mir):
        return None
    tot, n, mx = 0.0, 0, 0.0
    for a, b in zip(ref, mir):
        for x, y in zip(a, b):
            if x != x or y != y:
                return None
            e = abs(x - float(y))
            tot += e; n += 1; mx = max(mx, e)
    flip = int(np.argmax(ref[-1]) != np.argmax(mir[-1]))
    return tot, n, mx, flip


def sweep(name, configs, outdir):
    rows = []
    for c in configs:
        agg = {"n": 0, "sum": 0.0, "cnt": 0, "max": 0.0, "flips": 0, "bad": 0}
        for k in range(K):
            rng = np.random.default_rng(hash((c["sid"], c["nl"], c["seq"], k)) & 0xffffffff)
            ids = [int(x) for x in rng.integers(0, c["vocab"], size=c["seq"])]
            path = os.path.join(outdir, f"{name}_{c['tag']}_{c['nl']}_{k}.w")
            if name == "llama":
                W, hd = llama_weights(rng, c)
                dump(W, path)
                cfg = dict(c, hd=hd)
                mir = llama_forward(W, ids, cfg)
                ref = run("llama_ref", [path, str(c["d"]), str(c["nh"]), str(c["nkv"]),
                                        str(hd), str(c["ff"]), str(c["vocab"]),
                                        str(c["nl"]), ",".join(map(str, ids))])
            else:
                W = qwen_weights(rng, c)
                dump(W, path)
                mir = qwen_forward(W, ids, c)
                ref = run("qwen_ref", [path, str(c["d"]), str(c["nh"]), str(c["nkv"]),
                                       str(c["hd"]), str(c["rd"]), str(c["ff"]),
                                       str(c["vocab"]), str(c["nl"]), str(c["lnh"]),
                                       str(c["lhd"]), str(c["ck"]),
                                       ",".join(c["kinds"]), ",".join(map(str, ids))])
            agg["n"] += 1
            r = compare(ref, mir.tolist())
            if r is None:
                agg["bad"] += 1
                continue
            tot, n, mx, flip = r
            agg["sum"] += tot; agg["cnt"] += n
            agg["max"] = max(agg["max"], mx); agg["flips"] += flip
        mean = agg["sum"] / max(agg["cnt"], 1)
        ok = agg["n"] - agg["bad"]
        if name == "llama":
            shape = f"| {c['nl']} | {c['d']} | {c['nh']} | {c['nkv']} | {c['seq']} | {c['vocab']}"
        else:
            shape = (f"| {c['nl']} | {'/'.join(c['kinds'])} | {c['d']} | {c['lnh']}x{c['lhd']}"
                     f" | {c['ck']} | {c['seq']}")
        rows.append(f"{shape} | {agg['n']} | {mean:.3e} | {agg['max']:.3e} | "
                    f"{agg['flips']}/{ok} | {agg['bad']} |")
    return rows


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "archbatch")
    os.makedirs(outdir, exist_ok=True)
    lrows = sweep("llama", LLAMA, outdir)
    qrows = sweep("qwen", QWEN, outdir)
    out = ["", "## Llama path", "",
           "Reference: the inductive extraction of `f32_llama_forward`.",
           "",
           "| layers | d_model | heads | kv heads | seq | vocab | samples | mean abs-err | max abs-err | next-token flips | nan |",
           "|--------|---------|-------|----------|-----|-------|---------|--------------|-------------|------------------|-----|"]
    out += lrows
    out += ["", "## Qwen3.5 path", "",
            "Reference: the inductive extraction of `f32_qwen_forward`.",
            "",
            "| layers | kinds | d_model | deltanet | conv k | seq | samples | mean abs-err | max abs-err | next-token flips | nan |",
            "|--------|-------|---------|----------|--------|-----|---------|--------------|-------------|------------------|-----|"]
    out += qrows
    text = "\n".join(out) + "\n"
    print(text)
    # experiment_cmp.py writes the GPT-2 section of RESULTS.md; these append to it.
    with open(os.path.join(ROOT, "RESULTS.md"), "a", newline="\n", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
