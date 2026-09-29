"""What the residual between PyTorch and the extracted pass is made of.

Section 8 of the paper attributes llama.cpp's divergence by turning one
implementation choice off at a time. The residual against PyTorch is left
unattributed, and it has only two sources: PyTorch evaluates the elementary
functions with its own implementations rather than the series the development
composes, and it accumulates its matrix products in a different order.

This separates them on GPT-2. It runs the forward pass in numpy float32, over
the same checkpoint and the same windows:

  numpy      numpy's own reductions and numpy's own exp and GELU
  numpy+ef   numpy's own reductions, the extracted exp and GELU
  spec       the development's order: every sum a left fold from zero, every
             product rounded before it is added, the extracted exp and GELU
  spec+fma   the same order with each product fused into its sum
  tree       the rounded products, and the layer-norm and softmax terms,
             summed as a balanced binary tree
  extracted  the reference dump

The step from numpy to numpy+ef moves only the elementary functions. spec
reproduces the reference bit for bit, so spec+fma measures fused multiply-add
alone and tree the summation order alone, the two things numpy's BLAS changes
at once. PyTorch is reported beside them for scale.

  python residual_attr.py <weights.safetensors> <windows.txt> <ref dir>
                          [--torch DIR] [--limit N] [--passes numpy,numpy+ef,...]

The three orders' matrix products run in dots.c, compiled on first use with
gcc into the system temporary directory.
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import tempfile

import numpy as np
from safetensors import safe_open

f32 = np.float32


# --- the extracted elementary functions, vectorised -------------------------
# Each primitive is the binary64 operation narrowed to binary32, which is what
# the native extraction performs and what Narrow.v proves equal to the binary32
# operation. numpy float32 arithmetic on float32 operands is exactly that.

def _c(x):
    return np.float32(x)


ONE, TWO = _c(1.0), _c(2.0)
SIX, C24, C120, C720, C5040 = _c(6.0), _c(24.0), _c(120.0), _c(720.0), _c(5040.0)
LN2HI = np.float32(np.float32(355.0) / np.float32(512.0))
LN2LO = np.float32(np.float32(14581891.0) / np.float32(68719476736.0))
INVLN2 = np.float32(np.float32(12102203.0) / np.float32(8388608.0))
MAGIC = _c(12582912.0)
HI, LO = _c(88.0), _c(-88.0)
G1 = np.float32(np.float32(7978845608.0) / np.float32(10000000000.0))
G2 = np.float32(np.float32(44715.0) / np.float32(1000000.0))
HALF = np.float32(np.float32(1.0) / np.float32(2.0))


def _round_int(y):
    return np.float32(np.float32(y + MAGIC) - MAGIC)


def _pow2(k):
    """2**k for the integer-valued float32 k the reduction produces, by the
    same ladder of exact powers the definition walks."""
    a = np.abs(k).astype(f32)
    acc = np.full(a.shape, ONE, dtype=f32)
    p4 = np.float32(TWO * TWO); p16 = np.float32(p4 * p4)
    p256 = np.float32(p16 * p16); p65536 = np.float32(p256 * p256)
    p2_32 = np.float32(p65536 * p65536); p2_64 = np.float32(p2_32 * p2_32)
    for c, p in ((_c(64.0), p2_64), (_c(32.0), p2_32), (_c(16.0), p65536),
                 (_c(8.0), p256), (_c(4.0), p16), (TWO, p4), (ONE, TWO)):
        m = c <= a
        a = np.where(m, np.float32(a - c), a).astype(f32)
        acc = np.where(m, np.float32(acc * p), acc).astype(f32)
    return np.where(k < 0, np.float32(ONE / acc), acc).astype(f32)


def f32_exp(x):
    x = x.astype(f32)
    xc = np.where(HI < x, HI, np.where(x < LO, LO, x)).astype(f32)
    k = _round_int(np.float32(xc * INVLN2))
    r = np.float32(np.float32(xc - np.float32(k * LN2HI)) + np.float32(k * LN2LO))
    r2 = np.float32(r * r); r3 = np.float32(r2 * r); r4 = np.float32(r3 * r)
    r5 = np.float32(r4 * r); r6 = np.float32(r5 * r); r7 = np.float32(r6 * r)
    s = np.float32(ONE + np.float32(r + np.float32(np.float32(r2 / TWO)
        + np.float32(np.float32(r3 / SIX) + np.float32(np.float32(r4 / C24)
        + np.float32(np.float32(r5 / C120) + np.float32(np.float32(r6 / C720)
        + np.float32(r7 / C5040))))))))
    return np.float32(s * _pow2(k))


def f32_sigmoid(x):
    return np.float32(ONE / np.float32(ONE + f32_exp(np.float32(-x))))


def f32_tanh(x):
    return np.float32(np.float32(TWO * f32_sigmoid(np.float32(TWO * x))) - ONE)


def f32_gelu(x):
    x = x.astype(f32)
    x3 = np.float32(x * np.float32(x * x))
    inner = np.float32(G1 * np.float32(x + np.float32(G2 * x3)))
    return np.float32(np.float32(np.float32(HALF * x)) * np.float32(ONE + f32_tanh(inner)))


def np_gelu(x):
    t = np.tanh(np.sqrt(np.float64(2.0) / np.pi)
                * (np.float64(x) + 0.044715 * np.float64(x) ** 3))
    return (0.5 * np.float64(x) * (1.0 + t)).astype(f32)


# --- the GPT-2 forward in numpy float32 -------------------------------------

def layer_norm(x, w, b, eps=np.float32(1e-5)):
    mu = x.mean(axis=-1, keepdims=True, dtype=f32)
    c = (x - mu).astype(f32)
    var = (c * c).mean(axis=-1, keepdims=True, dtype=f32)
    return ((c / np.sqrt(var + eps)).astype(f32) * w + b).astype(f32)


def softmax(x, expf):
    m = x.max(axis=-1, keepdims=True)
    e = expf((x - m).astype(f32))
    return (e / e.sum(axis=-1, keepdims=True, dtype=f32)).astype(f32)


def forward(W, ids, nh, expf, geluf):
    d = W["wte.weight"].shape[1]
    hd = d // nh
    T = len(ids)
    h = (W["wte.weight"][ids] + W["wpe.weight"][:T]).astype(f32)
    mask = np.triu(np.full((T, T), np.float32(-1e10), dtype=f32), 1)
    nl = 1 + max(int(k.split(".")[1]) for k in W if k.startswith("h."))
    for i in range(nl):
        p = f"h.{i}."
        a = layer_norm(h, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = (a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).astype(f32)
        q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
        out = np.empty((T, d), dtype=f32)
        for hh in range(nh):
            qs = q[:, hh * hd:(hh + 1) * hd]
            ks = k[:, hh * hd:(hh + 1) * hd]
            vs = v[:, hh * hd:(hh + 1) * hd]
            s = ((qs @ ks.T).astype(f32) / np.float32(np.sqrt(hd))).astype(f32) + mask
            out[:, hh * hd:(hh + 1) * hd] = (softmax(s, expf) @ vs).astype(f32)
        h = (h + (out @ W[p + "attn.c_proj.weight"]
                  + W[p + "attn.c_proj.bias"]).astype(f32)).astype(f32)
        a = layer_norm(h, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        u = (a @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]).astype(f32)
        u = geluf(u)
        h = (h + (u @ W[p + "mlp.c_proj.weight"]
                  + W[p + "mlp.c_proj.bias"]).astype(f32)).astype(f32)
    h = layer_norm(h, W["ln_f.weight"], W["ln_f.bias"])
    return (h @ W["wte.weight"].T).astype(f32)


# --- the development's order, and two changes to it ---------------------------

_DOTS = None


def _dots_lib():
    global _DOTS
    if _DOTS is None:
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dots.c")
        lib = os.path.join(tempfile.gettempdir(),
                           "p2w_dots" + (".dll" if os.name == "nt" else ".so"))
        if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
            subprocess.run(["gcc", "-O2", "-mfma", "-ffp-contract=off", "-fopenmp",
                            "-shared", "-o", lib, src], check=True)
        if os.name == "nt":
            os.add_dll_directory(os.path.dirname(shutil.which("gcc")))
        _DOTS = ctypes.CDLL(lib)
        for name in ("dots_seq", "dots_fma", "dots_tree"):
            getattr(_DOTS, name).argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    return _DOTS


def tree_sum(x, axis=-1):
    """Sum along an axis in binary32 as a balanced binary tree."""
    x = np.moveaxis(x, axis, 0)
    while x.shape[0] > 1:
        n = x.shape[0] - x.shape[0] % 2
        pair = (x[0:n:2] + x[1:n:2]).astype(f32)
        x = np.concatenate([pair, x[n:]]) if n < x.shape[0] else pair
    return x[0]


def total(x, order):
    """Sum over the last axis, keeping it: a left fold from zero, or a tree."""
    if order == "tree":
        return tree_sum(x)[..., None]
    acc = np.zeros(x.shape[:-1], dtype=f32)
    for k in range(x.shape[-1]):
        acc = (acc + x[..., k]).astype(f32)
    return acc[..., None]


def dots(X, W, order):
    """Y[i, j] = sum_k X[i, k] W[k, j] in binary32: a left fold from zero of the
    rounded products (seq), of fused products (fma), or a tree of the rounded
    products (tree)."""
    X = np.ascontiguousarray(X, dtype=f32)
    W = np.ascontiguousarray(W, dtype=f32)
    T, n = X.shape
    m = W.shape[1]
    out = np.empty((T, m), dtype=f32)
    getattr(_dots_lib(), "dots_" + order)(X.ctypes.data, W.ctypes.data,
                                           out.ctypes.data, T, n, m)
    return out


def spec_forward(W, WTE_T, ids, nh, order):
    """The development's GPT-2 forward pass, with its sums in the given order."""
    d = WTE_T.shape[0]
    hd = d // nh
    T = len(ids)
    eps = np.float32(ONE / np.float32(100000.0))
    scale = np.float32(ONE / np.sqrt(np.float32(hd)))
    causal = np.tril(np.ones((T, T), dtype=bool))

    def ln(x, p):
        n = np.float32(x.shape[-1])
        mu = (total(x, order) / n).astype(f32)
        c = (x - mu).astype(f32)
        var = (total((c * c).astype(f32), order) / n).astype(f32)
        den = np.sqrt((var + eps).astype(f32)).astype(f32)
        return ((W[p + ".weight"] * (c / den).astype(f32)).astype(f32)
                + W[p + ".bias"]).astype(f32)

    def lin(x, p):
        return (dots(x, W[p + ".weight"], order) + W[p + ".bias"]).astype(f32)

    h = (W["wte.weight"][ids] + W["wpe.weight"][:T]).astype(f32)
    nl = 1 + max(int(k.split(".")[1]) for k in W if k.startswith("h."))
    for i in range(nl):
        p = f"h.{i}."
        qkv = lin(ln(h, p + "ln_1"), p + "attn.c_attn")
        out = np.empty((T, d), dtype=f32)
        for hh in range(nh):
            sl = slice(hh * hd, (hh + 1) * hd)
            q, k, v = qkv[:, :d][:, sl], qkv[:, d:2 * d][:, sl], qkv[:, 2 * d:][:, sl]
            s = (dots(q, np.ascontiguousarray(k.T), order) * scale).astype(f32)
            m = np.where(causal, s, -np.inf).max(axis=1, keepdims=True).astype(f32)
            e = np.where(causal, f32_exp((s - m).astype(f32)), f32(0)).astype(f32)
            w = (e / total(e, order)).astype(f32)
            out[:, sl] = dots(w, np.ascontiguousarray(v), order)
        h = (h + lin(out, p + "attn.c_proj")).astype(f32)
        u = f32_gelu(lin(ln(h, p + "ln_2"), p + "mlp.c_fc"))
        h = (h + lin(u, p + "mlp.c_proj")).astype(f32)
    return dots(ln(h, "ln_f"), WTE_T, order)


def stats(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    flips = int((a.argmax(axis=1) != b.argmax(axis=1)).sum())
    return d.max(), float(d.sum()), int(d.size), flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("windows")
    ap.add_argument("ref")
    ap.add_argument("--torch", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--passes", default="numpy,numpy+ef,spec,spec+fma,tree")
    args = ap.parse_args()
    passes = args.passes.split(",")

    with safe_open(args.weights, framework="np") as f:
        W = {k: f.get_tensor(k).astype(f32) for k in f.keys()}
    WTE_T = np.ascontiguousarray(W["wte.weight"].T)
    lines = [l.strip() for l in open(args.windows) if l.strip()]
    vocab = W["wte.weight"].shape[0]
    make = {
        "numpy": lambda ids: forward(W, ids, args.heads,
                                     lambda z: np.exp(z.astype(np.float64)).astype(f32), np_gelu),
        "numpy+ef": lambda ids: forward(W, ids, args.heads, f32_exp, f32_gelu),
        "spec": lambda ids: spec_forward(W, WTE_T, ids, args.heads, "seq"),
        "spec+fma": lambda ids: spec_forward(W, WTE_T, ids, args.heads, "fma"),
        "tree": lambda ids: spec_forward(W, WTE_T, ids, args.heads, "tree"),
    }

    acc = {}
    used = 0
    for w, line in enumerate(lines[:args.limit]):
        rp = os.path.join(args.ref, f"{w}.f32")
        if not os.path.exists(rp):
            continue
        ids = [int(t) for t in line.split(",")]
        ref = np.fromfile(rp, dtype="<f4").reshape(-1, vocab)
        if ref.shape[0] != len(ids):
            continue
        used += 1
        runs = {name: make[name](ids) for name in passes}
        if args.torch:
            tp = os.path.join(args.torch, f"{w}.f32")
            if os.path.exists(tp):
                runs["PyTorch"] = np.fromfile(tp, dtype="<f4").reshape(-1, vocab)
        for name, got in runs.items():
            mx, sm, n, fl = stats(got, ref)
            a = acc.setdefault(name, [0.0, 0.0, 0, 0])
            a[0] = max(a[0], mx); a[1] += sm; a[2] += n; a[3] += fl

    print(f"{used} windows of {len(lines[0].split(','))} tokens, against the "
          f"extracted reference\n")
    print(f"{'source':10s} {'max abs':>12s} {'mean abs':>12s} {'top-1 differs':>14s}")
    for name in passes + ["PyTorch"]:
        if name not in acc:
            continue
        mx, sm, n, fl = acc[name]
        print(f"{name:10s} {mx:12.3e} {sm / n:12.3e} {fl:14d}")


if __name__ == "__main__":
    main()
