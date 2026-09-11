"""Tiny GPT-2 reference for the float-pipeline equivalence check.

Builds a tiny model with deterministic weights, saves a real .safetensors
with GPT-2 tensor names, and computes a reference forward using the exact
operations, in the exact order, the Rocq float pipeline defines: the scaled
exponential, the tanh form of GELU, population-variance layernorm with
eps=1e-5, causal attention where each row attends over its prefix with a
max-shifted softmax, and tied-embedding logits. The extracted OCaml loads the
same file and must agree.
"""
import os
import numpy as np
from safetensors.numpy import save_file

# Fixtures land in the repository root, whatever the working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

f32 = np.float32

D = 4          # n_embd
H = 2          # n_head
HD = D // H    # head_dim
FF = 8         # n_inner
VOCAB = 5
NPOS = 4
EPS = f32(1) / f32(100000)

EXP_HI = f32(88)
EXP_LO = f32(-88)
LN2_HI = f32(355) / f32(512)
LN2_LO = f32(14581891) / f32(68719476736)
INV_LN2 = f32(12102203) / f32(8388608)
MAGIC = f32(12582912)
GELU_C1 = f32(7978845608) / f32(10000000000)
GELU_C2 = f32(44715) / f32(1000000)
HALF = f32(1) / f32(2)

def gen(shape):
    n = int(np.prod(shape))
    vals = [f32(((((k * 7 + 3) % 13) - 6)) / 25.0) for k in range(n)]
    return np.array(vals, dtype=f32).reshape(shape)

def my_exp(x):
    # exp(x) = 2^k exp(r) with r = x - k log 2, matching f32_exp_approx. The
    # right-associated accumulation mirrors the f32_plus nesting exactly.
    x = f32(x)
    xc = EXP_HI if EXP_HI < x else (EXP_LO if x < EXP_LO else x)
    k = f32(f32(f32(xc * INV_LN2) + MAGIC) - MAGIC)
    r = f32(f32(xc - f32(k * LN2_HI)) + f32(k * LN2_LO))
    r2 = f32(r * r); r3 = f32(r2 * r); r4 = f32(r3 * r)
    r5 = f32(r4 * r); r6 = f32(r5 * r); r7 = f32(r6 * r)
    s = f32(f32(r6 / f32(720)) + f32(r7 / f32(5040)))
    s = f32(f32(r5 / f32(120)) + s)
    s = f32(f32(r4 / f32(24)) + s)
    s = f32(f32(r3 / f32(6)) + s)
    s = f32(f32(r2 / f32(2)) + s)
    s = f32(r + s)
    s = f32(f32(1) + s)
    return f32(s * np.ldexp(f32(1), int(k)))

def my_sigmoid(x):
    return f32(f32(1.0) / f32(f32(1.0) + my_exp(f32(-x))))

def my_tanh(y):
    return f32(f32(f32(2) * my_sigmoid(f32(f32(2) * y))) - f32(1))

def my_gelu(x):
    x3 = f32(x * f32(x * x))
    inner = f32(GELU_C1 * f32(x + f32(GELU_C2 * x3)))
    return f32(f32(HALF * x) * f32(f32(1) + my_tanh(inner)))

def dot(a, b):
    acc = f32(0.0)
    for i in range(len(a)):
        acc = f32(acc + f32(a[i] * b[i]))
    return acc

def layernorm(v, g, b):
    n = f32(len(v))
    mean = f32(sum((f32(x) for x in v), f32(0.0)) / n)
    var = f32(sum((f32(f32(x - mean) * f32(x - mean)) for x in v), f32(0.0)) / n)
    denom = f32(np.sqrt(f32(var + EPS)))
    return [f32(f32(g[i] * f32(f32(v[i] - mean) / denom)) + b[i]) for i in range(len(v))]

def softmax(v):
    m = v[0]
    for x in v:
        if m < x:
            m = x
    exps = [my_exp(f32(x - m)) for x in v]
    s = f32(0.0)
    for e in exps:
        s = f32(s + e)
    return [f32(e / s) for e in exps]

def linear(x, W, b):
    # x: [in], W: [in, out], b: [out] -> [out]
    out_dim = W.shape[1]
    return [f32(dot([x[i] for i in range(len(x))], [W[i][j] for i in range(W.shape[0])]) + b[j]) for j in range(out_dim)]

def main():
    wte = gen((VOCAB, D)); wpe = gen((NPOS, D))
    ln1_w = gen((D,)); ln1_b = gen((D,))
    c_attn_w = gen((D, 3 * D)); c_attn_b = gen((3 * D,))
    c_proj_w = gen((D, D)); c_proj_b = gen((D,))
    ln2_w = gen((D,)); ln2_b = gen((D,))
    fc_w = gen((D, FF)); fc_b = gen((FF,))
    mlp_proj_w = gen((FF, D)); mlp_proj_b = gen((D,))
    lnf_w = gen((D,)); lnf_b = gen((D,))

    tensors = {
        "wte.weight": wte, "wpe.weight": wpe,
        "h.0.ln_1.weight": ln1_w, "h.0.ln_1.bias": ln1_b,
        "h.0.attn.c_attn.weight": c_attn_w, "h.0.attn.c_attn.bias": c_attn_b,
        "h.0.attn.c_proj.weight": c_proj_w, "h.0.attn.c_proj.bias": c_proj_b,
        "h.0.ln_2.weight": ln2_w, "h.0.ln_2.bias": ln2_b,
        "h.0.mlp.c_fc.weight": fc_w, "h.0.mlp.c_fc.bias": fc_b,
        "h.0.mlp.c_proj.weight": mlp_proj_w, "h.0.mlp.c_proj.bias": mlp_proj_b,
        "ln_f.weight": lnf_w, "ln_f.bias": lnf_b,
    }
    save_file(tensors, os.path.join(ROOT, "tiny_gpt2.safetensors"))

    toks = [0, 1, 2]
    # embeddings
    hidden = [[f32(wte[t][j] + wpe[p][j]) for j in range(D)] for p, t in enumerate(toks)]
    S = len(toks)
    # --- block 0, pre-norm ---
    ln1 = [layernorm(hidden[i], ln1_w, ln1_b) for i in range(S)]
    qkv = [linear(ln1[i], c_attn_w, c_attn_b) for i in range(S)]
    q = [row[0:D] for row in qkv]; k = [row[D:2*D] for row in qkv]; v = [row[2*D:3*D] for row in qkv]
    attn_out = [[f32(0.0)] * D for _ in range(S)]
    inv = f32(f32(1.0) / f32(np.sqrt(f32(HD))))
    for h in range(H):
        sl = slice(h*HD, (h+1)*HD)
        qh = [row[sl] for row in q]; kh = [row[sl] for row in k]; vh = [row[sl] for row in v]
        for i in range(S):
            # row i attends over positions 0..i
            scores = [f32(dot(qh[i], kh[j]) * inv) for j in range(i + 1)]
            w = softmax(scores)
            for c in range(HD):
                acc = f32(0.0)
                for j in range(i + 1):
                    acc = f32(acc + f32(w[j] * vh[j][c]))
                attn_out[i][h*HD + c] = acc
    proj = [linear(attn_out[i], c_proj_w, c_proj_b) for i in range(S)]
    hidden2 = [[f32(hidden[i][j] + proj[i][j]) for j in range(D)] for i in range(S)]
    ln2 = [layernorm(hidden2[i], ln2_w, ln2_b) for i in range(S)]
    fc = [[my_gelu(x) for x in linear(ln2[i], fc_w, fc_b)] for i in range(S)]
    mlp = [linear(fc[i], mlp_proj_w, mlp_proj_b) for i in range(S)]
    hidden3 = [[f32(hidden2[i][j] + mlp[i][j]) for j in range(D)] for i in range(S)]
    # final norm
    out = [layernorm(hidden3[i], lnf_w, lnf_b) for i in range(S)]
    # logits = out @ wte^T
    logits = [[dot(out[i], [wte[r][c] for c in range(D)]) for r in range(VOCAB)] for i in range(S)]

    flat = [float(x) for row in logits for x in row]
    with open(os.path.join(ROOT, "tiny_gpt2_ref_logits.txt"), "w") as fh:
        fh.write(" ".join("%.9g" % x for x in flat) + "\n")
    print("reference logits (%d x %d):" % (S, VOCAB))
    for row in logits:
        print("  " + " ".join("%.9g" % float(x) for x in row))

if __name__ == "__main__":
    main()
