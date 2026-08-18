"""Tiny GPT-2 reference for the float-pipeline equivalence check.

Builds a tiny model with deterministic weights, saves a real .safetensors
with GPT-2 tensor names, and computes a reference forward using the EXACT
operations the Coq f32 pipeline defines (range-reduced exp, exp(x/256)^256,
gelu(x)=x*sigmoid(1.702x), population-variance layernorm with eps=1e-5,
causal scaled-dot-product attention with max-shifted softmax, tied-embedding
logits). The extracted OCaml loads the same file and must agree.
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
EPS = f32(1e-5)

def gen(shape):
    n = int(np.prod(shape))
    vals = [f32(((((k * 7 + 3) % 13) - 6)) / 25.0) for k in range(n)]
    return np.array(vals, dtype=f32).reshape(shape)

def my_exp(x):
    # range-reduced: exp(x) = exp(x/256)^256, matching the Coq f32_exp_approx.
    # Saturate to the binary32 exponential range, divide by 256 so the argument
    # lands where a 6-term Taylor series is accurate, then square eight times.
    # Right-associated accumulation mirrors the Coq f32_plus nesting exactly.
    x = f32(x)
    xc = f32(min(max(x, f32(-88.0)), f32(88.0)))
    r = f32(xc / f32(256.0))
    r2 = f32(r * r); r3 = f32(r2 * r); r4 = f32(r3 * r); r5 = f32(r4 * r); r6 = f32(r5 * r)
    i = f32(r6 / f32(720.0))
    i = f32(f32(r5 / f32(120.0)) + i)
    i = f32(f32(r4 / f32(24.0)) + i)
    i = f32(f32(r3 / f32(6.0)) + i)
    i = f32(f32(r2 / f32(2.0)) + i)
    i = f32(r + i)
    s = f32(f32(1.0) + i)
    for _ in range(8):
        s = f32(s * s)
    return s

def my_sigmoid(x):
    return f32(f32(1.0) / f32(f32(1.0) + my_exp(f32(-x))))

def my_gelu(x):
    return f32(x * my_sigmoid(f32(f32(1702.0) / f32(1000.0) * x)))

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
        if x > m:
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
    rng_names = {}
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
            scores = []
            for j in range(S):
                sc = f32(dot(qh[i], kh[j]) * inv)
                if j > i:
                    sc = f32(sc + f32(-1000000000.0))
                scores.append(sc)
            w = softmax(scores)
            for c in range(HD):
                acc = f32(0.0)
                for j in range(S):
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
        fh.write(" ".join("%.6f" % x for x in flat) + "\n")
    print("reference logits (%d x %d):" % (S, VOCAB))
    for row in logits:
        print("  " + " ".join("%.6f" % float(x) for x in row))

if __name__ == "__main__":
    main()
