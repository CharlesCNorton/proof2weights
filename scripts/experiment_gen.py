"""Differential-testing experiment, generation step.

Builds random small GPT-2 models across a sweep of configurations, computes
logits with a numpy float32 implementation that uses the same elementwise math
as the Rocq definitions (the scaled exponential, the tanh form of GELU,
population-variance layernorm, causal attention where each row attends over its
prefix, scores scaled by a float32 1/sqrt(d_k)) but natural numpy reductions
(@, sum, mean, max). Writes a batch of safetensors, a manifest for the
reference runner, and the numpy logits for comparison. The extracted forward is
the reference; this measures how far numpy's reduction order lands from it.

The sweep moves one dimension at a time off a base model, so the report
separates the effect of depth, width, sequence length, and vocabulary.
"""
import os, json
import numpy as np
from safetensors.numpy import save_file

f32 = np.float32
EPS = f32(1e-5)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH = os.path.join(ROOT, "expbatch")
K = 16

# Each entry contributes one block of rows. Seeds are built from integers only:
# Python randomises the hash of a string per process, so a string-keyed seed
# would not reproduce across runs.
CONFIGS = [
    {"tag": "",    "sid": 0, "layers": [1, 2, 4, 8], "d": 8,  "h": 2, "ff": 32,  "vocab": 16, "npos": 16, "t": 8},
    {"tag": "d16", "sid": 1, "layers": [4], "d": 16, "h": 4, "ff": 64,  "vocab": 16, "npos": 16, "t": 8},
    {"tag": "d32", "sid": 2, "layers": [4], "d": 32, "h": 8, "ff": 128, "vocab": 16, "npos": 16, "t": 8},
    {"tag": "d64", "sid": 3, "layers": [4], "d": 64, "h": 8, "ff": 256, "vocab": 16, "npos": 16, "t": 8},
    {"tag": "t16", "sid": 4, "layers": [4], "d": 8,  "h": 2, "ff": 32,  "vocab": 16, "npos": 32, "t": 16},
    {"tag": "t32", "sid": 5, "layers": [4], "d": 8,  "h": 2, "ff": 32,  "vocab": 16, "npos": 32, "t": 32},
    {"tag": "v64", "sid": 6, "layers": [4], "d": 8,  "h": 2, "ff": 32,  "vocab": 64, "npos": 16, "t": 8},
]


EXP_HI = f32(88)
EXP_LO = f32(-88)
LN2_HI = f32(355) / f32(512)
LN2_LO = f32(14581891) / f32(68719476736)
INV_LN2 = f32(12102203) / f32(8388608)
MAGIC = f32(12582912)
GELU_C1 = f32(7978845608) / f32(10000000000)
GELU_C2 = f32(44715) / f32(1000000)
HALF = f32(1) / f32(2)

def my_exp(x):
    # exp(x) = 2^k exp(r) with r = x - k log 2, matching f32_exp_approx
    x = np.asarray(x, dtype=f32)
    xc = np.where(EXP_HI < x, EXP_HI, np.where(x < EXP_LO, EXP_LO, x)).astype(f32)
    k = (((xc*INV_LN2).astype(f32) + MAGIC).astype(f32) - MAGIC).astype(f32)
    r = ((xc - (k*LN2_HI).astype(f32)).astype(f32) + (k*LN2_LO).astype(f32)).astype(f32)
    r2 = (r*r).astype(f32); r3 = (r2*r).astype(f32); r4 = (r3*r).astype(f32)
    r5 = (r4*r).astype(f32); r6 = (r5*r).astype(f32); r7 = (r6*r).astype(f32)
    s = (r6/f32(720) + r7/f32(5040)).astype(f32)
    s = (r5/f32(120) + s).astype(f32)
    s = (r4/f32(24) + s).astype(f32)
    s = (r3/f32(6) + s).astype(f32)
    s = (r2/f32(2) + s).astype(f32)
    s = (r + s).astype(f32)
    s = (f32(1) + s).astype(f32)
    return (s*np.ldexp(f32(1), k.astype(np.int32))).astype(f32)

def my_sigmoid(x):
    return (f32(1.0)/(f32(1.0)+my_exp((-x).astype(f32)))).astype(f32)

def my_tanh(y):
    return ((f32(2)*my_sigmoid((f32(2)*y).astype(f32))).astype(f32) - f32(1)).astype(f32)

def my_gelu(x):
    # 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), matching f32_gelu
    x3 = (x*(x*x).astype(f32)).astype(f32)
    inner = (GELU_C1*(x + (GELU_C2*x3).astype(f32)).astype(f32)).astype(f32)
    return ((HALF*x).astype(f32)*(f32(1) + my_tanh(inner)).astype(f32)).astype(f32)

def layernorm(v, g, b):
    mean = v.mean(axis=-1, keepdims=True).astype(f32)
    dif = (v-mean).astype(f32)
    var = (dif*dif).astype(f32).mean(axis=-1, keepdims=True).astype(f32)
    denom = np.sqrt((var+EPS).astype(f32)).astype(f32)
    return (g*(dif/denom).astype(f32) + b).astype(f32)

def forward(w, n_layer, tokens, c):
    D, H = c["d"], c["h"]
    HD = D // H
    wte = w['wte.weight']; wpe = w['wpe.weight']
    n = len(tokens)
    hidden = (wte[np.array(tokens)] + wpe[:n]).astype(f32)
    inv = (f32(1.0)/np.sqrt(f32(HD))).astype(f32)
    for i in range(n_layer):
        p = f'h.{i}.'
        ln1 = layernorm(hidden, w[p+'ln_1.weight'], w[p+'ln_1.bias'])
        qkv = (ln1 @ w[p+'attn.c_attn.weight'] + w[p+'attn.c_attn.bias']).astype(f32)
        q, k, v = qkv[:, :D], qkv[:, D:2*D], qkv[:, 2*D:]
        attn = np.zeros((n, D), dtype=f32)
        for hh in range(H):
            sl = slice(hh*HD, (hh+1)*HD)
            qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
            for t in range(n):
                scores = ((kh[:t+1] @ qh[t]).astype(f32)*inv).astype(f32)
                e = my_exp((scores - scores.max()).astype(f32))
                ww = (e/e.sum().astype(f32)).astype(f32)
                attn[t, sl] = (ww @ vh[:t+1]).astype(f32)
        proj = (attn @ w[p+'attn.c_proj.weight'] + w[p+'attn.c_proj.bias']).astype(f32)
        hidden = (hidden + proj).astype(f32)
        ln2 = layernorm(hidden, w[p+'ln_2.weight'], w[p+'ln_2.bias'])
        fc = my_gelu((ln2 @ w[p+'mlp.c_fc.weight'] + w[p+'mlp.c_fc.bias']).astype(f32))
        mlp = (fc @ w[p+'mlp.c_proj.weight'] + w[p+'mlp.c_proj.bias']).astype(f32)
        hidden = (hidden + mlp).astype(f32)
    out = layernorm(hidden, w['ln_f.weight'], w['ln_f.bias'])
    return (out @ wte.T).astype(f32)

def gen_weights(rng, n_layer, c):
    # GPT-2-style init: layernorm gamma=1 / beta=0, biases 0, weights N(0,0.02).
    D, FF, VOCAB, NPOS = c["d"], c["ff"], c["vocab"], c["npos"]
    def t(*shape):
        return (rng.standard_normal(shape)*0.02).astype(f32)
    one = lambda *s: np.ones(s, dtype=f32)
    zero = lambda *s: np.zeros(s, dtype=f32)
    w = {'wte.weight': t(VOCAB, D), 'wpe.weight': t(NPOS, D),
         'ln_f.weight': one(D), 'ln_f.bias': zero(D)}
    for i in range(n_layer):
        p = f'h.{i}.'
        w[p+'ln_1.weight'] = one(D); w[p+'ln_1.bias'] = zero(D)
        w[p+'attn.c_attn.weight'] = t(D, 3*D); w[p+'attn.c_attn.bias'] = zero(3*D)
        w[p+'attn.c_proj.weight'] = t(D, D); w[p+'attn.c_proj.bias'] = zero(D)
        w[p+'ln_2.weight'] = one(D); w[p+'ln_2.bias'] = zero(D)
        w[p+'mlp.c_fc.weight'] = t(D, FF); w[p+'mlp.c_fc.bias'] = zero(FF)
        w[p+'mlp.c_proj.weight'] = t(FF, D); w[p+'mlp.c_proj.bias'] = zero(D)
    return w

def seed_for(c, L, kk):
    return hash((L, kk)) if c["sid"] == 0 else hash((L, kk, c["sid"]))

def main():
    os.makedirs(BATCH, exist_ok=True)
    manifest = []
    numpy_logits = {}
    configs = {}
    for c in CONFIGS:
        for L in c["layers"]:
            for kk in range(K):
                sid = f'L{L}_k{kk}' if not c["tag"] else f'{c["tag"]}_L{L}_k{kk}'
                rng = np.random.default_rng(seed_for(c, L, kk) & 0xffffffff)
                w = gen_weights(rng, L, c)
                tokens = [int(x) for x in rng.integers(0, c["vocab"], size=c["t"])]
                save_file(w, os.path.join(BATCH, sid+'.safetensors'))
                numpy_logits[sid] = forward(w, L, tokens, c).tolist()
                configs[sid] = dict(c, layers=L)
                manifest.append(
                    f"{sid} {L} {','.join(map(str, tokens))} "
                    f"{c['d']} {c['h']} {c['ff']} {c['vocab']} {c['npos']}")
    with open(os.path.join(BATCH, 'manifest.txt'), 'w', newline='\n') as f:
        f.write("\n".join(manifest) + "\n")
    with open(os.path.join(BATCH, 'numpy_logits.json'), 'w') as f:
        json.dump(numpy_logits, f)
    with open(os.path.join(BATCH, 'configs.json'), 'w') as f:
        json.dump(configs, f)
    print(f"generated {len(manifest)} samples across {len(CONFIGS)} configurations")

if __name__ == "__main__":
    main()
