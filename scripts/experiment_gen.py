"""Differential-testing experiment, generation step.

Builds random small GPT-2 models across a sweep of configurations, computes
logits with a numpy float32 implementation that uses the SAME elementwise math
as the Coq spec (6-term clamped Taylor exp, x*sigmoid(1.702x) GELU,
population-variance layernorm, scores scaled by a float32 1/sqrt(d_k)) but
NATURAL numpy reductions (@, sum, mean, max). Writes a batch of safetensors, a
manifest for the Coq reference runner, and the numpy logits for comparison. The
Coq-extracted reference is the canonical oracle; this measures how far numpy's
reduction order lands from it.

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
# would not reproduce across runs. The depth sweep carries seed id 0 and is
# seeded exactly as it always was, so its numbers are unchanged.
CONFIGS = [
    {"tag": "",    "sid": 0, "layers": [1, 2, 4, 8], "d": 8,  "h": 2, "ff": 32,  "vocab": 16, "npos": 16, "t": 8},
    {"tag": "d16", "sid": 1, "layers": [4], "d": 16, "h": 4, "ff": 64,  "vocab": 16, "npos": 16, "t": 8},
    {"tag": "d32", "sid": 2, "layers": [4], "d": 32, "h": 8, "ff": 128, "vocab": 16, "npos": 16, "t": 8},
    {"tag": "d64", "sid": 3, "layers": [4], "d": 64, "h": 8, "ff": 256, "vocab": 16, "npos": 16, "t": 8},
    {"tag": "t16", "sid": 4, "layers": [4], "d": 8,  "h": 2, "ff": 32,  "vocab": 16, "npos": 32, "t": 16},
    {"tag": "t32", "sid": 5, "layers": [4], "d": 8,  "h": 2, "ff": 32,  "vocab": 16, "npos": 32, "t": 32},
    {"tag": "v64", "sid": 6, "layers": [4], "d": 8,  "h": 2, "ff": 32,  "vocab": 64, "npos": 16, "t": 8},
]


def my_exp(x):
    # range-reduced: exp(x) = exp(x/256)^256, matching the Coq spec exactly
    x = x.astype(f32)
    xc = np.clip(x, f32(-88.0), f32(88.0)).astype(f32)
    r = (xc/f32(256.0)).astype(f32)
    r2 = (r*r).astype(f32); r3 = (r2*r).astype(f32); r4 = (r3*r).astype(f32)
    r5 = (r4*r).astype(f32); r6 = (r5*r).astype(f32)
    i = (r6/f32(720.0)).astype(f32)
    i = (r5/f32(120.0) + i).astype(f32)
    i = (r4/f32(24.0) + i).astype(f32)
    i = (r3/f32(6.0) + i).astype(f32)
    i = (r2/f32(2.0) + i).astype(f32)
    i = (r + i).astype(f32)
    s = (f32(1.0) + i).astype(f32)
    for _ in range(8):
        s = (s*s).astype(f32)
    return s.astype(f32)

def my_sigmoid(x):
    return (f32(1.0)/(f32(1.0)+my_exp((-x).astype(f32)))).astype(f32)

def my_gelu(x):
    return (x*my_sigmoid((f32(1.702)*x).astype(f32))).astype(f32)

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
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    for i in range(n_layer):
        p = f'h.{i}.'
        ln1 = layernorm(hidden, w[p+'ln_1.weight'], w[p+'ln_1.bias'])
        qkv = (ln1 @ w[p+'attn.c_attn.weight'] + w[p+'attn.c_attn.bias']).astype(f32)
        q, k, v = qkv[:, :D], qkv[:, D:2*D], qkv[:, 2*D:]
        attn = np.zeros((n, D), dtype=f32)
        for hh in range(H):
            sl = slice(hh*HD, (hh+1)*HD)
            qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
            scores = ((qh @ kh.T).astype(f32)*inv).astype(f32)
            scores = np.where(mask, f32(-1e9), scores).astype(f32)
            mx = scores.max(axis=-1, keepdims=True)
            e = my_exp((scores-mx).astype(f32))
            ww = (e/e.sum(axis=-1, keepdims=True).astype(f32)).astype(f32)
            attn[:, sl] = (ww @ vh).astype(f32)
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
    # Keeps the forward well-conditioned (exp arguments near zero), which is the
    # regime trained models occupy and where the spec's exp is in range.
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
