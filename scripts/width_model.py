"""A first-order model of the order-induced error in the random GPT-2 models.

Two float32 implementations that perform the same elementwise operations and
sum in different orders disagree by the rounding errors of their sums, which
experiment_gen.py and the reference runner measure. This model gives every sum
an independent normal error with standard deviation u * sqrt(sum_k s_k^2), the
s_k being the sum's partial sums, which is the typical size of a recursive
sum's rounding error; two independent orders disagree by sqrt(2) times that. It
evaluates each model of experiment_gen.py in float64, perturbs every sum (the
linear layers, the attention scores, the softmax normalisers, the
attention-weighted sums of values, the layer-norm means and variances, the
logits), and reports the mean absolute change over every logit of every sample,
with the two ratios the paper quotes.

  python width_model.py [--draws R]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_gen import CONFIGS, K, gen_weights, seed_for

U = 2.0 ** -24


class Sums:
    """Sums over an axis, each with an independent rounding error of the typical
    size when a generator is given, exact otherwise."""

    def __init__(self, rng=None):
        self.rng = rng

    def __call__(self, terms, axis):
        s = terms.sum(axis=axis)
        if self.rng is not None:
            sd = U * np.sqrt((np.cumsum(terms, axis=axis) ** 2).sum(axis=axis))
            s = s + sd * self.rng.standard_normal(np.shape(s))
        return s


def linear(S, x, W, b):
    return S(x[:, :, None] * W[None, :, :], axis=1) + b


def layernorm(S, v, g, b, eps=1e-5):
    n = v.shape[-1]
    dif = v - S(v, axis=-1)[..., None] / n
    var = S(dif * dif, axis=-1)[..., None] / n
    return g * dif / np.sqrt(var + eps) + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def forward(S, w, n_layer, tokens, c):
    D, H = c["d"], c["h"]
    HD = D // H
    w = {k: v.astype(np.float64) for k, v in w.items()}
    T = len(tokens)
    h = w["wte.weight"][np.array(tokens)] + w["wpe.weight"][:T]
    for i in range(n_layer):
        p = f"h.{i}."
        a = layernorm(S, h, w[p + "ln_1.weight"], w[p + "ln_1.bias"])
        qkv = linear(S, a, w[p + "attn.c_attn.weight"], w[p + "attn.c_attn.bias"])
        q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        attn = np.zeros((T, D))
        for hh in range(H):
            sl = slice(hh * HD, (hh + 1) * HD)
            for t in range(T):
                sc = S(k[:t + 1, sl] * q[t, sl], axis=1) / np.sqrt(HD)
                e = np.exp(sc - sc.max())
                wt = e / S(e, axis=0)
                attn[t, sl] = S(wt[:, None] * v[:t + 1, sl], axis=0)
        h = h + linear(S, attn, w[p + "attn.c_proj.weight"], w[p + "attn.c_proj.bias"])
        a = layernorm(S, h, w[p + "ln_2.weight"], w[p + "ln_2.bias"])
        f = gelu(linear(S, a, w[p + "mlp.c_fc.weight"], w[p + "mlp.c_fc.bias"]))
        h = h + linear(S, f, w[p + "mlp.c_proj.weight"], w[p + "mlp.c_proj.bias"])
    out = layernorm(S, h, w["ln_f.weight"], w["ln_f.bias"])
    return S(out[:, :, None] * w["wte.weight"].T[None, :, :], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=8)
    args = ap.parse_args()
    pred = {}
    print("| layers | d_model | seq | vocab | predicted mean abs-err |")
    print("|--------|---------|-----|-------|------------------------|")
    for c in CONFIGS:
        for L in c["layers"]:
            tot, cnt = 0.0, 0
            for kk in range(K):
                rng = np.random.default_rng(seed_for(c, L, kk) & 0xffffffff)
                w = gen_weights(rng, L, c)
                tokens = [int(x) for x in rng.integers(0, c["vocab"], size=c["t"])]
                exact = forward(Sums(), w, L, tokens, c)
                noise = np.random.default_rng(1000 * kk + L)
                for _ in range(args.draws):
                    got = forward(Sums(noise), w, L, tokens, c)
                    tot += float(np.abs(got - exact).sum()) * np.sqrt(2.0)
                    cnt += got.size
            pred[(L, c["d"], c["t"], c["vocab"])] = tot / cnt
            print(f"| {L} | {c['d']} | {c['t']} | {c['vocab']} | {tot / cnt:.3e} |")
    print(f"\nwidth 64 over width 8 at four layers: "
          f"{pred[(4, 64, 8, 16)] / pred[(4, 8, 8, 16)]:.1f}")
    print(f"eight layers over one at width 8: "
          f"{pred[(8, 8, 8, 16)] / pred[(1, 8, 8, 16)]:.2f}")


if __name__ == "__main__":
    main()
