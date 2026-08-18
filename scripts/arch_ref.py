"""numpy float32 mirrors of the Llama and Qwen3.5 forward passes.

These use the SAME elementwise math the Rocq definitions specify (the
range-reduced exponential, the arctanh logarithm, x * sigmoid(x) for SiLU,
RMSNorm over the mean square, causal attention with a max-shifted softmax) but
NATURAL numpy reductions (@, sum, mean, max) rather than the left folds the
definitions perform. The extracted verified forward is the canonical oracle;
this measures how far numpy's reduction order lands from it.
"""
import numpy as np

f32 = np.float32


def my_exp(x):
    """exp(x) = exp(x/256)^256, matching f32_exp_approx exactly."""
    x = np.asarray(x, dtype=f32)
    xc = np.clip(x, f32(-88.0), f32(88.0)).astype(f32)
    r = (xc / f32(256.0)).astype(f32)
    r2 = (r * r).astype(f32); r3 = (r2 * r).astype(f32); r4 = (r3 * r).astype(f32)
    r5 = (r4 * r).astype(f32); r6 = (r5 * r).astype(f32)
    i = (r6 / f32(720.0)).astype(f32)
    i = (r5 / f32(120.0) + i).astype(f32)
    i = (r4 / f32(24.0) + i).astype(f32)
    i = (r3 / f32(6.0) + i).astype(f32)
    i = (r2 / f32(2.0) + i).astype(f32)
    i = (r + i).astype(f32)
    s = (f32(1.0) + i).astype(f32)
    for _ in range(8):
        s = (s * s).astype(f32)
    return s.astype(f32)


def sigmoid(x):
    return (f32(1.0) / (f32(1.0) + my_exp((-np.asarray(x, dtype=f32)).astype(f32)))).astype(f32)


def silu(x):
    x = np.asarray(x, dtype=f32)
    return (x * sigmoid(x)).astype(f32)


def log_unit(m):
    """log m on (1, 2] by the arctanh series, matching f32_log_unit."""
    m = np.asarray(m, dtype=f32)
    u = ((m - f32(1.0)).astype(f32) / (m + f32(1.0)).astype(f32)).astype(f32)
    u2 = (u * u).astype(f32)
    u3 = (u2 * u).astype(f32)
    u5 = (u3 * u2).astype(f32)
    u7 = (u5 * u2).astype(f32)
    u9 = (u7 * u2).astype(f32)
    u11 = (u9 * u2).astype(f32)
    u13 = (u11 * u2).astype(f32)
    s = (u11 / f32(11.0) + u13 / f32(13.0)).astype(f32)
    s = (u9 / f32(9.0) + s).astype(f32)
    s = (u7 / f32(7.0) + s).astype(f32)
    s = (u5 / f32(5.0) + s).astype(f32)
    s = (u3 / f32(3.0) + s).astype(f32)
    s = (u + s).astype(f32)
    return (f32(2.0) * s).astype(f32)


def softplus(x):
    """max(x,0) + log(1 + exp(-|x|)), matching f32_softplus."""
    x = np.asarray(x, dtype=f32)
    e = my_exp((-np.abs(x)).astype(f32))
    return (np.maximum(x, f32(0.0)).astype(f32) + log_unit((f32(1.0) + e).astype(f32))).astype(f32)


def rmsnorm(w, x, eps):
    ms = (x * x).astype(f32).mean(axis=-1, keepdims=True).astype(f32)
    rs = (f32(1.0) / np.sqrt((ms + eps).astype(f32))).astype(f32)
    return (w * (x * rs).astype(f32)).astype(f32)


def rmsnorm_zc(w, x, eps):
    """The zero-centred variant: the weight is applied as 1 + w."""
    ms = (x * x).astype(f32).mean(axis=-1, keepdims=True).astype(f32)
    rs = (f32(1.0) / np.sqrt((ms + eps).astype(f32))).astype(f32)
    return ((f32(1.0) + w).astype(f32) * (x * rs).astype(f32)).astype(f32)


def rmsnorm_gated(w, gate, x, eps):
    ms = (x * x).astype(f32).mean(axis=-1, keepdims=True).astype(f32)
    rs = (f32(1.0) / np.sqrt((ms + eps).astype(f32))).astype(f32)
    return ((w * (x * rs).astype(f32)).astype(f32) * silu(gate)).astype(f32)


def l2norm(v, eps):
    ss = (v * v).astype(f32).sum().astype(f32)
    inv = (f32(1.0) / np.sqrt((ss + eps).astype(f32))).astype(f32)
    return (v * inv).astype(f32)


def rope(x, cos, sin, rd):
    """Rotate the first rd entries, pass the rest through."""
    rot, passthrough = x[:rd], x[rd:]
    half = rd // 2
    rotated = np.concatenate([(-rot[half:]).astype(f32), rot[:half]]).astype(f32)
    out = ((rot * cos[:rd]).astype(f32) + (rotated * sin[:rd]).astype(f32)).astype(f32)
    return np.concatenate([out, passthrough]).astype(f32)


def causal_attention(q, k, v, hd):
    """Scaled dot-product causal attention over the whole sequence."""
    n = q.shape[0]
    s = (f32(1.0) / np.sqrt(f32(hd))).astype(f32)
    scores = ((q @ k.T).astype(f32) * s).astype(f32)
    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    scores = np.where(mask, (scores + f32(-1000000000.0)).astype(f32), scores).astype(f32)
    mx = scores.max(axis=-1, keepdims=True)
    e = my_exp((scores - mx).astype(f32))
    w = (e / e.sum(axis=-1, keepdims=True).astype(f32)).astype(f32)
    return (w @ v).astype(f32)


def swiglu(wg, wu, wd, x):
    g = silu((x @ wg.T).astype(f32))
    u = (x @ wu.T).astype(f32)
    return ((g * u).astype(f32) @ wd.T).astype(f32)


def llama_forward(W, ids, cfg):
    d, nh, nkv, hd = cfg["d"], cfg["nh"], cfg["nkv"], cfg["hd"]
    eps = f32(1e-5)
    group = nh // nkv
    h = W["emb"][np.array(ids)].astype(f32)
    cos, sin = W["cos"], W["sin"]
    for i in range(cfg["nl"]):
        p = f"L{i}."
        hn = rmsnorm(W[p + "ln1"], h, eps)
        q = (hn @ W[p + "q"].T).astype(f32)
        k = (hn @ W[p + "k"].T).astype(f32)
        v = (hn @ W[p + "v"].T).astype(f32)
        qs = [np.stack([rope(q[t][c * hd:(c + 1) * hd], cos[t], sin[t], hd)
                        for t in range(len(ids))]) for c in range(nh)]
        ks = [np.stack([rope(k[t][c * hd:(c + 1) * hd], cos[t], sin[t], hd)
                        for t in range(len(ids))]) for c in range(nkv)]
        vs = [np.stack([v[t][c * hd:(c + 1) * hd] for t in range(len(ids))])
              for c in range(nkv)]
        heads = [causal_attention(qs[c], ks[c // group], vs[c // group], hd)
                 for c in range(nh)]
        concat = np.concatenate(heads, axis=1).astype(f32)
        attn = (concat @ W[p + "o"].T).astype(f32)
        h2 = (h + attn).astype(f32)
        hn2 = rmsnorm(W[p + "ln2"], h2, eps)
        mlp = np.stack([swiglu(W[p + "gate"], W[p + "up"], W[p + "down"], hn2[t])
                        for t in range(len(ids))]).astype(f32)
        h = (h2 + mlp).astype(f32)
    out = rmsnorm(W["norm"], h, eps)
    return (out @ W["emb"].T).astype(f32)


def qwen_delta_mix(W, p, hn, cfg):
    lnh, lhd, ck = cfg["lnh"], cfg["lhd"], cfg["ck"]
    ldim = lnh * lhd
    eps = f32(1e-6)
    T = hn.shape[0]
    qkv = (hn @ W[p + "in_qkv"].T).astype(f32)
    conv = W[p + "conv_w"]
    c = np.zeros_like(qkv)
    for t in range(T):
        win = np.zeros((ck, qkv.shape[1]), dtype=f32)
        for j in range(ck):
            idx = t - (ck - 1) + j
            if idx >= 0:
                win[j] = qkv[idx]
        acc = (conv * win.T).astype(f32).sum(axis=1).astype(f32)
        c[t] = silu((acc + f32(0.0)).astype(f32))
    z = (hn @ W[p + "in_z"].T).astype(f32)
    av = (hn @ W[p + "in_a"].T).astype(f32)
    bv = (hn @ W[p + "in_b"].T).astype(f32)
    heads = []
    for hh in range(lnh):
        st = np.zeros((lhd, lhd), dtype=f32)
        outs = []
        for t in range(T):
            qn = l2norm(c[t][hh * lhd:(hh + 1) * lhd], eps)
            q = (qn * (f32(1.0) / np.sqrt(f32(lhd))).astype(f32)).astype(f32)
            k = l2norm(c[t][ldim + hh * lhd: ldim + (hh + 1) * lhd], eps)
            v = c[t][2 * ldim + hh * lhd: 2 * ldim + (hh + 1) * lhd]
            beta = sigmoid(bv[t][hh])
            g = (-(my_exp(W[p + "a_log"][hh])
                   * softplus((av[t][hh] + W[p + "dt_bias"][hh]).astype(f32))).astype(f32)).astype(f32)
            eg = my_exp(g)
            st = (st * eg).astype(f32)
            kv = (st.T @ k).astype(f32)
            dv = ((v - kv).astype(f32) * beta).astype(f32)
            st = (st + np.outer(k, dv).astype(f32)).astype(f32)
            o = (st.T @ q).astype(f32)
            outs.append(rmsnorm_gated(W[p + "norm_w"],
                                      z[t][hh * lhd:(hh + 1) * lhd], o, eps))
        heads.append(np.stack(outs).astype(f32))
    cat = np.concatenate(heads, axis=1).astype(f32)
    return (cat @ W[p + "out"].T).astype(f32)


def qwen_attn_mix(W, p, hn, cos, sin, cfg):
    nh, nkv, hd, rd = cfg["nh"], cfg["nkv"], cfg["hd"], cfg["rd"]
    eps = f32(1e-6)
    T = hn.shape[0]
    group = nh // nkv
    qg = (hn @ W[p + "q"].T).astype(f32)
    kraw = (hn @ W[p + "k"].T).astype(f32)
    vraw = (hn @ W[p + "v"].T).astype(f32)
    ks = [np.stack([rope(rmsnorm_zc(W[p + "k_norm"], kraw[t][c * hd:(c + 1) * hd], eps),
                         cos[t], sin[t], rd) for t in range(T)]) for c in range(nkv)]
    vs = [np.stack([vraw[t][c * hd:(c + 1) * hd] for t in range(T)]) for c in range(nkv)]
    qs = [np.stack([rope(rmsnorm_zc(W[p + "q_norm"], qg[t][hh * hd * 2:hh * hd * 2 + hd], eps),
                         cos[t], sin[t], rd) for t in range(T)]) for hh in range(nh)]
    heads = [causal_attention(qs[hh], ks[hh // group], vs[hh // group], hd)
             for hh in range(nh)]
    cat = np.concatenate(heads, axis=1).astype(f32)
    gates = np.stack([np.concatenate([qg[t][hh * hd * 2 + hd: hh * hd * 2 + 2 * hd]
                                      for hh in range(nh)]) for t in range(T)]).astype(f32)
    gated = (cat * sigmoid(gates)).astype(f32)
    return (gated @ W[p + "o"].T).astype(f32)


def qwen_forward(W, ids, cfg):
    eps = f32(1e-6)
    h = W["emb"][np.array(ids)].astype(f32)
    cos, sin = W["cos"], W["sin"]
    for i in range(cfg["nl"]):
        p = f"L{i}."
        hn = rmsnorm_zc(W[p + "ln1"], h, eps)
        if cfg["kinds"][i] == "attn":
            mix = qwen_attn_mix(W, p, hn, cos, sin, cfg)
        else:
            mix = qwen_delta_mix(W, p, hn, cfg)
        h2 = (h + mix).astype(f32)
        hn2 = rmsnorm_zc(W[p + "ln2"], h2, eps)
        mlp = np.stack([swiglu(W[p + "gate"], W[p + "up"], W[p + "down"], hn2[t])
                        for t in range(len(ids))]).astype(f32)
        h = (h2 + mlp).astype(f32)
    out = rmsnorm_zc(W["norm"], h, eps)
    return (out @ W["emb"].T).astype(f32)
