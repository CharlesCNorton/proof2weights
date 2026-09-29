"""Agreement of float32 implementations with the extracted forward pass on held-out text.

Each source is a directory of logit dumps for the windows of one windows file
(agree_setup.py): <dir>/<index>.f32 holds little-endian binary32 logits, one
row of <vocab> entries for each position of the window, positions in order, and
a float64 reference writes <dir>/<index>.f64 in binary64 instead.
The first source named is the reference, and the pairs compared are the
reference with each other source, together with any pair named by --pair.

For every pair compared the report gives the fraction of positions whose
highest-scoring token agrees, the fraction whose ten highest-scoring tokens
agree in order, the fraction of logits that are bit-identical, quantiles and
the maximum of the absolute logit difference, the mean and largest
Kullback-Leibler divergence D(p_first || p_second) between the next-token
distributions, and the largest top-1 margin, in the first source of the pair,
at a position where the top-1 tokens differ. A quantile is reported as the
upper edge of the histogram bin holding it, bins being a hundredth of a decade
wide. For every source it gives the perplexity of the window tokens over the
positions that have a next token in the window, and the largest difference in
per-token negative log-likelihood from the reference.

  python agree_cmp.py <model> <windows.txt> <vocab> <name>=<dir> ... [--pair A:B]
                      [--title TEXT] [--note TEXT] [--write]

--title replaces the section heading and --note is appended to the section's
description; --write appends the section to RESULTS.md.
"""
import argparse
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGES = np.round(np.arange(-15.0, 3.0001, 0.01), 2)


def dump(d, w):
    """The dump of window w in directory d: binary64 where the source wrote
    one, binary32 otherwise."""
    p = os.path.join(d, f"{w}.f64")
    return p if os.path.exists(p) else os.path.join(d, f"{w}.f32")


def load(path, vocab):
    a = np.fromfile(path, dtype="<f8" if path.endswith(".f64") else "<f4")
    return a.reshape(-1, vocab)


def top10(x):
    idx = np.argpartition(-x, 10, axis=1)[:, :10]
    vals = np.take_along_axis(x, idx, 1)
    order = np.lexsort((idx, -vals), axis=1)
    return np.take_along_axis(idx, order, 1)


def log_softmax(x64):
    m = x64.max(axis=1, keepdims=True)
    return x64 - (m + np.log(np.exp(x64 - m).sum(axis=1, keepdims=True)))


class Pair:
    def __init__(self):
        self.pos = 0
        self.top1 = 0
        self.top10 = 0
        self.logits = 0
        self.exact = 0
        self.hist = np.zeros(len(EDGES) - 1, dtype=np.int64)
        self.below = 0
        self.dmax = 0.0
        self.flip_margin = 0.0
        self.kl_sum = 0.0
        self.kl_max = 0.0

    def add(self, a, b, ta, tb, a64, b64, la, lb):
        kl = (np.exp(la) * (la - lb)).sum(axis=1)
        self.kl_sum += float(kl.sum())
        self.kl_max = max(self.kl_max, float(kl.max()))
        self.pos += a.shape[0]
        a1 = a64.argmax(axis=1)
        b1 = b64.argmax(axis=1)
        same = a1 == b1
        self.top1 += int(same.sum())
        self.top10 += int((ta == tb).all(axis=1).sum())
        if not same.all():
            rows = np.nonzero(~same)[0]
            part = -np.partition(-a64[rows], 1, axis=1)[:, :2]
            self.flip_margin = max(self.flip_margin, float((part[:, 0] - part[:, 1]).max()))
        d = np.abs(a64 - b64).ravel()
        self.logits += d.size
        nz = d[d > 0]
        self.exact += d.size - nz.size
        if nz.size:
            self.dmax = max(self.dmax, float(nz.max()))
            lg = np.log10(nz)
            self.below += int((lg < EDGES[0]).sum())
            self.hist += np.histogram(lg, bins=EDGES)[0]

    def quantile(self, q):
        need = q * self.logits
        count = self.exact + self.below
        if count >= need:
            return 0.0 if self.exact >= need else 10 ** EDGES[0]
        for i, c in enumerate(self.hist):
            count += c
            if count >= need:
                return 10 ** EDGES[i + 1]
        return self.dmax


def fmt(v):
    return "0" if v == 0 else f"{v:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("windows")
    ap.add_argument("vocab", type=int)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--title", default="Agreement on held-out text")
    ap.add_argument("--note", default="")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    names, dirs = zip(*(s.split("=", 1) for s in args.sources))
    windows = [[int(t) for t in l.split(",")] for l in open(args.windows) if l.strip()]
    if args.limit:
        windows = windows[:args.limit]
    chosen = [(0, j) for j in range(1, len(names))]
    for p in args.pair:
        a, b = p.split(":", 1)
        chosen.append((names.index(a), names.index(b)))
    pairs = {p: Pair() for p in chosen}
    nll_sum = [0.0] * len(names)
    nll_max = [0.0] * len(names)
    predicted = 0
    used = 0
    for w, toks in enumerate(windows):
        paths = [dump(d, w) for d in dirs]
        if not all(os.path.exists(p) for p in paths):
            continue
        xs = [load(p, args.vocab) for p in paths]
        T = len(toks)
        for p, x in zip(paths, xs):
            if x.shape[0] != T:
                raise SystemExit(f"{p}: {x.shape[0]} rows for a window of {T} tokens")
        used += 1
        x64 = [x.astype(np.float64) for x in xs]
        ls = [log_softmax(x) for x in x64]
        tops = [top10(x) for x in xs]
        for (i, j), pr in pairs.items():
            pr.add(xs[i], xs[j], tops[i], tops[j], x64[i], x64[j], ls[i], ls[j])
        targets = np.array(toks[1:])
        nlls = [-l[np.arange(T - 1), targets] for l in ls]
        predicted += T - 1
        for s, v in enumerate(nlls):
            nll_sum[s] += float(v.sum())
            nll_max[s] = max(nll_max[s], float(np.abs(v - nlls[0]).max()))
    T = len(windows[0])
    note = f" {args.note}" if args.note else ""
    lines = ["", f"## {args.title}: {args.model}", "",
             f"{used} windows of {T} tokens from the WikiText-2 raw test split, every "
             f"position compared.{note}", "",
             "| pair | top-1 agrees | top-10 agrees in order | bit-identical logits "
             "| median abs diff | 99.9th pct | max abs diff | mean KL | max KL "
             "| largest margin at a top-1 difference |",
             "|------|--------------|------------------------|----------------------"
             "|-----------------|------------|--------------|---------|--------"
             "|--------------------------------------|"]
    for (i, j), pr in pairs.items():
        flips = pr.pos - pr.top1
        margin = f"{pr.flip_margin:.2e}" if flips else "-"
        lines.append(f"| {names[i]} / {names[j]} | {pr.top1}/{pr.pos} | {pr.top10}/{pr.pos} | "
                     f"{100.0 * pr.exact / pr.logits:.2f}% | {fmt(pr.quantile(0.5))} | "
                     f"{fmt(pr.quantile(0.999))} | {pr.dmax:.2e} | {pr.kl_sum / pr.pos:.1e} | "
                     f"{pr.kl_max:.1e} | {margin} |")
    lines += ["", "| source | predicted tokens | perplexity | max per-token NLL difference from "
              f"{names[0]} |", "|--------|------------------|------------|-----------------------|"]
    for s, name in enumerate(names):
        ppl = math.exp(nll_sum[s] / predicted) if predicted else float("nan")
        diff = "-" if s == 0 else f"{nll_max[s]:.2e}"
        lines.append(f"| {name} | {predicted} | {ppl:.6f} | {diff} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.write:
        with open(os.path.join(ROOT, "RESULTS.md"), "a", newline="\n", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
