"""Settle the inputs the exhaustive sweep leaves open.

scripts/exhaustive.c compares each result against a binary64 libm reference.
Binary64 carries 53 significand bits against binary32's 24, so an input whose
true value lies within a guard band of a binary32 midpoint cannot be decided
from it. Run with P2W_UNDECIDED set, the sweep writes those inputs, and the
result it computed for each, to <dir>/<function>.u32.

This reads them and decides each one against a reference at 120 decimal digits,
which is far beyond any midpoint they could be near, so nothing is left open.
Given the sweep's own output it folds the verdicts back in and writes the
merged table.

  python settle_undecided.py <dir> [--sweep paper/exhaustive.txt]
                                   [--out paper/exhaustive_settled.txt]
                                   [function ...]
"""
import argparse
import math
import os
import struct

from mpmath import mp, mpf, exp, log, sin, cos, sqrt, tanh, pi

mp.dps = 120

SAT = mpf(88)


def _sat(x):
    return SAT if x > SAT else (-SAT if x < -SAT else x)


def r_exp(x):
    return exp(_sat(x))


def r_sigmoid(x):
    return 1 / (1 + exp(_sat(-x)))


def r_tanh(x):
    return 2 * r_sigmoid(2 * x) - 1


def r_gelu(x):
    t = sqrt(mpf(2) / pi) * (x + mpf('0.044715') * x ** 3)
    return mpf('0.5') * x * (1 + tanh(t))


def r_softplus(x):
    return (x if x > 0 else mpf(0)) + log(1 + exp(-abs(x)))


REF = {"exp": r_exp, "sigmoid": r_sigmoid, "tanh": r_tanh, "gelu": r_gelu,
       "log": log, "softplus": r_softplus, "sin": sin, "cos": cos,
       "sqrt": sqrt}

ORDER = ["exp", "sigmoid", "tanh", "gelu", "log", "softplus", "sin", "cos",
         "sqrt"]


def f32(x):
    """The binary32 nearest to an mpmath value, ties to even."""
    return struct.unpack("<f", struct.pack("<f", float(mp.nstr(x, 40))))[0]


def bits_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def ulp32(x):
    """The spacing of binary32 at a real value, the sweep's own definition."""
    a = abs(float(x))
    if a == 0.0:
        return math.ldexp(1.0, -149)
    _, e = math.frexp(a)
    return max(math.ldexp(1.0, e - 24), math.ldexp(1.0, -149))


def settle(path, name):
    """Decide every recorded input. Returns (count, wrong, max ulp error)."""
    raw = open(path, "rb").read()
    words = struct.unpack("<%dI" % (len(raw) // 4), raw)
    n = wrong = 0
    worst = 0.0
    for i in range(0, len(words), 2):
        x = mpf(bits_to_f32(words[i]))
        got = bits_to_f32(words[i + 1])
        ref = REF[name](x)
        n += 1
        if struct.pack("<f", got) != struct.pack("<f", f32(ref)):
            wrong += 1
        worst = max(worst, abs(got - float(ref)) / ulp32(ref))
    return n, wrong, worst


def read_sweep(path):
    """The sweep's own table, keyed by function."""
    rows = {}
    for line in open(path):
        f = line.split()
        if len(f) < 7 or f[0] == "function":
            continue
        rows[f[0]] = dict(inputs=int(f[1]), ulp=float(f[2]), abs=f[3],
                          rel=f[4], bad=int(f[5]), undec=int(f[6]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("functions", nargs="*")
    ap.add_argument("--sweep")
    ap.add_argument("--out")
    args = ap.parse_args()

    names = args.functions or [n for n in ORDER
                               if os.path.exists(os.path.join(args.dir, n + ".u32"))]
    rows = read_sweep(args.sweep) if args.sweep else {}

    settled = {}
    print("%-10s %10s %10s %12s" % ("function", "undecided", "not c.r.", "max ulp"))
    for name in names:
        n, wrong, worst = settle(os.path.join(args.dir, name + ".u32"), name)
        settled[name] = (n, wrong, worst)
        print("%-10s %10d %10d %12.4f" % (name, n, wrong, worst))
    total = sum(v[0] for v in settled.values())
    bad = sum(v[1] for v in settled.values())
    print("\n%d settled, %d of them not correctly rounded, none left open"
          % (total, bad))

    if not rows:
        return

    order = [n for n in ORDER if n in rows]
    lines = ["%-9s %13s %12s %11s %11s %12s" %
             ("function", "inputs", "max ulp", "max abs", "max rel", "not c.r.")]
    for name in order:
        r = rows[name]
        n, wrong, worst = settled.get(name, (0, 0, 0.0))
        if n != r["undec"]:
            raise SystemExit("%s: sweep left %d open, %d recorded"
                             % (name, r["undec"], n))
        lines.append("%-9s %13d %12.4f %11s %11s %12d" %
                     (name, r["inputs"], max(r["ulp"], worst), r["abs"],
                      r["rel"], r["bad"] + wrong))
    text = "\n".join(lines) + "\n"
    print()
    print(text, end="")
    if args.out:
        open(args.out, "w").write(text)


if __name__ == "__main__":
    main()
