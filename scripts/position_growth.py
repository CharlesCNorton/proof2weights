"""Whether the difference from a float32 implementation grows with position.

A window's later positions attend over more keys and carry larger rotary
angles, so if rounding accumulated with context length it would show as a
difference that grows along the window. This reports the maximum and mean
absolute logit difference at each position, averaged over windows, in a few
bands.

  python position_growth.py <vocab> <windows.txt> <ref dir> <other dir> [--bands N]
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocab", type=int)
    ap.add_argument("windows")
    ap.add_argument("ref")
    ap.add_argument("other")
    ap.add_argument("--bands", type=int, default=8)
    args = ap.parse_args()

    lines = [l.strip() for l in open(args.windows) if l.strip()]
    T = len(lines[0].split(","))
    mx = np.zeros(T)
    sm = np.zeros(T)
    n = 0
    for w in range(len(lines)):
        a = os.path.join(args.ref, f"{w}.f32")
        b = os.path.join(args.other, f"{w}.f32")
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        A = np.fromfile(a, dtype="<f4").reshape(-1, args.vocab).astype(np.float64)
        B = np.fromfile(b, dtype="<f4").reshape(-1, args.vocab).astype(np.float64)
        if A.shape != B.shape or A.shape[0] != T:
            continue
        d = np.abs(A - B)
        mx = np.maximum(mx, d.max(axis=1))
        sm += d.mean(axis=1)
        n += 1
    if n == 0:
        raise SystemExit("no window pair found")
    sm /= n
    step = max(1, T // args.bands)
    print(f"{n} windows of {T} tokens\n")
    print(f"{'positions':>14s} {'max abs':>12s} {'mean abs':>12s}")
    for lo in range(0, T, step):
        hi = min(lo + step, T)
        print(f"{lo:6d}..{hi - 1:<6d} {mx[lo:hi].max():12.3e} "
              f"{sm[lo:hi].mean():12.3e}")
    first = sm[:step].mean()
    last = sm[-step:].mean()
    print(f"\nmean over the last band divided by the first: {last / first:.3f}")


if __name__ == "__main__":
    main()
