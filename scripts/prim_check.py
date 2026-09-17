"""Differential test of the extracted elementary functions.

Feeds binary32 bit patterns to runners/prim_sweep, built against the inductive
extraction, reads the results back, and compares them against the mathematical
function computed in double precision.

Two quantities are reported per primitive: the maximum error against the true
function, and the number of inputs whose sign disagrees with it. The second
detects an argument reduction that lands on the wrong multiple of the period.
GELU is compared against the tanh form GPT-2 is trained with.

  python prim_check.py [--bin DIR] [--remote HOST]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

f32 = np.float32

# primitive -> (sample range, reference function, kind)
CASES = {
    # sin and cos are sampled over the whole rotary range: an angle is a
    # position times an inverse frequency, the largest inverse frequency is
    # one, so the range is the longest context of the three checkpoints.
    "sin":      ((-262144.0, 262144.0), np.sin,                   "abs"),
    "cos":      ((-262144.0, 262144.0), np.cos,                   "abs"),
    "exp":      ((-80.0, 80.0),     np.exp,                       "rel"),
    "sigmoid":  ((-40.0, 40.0),     lambda x: 1.0 / (1.0 + np.exp(-x)), "abs"),
    "tanh":     ((-20.0, 20.0),     np.tanh,                      "abs"),
    "gelu":     ((-20.0, 20.0),
                 lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi)
                                                    * (x + 0.044715 * x ** 3))),
                 "abs"),
    "log":      ((1.0, 2.0),        np.log,                       "abs"),
    "softplus": ((-30.0, 30.0),     lambda x: np.logaddexp(0.0, x), "abs"),
    "sqrt":     ((0.0, 1e6),        np.sqrt,                      "rel"),
}


def run(binary, name, xs, remote=None):
    bits = np.asarray(xs, dtype=f32).view(np.uint32)
    payload = "\n".join(str(int(b)) for b in bits) + "\n"
    if remote:
        cmd = ["ssh", remote, "%s %s" % (binary, name)]
    else:
        cmd = [binary, name]
    out = subprocess.run(cmd, input=payload, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("prim_sweep %s failed: %s" % (name, out.stderr[:400]))
    vals = [float(t) for t in out.stdout.split()]
    if len(vals) != len(xs):
        raise SystemExit("prim_sweep %s returned %d of %d values"
                         % (name, len(vals), len(xs)))
    return np.array(vals, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=".", help="directory holding prim_sweep")
    ap.add_argument("--remote", default=None, help="ssh host to run it on")
    ap.add_argument("--n", type=int, default=20001)
    args = ap.parse_args()
    binary = os.path.join(args.bin, "prim_sweep") if not args.remote \
        else "%s/prim_sweep" % args.bin

    print("%-10s %-18s %12s %12s %10s" %
          ("primitive", "range", "max err", "kind", "sign flips"))
    bad = 0
    for name, ((lo, hi), ref, kind) in CASES.items():
        xs = np.linspace(lo, hi, args.n).astype(f32)
        got = run(binary, name, xs, args.remote)
        want = ref(xs.astype(np.float64))
        if kind == "rel":
            ok = want != 0
            err = np.max(np.abs(got[ok] - want[ok]) / np.abs(want[ok]))
        else:
            err = np.max(np.abs(got - want))
        # a reduction that lands on the wrong multiple of the period shows up
        # here and not in a mean error
        sig = np.sum((np.sign(got) != np.sign(want)) & (np.abs(want) > 1e-3))
        flag = "" if sig == 0 else "  <-- SIGN"
        if sig:
            bad += 1
        print("%-10s [%7.1f,%7.1f] %12.3e %12s %10d%s"
              % (name, lo, hi, err, kind, sig, flag))
    if bad:
        print("\n%d primitive(s) disagree in sign with the true function." % bad)
        sys.exit(1)
    print("\nno sign disagreements")


if __name__ == "__main__":
    main()
