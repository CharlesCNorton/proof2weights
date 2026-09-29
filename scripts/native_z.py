"""The native build's operations on Z.

The native build maps Z, positive and N to the host's 63-bit integers.
native_z.v extracts the build's roots with those types left inductive, so every
operation on them the build performs appears in the output, and this lists the
Z, positive and N functions each extracted definition calls outside the modules
that define them. The check passes when the only calls are Z.of_nat, applied to
a length, a dimension or a position before f32_of_Z, and the conversions
between characters and N of the header parser, whose values are below 256.

  python native_z.py [theories dir]

Needs coqc on PATH and the compiled theories.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
THEORIES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "theories")
ALLOWED = {"Z.of_nat", "N.of_nat", "N.to_nat", "N.add", "N.mul"}
HELPERS = {"Nat", "Pos", "N", "Z", "Coq_Pos", "Coq_N", "Coq_Z"}


def calls(ml):
    out, cur, body, mod = {}, None, [], None
    for ln in ml.splitlines():
        m = re.match(r"^module (\w+) =", ln)
        if m:
            mod = m.group(1)
            continue
        if mod and re.match(r"^ end", ln):
            mod = None
            continue
        if mod in HELPERS:
            continue
        m = re.match(r"^(?:let|and)(?: rec)? (\w+)", ln)
        if m:
            if cur:
                out[cur] = "\n".join(body)
            cur, body = m.group(1), [ln]
        elif cur:
            body.append(ln)
    if cur:
        out[cur] = "\n".join(body)
    found = {}
    for name, b in out.items():
        for mo, f in re.findall(r"\b(Z|Pos|N|Coq_Pos|Coq_N|Coq_Z)\.(\w+)", b):
            found.setdefault(mo.replace("Coq_", "") + "." + f, set()).add(name)
    return found


def main():
    work = tempfile.mkdtemp(prefix="native_z_")
    try:
        shutil.copy(os.path.join(HERE, "native_z.v"), work)
        subprocess.run(["coqc", "-R", THEORIES, "", "native_z.v"], cwd=work,
                       check=True, capture_output=True)
        found = calls(open(os.path.join(work, "nativez.ml"), encoding="utf-8").read())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for f in sorted(found):
        print(f"{f:14s} {', '.join(sorted(found[f]))}")
    bad = set(found) - ALLOWED
    print("pass" if not bad else f"FAIL: {', '.join(sorted(bad))}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
