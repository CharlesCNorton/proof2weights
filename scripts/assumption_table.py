"""Classify every result the assumption report covers.

Pairs each [Print Assumptions] of theories/Audit.v with the block the report
prints for it, and sorts the results into the three classes the report can
produce: closed under the global context, resting on the classical axioms of
the Rocq real-number library, or resting in addition on the primitive-float,
63-bit-integer and primitive-array interface CoqInterval computes with.

  python assumption_table.py <audit_output.txt> [--names a,b,c]

Without --names every result is listed; with it, only those named.
"""
import argparse
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRIM = ("PrimFloat", "FloatAxioms", "PrimInt63", "Uint63", "Sint63", "PrimArray",
        "SpecFloat.Prim")

CLASS = {
    "closed": "closed",
    "classical": "classical",
    "interval": "interval",
}


def names_in_audit(path):
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*Print Assumptions\s+([A-Za-z0-9_'.]+)\s*\.", line)
        if m:
            out.append(m.group(1))
    return out


def blocks(path):
    """The report's output, split into one block per result, in order."""
    text = open(path, encoding="utf-8", errors="replace").read()
    parts = []
    cur = None
    for line in text.splitlines():
        if line.startswith("Closed under the global context"):
            if cur is not None:
                parts.append(cur)
            parts.append(("closed", []))
            cur = None
        elif line.startswith("Axioms:"):
            if cur is not None:
                parts.append(cur)
            cur = ("axioms", [])
        elif cur is not None:
            cur[1].append(line)
    if cur is not None:
        parts.append(cur)
    return parts


def classify(kind, body):
    if kind == "closed":
        return "closed"
    joined = "\n".join(body)
    return "interval" if any(k in joined for k in PRIM) else "classical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--audit", default=os.path.join(ROOT, "theories", "Audit.v"))
    ap.add_argument("--names", default="")
    args = ap.parse_args()

    names = names_in_audit(args.audit)
    parts = blocks(args.report)
    if len(names) != len(parts):
        raise SystemExit(f"{len(names)} results named, {len(parts)} reported; "
                         "the report is not the one this Audit.v produces")
    rows = [(n, classify(k, b)) for n, (k, b) in zip(names, parts)]

    counts = {c: sum(1 for _, k in rows if k == c)
              for c in ("closed", "classical", "interval")}
    print(f"{len(rows)} results: {counts['closed']} closed, "
          f"{counts['classical']} classical, {counts['interval']} CoqInterval")

    wanted = [s.strip() for s in args.names.split(",") if s.strip()]
    if wanted:
        by = dict(rows)
        missing = [w for w in wanted if w not in by]
        if missing:
            raise SystemExit(f"not in the report: {', '.join(missing)}")
        rows = [(w, by[w]) for w in wanted]

    label = {"closed": "closed", "classical": "classical",
             "interval": "classical + CoqInterval"}
    for n, k in rows:
        print(f"  {n:38s} {label[k]}")


if __name__ == "__main__":
    main()
