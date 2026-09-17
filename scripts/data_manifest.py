"""The digests of a published logit dump, and a seeded re-check of part of it.

The per-window logit dumps behind the agreement tables are large, so they are
published as a release asset rather than committed. This writes the manifest
that names them and verifies one against the files on disk, and picks a seeded
subset for a reader who wants to regenerate rather than trust: the subset is a
function of the seed and the window count alone, so naming the seed names the
windows.

  python data_manifest.py write <manifest.json> <name>=<dir> ...
  python data_manifest.py check <manifest.json> [--root DIR]
  python data_manifest.py subset <manifest.json> <name> [--seed N] [--count K]

A dump directory holds <index>.f32 per window: the logits of every position as
little-endian binary32, positions in order, which is the layout every runner's
dump mode writes and every comparison reads.
"""
import argparse
import hashlib
import json
import os
import random
import sys


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write(args):
    out = {"sources": {}}
    for spec in args.sources:
        name, d = spec.split("=", 1)
        files = sorted((f for f in os.listdir(d) if f.endswith(".f32")),
                       key=lambda f: int(f.split(".")[0]))
        entry = {"dir": os.path.basename(d.rstrip("/\\")), "files": {}}
        total = 0
        for f in files:
            p = os.path.join(d, f)
            entry["files"][f] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
            total += os.path.getsize(p)
        entry["count"] = len(files)
        entry["bytes"] = total
        out["sources"][name] = entry
        print(f"  {name:22s} {len(files):4d} files {total / 1e6:9.1f} MB")
    with open(args.manifest, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {args.manifest}")
    return 0


def check(args):
    man = json.load(open(args.manifest))
    bad = missing = ok = 0
    for name, entry in man["sources"].items():
        d = os.path.join(args.root, entry["dir"])
        for f, rec in entry["files"].items():
            p = os.path.join(d, f)
            if not os.path.exists(p):
                missing += 1
                continue
            if sha256(p) == rec["sha256"]:
                ok += 1
            else:
                bad += 1
                print(f"  DIGEST DIFFERS {name}/{f}")
    print(f"{ok} verified, {bad} differ, {missing} absent")
    return 1 if bad else 0


def subset(args):
    man = json.load(open(args.manifest))
    entry = man["sources"][args.name]
    idx = sorted(int(f.split(".")[0]) for f in entry["files"])
    rng = random.Random(args.seed)
    pick = sorted(rng.sample(idx, min(args.count, len(idx))))
    print(f"seed {args.seed}, {len(pick)} of {len(idx)} windows of {args.name}:")
    print(",".join(str(i) for i in pick))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("manifest")
    w.add_argument("sources", nargs="+"); w.set_defaults(fn=write)
    c = sub.add_parser("check"); c.add_argument("manifest")
    c.add_argument("--root", default="."); c.set_defaults(fn=check)
    s = sub.add_parser("subset"); s.add_argument("manifest"); s.add_argument("name")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--count", type=int, default=5); s.set_defaults(fn=subset)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
