"""Each checkpoint's extracted forward pass against its stored PyTorch oracle.

The setup scripts record, for each model, the prompt token ids and what PyTorch
in float32 generates from them. This runs the built runner on the same ids and
compares the greedy continuation token for token. It is the check that says the
extracted pass still runs the published weights, and it takes seconds to
minutes rather than the hours a full sweep takes.

  python oracle_check.py [smollm|qwen ...] [--bin DIR] [--remote HOST]

With no model named, every model whose oracle and weights are present is run.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODELS, select, runner_command, runner_argv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def oracle_path(name):
    return os.path.join(ROOT, f"{name}_prompt.json")


def generate(cfg, ids, max_new):
    cmd = f"{runner_command(cfg)} {','.join(map(str, ids))} {max_new} -1"
    fd, out = tempfile.mkstemp()
    os.close(fd)
    try:
        with open(out, "w") as fh:
            rc = subprocess.run(runner_argv(cmd), stdout=fh,
                                stderr=subprocess.PIPE, text=True)
        if rc.returncode != 0:
            raise SystemExit(f"runner failed: {rc.stderr[:400]}")
        text = open(out).read()
    finally:
        os.unlink(out)
    got = []
    for tok in text.replace(",", " ").split():
        try:
            got.append(int(tok))
        except ValueError:
            pass
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None)
    ap.add_argument("--bin", default=None)
    ap.add_argument("--remote", default=None)
    args = ap.parse_args()
    if args.bin:
        os.environ["P2W_RUN_DIR"] = args.bin
    if args.remote:
        os.environ["P2W_REMOTE"] = args.remote

    names = args.models or list(MODELS)
    ran = failed = 0
    for name in names:
        op = oracle_path(name)
        if not os.path.exists(op):
            print(f"  {name:8s} skipped, no oracle at {os.path.basename(op)}")
            continue
        meta = json.load(open(op))
        ids = meta["ids"]
        # the setup scripts record either the whole greedy continuation or,
        # where they were run for a single step, the first token alone
        want = meta.get("greedy") or [meta["first_top1"]]
        cfg = select(name)
        got = generate(cfg, ids, len(want))
        ran += 1
        if got[:len(want)] == want:
            print(f"  {name:8s} {len(want)} token(s) identical to the oracle")
        else:
            failed += 1
            print(f"  {name:8s} DIFFERS\n      oracle {want}\n      got    {got[:len(want)]}")
    if ran == 0:
        print("  no oracle present; nothing decided")
        return 0
    if failed:
        raise SystemExit(f"{failed} of {ran} models differ from their oracle")
    print(f"  {ran} model(s) reproduce their oracle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
