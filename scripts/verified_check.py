"""The extracted forward pass itself, on each checkpoint, against its oracle.

The runners named here call f32_gpt2_logits_pre, f32_llama_forward and
f32_qwen_forward, so the composition they run is extracted code rather than a
loop rebuilt in OCaml. This compares what they return on the stored prompt with
the PyTorch oracle the setup scripts recorded: the highest-scoring token for
every model, and the eight highest for Qwen3.5, which is the depth its oracle
holds.

  python verified_check.py [gpt2|smollm|qwen ...] [--bin DIR] [--remote HOST]
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# runner, weight file, and the arguments it takes before the token ids
RUNNERS = {
    "gpt2":   ("gpt2_verified", "gpt2.safetensors",
               ["768", "12", "12", "3072", "50257", "1024"]),
    "smollm": ("llama_verified", "smollm.safetensors",
               ["576", "30", "9", "3", "1536", "49152"]),
    "qwen":   ("qwen_verified", "qwen.safetensors",
               ["1024", "24", "8", "2", "256", "64", "3584", "248320",
                "16", "128", "4"]),
}

# GPT-2's oracle is not in the repository; its prompt is "The quick brown",
# whose highest-scoring next token the setup script records as 494.
FALLBACK = {"gpt2": {"ids": [464, 2068, 7586], "first_top1": 494}}


def oracle(name):
    path = os.path.join(ROOT, f"{name}_prompt.json")
    if os.path.exists(path):
        return json.load(open(path))
    return FALLBACK.get(name)


def run(bindir, remote, runner, weights, args, ids):
    cmd = (f"cd {bindir} && ./{runner} {weights} " + " ".join(args) + " "
           + ",".join(str(i) for i in ids))
    argv = ["sh", "-c", cmd] if not remote else ["ssh", remote, cmd]
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{runner} failed: {out.stderr[-400:]}")
    got = []
    for line in out.stdout.splitlines():
        m = re.match(r"\s*\d+\s+(\d+)\s+(-?[\d.eE+]+)\s*$", line)
        if m:
            got.append((int(m.group(1)), float(m.group(2))))
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None)
    ap.add_argument("--bin", default=".")
    ap.add_argument("--remote", default=None)
    args = ap.parse_args()

    names = args.models or list(RUNNERS)
    ran = failed = 0
    for name in names:
        runner, weights, cfg = RUNNERS[name]
        meta = oracle(name)
        if meta is None:
            print(f"  {name:8s} skipped, no oracle")
            continue
        got = run(args.bin, args.remote, runner, weights, cfg, meta["ids"])
        if not got:
            failed += 1
            print(f"  {name:8s} produced no logits")
            continue
        ran += 1
        want = meta.get("top8_ids") or [meta["first_top1"]]
        have = [t for t, _ in got][:len(want)]
        if have == want:
            print(f"  {name:8s} top {len(want)} identical to the oracle")
        else:
            failed += 1
            print(f"  {name:8s} DIFFERS\n      oracle {want}\n      got    {have}")
    if ran == 0:
        print("  no runner present; nothing decided")
        return 0
    if failed:
        raise SystemExit(f"{failed} of {ran} differ from their oracle")
    print(f"  {ran} model(s) reproduce their oracle through the extracted pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
