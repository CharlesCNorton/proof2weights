"""The gain each linear layer applies to an absolute error on its input.

A linear layer maps an input error bounded componentwise by eps to an output
error bounded by (max row sum of |W|) eps, and that bound is attained by the
error vector whose signs match a row of W. It is therefore not slack in the
analysis: it is the layer's own sensitivity in the infinity norm, and any
forward bound that holds for every input error pattern pays it.

This prints the max row sum of each linear layer of a checkpoint, which is what
the annotated pass of RunErr.v grows by from one layer to the next.

  python layer_gain.py [gpt2.safetensors]
"""
import sys

import numpy as np
from safetensors.numpy import load_file

# GPT-2 stores its linear layers transposed, as (in, out); llama and qwen do not.
TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight",
              "mlp.c_proj.weight")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "gpt2.safetensors"
    t = load_file(path)
    rows = []
    for name, w in sorted(t.items()):
        if w.ndim != 2 or not name.endswith(".weight"):
            continue
        if any(name.endswith(s) for s in TRANSPOSED):
            w = w.T
        gain = float(np.abs(w.astype(np.float64)).sum(axis=1).max())
        rows.append((name, w.shape, gain))

    print("%-34s %14s %12s" % ("layer", "shape", "max row sum"))
    for name, shape, gain in rows:
        print("%-34s %14s %12.2f" % (name, "%dx%d" % shape, gain))

    per_block = {}
    for name, _, gain in rows:
        if name.startswith("h."):
            per_block.setdefault(name.split(".")[1], []).append(gain)
    if per_block:
        prod = [np.prod(v) for v in per_block.values()]
        print("\nproduct of the gains within one block: min %.3g, median %.3g, "
              "max %.3g" % (min(prod), float(np.median(prod)), max(prod)))


if __name__ == "__main__":
    main()
