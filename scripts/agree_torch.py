"""PyTorch float32 logits for the agreement evaluation.

Runs the Hugging Face model in float32 on each window of a windows file and
writes <outdir>/<index>.f32: the logits of every position, as little-endian
float32, positions in order, the layout the runners' dump mode writes. On CUDA,
TF32 is disabled so matrix products stay in float32.

  python agree_torch.py <gpt2|smollm|qwen> <windows.txt> <outdir> <cpu|cuda>
                        [first] [last] [f32|bf16|f16]

The dtype argument names what the model runs in; the logits are written as
float32 either way, so a bf16 run is comparable with the float32 one entry for
entry. bf16 is what deployments use, and it is the row the float32 comparison
does not reach.
"""
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODELS = {
    "gpt2": "gpt2",
    "smollm": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen": "Qwen/Qwen3.5-0.8B",
}


def main():
    name, windows, outdir, device = sys.argv[1:5]
    lines = [l.strip() for l in open(windows) if l.strip()]
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    last = int(sys.argv[6]) if len(sys.argv) > 6 else len(lines)
    dt = sys.argv[7] if len(sys.argv) > 7 else "f32"
    torch_dt = {"f32": torch.float32, "bf16": torch.bfloat16,
                "f16": torch.float16}[dt]
    os.makedirs(outdir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[name], dtype=torch_dt, attn_implementation="eager").to(device).eval()
    with torch.no_grad():
        for w in range(first, min(last, len(lines))):
            ids = torch.tensor([[int(t) for t in lines[w].split(",")]], device=device)
            logits = model(ids).logits[0].to(torch.float32).cpu().numpy()
            logits.astype("<f4").tofile(os.path.join(outdir, f"{w}.f32"))
    print(f"{name} {device} {dt}: windows {first}..{min(last, len(lines)) - 1}")


if __name__ == "__main__":
    main()
