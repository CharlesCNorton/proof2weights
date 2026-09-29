"""PyTorch float32 logits for the agreement evaluation.

Runs the Hugging Face model in float32 on each window of a windows file and
writes <outdir>/<index>.f32: the logits of every position, as little-endian
float32, positions in order, the layout the runners' dump mode writes. On CUDA,
TF32 is disabled so matrix products stay in float32.

  python agree_torch.py <gpt2|smollm|qwen> <windows.txt> <outdir> <cpu|cuda>
                        [first] [last] [f32|bf16|f16|f64]

The dtype argument names what the model runs in; the logits are written as
float32 for f32, bf16 and f16, so a bf16 run is comparable with the float32 one
entry for entry. bf16 is what deployments use, and it is the row the float32
comparison does not reach. f64 runs the model in float64 and writes
<outdir>/<index>.f64, little-endian binary64, the reference against which each
float32 implementation's distance from exact arithmetic is measured.

flash-linear-attention's Triton kernels take float32 dot products in TF32
unless told otherwise: Triton's default for tl.dot on float32 operands, and a
TF32 triangular solve in the chunked gated delta rule. P2W_IEEE_DOTS=1 makes
both IEEE and compiles into a Triton cache of its own.
"""
import os
import sys

IEEE_DOTS = os.environ.get("P2W_IEEE_DOTS") == "1"
if IEEE_DOTS:
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        os.path.expanduser("~"), ".triton", "cache_ieee")

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
                "f16": torch.float16, "f64": torch.float64}[dt]
    os.makedirs(outdir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[name], dtype=torch_dt, attn_implementation="eager").to(device).eval()
    if IEEE_DOTS:
        import triton.language as tl
        import fla.ops.gated_delta_rule.chunk_fwd as chunk_fwd
        chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    with torch.no_grad():
        for w in range(first, min(last, len(lines))):
            ids = torch.tensor([[int(t) for t in lines[w].split(",")]], device=device)
            logits = model(ids).logits[0]
            if dt == "f64":
                logits.cpu().numpy().astype("<f8").tofile(os.path.join(outdir, f"{w}.f64"))
            else:
                logits.to(torch.float32).cpu().numpy().astype("<f4").tofile(
                    os.path.join(outdir, f"{w}.f32"))
    print(f"{name} {device} {dt}: windows {first}..{min(last, len(lines)) - 1}")


if __name__ == "__main__":
    main()
