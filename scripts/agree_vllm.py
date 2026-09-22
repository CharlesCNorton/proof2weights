"""vLLM logits for the agreement evaluation.

Runs the Hugging Face checkpoint through vLLM, in bfloat16 by default, on each
window of a windows file and writes <outdir>/<index>.f32: the logits of every
position, as little-endian float32, positions in order, the layout the
runners' dump mode writes. vLLM computes logits for the last position of a
prompt only, so each window becomes one request per prefix, and a logits
processor copies the row the sampler receives for the prefix of p + 1 tokens
into row p of the window's dump. Prefix caching is off, so every prefix is a
full prefill, and the engine runs in this process so the processor can keep
the rows.

  python agree_vllm.py <gpt2|smollm|qwen> <windows.txt> <outdir> [first] [last] [bf16|f16|f32]
"""
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

import numpy as np
from vllm import LLM, SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessor, process_dict_updates

MODELS = {
    "gpt2": "gpt2",
    "smollm": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen": "Qwen/Qwen3.5-0.8B",
}


class Capture(LogitsProcessor):
    """Keeps the sampler's logits row of every tagged request, keyed by (window, position)."""

    rows = {}

    def __init__(self, vllm_config, device, is_pin_memory):
        self.tags = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        process_dict_updates(
            self.tags, batch_update,
            lambda params, prompt, out: params.extra_args.get("tag") if params.extra_args else None)

    def apply(self, logits):
        for index, tag in self.tags.items():
            Capture.rows[tag] = logits[index].to("cpu", copy=True).numpy().astype("<f4")
        return logits


def main():
    name, windows, outdir = sys.argv[1:4]
    lines = [l.strip() for l in open(windows) if l.strip()]
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    last = int(sys.argv[5]) if len(sys.argv) > 5 else len(lines)
    dt = sys.argv[6] if len(sys.argv) > 6 else "bf16"
    dtype = {"bf16": "bfloat16", "f16": "float16", "f32": "float32"}[dt]
    T = max(len(l.split(",")) for l in lines)
    os.makedirs(outdir, exist_ok=True)
    llm = LLM(model=MODELS[name], dtype=dtype, max_model_len=T + 1,
              enable_prefix_caching=False, gpu_memory_utilization=0.5,
              logits_processors=[Capture], seed=0)
    for w in range(first, min(last, len(lines))):
        toks = [int(t) for t in lines[w].split(",")]
        prompts = [{"prompt_token_ids": toks[:p + 1]} for p in range(len(toks))]
        params = [SamplingParams(max_tokens=1, temperature=0.0, detokenize=False,
                                 extra_args={"tag": (w, p)}) for p in range(len(toks))]
        Capture.rows.clear()
        llm.generate(prompts, params, use_tqdm=False)
        out = np.stack([Capture.rows[(w, p)] for p in range(len(toks))])
        out.astype("<f4").tofile(os.path.join(outdir, f"{w}.f32"))
    print(f"{name} vllm {dt}: windows {first}..{min(last, len(lines)) - 1}")


if __name__ == "__main__":
    main()
