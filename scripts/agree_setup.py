"""Held-out token windows for the agreement evaluation.

Tokenizes the WikiText-2 raw test split with a model's tokenizer, joining the
lines as the Hugging Face perplexity recipe does, cuts the token stream into
non-overlapping blocks of T tokens, keeps N blocks spread evenly over the
split, and takes the first P tokens of each kept block as a window. Each
window is one prompt: every position predicts the next token of the window.
Writes one comma-separated window per line.

  python agree_setup.py <gpt2|smollm|qwen> <out.txt> [T] [N] [P]

The defaults, T = 64, N = 300 and P = 16, give the windows of the reported
evaluation.
"""
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

TOKENIZERS = {
    "gpt2": "gpt2",
    "smollm": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen": "Qwen/Qwen3.5-0.8B",
}


def main():
    name, out = sys.argv[1], sys.argv[2]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    P = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    path = hf_hub_download("wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    tok = AutoTokenizer.from_pretrained(TOKENIZERS[name])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    n_blocks = len(ids) // T
    step = n_blocks / N
    picks = [int(i * step) for i in range(N)]
    with open(out, "w", newline="\n") as fh:
        for w in picks:
            fh.write(",".join(str(t) for t in ids[w * T:w * T + P]) + "\n")
    print(f"{name}: {len(ids)} tokens, {n_blocks} blocks of {T}, "
          f"first {P} tokens of {N} kept")


if __name__ == "__main__":
    main()
