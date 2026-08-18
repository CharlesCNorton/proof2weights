"""Fetch real GPT-2 (124M), re-save its base weights as an f32 safetensors with
the exact tensor names the Coq loader builds, and confirm with the torch
reference that greedy decoding reproduces the brown-fox pangram. The torch
continuation is the oracle the verified extracted forward must match."""
import os, json
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from safetensors.torch import save_file

# Artifacts land in the repository root unless P2W_WORK names another directory.
WORK = os.environ.get("P2W_WORK") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(WORK, exist_ok=True)
OUT = os.path.join(WORK, "gpt2.safetensors")
IDS = os.path.join(WORK, "gpt2_prompt.json")

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

# Save base-transformer weights as f32, dropping the causal-mask buffers the
# Coq loader never looks up. Names match wte/wpe/h.<i>.*/ln_f.* exactly.
sd = model.transformer.state_dict()
tensors = {}
for k, v in sd.items():
    if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
        continue
    tensors[k] = v.to(torch.float32).contiguous()
save_file(tensors, OUT)
print(f"saved {len(tensors)} tensors -> {OUT} ({os.path.getsize(OUT)} bytes)")

@torch.no_grad()
def greedy(prompt, n):
    ids = tok.encode(prompt)
    out = list(ids)
    steps = []
    for _ in range(n):
        logits = model(torch.tensor([out])).logits[0, -1]
        nxt = int(logits.argmax())
        steps.append((nxt, tok.decode([nxt])))
        out.append(nxt)
    return ids, steps, tok.decode(out)

for prompt in ["The quick brown", "The quick brown fox jumps over the lazy"]:
    ids, steps, full = greedy(prompt, 5)
    print(f"\nprompt {prompt!r}")
    print(f"  token ids: {ids}")
    print(f"  greedy next 5: {steps}")
    print(f"  -> {full!r}")

# Pin the single-step target for the verified runner: prompt "The quick brown",
# expected next token " fox".
ids = tok.encode("The quick brown")
with open(IDS, "w") as f:
    json.dump({"prompt": "The quick brown", "ids": ids,
               "id_to_str_sample": {str(i): tok.decode([i]) for i in ids}}, f)
print(f"\nwrote prompt ids {ids} -> {IDS}")
