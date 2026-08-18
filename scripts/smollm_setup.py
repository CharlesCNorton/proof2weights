"""Fetch SmolLM2-135M-Instruct (Llama architecture), report its config and
tensor names, save its weights as an f32 safetensors with clean names, and
capture a PyTorch oracle: the token ids of a chat prompt and the greedy
continuation. The verified Llama forward must reproduce this prediction."""
import os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

# Optional args: <model id> <tag>. Default is the 135M model with no tag.
MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M-Instruct"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
# Artifacts land in the repository root unless P2W_WORK names another directory.
WORK = os.environ.get("P2W_WORK") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(WORK, exist_ok=True)
OUT = os.path.join(WORK, f"smollm{TAG}.safetensors")
META = os.path.join(WORK, f"smollm{TAG}_prompt.json")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
cfg = model.config

print("=== config ===")
for k in ["hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads",
          "intermediate_size","vocab_size","rope_theta","rms_norm_eps","hidden_act",
          "tie_word_embeddings","head_dim","attention_bias","max_position_embeddings"]:
    print(f"  {k} = {getattr(cfg, k, None)}")

print("=== state_dict tensor names (first layer + global) ===")
sd = model.state_dict()
for k in sd:
    if k.startswith("model.layers.0.") or not k.startswith("model.layers."):
        print(f"  {k}  {tuple(sd[k].shape)}")

# Save weights with clean names (strip "model." prefix; drop tied lm_head and
# any rotary buffers, which the forward recomputes).
tensors = {}
for k, v in sd.items():
    if "rotary" in k or "inv_freq" in k:
        continue
    if k == "lm_head.weight":
        continue
    name = k[len("model."):] if k.startswith("model.") else k
    tensors[name] = v.to(torch.float32).contiguous()
# Save the exact rotary frequencies the model uses, so the verified forward
# computes RoPE angles from the same inv_freq rather than reconstructing theta.
inv_freq = model.model.rotary_emb.inv_freq.detach().to(torch.float32).contiguous()
tensors["rope.inv_freq"] = inv_freq
print(f"\nrope.inv_freq shape {tuple(inv_freq.shape)}; first {inv_freq[:3].tolist()}, last {inv_freq[-1].item()}")
save_file(tensors, OUT)
print(f"\nsaved {len(tensors)} tensors -> {OUT} ({os.path.getsize(OUT)} bytes)")

# Chat prompt via the model's template.
msgs = [{"role": "user", "content": "What is the capital of France?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
if not isinstance(ids, list):
    ids = list(ids["input_ids"])
print(f"\nchat prompt ids ({len(ids)}): {ids}")
print("prompt text:\n" + tok.decode(ids))

@torch.no_grad()
def greedy(ids, n):
    out = list(ids); steps = []
    for _ in range(n):
        nxt = int(model(torch.tensor([out])).logits[0, -1].argmax())
        steps.append((nxt, tok.decode([nxt]))); out.append(nxt)
    return steps, tok.decode(out[len(ids):])

steps, cont = greedy(ids, 12)
print(f"\ngreedy next 12: {steps}")
print(f"continuation: {cont!r}")

# Top-8 next-token logits at the first step, the oracle for the verified forward.
with torch.no_grad():
    lg = model(torch.tensor([ids])).logits[0, -1]
v, i = lg.topk(8)
print("\ntop-8 first next-token logits:")
for vv, ii in zip(v.tolist(), i.tolist()):
    print(f"  {ii:6d} {tok.decode([ii])!r:14} {vv:8.4f}")

with open(META, "w") as f:
    json.dump({"ids": ids, "first_top1": int(i[0]), "first_top1_str": tok.decode([int(i[0])])}, f)
print(f"\nwrote {META}")
