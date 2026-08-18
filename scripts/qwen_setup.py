"""Fetch Qwen3.5-0.8B, save its text-model weights as an f32 safetensors with
the names the Qwen runner looks up, and capture a PyTorch oracle: the token ids
of a chat prompt, the top-eight next-token logits, and the greedy continuation.
The verified Qwen forward must reproduce this prediction.

Qwen3.5 alternates three gated-DeltaNet layers with one gated full-attention
layer. Only the text model is saved: the vision tower and the multi-token
prediction head take no part in a text-only greedy decode.
"""
import os, sys, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
# Artifacts land in the repository root unless P2W_WORK names another directory.
WORK = os.environ.get("P2W_WORK") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(WORK, exist_ok=True)
OUT = os.path.join(WORK, "qwen.safetensors")
META = os.path.join(WORK, "qwen_prompt.json")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config
tcfg = getattr(cfg, "text_config", cfg)

print("=== text config ===")
for k in ["hidden_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
          "rms_norm_eps", "tie_word_embeddings", "attn_output_gate",
          "linear_conv_kernel_dim", "linear_key_head_dim", "linear_value_head_dim",
          "linear_num_key_heads", "linear_num_value_heads"]:
    print(f"  {k} = {getattr(tcfg, k, None)}")
rope = getattr(tcfg, "rope_parameters", {}) or {}
print(f"  rope_theta = {rope.get('rope_theta')}")
print(f"  partial_rotary_factor = {rope.get('partial_rotary_factor')}")
layer_types = list(getattr(tcfg, "layer_types", []))
print(f"  layer_types = {layer_types}")

# Save the text decoder with its "model." prefix stripped. AutoModelForCausalLM
# already resolves to the text-only Qwen3_5ForCausalLM, so the vision tower and
# the multi-token prediction head are absent. lm_head is tied to the embedding
# and would only duplicate a gigabyte.
sd = model.state_dict()
tensors = {}
for k, v in sd.items():
    if k.startswith("lm_head."):
        continue
    name = k[len("model."):] if k.startswith("model.") else k
    t = v.to(torch.float32).contiguous()
    if name.endswith("conv1d.weight"):
        t = t.squeeze(1).contiguous()      # [C,1,K] -> [C,K]
    tensors[name] = t
assert tensors, "no tensors selected; state_dict layout changed"

# The rotary frequencies the model actually uses, saved alongside the weights so
# the runner does not have to raise theta to a fractional power. For text-only
# input the interleaved multimodal RoPE collapses to ordinary RoPE, because the
# three positional axes carry identical indices.
head_dim = getattr(tcfg, "head_dim", None) or tcfg.hidden_size // tcfg.num_attention_heads
prf = rope.get("partial_rotary_factor", 1.0)
rotary_dim = int(head_dim * prf)
base = rope.get("rope_theta", 10000.0)
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
tensors["rope.inv_freq"] = inv_freq.contiguous()
print(f"rotary_dim {rotary_dim}, inv_freq {tuple(inv_freq.shape)}, "
      f"first {inv_freq[:3].tolist()}, last {inv_freq[-1].item()}")

save_file(tensors, OUT)
print(f"\nsaved {len(tensors)} tensors -> {OUT} ({os.path.getsize(OUT)} bytes)")

msgs = [{"role": "user", "content": "What is the capital of France?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                              return_dict=False)
if not isinstance(ids, list):
    ids = list(ids["input_ids"])
print(f"\nchat prompt ids ({len(ids)}): {ids}")

@torch.no_grad()
def greedy(ids, n):
    out = list(ids); steps = []
    for _ in range(n):
        nxt = int(model(torch.tensor([out])).logits[0, -1].argmax())
        steps.append((nxt, tok.decode([nxt]))); out.append(nxt)
    return steps, tok.decode(out[len(ids):])

steps, cont = greedy(ids, 12)
print(f"greedy next 12: {steps}")
print(f"continuation: {cont!r}")

with torch.no_grad():
    lg = model(torch.tensor([ids])).logits[0, -1]
v, i = lg.topk(8)
print("\ntop-8 first next-token logits:")
for vv, ii in zip(v.tolist(), i.tolist()):
    print(f"  {ii:7d} {tok.decode([ii])!r:14} {vv:9.4f}")

with open(META, "w") as f:
    json.dump({"ids": ids,
               "first_top1": int(i[0]),
               "first_top1_str": tok.decode([int(i[0])]),
               "top8_ids": [int(x) for x in i.tolist()],
               "top8_logits": [float(x) for x in v.tolist()],
               "greedy": [int(s[0]) for s in steps],
               "continuation": cont,
               "rotary_dim": rotary_dim,
               "inv_freq": inv_freq.tolist(),
               "layer_types": layer_types}, f)
print(f"\nwrote {META}")
