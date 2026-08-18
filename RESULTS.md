# Differential test: numpy float32 vs verified IEEE-754 reference

Each row is a model configuration. The reference is the extracted
verified forward; numpy runs the same elementwise math with its own
reduction order. `abs-err` is over every logit of every sample.

| layers | d_model | heads | seq | vocab | samples | mean abs-err | max abs-err | next-token flips | nan |
|--------|---------|-------|-----|-------|---------|--------------|-------------|------------------|-----|
| 1 | 8 | 2 | 8 | 16 | 16 | 4.113e-09 | 4.434e-08 | 0/16 | 0 |
| 2 | 8 | 2 | 8 | 16 | 16 | 4.489e-09 | 3.021e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 8 | 16 | 16 | 5.920e-09 | 1.415e-07 | 0/16 | 0 |
| 4 | 8 | 2 | 8 | 64 | 16 | 5.570e-09 | 8.566e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 16 | 16 | 16 | 5.128e-09 | 4.502e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 32 | 16 | 16 | 4.989e-09 | 4.847e-08 | 0/16 | 0 |
| 4 | 16 | 4 | 8 | 16 | 16 | 1.233e-08 | 1.267e-07 | 0/16 | 0 |
| 4 | 32 | 8 | 8 | 16 | 16 | 3.657e-08 | 3.306e-07 | 0/16 | 0 |
| 4 | 64 | 8 | 8 | 16 | 16 | 1.123e-07 | 8.938e-07 | 0/16 | 0 |
| 8 | 8 | 2 | 8 | 16 | 16 | 6.623e-09 | 9.681e-08 | 0/16 | 0 |

## Llama path

Reference: the inductive extraction of `f32_llama_forward`.

| layers | d_model | heads | kv heads | seq | vocab | samples | mean abs-err | max abs-err | next-token flips | nan |
|--------|---------|-------|----------|-----|-------|---------|--------------|-------------|------------------|-----|
| 1 | 8 | 2 | 1 | 8 | 16 | 16 | 7.014e-09 | 1.196e-07 | 0/16 | 0 |
| 2 | 8 | 2 | 1 | 8 | 16 | 16 | 6.757e-09 | 5.999e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 1 | 8 | 16 | 16 | 9.560e-09 | 8.981e-08 | 0/16 | 0 |
| 2 | 16 | 4 | 2 | 8 | 16 | 16 | 1.613e-08 | 1.495e-07 | 0/16 | 0 |
| 2 | 32 | 8 | 4 | 8 | 16 | 16 | 4.692e-08 | 3.129e-07 | 0/16 | 0 |
| 2 | 8 | 2 | 1 | 16 | 16 | 16 | 7.624e-09 | 8.990e-08 | 0/16 | 0 |
| 2 | 8 | 2 | 1 | 8 | 64 | 16 | 7.505e-09 | 1.491e-07 | 0/16 | 0 |

## Qwen3.5 path

Reference: the inductive extraction of `f32_qwen_forward`. The four-layer row
carries the layer pattern the real model uses, three gated DeltaNet blocks
followed by one gated full-attention block.

| layers | kinds | d_model | deltanet | conv k | seq | samples | mean abs-err | max abs-err | next-token flips | nan |
|--------|-------|---------|----------|--------|-----|---------|--------------|-------------|------------------|-----|
| 1 | delta | 8 | 1x4 | 2 | 8 | 16 | 1.049e-07 | 3.741e-06 | 0/16 | 0 |
| 1 | attn | 8 | 1x4 | 2 | 8 | 16 | 1.133e-07 | 3.308e-06 | 0/16 | 0 |
| 4 | delta/delta/delta/attn | 8 | 1x4 | 2 | 8 | 16 | 3.100e-07 | 5.543e-06 | 0/16 | 0 |
| 1 | delta | 8 | 1x4 | 2 | 16 | 16 | 9.046e-08 | 2.832e-06 | 0/16 | 0 |
| 2 | delta/attn | 16 | 2x4 | 3 | 8 | 16 | 5.848e-07 | 8.047e-06 | 0/16 | 0 |
| 2 | delta/attn | 32 | 4x4 | 3 | 8 | 16 | 1.074e-06 | 3.248e-05 | 0/16 | 0 |
