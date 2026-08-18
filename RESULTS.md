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
| 1 | 8 | 2 | 1 | 4 | 16 | 4 | 6.029e-09 | 5.995e-08 | 0/4 | 0 |
| 2 | 8 | 2 | 1 | 4 | 16 | 4 | 9.612e-09 | 5.998e-08 | 0/4 | 0 |
| 1 | 16 | 4 | 2 | 4 | 16 | 4 | 1.402e-08 | 8.927e-08 | 0/4 | 0 |
| 1 | 8 | 2 | 1 | 8 | 16 | 4 | 7.387e-09 | 5.976e-08 | 0/4 | 0 |
| 1 | 8 | 2 | 1 | 4 | 32 | 4 | 7.397e-09 | 7.455e-08 | 0/4 | 0 |

## Qwen3.5 path

Reference: the inductive extraction of `f32_qwen_forward`.

| layers | kinds | d_model | deltanet | conv k | seq | samples | mean abs-err | max abs-err | next-token flips | nan |
|--------|-------|---------|----------|--------|-----|---------|--------------|-------------|------------------|-----|
| 1 | delta | 8 | 1x4 | 2 | 4 | 4 | 6.165e-08 | 3.622e-07 | 0/4 | 0 |
| 1 | attn | 8 | 1x4 | 2 | 4 | 4 | 9.809e-08 | 1.133e-06 | 0/4 | 0 |
| 2 | delta/attn | 8 | 1x4 | 2 | 4 | 4 | 1.468e-07 | 1.610e-06 | 0/4 | 0 |
| 1 | delta | 8 | 1x4 | 2 | 8 | 4 | 5.408e-08 | 3.572e-07 | 0/4 | 0 |
| 1 | delta | 16 | 2x4 | 3 | 4 | 4 | 1.488e-07 | 7.751e-07 | 0/4 | 0 |
