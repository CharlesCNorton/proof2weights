# Differential test: numpy float32 vs verified IEEE-754 reference

Each row is a model configuration. The reference is the extracted
verified forward; numpy runs the same elementwise math with its own
reduction order. `abs-err` is over every logit of every sample.

| layers | d_model | heads | seq | vocab | samples | mean abs-err | max abs-err | next-token flips | nan |
|--------|---------|-------|-----|-------|---------|--------------|-------------|------------------|-----|
| 1 | 8 | 2 | 8 | 16 | 16 | 4.308e-09 | 4.462e-08 | 0/16 | 0 |
| 2 | 8 | 2 | 8 | 16 | 16 | 4.654e-09 | 4.440e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 8 | 16 | 16 | 5.144e-09 | 4.445e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 8 | 64 | 16 | 5.082e-09 | 3.724e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 16 | 16 | 16 | 4.960e-09 | 3.353e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 32 | 16 | 16 | 4.886e-09 | 3.725e-08 | 0/16 | 0 |
| 4 | 16 | 4 | 8 | 16 | 16 | 1.094e-08 | 5.982e-08 | 0/16 | 0 |
| 4 | 32 | 8 | 8 | 16 | 16 | 2.635e-08 | 1.415e-07 | 0/16 | 0 |
| 4 | 64 | 8 | 8 | 16 | 16 | 6.419e-08 | 3.278e-07 | 0/16 | 0 |
| 8 | 8 | 2 | 8 | 16 | 16 | 6.069e-09 | 3.017e-08 | 0/16 | 0 |

## Llama path

Reference: the inductive extraction of `f32_llama_forward`.

| layers | d_model | heads | kv heads | seq | vocab | samples | mean abs-err | max abs-err | next-token flips | nan |
|--------|---------|-------|----------|-----|-------|---------|--------------|-------------|------------------|-----|
| 1 | 8 | 2 | 1 | 8 | 16 | 16 | 7.312e-09 | 1.189e-07 | 0/16 | 0 |
| 2 | 8 | 2 | 1 | 8 | 16 | 16 | 6.175e-09 | 8.938e-08 | 0/16 | 0 |
| 4 | 8 | 2 | 1 | 8 | 16 | 16 | 9.149e-09 | 1.196e-07 | 0/16 | 0 |
| 2 | 16 | 4 | 2 | 8 | 16 | 16 | 1.577e-08 | 1.188e-07 | 0/16 | 0 |
| 2 | 32 | 8 | 4 | 8 | 16 | 16 | 4.298e-08 | 2.389e-07 | 0/16 | 0 |
| 2 | 8 | 2 | 1 | 16 | 16 | 16 | 7.496e-09 | 8.987e-08 | 0/16 | 0 |
| 2 | 8 | 2 | 1 | 8 | 64 | 16 | 7.622e-09 | 1.195e-07 | 0/16 | 0 |

## Qwen3.5 path

Reference: the inductive extraction of `f32_qwen_forward`.

| layers | kinds | d_model | deltanet | conv k | seq | samples | mean abs-err | max abs-err | next-token flips | nan |
|--------|-------|---------|----------|--------|-----|---------|--------------|-------------|------------------|-----|
| 1 | delta | 8 | 1x4 | 2 | 8 | 16 | 8.168e-08 | 4.812e-07 | 0/16 | 0 |
| 1 | attn | 8 | 1x4 | 2 | 8 | 16 | 8.296e-08 | 6.852e-07 | 0/16 | 0 |
| 4 | delta/delta/delta/attn | 8 | 1x4 | 2 | 8 | 16 | 1.656e-07 | 3.234e-06 | 0/16 | 0 |
| 1 | delta | 8 | 1x4 | 2 | 16 | 16 | 7.563e-08 | 5.810e-07 | 0/16 | 0 |
| 2 | delta/attn | 16 | 2x4 | 3 | 8 | 16 | 2.785e-07 | 2.280e-06 | 0/16 | 0 |
| 2 | delta/attn | 32 | 4x4 | 3 | 8 | 16 | 5.017e-07 | 3.457e-06 | 0/16 | 0 |

## Verified bound against the actual error (GPT-2 path)

Bounds from `runners/gpt2_bound_ref`, the inductive extraction of the annotated
forward pass `gpt2_logits_bounded` is stated about. The actual error is the
distance from the binary32 logit to a float64 evaluation of the reference
network: exact arithmetic, true exponential of the saturated argument, program
constants as stored.

| layers | d_model | heads | seq | vocab | samples | logits bounded | max bound | max actual error | median bound/error | exceeded |
|--------|---------|-------|-----|-------|---------|----------------|-----------|------------------|--------------------|----------|
| 1 | 8 | 2 | 8 | 16 | 16 | 2048/2048 | 1.838e-05 | 3.444e-08 | 970 | 0 |
| 2 | 8 | 2 | 8 | 16 | 16 | 2048/2048 | 6.251e-04 | 2.636e-08 | 1.95e+04 | 0 |
| 4 | 8 | 2 | 8 | 16 | 16 | 1664/2048 | 1.855e+00 | 3.127e-08 | 7.33e+06 | 0 |
| 4 | 8 | 2 | 8 | 64 | 16 | 7232/8192 | 7.379e-01 | 2.860e-08 | 9.31e+06 | 0 |
| 4 | 8 | 2 | 16 | 16 | 16 | 3632/4096 | 3.805e+00 | 3.091e-08 | 7.99e+06 | 0 |
| 4 | 8 | 2 | 32 | 16 | 16 | 7248/8192 | 9.984e-01 | 3.408e-08 | 8.24e+06 | 0 |
| 4 | 16 | 4 | 8 | 16 | 16 | 0/2048 | - | - | - | 0 |
| 4 | 32 | 8 | 8 | 16 | 16 | 0/2048 | - | - | - | 0 |
| 4 | 64 | 8 | 8 | 16 | 16 | 0/2048 | - | - | - | 0 |
| 8 | 8 | 2 | 8 | 16 | 16 | 0/2048 | - | - | - | 0 |

## Agreement on held-out text: gpt2

300 windows of 16 tokens from the WikiText-2 raw test split, every position compared.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / PyTorch CPU | 4800/4800 | 4794/4800 | 5.23% | 3.9e-05 | 5.2e-04 | 1.57e-03 | 6.6e-10 | 7.8e-09 | - |
| extracted / PyTorch CUDA | 4800/4800 | 4791/4800 | 5.06% | 4.7e-05 | 5.1e-04 | 1.51e-03 | 7.7e-10 | 7.4e-09 | - |
| extracted / llama.cpp CPU | 4798/4800 | 4685/4800 | 0.03% | 6.6e-03 | 1.8e-01 | 3.12e-01 | 3.9e-07 | 4.4e-05 | 9.16e-04 |
| extracted / llama.cpp CUDA | 4793/4800 | 4039/4800 | 0.01% | 2.5e-02 | 8.9e-01 | 1.70e+00 | 1.9e-05 | 2.8e-04 | 2.23e-02 |
| PyTorch CPU / PyTorch CUDA | 4800/4800 | 4793/4800 | 11.58% | 1.5e-05 | 2.8e-04 | 4.96e-04 | 1.3e-10 | 1.3e-09 | - |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 4500 | 246.928477 | - |
| PyTorch CPU | 4500 | 246.928159 | 2.59e-04 |
| PyTorch CUDA | 4500 | 246.928159 | 3.20e-04 |
| llama.cpp CPU | 4500 | 246.929787 | 1.54e-02 |
| llama.cpp CUDA | 4500 | 246.898478 | 3.82e-02 |

## llama.cpp configurations against the extracted forward pass: gpt2

300 windows of 16 tokens from the WikiText-2 raw test split, every position compared. Cache is the key/value cache type; the GELU table is ggml-cpu's float16 table (GGML_GELU_FP16); TF32 is the cuBLAS math mode ggml-cuda sets, disabled with NVIDIA_TF32_OVERRIDE=0; mul_mat_f is ggml-cuda's kernel for float matrix products over at most 16 tokens, removed from ggml_cuda_mul_mat in the builds without it.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / CPU, f16 cache, GELU table | 4798/4800 | 4685/4800 | 0.03% | 6.6e-03 | 1.8e-01 | 3.12e-01 | 3.9e-07 | 4.4e-05 | 9.16e-04 |
| extracted / CPU, f32 cache, GELU table | 4797/4800 | 4709/4800 | 0.05% | 4.8e-03 | 1.2e-01 | 2.83e-01 | 2.0e-07 | 4.9e-06 | 2.44e-03 |
| extracted / CPU, f16 cache, no GELU table | 4800/4800 | 4721/4800 | 0.06% | 4.4e-03 | 1.0e-01 | 2.17e-01 | 1.7e-07 | 8.8e-06 | - |
| extracted / CPU, f32 cache, no GELU table | 4800/4800 | 4793/4800 | 5.26% | 3.9e-05 | 5.6e-04 | 1.79e-03 | 6.8e-10 | 6.8e-09 | - |
| extracted / GPU, f16 cache, TF32, mul_mat_f | 4793/4800 | 4039/4800 | 0.01% | 2.5e-02 | 8.9e-01 | 1.70e+00 | 1.9e-05 | 2.8e-04 | 2.23e-02 |
| extracted / GPU, f32 cache, TF32, mul_mat_f | 4792/4800 | 4011/4800 | 0.01% | 2.6e-02 | 9.5e-01 | 1.37e+00 | 2.0e-05 | 3.7e-04 | 2.23e-02 |
| extracted / GPU, f16 cache, no TF32, mul_mat_f | 4795/4800 | 4400/4800 | 0.01% | 2.2e-02 | 8.9e-01 | 1.70e+00 | 5.4e-06 | 2.9e-04 | 1.18e-02 |
| extracted / GPU, f32 cache, no TF32, mul_mat_f | 4793/4800 | 4364/4800 | 0.01% | 2.4e-02 | 9.3e-01 | 1.37e+00 | 6.6e-06 | 3.1e-04 | 7.36e-03 |
| extracted / GPU, f16 cache, TF32, no mul_mat_f | 4789/4800 | 4109/4800 | 0.01% | 1.5e-02 | 2.5e-01 | 8.53e-01 | 1.5e-05 | 1.6e-04 | 1.33e-02 |
| extracted / GPU, f32 cache, TF32, no mul_mat_f | 4789/4800 | 4113/4800 | 0.02% | 1.4e-02 | 2.0e-01 | 8.04e-01 | 1.4e-05 | 1.5e-04 | 2.23e-02 |
| extracted / GPU, f16 cache, no TF32, no mul_mat_f | 4799/4800 | 4713/4800 | 0.06% | 4.2e-03 | 8.7e-02 | 1.20e-01 | 1.8e-07 | 2.3e-05 | 2.21e-03 |
| extracted / GPU, f32 cache, no TF32, no mul_mat_f | 4800/4800 | 4791/4800 | 5.06% | 4.7e-05 | 5.1e-04 | 1.34e-03 | 7.7e-10 | 7.5e-09 | - |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 4500 | 246.928477 | - |
| CPU, f16 cache, GELU table | 4500 | 246.929787 | 1.54e-02 |
| CPU, f32 cache, GELU table | 4500 | 246.925949 | 7.56e-03 |
| CPU, f16 cache, no GELU table | 4500 | 246.927292 | 1.26e-02 |
| CPU, f32 cache, no GELU table | 4500 | 246.928129 | 2.90e-04 |
| GPU, f16 cache, TF32, mul_mat_f | 4500 | 246.898478 | 3.82e-02 |
| GPU, f32 cache, TF32, mul_mat_f | 4500 | 246.931036 | 4.81e-02 |
| GPU, f16 cache, no TF32, mul_mat_f | 4500 | 246.841989 | 3.58e-02 |
| GPU, f32 cache, no TF32, mul_mat_f | 4500 | 246.873136 | 3.47e-02 |
| GPU, f16 cache, TF32, no mul_mat_f | 4500 | 246.985689 | 3.42e-02 |
| GPU, f32 cache, TF32, no mul_mat_f | 4500 | 246.969572 | 3.41e-02 |
| GPU, f16 cache, no TF32, no mul_mat_f | 4500 | 246.924304 | 7.28e-03 |
| GPU, f32 cache, no TF32, no mul_mat_f | 4500 | 246.928160 | 3.20e-04 |

## Agreement on held-out text: smollm

300 windows of 16 tokens from the WikiText-2 raw test split, every position compared.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / PyTorch CPU | 4800/4800 | 4797/4800 | 1.36% | 1.3e-05 | 2.8e-04 | 5.10e-04 | 4.8e-11 | 8.8e-10 | - |
| extracted / PyTorch CUDA | 4800/4800 | 4797/4800 | 1.38% | 1.2e-05 | 3.7e-04 | 6.93e-04 | 5.0e-11 | 1.4e-09 | - |
| extracted / llama.cpp CPU | 4797/4800 | 4644/4800 | 0.01% | 1.9e-03 | 3.1e-02 | 9.33e-02 | 7.7e-07 | 6.7e-05 | 4.03e-03 |
| extracted / llama.cpp CUDA | 4793/4800 | 4373/4800 | 0.00% | 6.5e-03 | 7.9e-02 | 2.25e-01 | 5.9e-06 | 1.1e-03 | 6.70e-03 |
| PyTorch CPU / PyTorch CUDA | 4800/4800 | 4800/4800 | 3.10% | 5.8e-06 | 9.5e-05 | 1.87e-04 | 1.2e-11 | 1.7e-10 | - |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 4500 | 272.438622 | - |
| PyTorch CPU | 4500 | 272.438619 | 8.72e-05 |
| PyTorch CUDA | 4500 | 272.438619 | 9.14e-05 |
| llama.cpp CPU | 4500 | 272.428655 | 1.55e-02 |
| llama.cpp CUDA | 4500 | 272.325178 | 5.86e-02 |

## llama.cpp configurations against the extracted forward pass: smollm

300 windows of 16 tokens from the WikiText-2 raw test split, every position compared. Cache is the key/value cache type; the GELU table is ggml-cpu's float16 table (GGML_GELU_FP16); TF32 is the cuBLAS math mode ggml-cuda sets, disabled with NVIDIA_TF32_OVERRIDE=0; mul_mat_f is ggml-cuda's kernel for float matrix products over at most 16 tokens, removed from ggml_cuda_mul_mat in the builds without it.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / CPU, f16 cache, GELU table | 4797/4800 | 4644/4800 | 0.01% | 1.9e-03 | 3.1e-02 | 9.33e-02 | 7.7e-07 | 6.7e-05 | 4.03e-03 |
| extracted / CPU, f32 cache, GELU table | 4800/4800 | 4797/4800 | 1.40% | 1.2e-05 | 3.4e-04 | 6.31e-04 | 4.9e-11 | 1.1e-09 | - |
| extracted / CPU, f16 cache, no GELU table | 4797/4800 | 4644/4800 | 0.01% | 1.9e-03 | 3.1e-02 | 9.33e-02 | 7.7e-07 | 6.7e-05 | 4.03e-03 |
| extracted / CPU, f32 cache, no GELU table | 4800/4800 | 4797/4800 | 1.40% | 1.2e-05 | 3.4e-04 | 6.31e-04 | 4.9e-11 | 1.1e-09 | - |
| extracted / GPU, f16 cache, TF32, mul_mat_f | 4793/4800 | 4373/4800 | 0.00% | 6.5e-03 | 7.9e-02 | 2.25e-01 | 5.9e-06 | 1.1e-03 | 6.70e-03 |
| extracted / GPU, f32 cache, TF32, mul_mat_f | 4788/4800 | 4308/4800 | 0.00% | 8.3e-03 | 9.3e-02 | 3.49e-01 | 1.0e-05 | 1.9e-03 | 2.52e-02 |
| extracted / GPU, f16 cache, no TF32, mul_mat_f | 4793/4800 | 4373/4800 | 0.00% | 6.5e-03 | 7.9e-02 | 2.25e-01 | 5.9e-06 | 1.1e-03 | 6.70e-03 |
| extracted / GPU, f32 cache, no TF32, mul_mat_f | 4788/4800 | 4308/4800 | 0.00% | 8.3e-03 | 9.3e-02 | 3.49e-01 | 1.0e-05 | 1.9e-03 | 2.52e-02 |
| extracted / GPU, f16 cache, TF32, no mul_mat_f | 4797/4800 | 4487/4800 | 0.00% | 4.9e-03 | 6.3e-02 | 2.01e-01 | 3.2e-06 | 6.7e-04 | 7.29e-04 |
| extracted / GPU, f32 cache, TF32, no mul_mat_f | 4795/4800 | 4511/4800 | 0.00% | 4.5e-03 | 5.8e-02 | 2.00e-01 | 2.7e-06 | 7.1e-04 | 4.03e-03 |
| extracted / GPU, f16 cache, no TF32, no mul_mat_f | 4798/4800 | 4637/4800 | 0.01% | 1.7e-03 | 3.2e-02 | 9.20e-02 | 8.5e-07 | 1.3e-04 | 1.64e-03 |
| extracted / GPU, f32 cache, no TF32, no mul_mat_f | 4799/4800 | 4796/4800 | 1.39% | 1.3e-05 | 3.5e-04 | 6.69e-04 | 5.0e-11 | 1.2e-09 | 3.24e-05 |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 4500 | 272.438622 | - |
| CPU, f16 cache, GELU table | 4500 | 272.428655 | 1.55e-02 |
| CPU, f32 cache, GELU table | 4500 | 272.438588 | 9.71e-05 |
| CPU, f16 cache, no GELU table | 4500 | 272.428655 | 1.55e-02 |
| CPU, f32 cache, no GELU table | 4500 | 272.438588 | 9.71e-05 |
| GPU, f16 cache, TF32, mul_mat_f | 4500 | 272.325178 | 5.86e-02 |
| GPU, f32 cache, TF32, mul_mat_f | 4500 | 272.283624 | 5.31e-02 |
| GPU, f16 cache, no TF32, mul_mat_f | 4500 | 272.325178 | 5.86e-02 |
| GPU, f32 cache, no TF32, mul_mat_f | 4500 | 272.283624 | 5.31e-02 |
| GPU, f16 cache, TF32, no mul_mat_f | 4500 | 272.439873 | 5.37e-02 |
| GPU, f32 cache, TF32, no mul_mat_f | 4500 | 272.419929 | 4.94e-02 |
| GPU, f16 cache, no TF32, no mul_mat_f | 4500 | 272.438365 | 2.23e-02 |
| GPU, f32 cache, no TF32, no mul_mat_f | 4500 | 272.438581 | 7.52e-05 |

## Agreement on held-out text: qwen

300 windows of 16 tokens from the WikiText-2 raw test split, every position compared.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / PyTorch CPU | 4800/4800 | 4800/4800 | 1.31% | 4.8e-06 | 4.1e-05 | 1.55e-04 | 4.2e-11 | 7.8e-10 | - |
| extracted / PyTorch CUDA | 4800/4800 | 4799/4800 | 1.31% | 4.8e-06 | 4.1e-05 | 1.55e-04 | 4.3e-11 | 7.5e-10 | - |
| extracted / llama.cpp CPU | 4800/4800 | 4747/4800 | 0.02% | 2.7e-04 | 4.1e-03 | 2.16e-02 | 1.1e-07 | 1.4e-05 | - |
| extracted / llama.cpp CUDA | 4794/4800 | 4546/4800 | 0.00% | 1.5e-03 | 1.3e-02 | 5.55e-02 | 2.2e-06 | 6.1e-05 | 1.61e-03 |
| PyTorch CPU / PyTorch CUDA | 4800/4800 | 4799/4800 | 3.65% | 1.9e-06 | 1.3e-05 | 7.68e-05 | 5.5e-12 | 1.8e-10 | - |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 4500 | 213.216541 | - |
| PyTorch CPU | 4500 | 213.216586 | 8.50e-05 |
| PyTorch CUDA | 4500 | 213.216562 | 8.70e-05 |
| llama.cpp CPU | 4500 | 213.219491 | 5.93e-03 |
| llama.cpp CUDA | 4500 | 213.199794 | 3.21e-02 |

## llama.cpp configurations against the extracted forward pass: qwen

300 windows of 16 tokens from the WikiText-2 raw test split, every position compared. Cache is the key/value cache type; the GELU table is ggml-cpu's float16 table (GGML_GELU_FP16); TF32 is the cuBLAS math mode ggml-cuda sets, disabled with NVIDIA_TF32_OVERRIDE=0; mul_mat_f is ggml-cuda's kernel for float matrix products over at most 16 tokens, removed from ggml_cuda_mul_mat in the builds without it.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / CPU, f16 cache, GELU table | 4800/4800 | 4747/4800 | 0.02% | 2.7e-04 | 4.1e-03 | 2.16e-02 | 1.1e-07 | 1.4e-05 | - |
| extracted / CPU, f32 cache, GELU table | 4800/4800 | 4800/4800 | 1.31% | 4.8e-06 | 4.0e-05 | 1.67e-04 | 4.1e-11 | 8.5e-10 | - |
| extracted / CPU, f16 cache, no GELU table | 4800/4800 | 4747/4800 | 0.02% | 2.7e-04 | 4.1e-03 | 2.16e-02 | 1.1e-07 | 1.4e-05 | - |
| extracted / CPU, f32 cache, no GELU table | 4800/4800 | 4800/4800 | 1.31% | 4.8e-06 | 4.0e-05 | 1.67e-04 | 4.1e-11 | 8.5e-10 | - |
| extracted / GPU, f16 cache, TF32, mul_mat_f | 4794/4800 | 4546/4800 | 0.00% | 1.5e-03 | 1.3e-02 | 5.55e-02 | 2.2e-06 | 6.1e-05 | 1.61e-03 |
| extracted / GPU, f32 cache, TF32, mul_mat_f | 4793/4800 | 4550/4800 | 0.00% | 1.6e-03 | 1.5e-02 | 6.76e-02 | 2.7e-06 | 1.5e-04 | 1.87e-03 |
| extracted / GPU, f16 cache, no TF32, mul_mat_f | 4792/4800 | 4556/4800 | 0.00% | 1.5e-03 | 1.3e-02 | 5.41e-02 | 2.3e-06 | 5.8e-05 | 1.87e-03 |
| extracted / GPU, f32 cache, no TF32, mul_mat_f | 4789/4800 | 4527/4800 | 0.00% | 1.6e-03 | 1.4e-02 | 7.07e-02 | 2.6e-06 | 1.2e-04 | 1.87e-03 |
| extracted / GPU, f16 cache, TF32, no mul_mat_f | 4799/4800 | 4654/4800 | 0.01% | 9.3e-04 | 8.5e-03 | 3.98e-02 | 8.3e-07 | 4.7e-05 | 2.96e-04 |
| extracted / GPU, f32 cache, TF32, no mul_mat_f | 4795/4800 | 4663/4800 | 0.01% | 8.5e-04 | 7.6e-03 | 4.58e-02 | 7.1e-07 | 5.0e-05 | 1.87e-03 |
| extracted / GPU, f16 cache, no TF32, no mul_mat_f | 4800/4800 | 4743/4800 | 0.02% | 3.5e-04 | 4.1e-03 | 2.16e-02 | 1.3e-07 | 1.4e-05 | - |
| extracted / GPU, f32 cache, no TF32, no mul_mat_f | 4800/4800 | 4799/4800 | 1.31% | 4.8e-06 | 4.1e-05 | 1.54e-04 | 4.3e-11 | 7.2e-10 | - |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 4500 | 213.216541 | - |
| CPU, f16 cache, GELU table | 4500 | 213.219491 | 5.93e-03 |
| CPU, f32 cache, GELU table | 4500 | 213.216582 | 7.78e-05 |
| CPU, f16 cache, no GELU table | 4500 | 213.219491 | 5.93e-03 |
| CPU, f32 cache, no GELU table | 4500 | 213.216582 | 7.78e-05 |
| GPU, f16 cache, TF32, mul_mat_f | 4500 | 213.199794 | 3.21e-02 |
| GPU, f32 cache, TF32, mul_mat_f | 4500 | 213.129891 | 3.39e-02 |
| GPU, f16 cache, no TF32, mul_mat_f | 4500 | 213.187508 | 3.09e-02 |
| GPU, f32 cache, no TF32, mul_mat_f | 4500 | 213.123567 | 3.56e-02 |
| GPU, f16 cache, TF32, no mul_mat_f | 4500 | 213.222825 | 1.26e-02 |
| GPU, f32 cache, TF32, no mul_mat_f | 4500 | 213.213979 | 1.03e-02 |
| GPU, f16 cache, no TF32, no mul_mat_f | 4500 | 213.221885 | 5.95e-03 |
| GPU, f32 cache, no TF32, no mul_mat_f | 4500 | 213.216583 | 8.14e-05 |

## What the PyTorch residual is made of

The llama.cpp attribution turns one implementation choice off at a time. The
residual against PyTorch has only two sources: PyTorch evaluates the elementary
functions with its own implementations rather than the series the development
composes, and it accumulates its matrix products in a different order.
`scripts/residual_attr.py` separates them on GPT-2 by running the forward pass
in numpy float32 twice over the same checkpoint and windows, once with numpy's
own exp and GELU and once with the extracted ones, against the extracted
reference. Twenty windows of 64 tokens:

| source | max abs | mean abs | top-1 differs |
|---|---|---|---|
| numpy | 1.766e-03 | 6.159e-05 | 0 |
| numpy, extracted exp and GELU | 1.678e-03 | 6.112e-05 | 0 |
| PyTorch CPU | 1.678e-03 | 6.199e-05 | 0 |

Substituting the extracted elementary functions moves the maximum by 5 percent
and the mean by under 1 percent, and PyTorch lands where the substituted numpy
lands. The residual against PyTorch is therefore reduction order, not the
elementary functions.

## The dumps behind these tables

The logit dumps above are produced by `scripts/agree_setup.py`, the runners'
dump mode, `scripts/agree_torch.py`, `scripts/llamacpp_logits` and
`scripts/agree_vllm.py`. The windows are determined by the split,
the tokenizer and the three integers the first of those takes, and the dumps
come back byte for byte on the same host.

## GPT-2 on whole blocks of 64 tokens: gpt2

60 windows of 64 tokens from the WikiText-2 raw test split, every position compared. The bf16 rows read bfloat16 weights; the PyTorch one computes in bfloat16 as well.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / PyTorch CPU | 3840/3840 | 3836/3840 | 5.20% | 4.7e-05 | 1.0e-03 | 1.68e-03 | 7.2e-10 | 7.5e-09 | - |
| extracted / PyTorch CUDA | 3840/3840 | 3838/3840 | 5.53% | 3.9e-05 | 1.1e-03 | 1.56e-03 | 6.4e-10 | 1.1e-08 | - |
| extracted / PyTorch bf16 | 3372/3840 | 92/3840 | 0.00% | 3.1e-01 | 8.7e+00 | 1.25e+01 | 1.6e-02 | 8.6e-02 | 7.89e-01 |
| extracted / llama.cpp CPU | 3838/3840 | 3728/3840 | 0.04% | 7.1e-03 | 3.3e-01 | 7.58e-01 | 4.1e-07 | 8.4e-06 | 1.45e-03 |
| extracted / llama.cpp CUDA | 3826/3840 | 3305/3840 | 0.02% | 1.6e-02 | 6.5e-01 | 1.26e+00 | 1.6e-05 | 1.8e-04 | 1.38e-02 |
| extracted / llama.cpp CPU bf16 | 3631/3840 | 508/3840 | 0.00% | 2.2e-01 | 3.2e+00 | 6.81e+00 | 6.2e-03 | 2.6e-02 | 4.44e-01 |
| extracted / llama.cpp CUDA bf16 | 3630/3840 | 509/3840 | 0.00% | 2.2e-01 | 3.2e+00 | 7.39e+00 | 6.2e-03 | 2.7e-02 | 4.44e-01 |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 3780 | 80.256641 | - |
| PyTorch CPU | 3780 | 80.256637 | 2.59e-04 |
| PyTorch CUDA | 3780 | 80.256674 | 2.90e-04 |
| PyTorch bf16 | 3780 | 80.495299 | 1.15e+00 |
| llama.cpp CPU | 3780 | 80.255019 | 1.44e-02 |
| llama.cpp CUDA | 3780 | 80.281634 | 3.27e-02 |
| llama.cpp CPU bf16 | 3780 | 79.954552 | 6.99e-01 |
| llama.cpp CUDA bf16 | 3780 | 79.943026 | 7.01e-01 |

## SmolLM2 on whole blocks of 64 tokens: smollm

60 windows of 64 tokens from the WikiText-2 raw test split, every position compared. The bf16 rows read bfloat16 weights; the PyTorch one computes in bfloat16 as well.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / PyTorch CPU | 3840/3840 | 3840/3840 | 1.35% | 1.1e-05 | 1.4e-04 | 1.17e-03 | 4.7e-11 | 4.1e-08 | - |
| extracted / PyTorch CUDA | 3840/3840 | 3840/3840 | 1.35% | 1.1e-05 | 1.4e-04 | 9.61e-04 | 4.1e-11 | 2.7e-08 | - |
| extracted / PyTorch bf16 | 3691/3840 | 961/3840 | 0.00% | 8.9e-02 | 1.5e+00 | 7.85e+00 | 1.6e-03 | 2.7e-01 | 3.16e-01 |
| extracted / llama.cpp CPU | 3838/3840 | 3749/3840 | 0.01% | 1.9e-03 | 2.6e-02 | 1.62e-01 | 8.1e-07 | 7.3e-04 | 1.41e-03 |
| extracted / llama.cpp CUDA | 3836/3840 | 3648/3840 | 0.00% | 4.0e-03 | 4.9e-02 | 3.29e-01 | 2.9e-06 | 2.9e-03 | 3.18e-03 |
| extracted / llama.cpp CPU bf16 | 3814/3840 | 2679/3840 | 0.00% | 3.2e-02 | 2.8e-01 | 1.22e+00 | 1.3e-04 | 4.8e-02 | 7.34e-02 |
| extracted / llama.cpp CUDA bf16 | 3813/3840 | 2668/3840 | 0.00% | 3.2e-02 | 2.8e-01 | 1.59e+00 | 1.3e-04 | 7.8e-02 | 3.91e-02 |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 3780 | 62.922808 | - |
| PyTorch CPU | 3780 | 62.922790 | 1.83e-04 |
| PyTorch CUDA | 3780 | 62.922791 | 1.55e-04 |
| PyTorch bf16 | 3780 | 63.001049 | 1.31e+00 |
| llama.cpp CPU | 3780 | 62.922448 | 2.51e-02 |
| llama.cpp CUDA | 3780 | 62.923297 | 4.79e-02 |
| llama.cpp CPU bf16 | 3780 | 62.924422 | 2.93e-01 |
| llama.cpp CUDA bf16 | 3780 | 62.925257 | 3.78e-01 |

## Qwen3.5 on whole blocks of 64 tokens: qwen

60 windows of 64 tokens from the WikiText-2 raw test split, every position compared. The bf16 rows read bfloat16 weights; the PyTorch and vLLM ones compute in bfloat16 as well. The fused row runs the gated delta rule through flash-linear-attention's Triton kernels in float32, with the depthwise convolution on the torch path.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / PyTorch CPU | 3840/3840 | 3840/3840 | 1.49% | 4.6e-06 | 3.5e-05 | 1.65e-04 | 3.7e-11 | 8.3e-10 | - |
| extracted / PyTorch CUDA | 3840/3840 | 3840/3840 | 1.46% | 4.7e-06 | 3.4e-05 | 1.61e-04 | 3.6e-11 | 8.2e-10 | - |
| extracted / llama.cpp CPU | 3840/3840 | 3798/3840 | 0.03% | 2.7e-04 | 2.5e-03 | 1.96e-02 | 6.7e-08 | 4.9e-06 | - |
| extracted / llama.cpp CUDA | 3838/3840 | 3723/3840 | 0.01% | 9.3e-04 | 6.8e-03 | 3.04e-02 | 6.7e-07 | 1.5e-05 | 1.57e-03 |
| extracted / PyTorch fused | 3834/3840 | 3630/3840 | 0.00% | 1.5e-03 | 1.8e-02 | 1.14e-01 | 2.9e-06 | 1.9e-04 | 1.61e-03 |
| extracted / llama.cpp CPU bf16 | 3827/3840 | 3084/3840 | 0.00% | 6.8e-03 | 4.9e-02 | 2.44e-01 | 3.6e-05 | 2.1e-03 | 1.10e-02 |
| extracted / llama.cpp CUDA bf16 | 3827/3840 | 3075/3840 | 0.00% | 6.8e-03 | 4.9e-02 | 2.08e-01 | 3.6e-05 | 1.1e-03 | 1.46e-02 |
| extracted / PyTorch bf16 | 3756/3840 | 1505/3840 | 0.00% | 1.9e-02 | 1.4e-01 | 5.49e-01 | 5.2e-04 | 8.1e-03 | 1.80e-01 |
| extracted / vLLM bf16 | 3760/3840 | 1610/3840 | 0.00% | 1.7e-02 | 1.2e-01 | 5.25e-01 | 4.3e-04 | 7.9e-03 | 1.14e-01 |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 3780 | 58.171639 | - |
| PyTorch CPU | 3780 | 58.171641 | 6.13e-05 |
| PyTorch CUDA | 3780 | 58.171642 | 5.10e-05 |
| llama.cpp CPU | 3780 | 58.171435 | 5.40e-03 |
| llama.cpp CUDA | 3780 | 58.171916 | 9.83e-03 |
| PyTorch fused | 3780 | 58.169333 | 5.53e-02 |
| llama.cpp CPU bf16 | 3780 | 58.178980 | 1.28e-01 |
| llama.cpp CUDA bf16 | 3780 | 58.176583 | 9.93e-02 |
| PyTorch bf16 | 3780 | 58.235540 | 2.77e-01 |
| vLLM bf16 | 3780 | 58.172501 | 2.18e-01 |

## GPT-2 at its full 1024 positions: gpt2

3 windows of 1024 tokens from the WikiText-2 raw test split, every position compared. 1024 positions is the whole of GPT-2's position table.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / PyTorch CPU | 3072/3072 | 3069/3072 | 4.68% | 6.2e-05 | 5.4e-03 | 9.89e-03 | 8.2e-10 | 7.8e-09 | - |
| extracted / PyTorch CUDA | 3072/3072 | 3071/3072 | 6.92% | 4.7e-05 | 3.8e-03 | 8.87e-03 | 9.8e-11 | 1.5e-09 | - |
| extracted / PyTorch bf16 | 2754/3072 | 113/3072 | 0.00% | 4.1e-01 | 7.6e+01 | 2.98e+02 | 1.7e-02 | 1.0e-01 | 8.86e-01 |
| extracted / llama.cpp CPU | 3070/3072 | 2965/3072 | 0.03% | 1.1e-02 | 2.0e+00 | 3.54e+00 | 7.5e-07 | 9.6e-05 | 8.77e-04 |
| extracted / llama.cpp CUDA | 3066/3072 | 2580/3072 | 0.01% | 2.3e-02 | 2.7e+00 | 6.31e+00 | 1.8e-05 | 4.7e-04 | 1.59e-02 |
| extracted / llama.cpp CPU bf16 | 2891/3072 | 365/3072 | 0.00% | 2.6e-01 | 1.6e+01 | 4.61e+01 | 6.3e-03 | 3.3e-02 | 4.82e-01 |
| extracted / llama.cpp CUDA bf16 | 2900/3072 | 366/3072 | 0.00% | 2.6e-01 | 2.6e+01 | 3.94e+01 | 6.4e-03 | 4.2e-02 | 5.52e-01 |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 3069 | 28.277054 | - |
| PyTorch CPU | 3069 | 28.277070 | 3.81e-04 |
| PyTorch CUDA | 3069 | 28.277057 | 6.92e-05 |
| PyTorch bf16 | 3069 | 28.559767 | 1.48e+00 |
| llama.cpp CPU | 3069 | 28.277568 | 3.62e-02 |
| llama.cpp CUDA | 3069 | 28.281877 | 7.64e-02 |
| llama.cpp CPU bf16 | 3069 | 28.224309 | 7.33e-01 |
| llama.cpp CUDA bf16 | 3069 | 28.224357 | 7.15e-01 |

## SmolLM2 on windows of 2048 tokens: smollm

3 windows of 2048 tokens from the WikiText-2 raw test split, every position compared. The rotary angles reach 2047.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / PyTorch CPU | 6144/6144 | 6144/6144 | 1.37% | 1.0e-05 | 7.8e-05 | 2.15e-04 | 2.8e-11 | 1.4e-09 | - |
| extracted / llama.cpp CPU | 6143/6144 | 5957/6144 | 0.01% | 2.0e-03 | 2.1e-02 | 6.29e-02 | 7.4e-07 | 4.3e-05 | 5.16e-04 |
| extracted / llama.cpp CPU bf16 | 6118/6144 | 4467/6144 | 0.00% | 2.9e-02 | 1.7e-01 | 5.28e-01 | 7.6e-05 | 1.1e-02 | 2.09e-01 |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 6141 | 17.113821 | - |
| PyTorch CPU | 6141 | 17.113819 | 7.88e-05 |
| llama.cpp CPU | 6141 | 17.114370 | 2.39e-02 |
| llama.cpp CPU bf16 | 6141 | 17.110629 | 3.82e-01 |

## Fused gated delta rule with TF32 and with IEEE dot products: qwen

60 windows of 64 tokens from the WikiText-2 raw test split, every position compared. The fused rows run the gated delta rule through flash-linear-attention 0.5.2's Triton kernels; TF32 is Triton's default for float32 dot products, and the IEEE row runs agree_torch.py with P2W_IEEE_DOTS=1.

| pair | top-1 agrees | top-10 agrees in order | bit-identical logits | median abs diff | 99.9th pct | max abs diff | mean KL | max KL | largest margin at a top-1 difference |
|------|--------------|------------------------|----------------------|-----------------|------------|--------------|---------|--------|--------------------------------------|
| extracted / PyTorch fused TF32 | 3834/3840 | 3630/3840 | 0.00% | 1.5e-03 | 1.8e-02 | 1.14e-01 | 2.9e-06 | 1.9e-04 | 1.61e-03 |
| extracted / PyTorch fused IEEE | 3840/3840 | 3840/3840 | 1.46% | 4.7e-06 | 3.4e-05 | 1.59e-04 | 3.6e-11 | 6.4e-10 | - |

| source | predicted tokens | perplexity | max per-token NLL difference from extracted |
|--------|------------------|------------|-----------------------|
| extracted | 3780 | 58.171639 | - |
| PyTorch fused TF32 | 3780 | 58.169333 | 5.53e-02 |
| PyTorch fused IEEE | 3780 | 58.171642 | 5.18e-05 |
