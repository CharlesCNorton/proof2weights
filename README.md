# proof2weights

proof2weights defines the forward passes of three language-model architectures
in the Rocq prover, with every floating-point operation in IEEE-754 binary32
through the Flocq library, and extracts the definitions to OCaml. The extracted
programs load published `.safetensors` checkpoints and run inference on GPT-2
(124M parameters), SmolLM2-135M-Instruct and Qwen3.5-0.8B, whose layers are
mostly gated DeltaNet linear-attention recurrences. On all three, the extracted
forward pass returns the same next-token ranking as the PyTorch reference, with
logits within 7.3e-5 absolute. On SmolLM2 and Qwen3.5, greedy generation
reproduces PyTorch's continuation token for token. Over 300 windows of held-out
text per model, 4,800 next-token distributions each, PyTorch in float32 returns
the extracted pass's top-1 token at every position, with every logit within
1.6e-3. That holds at longer windows too: at whole blocks of 64 tokens on all
three models, at GPT-2's full 1024 positions, and at 2048 tokens on SmolLM2.

The arithmetic executed at inference time is the arithmetic the proofs concern.
The development proves that cached autoregressive decoding computes exactly the
values full recomputation computes, that each of the three forward passes is
causal, that the two extraction modes compute the same binary32 value at every
operation for every input, with overflow, infinities, NaNs and signed zeros
inside the statement, a composed rounding-error bound from the arithmetic
primitives to the logits of all three architectures, and a bound from each
elementary function the models use to the corresponding mathematical function.

A paper describing the development is in `paper/paper.tex`.

## Arithmetic

`binary32` is Flocq's `binary_float` at precision 24 and exponent bound 128.
Addition, multiplication, division, square root, negation, absolute value and
comparison are Flocq's `Bplus`, `Bmult`, `Bdiv`, `Bsqrt`, `Bopp`, `Babs` and
`Bcompare` under round-to-nearest, ties-to-even. Every scalar the networks
compute comes from one of these operations; everything else is list plumbing.
A weight is decoded from its four little-endian bytes through Flocq's
`b32_of_bits`, so a loaded value is the IEEE-754 value its bytes denote.

The elementary functions are compositions of those operations, and their
constants are quotients of integers.

- `f32_exp_approx` saturates its argument to [-88, 88], writes it as
  `k log 2 + r` with `k` the nearest integer to `x / log 2`, evaluates the
  degree-seven Taylor polynomial at `|r| <= 0.35`, and multiplies by `2^k`,
  which is exact. `log 2` is split as `355/512 - 14581891/2^36`, so `k * 355/512` is
  exact for every `k` the reduction produces.
- `f32_round_int` rounds to the nearest integer by adding and subtracting
  `1.5 * 2^23`, which for arguments of either sign below `2^22` in magnitude
  keeps the sum in the binade whose unit in the last place is one.
- `f32_sin` and `f32_cos` reduce the argument by the nearest integer multiple
  of a split `2 pi = 201/32 + 8312081/2^32` and evaluate Taylor polynomials
  through degree 19 and 18. The divisors are the factorials as binary32 stores
  them; from 14! upward those differ from the factorials.
- `f32_log_unit` evaluates `log m = 2 artanh ((m-1)/(m+1))` by a seven-term
  series on (1, 2].
- `f32_sigmoid` is `1 / (1 + exp (-x))`, `f32_tanh` is `2 sigmoid (2x) - 1`,
  `f32_gelu` is the form GPT-2 is trained with,
  `x/2 (1 + tanh (sqrt (2/pi) (x + 0.044715 x^3)))`, SiLU is `x sigmoid x`,
  `f32_softplus` is `max x 0 + log (1 + exp (-|x|))`, and `f32_softmax`
  subtracts the row maximum before exponentiating.

## Architectures

The GPT-2 path (`theories/Phases1_15_complete.v`) is a pre-norm decoder stack:
layer normalization, fused query/key/value projection, multi-head causal
attention, residual, layer normalization, GELU feed-forward, residual, a final
layer normalization and a tied-embedding logit projection. Causal attention
computes row `i` by attending the `i`-th query over the first `i + 1` key and
value rows with a max-shifted softmax. Parameter counts are pinned by
reflection: `gpt2_total_params gpt2_small` reduces to 124,439,808.

The Llama path (`theories/Llama.v`) replaces layer normalization with RMSNorm,
learned positions with rotary embeddings, multi-head with grouped-query
attention, and the GELU feed-forward with SwiGLU. `f32_llama_layer`,
`f32_llama_stack` and `f32_llama_forward` name the composition.

The Qwen3.5 path (`theories/Qwen.v`) alternates three gated DeltaNet layers
with one gated full-attention layer. A DeltaNet layer projects to a fused
query/key/value triple, applies a depthwise causal convolution, normalizes
query and key, and runs a recurrence over a per-head state matrix: the state
is decayed by `exp g` with `g = -exp(A_log) softplus(a + dt_bias)`, corrected
towards the value by `beta = sigmoid(b)`, and read out by the query. The
full-attention layers gate their output elementwise by the sigmoid of a stream
carried in the query projection and rotate a quarter of each head. For
text-only input the multimodal rotary embedding coincides with ordinary
rotation on that prefix.

## Two extraction modes

The inductive extraction (`Phases1_15_complete.v`, `Llama_inductive.v`,
`Qwen_inductive.v`) keeps `binary32` as Flocq's inductive `binary_float` and
`Z` and `positive` as inductive datatypes, so every float operation is the
computational content of its definition and no floating-point hardware is
trusted. Only `nat`, which carries indices, dimensions and token identifiers,
extracts to machine `int`.

The native extraction (`Extract.v`) maps `binary32` to OCaml `float`: each
operation is the binary64 result rounded to binary32 through an `Int32` bit
round-trip. `Float_error.v` proves, by instantiating Flocq's double-rounding
theorems at the two formats, that for binary32 operands rounding the exact
result of `+`, `-`, `*`, `/` or `sqrt` to binary64 and then to binary32 equals
rounding it directly to binary32.

That statement is about real numbers, so it settles the two modes' agreement
only where both results are finite. `Narrow.v` moves it to the floats
themselves. It defines `widen`, the exact embedding of a binary32 into
binary64 that happens when an OCaml `float` holding a binary32 value enters an
operation, and `narrow`, the round-to-nearest-even conversion back that the
`Int32` round-trip performs, and proves

```
narrow (b64_plus (widen x) (widen y))          = f32_plus x y
narrow (b64_plus (widen x) (Bopp (widen y)))   = f32_minus x y
narrow (b64_mult (widen x) (widen y))          = f32_mult x y
narrow (b64_div  (widen x) (widen y))          = f32_div  x y
narrow (b64_sqrt (widen x))                    = f32_sqrt x
narrow (widen x)                               = x
```

for every binary32 `x` and `y`, with no side condition. The statements quantify
over the constructors of `binary_float`, so overflow, infinities, NaNs, signed
zeros and underflow are inside them. The two modes therefore compute the same
binary32 value at every operation for every input, and a number measured on the
native build is a number of the inductive build.

The trusted boundary of the native mode is that the host float is binary64 with
round-to-nearest-even, with no excess precision and no contraction of a
multiply and an add, and that `Int32.bits_of_float` rounds to nearest.
`runners/fp_selftest.ml` decides both on the host that runs it. The elementary
functions extract structurally in both modes, as the same compositions of the
primitives.

## Two runners per model

Each model has two runners. The cached one (`gpt2_talk_native`,
`llama_talk_native`, `qwen_talk_native`) writes the layer loop in OCaml and
calls the extracted operators from it, so its arithmetic is verified and its
composition is not; it is the one that decodes autoregressively, and the cache
equalities of `Cache.v` are what cover it.

The other (`gpt2_verified`, `llama_verified`, `qwen_verified`) does not write
the loop. It hands the extracted loader a fetch of one byte of the data section,
and `f32_load_model_pre`, `f32_load_llama` or `f32_load_qwen` looks up each
tensor's offset through `json_tensor_offsets`, decodes it and assembles the
weight record; `f32_gpt2_logits_pre`, `f32_llama_logits_of` or
`f32_qwen_logits_of` then runs the pass. `f32_load_model_pre_correct` proves the
assembled record is the one `f32_load_model` defines, transposed, and
`f32_load_llama_correct` and `f32_load_qwen_correct` say the same of the other two,
field by field. What remains native is reading the file and answering the fetch.
It does not cache, so it recomputes the sequence at every step and is not the
runner to generate with.

The two agree: on GPT-2 and SmolLM2 the ten highest logits of the stored prompt
agree to every digit printed, and each verified runner reproduces its model's
stored PyTorch oracle, Qwen3.5's through the eighth rank its oracle records.
`scripts/verified_check.py` is that comparison.

`f32_linear_forward` transposes its weight, and the extracted transpose indexes
a list of lists, so it is quadratic in the output dimension; that is why the
GPT-2 runner cannot call `f32_gpt2_logits` directly. `Pretransposed.v` gives the
same pass over weights already stored transposed, which is the order the runner
decodes them in, and proves that on the transpose of a model it returns what the
original returns on the model (`f32_gpt2_logits_pre_correct`). The Llama and
Qwen3.5 paths need no such variant.

## Results on published checkpoints

Agreement is against PyTorch in float32 on the same prompt. Timings are wall
clock on an i9-12900H, single-threaded.

| | GPT-2 | SmolLM2-135M-Instruct | Qwen3.5-0.8B |
|---|---|---|---|
| Prompt | "The quick brown" | chat: "What is the capital of France?" (37 tokens) | same question (19 tokens) |
| Ranking | top 10 identical | top 10 identical | top 8 identical |
| Max logit difference | 7.3e-5 (relative 1.2e-6) | 3.2e-5 (1.2e-6) | 2.7e-5 (1.5e-6) |
| Greedy continuation | | 12 tokens identical: "The capital of France is Paris." | 12 tokens identical: "The capital of France is **Paris**." |
| Native forward | 10.3 s, 1.1 GB | 51.3 s, 6.3 GB | 244.6 s, 4.1 GB |
| Inductive forward | 58.8 min, 7.1 GB | 4.17 h, 2.8 GB | 11.1 h, 3.5 GB |

## What is proved

Serialization and storage. Decoding the encoding of any 32-bit integer returns
it (`roundtrip_z`), the binary32 and binary16 bit layouts round trip
(`roundtrip_f32`, `roundtrip_f16`), and chunking, run-length compression and
sharding under a byte budget invert on every input
(`reassemble_split_into_chunks`, `rle_roundtrip`,
`decompress_compress_network`, `unshard_shard_network`).

Shapes and loading. The float forward pass returns one row per token and each
logit row has one entry per vocabulary item (`f32_gpt2_forward_rows`,
`f32_gpt2_logits_rows`, `f32_gpt2_logits_row_width`), and a model that passes
validation has logit rows of the configured width (`validated_logits_shape`).
Entry `i` of a named tensor is the binary32 value of the four bytes at offset
`a + 4i` of the data section (`f32_load_named_nth`, `Loader.v`).

Per operation. Each primitive returns the correctly rounded result of the exact
operation and lies within half a ULP of it (`f32_mult_correct`,
`f32_plus_error` and the others in `Float_error.v`), and double rounding
through binary64 is harmless (`f32_double_round_plus`, `_minus`, `_mult`,
`_div`, `_sqrt`).

The composed forward bound. Rounding is modeled as `z (1 + d) + e` with
`|d| <= 2^-24` and `|e| <= 2^-150`, which holds for every real `z`, zero and the
subnormal range included (`f32_round_mixed`, from Flocq's `error_N_FLT`). The
relation `ok d x r` states that a float `x` is finite and within `d` of the
real value `r`; each primitive maps inputs within `d` to an output within
`u M + eta + L d`, where `M` bounds magnitudes and `L` bounds how much an
operation can amplify an existing error. One lemma per primitive and structural
lemmas for the plumbing carry the relation through layer normalization, the
exponential, softmax, causal attention, the transformer block and stack
(`ok_block_forward`, `ok_blocks_forward`) to the GPT-2 logits
(`ok_gpt2_logits_full`); through RMSNorm, rotary embedding, SwiGLU and the Llama
layer to its logits (`ok_llama_logits_full`); and through the logarithm,
softplus, both extra RMSNorm variants, the depthwise convolution, the gated
delta step and scan, and both Qwen3.5 mixers to its logits
(`ok_qwen_logits_full`). The side conditions are collected per stage in
records, and `Witness.v` discharges the records of the exponential, sigmoid,
tanh and GELU at an exact zero, and that of the exponential at an argument it
saturates, deciding every binary32 comparison by computation on the rational
the float denotes (`Qb`, `Qb_correct`). `Dot.v` gives the sharper running bound for the
dot product (`f32_dot_error_mixed`) and witnesses it on a product that is
exactly zero.

Backward error. If no rounding in a dot product of length `n` overflows, the
computed value is the exact inner product of its operands with each product
scaled by a factor within `(1 + u)^(n+1) - 1` of one, plus a displacement of at
most `2 n eta (1 + u)^n` (`f32_dot_backward_mixed`), and the statement lifts to
the matrix-vector product and the tied-embedding projection of all three models
(`f32_mat_vec_mul_backward_mixed`, `logits_backward_mixed`).

A forward pass that carries its bound. `RunErr.v` writes the GPT-2 forward pass
once over an abstract arithmetic of nine operations and instantiates it at
binary32, where it is `f32_gpt2_logits` by conversion (`g_gpt2_logits_f32`), at
exact real arithmetic with the exponential taken on its argument saturated to
[-88, 88], and at the annotated arithmetic of `Annot.v` and `AnnExp.v`, where
each value is a binary32 result paired with a binary64 bound. The bound is
computed with outward rounding (`Bound64.v`); a condition the analysis needs is
tested on the values the pass computes, and a failed test returns an infinite
bound, so every annotated operation preserves the relation "when the bound is
finite, the value is finite and within the bound of the real result" with no
hypothesis (`aok_plus`, `aok_mult`, `aok_div`, `aok_sqrt`, `aok_exp`). The
exponential's bound holds for whatever integer the float reduction selects,
provided it is at most 127 in magnitude and the widened reduced argument lies
within 179/512 of zero, both of which are tested. One relational theorem gives
`gpt2_logits_bounded`: the binary32 parts of the annotated pass are the logits
`f32_gpt2_logits` returns, and each lies within its bound of the logit exact
real arithmetic produces from the same weights and tokens. The bound is computed
from the values the pass produces, so it holds for the input the pass was run
on.

Elementary functions against mathematics. `Series.v` bounds each series the
code evaluates, with its stored divisors, against the function it approximates,
using CoqInterval, and `Truth.v` composes those bounds with the propagation
relation. The real evaluation of the exponential is within a relative `1.6e-8`
of `exp` on [-88, 88] (`exp_core_vs_true`). For `|x| <= 262144` the reduction
lands within 3.16 of zero, the stored split of `2 pi` costs at most `8.4e-7`,
and the extracted sine and cosine are within `8.5e-7` and `8.7e-7` of `sin x`
and `cos x` beyond the rounding term (`ok_sin_true_full`, `ok_cos_true_full`).
That range is the largest rotary angle any of the three checkpoints forms: an
angle is a position times an inverse frequency, the largest inverse frequency
is one, so the range is the longest context, and Qwen3.5's is 262144. Both
constants grow with the range, the first because the split's error is charged
once per multiple of `2 pi` removed and the second because the interval the
reduced argument occupies widens with the error in the stored `1/(2 pi)`, which
is why `Series.v` states its truncation bounds on [-3.17, 3.17].
The logarithm is within `2e-8` of `ln` on (1, 2] (`ok_log_true`), sigmoid
within `1.6e-8` (`ok_sigmoid_true`), tanh within `3.2e-8` (`ok_tanh_true`),
softplus within `4e-8` of `ln (1 + exp x)` (`ok_softplus_true`), GELU within
`6e-6` of the tanh form on [-8, 8] (`ok_gelu_true`), and square root within
half a ULP (`ok_sqrt_true`).

Cached decoding. The gated delta scan decomposes at any split point into a
prefill and a continuation from the state the prefill produced
(`delta_scan_app`); extending the sequence by one token appends exactly the
output of one step (`delta_scan_snoc`); outputs after a split depend on the
prefix only through the state (`delta_markov`); the runner's incrementally
maintained convolution window is the window the definition reads
(`conv_window_cached_correct`, `conv_hist_step`); and a decode step of causal
attention appends the new query attending over every cached key and value
(`causal_attention_snoc`). These are equalities of binary32 values.

Causality. A sequence function is causal when it preserves length and its rows
for a sequence are the first rows of its output on every extension. Row-wise
maps, causal attention, the convolution and the delta scan are causal, and
causality is closed under composition, zipping and stacking, so the three
forward passes are causal (`causal_gpt2_forward`, `causal_llama_forward`,
`causal_qwen_forward`): running a model on a longer sequence reproduces bit for
bit every row it produced for the shorter one (`gpt2_logits_prefix`,
`llama_forward_prefix`, `qwen_forward_prefix`). Emitting the last row of each
prefix, which is what a decoder does, therefore produces the rows one
evaluation of the whole sequence produces (`decode_stream_correct`, and
`gpt2_decode_step`, `llama_decode_step`, `qwen_decode_step` at the three
models).

Runners and generation. The GPT-2 runners decode each weight matrix directly in
transposed order, and that decode equals `f32_mat_transpose` of the reshape, so
their linear layer is `f32_linear_forward` (`decode_transposed_correct`,
`runner_linear_correct`). Greedy generation always extends the prompt
(`gpt2_generation_preserves_prompt`, `f32_generation_preserves_prompt`).

`theories/Audit.v` prints the assumptions behind each of these results. Beyond
the four classical axioms of the Rocq real-number library (`classic`,
`functional_extensionality_dep`, `sig_forall_dec`, `sig_not_dec`), which Flocq
inherits, only the results that call CoqInterval assume anything: the
axiomatization of primitive floats, 63-bit integers and primitive arrays it
computes with. `scripts/assumption_table.py` classifies the report.

## Bounds computed during a forward pass

`runners/gpt2_bound_ref` runs the inductive extraction of the annotated forward
pass `gpt2_logits_bounded` is stated about, on models loaded through the verified
loader. Over the 160 random GPT-2 models of the differential sweep, the bound is
compared with the actual error, the distance from the binary32 logit to a
float64 evaluation of the reference network; evaluating one sample per
configuration at 50 digits moves that reference by at most 3.1e-16. No logit
exceeds its bound. `RESULTS.md` holds every configuration.

| layers | d_model | seq | logits bounded | max bound | max actual error | median bound/error |
|---|---|---|---|---|---|---|
| 1 | 8 | 8 | 2048/2048 | 1.8e-5 | 3.4e-8 | 970 |
| 2 | 8 | 8 | 2048/2048 | 6.3e-4 | 2.6e-8 | 2.0e4 |
| 4 | 8 | 8 | 1664/2048 | 1.9 | 3.1e-8 | 7.3e6 |
| 4 | 8 | 32 | 7248/8192 | 1.0 | 3.4e-8 | 8.2e6 |
| 4 | 16 | 8 | 0/2048 | | | |
| 8 | 8 | 8 | 0/2048 | | | |

The bound grows thirty- to fifty-fold per layer. On GPT-2 small,
`runners/gpt2_bound_native` reports the bound after each stage: 2.7e-7 on the
embeddings, then through the first block 8.7e-6 after layer normalization,
4.3e-4 after the query/key/value projection, 1.5e-2 after attention, 0.49 after
the output projection, 0.20 after the second layer normalization and 17 after
the feed-forward projection, whose values reach 11.5. The exponentials inside
the first GELU then return infinite bounds, and every later bound is infinite.

## Agreement on held-out text

For each model, the WikiText-2 raw test split, tokenized by the model's own
tokenizer, is cut into blocks of 64 tokens, and the first 16 tokens of 300
blocks spread evenly over the split form the windows. Each window runs through
the native extraction, through PyTorch in float32 on an i9-13900KF and an RTX
6000 Ada with eager attention and TF32 off, and through llama.cpp from a float32
GGUF on the same CPU and GPU with flash attention off and otherwise its
defaults. `scripts/provenance.py --gguf smollm=<file>` compares that conversion
against the `safetensors` the extracted pass loads, tensor by tensor: on
SmolLM2 all 272 tensors hold the same values, 212 bit for bit and the other 60,
the query and key projections of the thirty layers, after undoing the row
permutation the converter applies so that llama.cpp's rotary convention matches
the checkpoint's. The deviations below are differences in arithmetic, not in
the weights. Every implementation returns the logits of all 16 positions, 4,800
next-token distributions per model. Against the extracted pass, where top-1
differs counts positions of 4,800, KL is D(p_extracted || p) averaged over
positions, and perplexity covers the 4,500 positions with a following token:

| model | implementation | top-1 differs | median abs diff | max abs diff | mean KL | perplexity |
|---|---|---|---|---|---|---|
| GPT-2 | extracted | | | | | 246.928477 |
| | PyTorch CPU | 0 | 3.9e-5 | 1.57e-3 | 6.6e-10 | 246.928159 |
| | PyTorch GPU | 0 | 4.7e-5 | 1.51e-3 | 7.7e-10 | 246.928159 |
| | llama.cpp CPU | 2 | 6.6e-3 | 0.312 | 3.9e-7 | 246.929787 |
| | llama.cpp GPU | 7 | 2.5e-2 | 1.70 | 1.9e-5 | 246.898478 |
| SmolLM2 | extracted | | | | | 272.438622 |
| | PyTorch CPU | 0 | 1.3e-5 | 5.10e-4 | 4.8e-11 | 272.438619 |
| | PyTorch GPU | 0 | 1.2e-5 | 6.93e-4 | 5.0e-11 | 272.438619 |
| | llama.cpp CPU | 3 | 1.9e-3 | 9.33e-2 | 7.7e-7 | 272.428655 |
| | llama.cpp GPU | 7 | 6.5e-3 | 0.225 | 5.9e-6 | 272.325178 |
| Qwen3.5 | extracted | | | | | 213.216541 |
| | PyTorch CPU | 0 | 4.8e-6 | 1.55e-4 | 4.2e-11 | 213.216586 |
| | PyTorch GPU | 0 | 4.8e-6 | 1.55e-4 | 4.3e-11 | 213.216562 |
| | llama.cpp CPU | 0 | 2.7e-4 | 2.16e-2 | 1.1e-7 | 213.219491 |
| | llama.cpp GPU | 6 | 1.5e-3 | 5.55e-2 | 2.2e-6 | 213.199794 |

`RESULTS.md` adds the fraction of positions whose ten highest tokens agree in
order, the fraction of bit-identical logits, the 99.9th percentile of the
absolute difference and the largest KL.

## Differential testing

Because the extracted forward pass fixes every rounding and every reduction
order, it serves as a reference against which other float32 implementations
can be measured. The harness builds random small models, runs each through the
inductive extraction and through a numpy float32 implementation of the same
elementwise operations with numpy's own reductions, and reports the divergence
and any final-position next-token disagreement. Weights cross as raw binary32
bit patterns, so both sides start from identical values. `RESULTS.md` holds the
sweep: 368 models across GPT-2, Llama and Qwen3.5 configurations, with no
next-token disagreement. Divergence tracks width: moving `d_model` from 8 to 64
at four layers raises the mean absolute logit error by a factor of 12.5, while
moving from one layer to eight at width 8 raises it by 1.4.

The architecture sweep supplies the rotary tables as data, so it does not
evaluate `f32_sin` or `f32_cos`. `runners/prim_sweep.ml` and
`scripts/prim_check.py` evaluate the elementary functions themselves on the
inductive extraction at 40,001 points each against the mathematical functions
in double precision, sine and cosine over the whole rotary range, and
`scripts/exhaustive.c` does the same on the native build at every one of the
2^32 binary32 inputs; `paper/exhaustive.txt` holds that sweep.

On the checkpoints the same reference attributes llama.cpp's divergence to four
choices: the float16 key/value cache it uses by default, ggml-cpu's float16 GELU
table (`GGML_GELU_FP16`), the TF32 math mode ggml-cuda sets for cuBLAS, and
ggml-cuda's `mul_mat_f` kernel, which for batches of at most 16 tokens
multiplies float32 weight matrices of suitable shape on the GPU's tensor cores
outside cuBLAS, where `NVIDIA_TF32_OVERRIDE=0` does not reach. `RESULTS.md`
crosses the four on every model. Each alone, with the other three removed,
multiplies the largest logit difference by about 90 to 1,000 on every model it
applies to. With all four removed, llama.cpp agrees with the extracted pass as
closely as PyTorch on both backends, every logit within 1.8e-3, with one top-1
difference, on SmolLM2 on the GPU, where the two highest extracted logits are
3.2e-5 apart.

## Scope

The development is the specification of what these models compute in binary32,
and its agreement with PyTorch is a measurement. The cached checkpoint runners
compose the verified primitives in OCaml, reading bytes natively and streaming
weights layer by layer; the verified runners read the file and answer a byte
fetch, and everything after that is extracted. The
transposed decode and the caches, which regroup the arithmetic, are proved to
compute the values the definitions
compute; the rest of the composition is ordinary OCaml, checked by the fixture
comparison above and by the inductive references, which call
`f32_llama_forward` and `f32_qwen_forward` directly. The composed error bound
holds under its side-condition records, which are hypotheses about the inputs;
they are discharged at specific points, not for a checkpoint. The bound the
annotated forward pass computes needs no hypothesis, but it propagates
intervals: it is informative on one- and two-layer models and infinite on GPT-2
small from the first GELU on, and it is proved for the GPT-2 path only.

## Building and running

The theories need Rocq 9.0 with `coq-flocq` and `coq-interval`; the runners need
OCaml 4.14 or later; the setup scripts and the harness need Python with `torch`,
`transformers`, `numpy` and `safetensors`.

`scripts/check.sh` runs the checks and prints the time each takes:
the theories, the assumption report, the host self-test, the provenance record,
the elementary functions against the mathematical ones, each checkpoint against
its stored PyTorch oracle, and the bound the annotated pass carries. A step
whose input is absent is reported as skipped and named, so the output says what
was decided and what was not. On a machine with the theories built and the
runners and checkpoints present it takes about fifteen minutes, of which the
checkpoint oracles are nine and the assumption report is three.

```bash
scripts/check.sh            # everything
scripts/check.sh --quick    # everything but the assumption report
```

Compiling the theories rewrites the extraction output in `theories/`, which the
runners link against, so build the runners after the theories and not before.



The individual steps:

```bash
# Compile every theory in dependency order. Extraction output
# (phases1_15_complete, phases1_15_native, llama_native, qwen_native,
# llama_inductive, qwen_inductive, runerr_native, runerr_inductive, each .ml and
# .mli) lands in theories/.
make -C theories

# Runners against the native extraction.
ocamlopt -rectypes -w -a -I theories theories/phases1_15_native.mli theories/phases1_15_native.ml runners/gpt2_talk_native.ml -o gpt2_talk_native
ocamlopt -rectypes -w -a -I theories theories/llama_native.mli theories/llama_native.ml runners/llama_talk_native.ml -o llama_talk_native
ocamlopt -rectypes -w -a -I theories theories/qwen_native.mli theories/qwen_native.ml runners/qwen_talk_native.ml -o qwen_talk_native

# Runners against the inductive extraction.
ocamlopt -rectypes -w -a -I theories theories/phases1_15_complete.mli theories/phases1_15_complete.ml runners/gpt2_talk.ml -o gpt2_talk
ocamlopt -rectypes -w -a -I theories theories/llama_inductive.mli theories/llama_inductive.ml runners/llama_talk_inductive.ml -o llama_talk_inductive
ocamlopt -rectypes -w -a -I theories theories/qwen_inductive.mli theories/qwen_inductive.ml runners/qwen_talk_inductive.ml -o qwen_talk_inductive

# Fetch each model, save f32 weights under the names the runners look up, and
# print the PyTorch reference.
python scripts/gpt2_setup.py
python scripts/smollm_setup.py
python scripts/qwen_setup.py

# GPT-2: mode (full|next), file, n_embd, n_head, n_layer, n_inner, vocab,
# n_positions, token ids.
./gpt2_talk_native next gpt2.safetensors 768 12 12 3072 50257 1024 464,2068,7586

# SmolLM2: file, d, n_layer, n_head, n_kv, ff, vocab, token ids, then optionally
# max_new and eos for greedy generation.
./llama_talk_native smollm.safetensors 576 30 9 3 1536 49152 <ids> 12 2

# Qwen3.5: file, d, n_layer, n_head, n_kv, head_dim, rotary_dim, ff, vocab,
# DeltaNet heads, DeltaNet head_dim, conv kernel, token ids, then optionally
# max_new and eos.
./qwen_talk_native qwen.safetensors 1024 24 8 2 256 64 3584 248320 16 128 4 <ids>

# Chat against either model; tokenization runs in the script.
python scripts/chat.py smollm "What is the capital of France?"
python scripts/chat.py qwen "What is the capital of France?"
```

The native runners also accept `serve <eos>` in place of the token ids, which
keeps the weights resident and answers one query per line on standard input,
and `dump <windows> <outdir> <first> <last>`, which runs each line of a file of
comma-separated windows and writes the logits of every position to
`<outdir>/<index>.f32` as little-endian binary32 (GPT-2 takes `dump` as its
mode argument, ahead of the file).

The differential harness:

```bash
ocamlopt -rectypes -w -a -I theories theories/phases1_15_complete.mli theories/phases1_15_complete.ml runners/ref_logits.ml -o ref_logits
ocamlopt -rectypes -w -a -I theories theories/llama_inductive.mli theories/llama_inductive.ml runners/llama_ref.ml -o llama_ref
ocamlopt -rectypes -w -a -I theories theories/qwen_inductive.mli theories/qwen_inductive.ml runners/qwen_ref.ml -o qwen_ref
ocamlopt -rectypes -w -a -I theories theories/qwen_inductive.mli theories/qwen_inductive.ml runners/prim_sweep.ml -o prim_sweep

python scripts/experiment_gen.py     # GPT-2 models and numpy logits in expbatch/
bash scripts/run_batch.sh            # reference logits into coq_out.txt
python scripts/experiment_cmp.py     # GPT-2 section of RESULTS.md
python scripts/experiment_arch.py    # Llama and Qwen3.5 sections
python scripts/prim_check.py         # the elementary functions
```

The annotated forward pass:

```bash
ocamlopt -rectypes -w -a -I theories theories/runerr_native.mli theories/runerr_native.ml runners/gpt2_bound_native.ml -o gpt2_bound_native
ocamlopt -rectypes -w -a -I theories theories/runerr_inductive.mli theories/runerr_inductive.ml runners/gpt2_bound_ref.ml -o gpt2_bound_ref

# Per-stage bounds on a checkpoint: file, n_embd, n_head, n_layer, n_inner,
# vocab, token ids.
./gpt2_bound_native gpt2.safetensors 768 12 12 3072 50257 464,2068,7586

bash scripts/run_bound_batch.sh      # bounds for every model of the GPT-2 sweep
python scripts/bound_cmp.py --write  # bound against actual error, into RESULTS.md
```

Agreement on held-out text. `agree_setup.py` writes the windows of the reported
evaluation by default. The extracted dumps take about 2, 1.5 and 18 hours of
CPU time for GPT-2, SmolLM2 and Qwen3.5, and each runner takes a range of window
indices, so the 300 windows split across processes. PyTorch, llama.cpp and the
report are shown for GPT-2; SmolLM2 and Qwen3.5 take the same commands with
`smollm` or `qwen`, their own GGUF, and vocabulary sizes 49152 and 248320.
`scripts/llamacpp_logits` builds against a llama.cpp checkout, and the GGUF
comes from llama.cpp's `convert_hf_to_gguf.py` with `--outtype f32`, run on a
directory holding the checkpoint's configuration and tokenizer files beside a
weight file without the attention-mask buffers (`gpt2_setup.py` saves one).

```bash
python scripts/agree_setup.py gpt2 agree/gpt2_windows.txt
python scripts/agree_setup.py smollm agree/smollm_windows.txt
python scripts/agree_setup.py qwen agree/qwen_windows.txt
./gpt2_talk_native dump gpt2.safetensors 768 12 12 3072 50257 1024 agree/gpt2_windows.txt agree/gpt2 0 300
./llama_talk_native smollm.safetensors 576 30 9 3 1536 49152 dump agree/smollm_windows.txt agree/smollm 0 300
./qwen_talk_native qwen.safetensors 1024 24 8 2 256 64 3584 248320 16 128 4 dump agree/qwen_windows.txt agree/qwen 0 300

python scripts/agree_torch.py gpt2 agree/gpt2_windows.txt agree/torch_cpu_gpt2 cpu
python scripts/agree_torch.py gpt2 agree/gpt2_windows.txt agree/torch_cuda_gpt2 cuda

cmake -S scripts/llamacpp_logits -B build -DLLAMA_CPP=<llama.cpp checkout>
cmake --build build --config Release --target llamacpp_logits
cmake -S scripts/llamacpp_logits -B build_cuda -DLLAMA_CPP=<llama.cpp checkout> -DGGML_CUDA=ON
cmake --build build_cuda --config Release --target llamacpp_logits
# model, windows, outdir, first, last, GPU layers, threads, cache type
./build/llamacpp_logits gpt2-f32.gguf agree/gpt2_windows.txt agree/llamacpp_cpu_gpt2 0 300 0 16
./build_cuda/llamacpp_logits gpt2-f32.gguf agree/gpt2_windows.txt agree/llamacpp_cuda_gpt2 0 300 99 8

python scripts/agree_cmp.py gpt2 agree/gpt2_windows.txt 50257 extracted=agree/gpt2 \
  "PyTorch CPU=agree/torch_cpu_gpt2" "PyTorch CUDA=agree/torch_cuda_gpt2" \
  "llama.cpp CPU=agree/llamacpp_cpu_gpt2" "llama.cpp CUDA=agree/llamacpp_cuda_gpt2" \
  --pair "PyTorch CPU:PyTorch CUDA" --write
```

The configuration report, written with
`--title "llama.cpp configurations against the extracted forward pass"`, crosses
the cache type (`f16` or `f32` as the last argument) with, on the CPU, a build
whose `ggml/src/ggml-cpu/vec.h` has no `#define GGML_GELU_FP16`, and on the GPU,
`NVIDIA_TF32_OVERRIDE=0` in the environment and a build whose
`ggml_cuda_mul_mat` in `ggml/src/ggml-cuda/ggml-cuda.cu` has no
`ggml_cuda_should_use_mmf` branch, all at llama.cpp commit 8172e65.

The longer evaluations take the same commands with different window files.
`agree_setup.py <model> <out> <T> <N> <P>` with `T` equal to `P` cuts whole
blocks: `64 60 64` gives the sixty blocks of 64 tokens, `1024 3 1024` GPT-2's
full position table and `2048 3 2048` SmolLM2's long windows. The bfloat16 rows
read a GGUF converted with `--outtype bf16`, which llama.cpp computes from in
float32, while `agree_torch.py` takes `bf16` as its last argument and computes
in bfloat16 as well. Qwen3.5's fused row runs `agree_torch.py` from an
environment with `flash-linear-attention` installed, which routes the gated
delta rule through its Triton kernels. Its vLLM row runs `agree_vllm.py`, which
takes the model, windows file, output directory and window range and computes
in bfloat16, from an environment with `vllm` installed; vLLM scores the last
position of a prompt only, so the script sends one request per prefix of each
window and a logits processor keeps the row the sampler receives.

`scripts/run_float_demo.sh` builds the small float drivers and runs the fixture
comparison against `scripts/tiny_gpt2_ref.py`. The integer export path has its
own build: `make -C tools` writes the two example integer networks to
`.safetensors` and checks each file against the bytes `serialize_list` produces,
and `make -C tools verify` reads them back with the Python `safetensors`
library.

## Repository layout

| Path | Contents |
|------|----------|
| `theories/Phases1_15_complete.v` | Serialization, binary32 arithmetic, the elementary functions, the layer library, GPT-2, the safetensors loader, generation, and the inductive extraction. |
| `theories/Llama.v` | RMSNorm, SiLU, sine and cosine, slicing, partial rotary embedding, SwiGLU, and the Llama layer, stack and forward pass. |
| `theories/Qwen.v` | The logarithm and softplus, Euclidean normalization, the two extra RMSNorm variants, the depthwise causal convolution, the gated delta rule, and the Qwen3.5 mixers, layer wrapper, stack and forward pass. |
| `theories/Float_error.v` | Correct rounding per operation, double rounding through binary64, the rounding model, and the composed error bounds up to the logits of all three architectures, with the backward-error statements. |
| `theories/Backward.v` | The backward error of the dot product with the underflow term, whose only hypothesis is that nothing overflows, and its lifting to the linear layers and the logit projection. |
| `theories/Pretransposed.v` | The GPT-2 forward pass over weights already stored transposed, and the proof that it returns what the original returns. |
| `runners/*_verified.ml` | Each checkpoint through the extracted forward pass itself rather than a loop rebuilt around it. |
| `scripts/verified_check.py` | Those runners against the stored PyTorch oracles. |
| `scripts/exhaustive.c` | Every elementary function at every one of the 2^32 binary32 inputs. |
| `theories/Narrow.v` | The widening and narrowing the native build performs, and the proof that narrowing the binary64 result of an operation on widened binary32 operands is the binary32 result, for every input, overflow and infinities and NaNs and signed zeros included. |
| `runners/fp_selftest.ml` | The assumptions the native build makes about its host, decided on that host. |
| `scripts/check.sh` | Runs the checks and prints the time each takes. |
| `scripts/provenance.py` | The checkpoint revisions, file digests and tool versions every reported number was produced from. |
| `scripts/oracle_check.py` | Each checkpoint's extracted forward pass against its stored PyTorch oracle. |
| `scripts/assumption_table.py` | The assumption class of each result, from the report `Audit.v` produces. |
| `theories/Dot.v` | The running bound for the dot product and its witnesses at an exact zero. |
| `theories/Series.v` | CoqInterval bounds on each series and constant the code evaluates. |
| `theories/Truth.v` | The elementary functions against the mathematical functions. |
| `theories/Witness.v` | Side-condition records discharged by computation at an exact zero and at a saturated argument. |
| `theories/Bound64.v` | Binary64 arithmetic with outward rounding, and what each operation bounds. |
| `theories/Annot.v`, `theories/AnnExp.v` | Annotated binary32 operations carrying a binary64 bound against real arithmetic, the exponential included. |
| `theories/RunErr.v` | The GPT-2 forward pass over an abstract arithmetic, its binary32, real and annotated instances, and `gpt2_logits_bounded`. |
| `theories/Cache.v`, `theories/Cache_attn.v` | Prefill and decode for the delta scan, the convolution window, and causal attention; transposition. |
| `theories/Causal.v` | Causality of the three forward passes, and that one decode step per token produces the rows one evaluation of the whole sequence produces. |
| `theories/Runner.v` | The transposed decode the checkpoint runners perform. |
| `theories/Loader.v` | What a named load returns, the dtype check, and validation connected to the shape theorems. |
| `theories/Loadpre.v` | The same load through a byte fetch rather than a list of bytes, and the proof that the record it assembles is the one `f32_load_model` defines. |
| `theories/RoundChk.v` | Rounding to the nearest integer, checked by computation. |
| `theories/Extract.v` | The native extraction of the GPT-2, Llama and Qwen3.5 targets and of the annotated GPT-2 forward pass. |
| `theories/Llama_inductive.v`, `theories/Qwen_inductive.v`, `theories/RunErr_inductive.v` | The inductive extraction of the Llama and Qwen3.5 definitions and of the annotated GPT-2 forward pass. |
| `theories/Audit.v` | `Print Assumptions` for the headline results. |
| `theories/_CoqProject`, `theories/Makefile` | Build order and build. |
| `runners/gpt2_talk.ml`, `runners/gpt2_talk_native.ml` | GPT-2 on the inductive and native extractions; the native runner also dumps the logits of every position of a file of windows. |
| `runners/llama_talk_native.ml`, `runners/llama_talk_inductive.ml` | SmolLM2 with a key/value cache, a serve mode and a dump mode; the inductive forward streams one layer at a time. |
| `runners/qwen_talk_native.ml`, `runners/qwen_talk_inductive.ml` | Qwen3.5 with attention, recurrent-state and convolution caches, a serve mode and a dump mode; the inductive forward decodes one projection row at a time. |
| `runners/gpt2_bound_native.ml`, `runners/gpt2_bound_ref.ml` | The annotated GPT-2 forward pass on the native extraction, with per-stage bound statistics, and on the inductive extraction through the verified loader. |
| `runners/ref_logits.ml`, `runners/llama_ref.ml`, `runners/qwen_ref.ml` | References for the differential harness on the inductive extraction; `ref_logits` goes through the verified list-based loader, and the other two call `f32_llama_forward` and `f32_qwen_forward`. |
| `runners/prim_sweep.ml` | The elementary functions on the inductive extraction. |
| `runners/float_smoke.ml`, `runners/float_load_run.ml`, `runners/test_bplus.ml` | Small drivers for the float path. |
| `scripts/gpt2_setup.py`, `scripts/smollm_setup.py`, `scripts/qwen_setup.py` | Fetch a model, save f32 weights, print the PyTorch reference. |
| `scripts/models.py`, `scripts/chat.py` | The model registry and interactive chat. |
| `scripts/tiny_gpt2_ref.py`, `scripts/experiment_gen.py`, `scripts/experiment_cmp.py`, `scripts/run_batch.sh` | The GPT-2 fixture and the GPT-2 sweep. |
| `scripts/arch_ref.py`, `scripts/experiment_arch.py` | numpy mirrors of the Llama and Qwen3.5 forwards, and their sweep. |
| `scripts/prim_check.py` | The elementary-function sweep. |
| `scripts/run_bound_batch.sh`, `scripts/bound_cmp.py` | The annotated forward pass over the GPT-2 sweep, and its bounds against the actual error. |
| `scripts/layer_gain.py` | The gain each linear layer applies to an absolute error on its input. |
| `scripts/agree_setup.py`, `scripts/agree_torch.py`, `scripts/agree_vllm.py`, `scripts/llamacpp_logits/`, `scripts/agree_cmp.py` | Held-out windows, PyTorch, vLLM and llama.cpp logit dumps, and the agreement report. |
| `scripts/run_float_demo.sh` | The fixture demonstration. |
| `tools/` | Export of the integer example networks. |
| `paper/paper.tex` | The paper. |

## Related work

- [Flocq](https://flocq.gitlabpages.inria.fr/) is the IEEE-754 formalization the
  development computes in, and its double-rounding results are instantiated in
  `Float_error.v`.
- [CompCert's verified floating point](https://xavierleroy.org/publi/floating-point-compcert.pdf)
  is the model for extracting Flocq arithmetic and for the native float
  boundary.
- [CoqInterval](https://coqinterval.gitlabpages.inria.fr/) discharges the series
  bounds.
- [LAProof](https://github.com/VeriNum/LAProof) proves forward and mixed backward
  error bounds for dot products and matrix-vector products.
- [MLCert](https://github.com/OUPL/MLCert) certifies generalization bounds for
  extracted machine-learning programs in Coq.
- [Cheerios](https://github.com/uwplse/cheerios) is verified serialization for
  Coq.

## License

MIT
