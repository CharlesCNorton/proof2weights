# proof2weights

proof2weights defines a neural network, up to and including a GPT-2 transformer,
in the Rocq prover, with arithmetic in IEEE-754 binary32 through the Flocq
library, and extracts it to OCaml. The extracted program loads a `.safetensors`
file and performs inference using the rounding behavior the proofs specify. The
development loads the published 124-million-parameter GPT-2 weights, runs the
forward pass over all twelve transformer blocks, and produces the same
next-token prediction as the PyTorch reference implementation. The same approach
extends to the Llama architecture: the development also runs SmolLM2, an
instruction-tuned model, and the verified forward generates a coherent chat
response. It also runs Qwen3.5, whose layers are mostly linear-attention state
recurrences rather than softmax attention, and reproduces the reference's
next-token ranking there too. The weights written to disk are computed by the
same definitions the proofs concern, and the floating-point arithmetic executed
at inference time is the arithmetic the development reasons about.

The core development is contained in a single file,
`theories/Phases1_15_complete.v`. It compiles under Rocq 9 with `coq-flocq` and
extracts to standalone OCaml. `theories/Audit.v` runs `Print Assumptions` over
the headline theorems: the integer serialization, storage, and inference results
report "Closed under the global context", and the float results report the four
classical axioms that `Coq.Reals` introduces and that Flocq's operations
inherit. A second file, `theories/Extract.v`, re-extracts the same development
with floating-point arithmetic mapped to the host's hardware float, which
reduces the time for one forward pass from minutes to seconds.

## Motivation

A common approach to verifying a neural network establishes a property in a
proof assistant and then implements the weights and inference code separately in
Python or C for deployment. The deployed numbers are a transcription of the
proven numbers, and the deployed arithmetic is a separate implementation of the
proven arithmetic. The two are not guaranteed to agree, and in floating point
they frequently differ.

In proof2weights the weights are defined in Rocq, serialized by a Rocq function,
and that serializer is extracted and executed, so the bytes written to disk are
produced by the definitions the proofs concern. The reader is also Rocq: a
function parses `.safetensors`, decodes IEEE-754 values, and assembles a typed
model, and the forward pass that consumes the model is extracted from the same
source. No separate implementation is introduced.

```
        Rocq definitions + Flocq IEEE-754
                     |
              extraction (verbatim)
                     |
                 OCaml binary
                 /          \
   write .safetensors      read .safetensors, run inference
```

## Integer-exact core

The foundation is exact integer serialization. Tensors are records of a name, a
shape (`list nat`), and data (`list Z`); a network is a list of tensors. Values
serialize to little-endian `i32`, and the round trip is proved: decoding the
encoding of any 32-bit integer returns that integer (`roundtrip_z`). On top of
this sits a safetensors writer (an 8-byte little-endian header length, a JSON
header, then the concatenated tensor bytes) and the inverse readers.

The development also includes the serialization and validation machinery a
deployment uses, each with its definitions and the lemmas that state their
properties: shape and bounds validation with sound boolean checkers, signed
`int8` and packed `int4` quantization with proved range containment, lazy
tensors whose force-after-defer is the identity, and JSON proof certificates
with attestations and provenance chains.

The three storage transformations are proved to round trip on every input.
Splitting a tensor into fixed-size chunks and reassembling them returns the
original data (`reassemble_split_into_chunks`). Run-length decoding inverts
run-length encoding (`rle_roundtrip`), and that lifts to whole tensors and
whole networks (`decompress_compress_tensor`, `decompress_compress_network`).
Splitting a network into shards under a byte budget and unsharding them returns
the original network (`unshard_shard_network`).

## IEEE-754 floating point

The floating-point type is binary32, not a fixed-point approximation. The
`binary32` type is Flocq's `binary_float` at precision 24 and exponent bound
128, and the arithmetic (`f32_plus`, `f32_mult`, `f32_div`, `f32_neg`,
`f32_abs`, `f32_compare`, and square root via Flocq's `Bsqrt`) is
round-to-nearest, ties-to-even. Bit patterns convert both ways for binary32,
binary16, and bfloat16, and the encode/decode round trips are proved
(`roundtrip_f32`, `roundtrip_f16`), as is `B2R f32_one = 1`.

The transcendentals are built from these primitives. The exponential is range
reduced: the argument is saturated to the binary32 exponential range, divided by
256 so it lands where a short Taylor series is accurate, evaluated by a six-term
Taylor series, and squared eight times, which remains finite across the input
range. Sigmoid is `1 / (1 + exp(-x))`, GELU is `x * sigmoid(1.702 x)`, `tanh` is
`2 sigmoid(2x) - 1`, ReLU is a clamp, and softmax subtracts the row maximum
before exponentiating. These run in binary32 and preserve vector and matrix
dimensions, which is proved for each.

## Neural-network library

The library includes the following components, each with a dimension-
preservation lemma:

- Dense layers, residual blocks, and bottleneck blocks.
- Convolution weight records with `im2col`-style flattening, max and average
  pooling with non-negativity bounds.
- Batch, layer, and group normalization; token and learned position embeddings
  and their sum.
- Vanilla RNN, LSTM, and GRU cells, their sequence unrollings (output length
  equals input length, proved), bidirectional wrapping, and an RNN sequence
  classifier.
- Scaled dot-product attention, multi-head attention with head splitting and
  concatenation, causal masking, and cross-attention, each shown to preserve
  sequence length.
- Pre-norm and post-norm transformer blocks, feed-forward sublayers, full
  encoder and decoder layers, and assembled GPT-style, BERT-style, and full
  encoder-decoder models.

Both an integer fixed-point path and an IEEE-754 float path exist for the
transformer operations; the float path is used for the GPT-2 model below.

## The float GPT-2

The transformer is assembled end to end in binary32. The configuration record
matches the published GPT-2 family, and the parameter counts are proved by
reflection: `gpt2_total_params gpt2_small` reduces to `124439808`, with the
medium, large, and XL counts likewise pinned, alongside head-dimension and
feed-forward-expansion checks.

A typed weight model (`f32_model_weights`) holds the token and position
embeddings, a list of per-block weights (two layer norms, the fused QKV and
output attention projections, and the two MLP projections), and the final layer
norm. The forward pass embeds tokens and positions, runs the pre-norm decoder
stack (layer norm, multi-head causal attention, residual, layer norm, MLP,
residual), applies the final layer norm, and projects through the tied embedding
to logits. Greedy generation decodes over those logits, and the generated
sequence is proved to always extend the prompt, so the prompt is a prefix of the
output. A finiteness certificate checks that an output contains no NaN or
infinity, with a soundness lemma that a passing check implies every entry is a
finite IEEE-754 value, and shape validators reject weights whose dimensions do
not match the configuration.

## The safetensors loader

The loader is implemented in Rocq. The header length is read as a little-endian
`u64`, the JSON header is parsed into a string, and a JSON scanner (whitespace,
natural numbers, quoted strings, integer arrays, and a substring key search)
extracts each tensor's `dtype`, `shape`, and `data_offsets`. A named tensor is
loaded by scoping to its key, reading its byte offsets, slicing the data
section, and decoding little-endian f32 into `binary32`. The model loader
constructs every GPT-2 tensor name (including the per-layer `h.<i>.` prefixes,
built with a verified `nat`-to-string), loads and reshapes each, and assembles
`f32_model_weights`. The token-embedding matrix is proved to have `vocab_size`
rows and the assembled model to have exactly `n_layer` blocks. Decoding bytes to
`binary32` goes through Flocq's `b32_of_bits` composed with the single-NaN
collapse, so the loaded value is the IEEE-754 value the bytes denote.

## Inference on GPT-2 weights

The development loads the published 124-million-parameter GPT-2 base weights,
runs the IEEE-754 forward over all twelve blocks, projects every one of the
50257 logits through the tied embedding, and predicts a next token. For the
prompt "The quick brown" the greedy next token is "ie" (token 494), which
matches the prediction of the PyTorch reference for the same prompt, and the top
five candidates are returned in the same order. The logits computed here are
offset from the reference's by approximately one unit, because the MLP uses the
`x * sigmoid(1.702 x)` form of GELU while GPT-2 was trained with the tanh-based
`gelu_new`; the offset is uniform and does not change the ranking or the argmax.

Applying the verified definitions to a model of this size requires addressing
two performance constraints. The list-based loader cannot ingest a file of
roughly 500 MB, because representing it as a `list byte` builds a linked list of
hundreds of millions of boxed values, and the extracted matrix transpose inside
the linear layer is quadratic in the output dimension through list indexing,
which is acceptable at small dimensions but not at 768 and 3072. The GPT-2
runner therefore reads the file bytes natively, decodes each value with the
verified `f32_bytes_to_binary32`, and composes the verified primitives
(`f32_mat_vec_mul`, `f32_dot`, `f32_layer_norm_2d`, `f32_causal_attention`,
`f32_concat_heads`, `f32_gelu_vec`, `f32_add_matrices`) in the order the proven
block specifies, decoding each weight matrix already transposed so the verified
matrix-vector product computes the same dot products the verified linear layer
would. Weights are streamed block by block, so memory remains in the low
gigabytes. Every floating-point value is produced by the verified operators;
only the byte addressing is native. In a mode that prints the full logit matrix,
the runner produces output bit-identical to the top-level verified
`f32_gpt2_logits` on a small fixture, confirming the composition agrees with the
verified forward.

## Inference and generation on a Llama model

The same approach extends to the Llama architecture, which differs from GPT-2 in
four respects: RMSNorm in place of layer normalization, rotary position
embeddings (RoPE) in place of learned position embeddings, grouped-query
attention, and a SwiGLU feed-forward network. `theories/Llama.v` adds the primitives
these require on top of the binary32 development. RMSNorm and SiLU compose
existing operations; `f32_sin` and `f32_cos`, which RoPE needs, are defined by
argument reduction modulo 2*pi and a Taylor polynomial. The runner loads
SmolLM2-135M-Instruct (576 hidden, 30 layers, 9 query and 3 key/value heads,
intermediate 1536, tied embeddings) and composes these primitives into the
forward pass: RMSNorm, the query/key/value projections, RoPE applied to the
per-head query and key vectors, grouped-query causal attention, the output
projection, and the SwiGLU block, with weights streamed per layer.

On the chat prompt for "What is the capital of France?", the verified forward
reproduces the PyTorch reference's full top-eight next-token ranking, with logits
agreeing to four decimal places. Greedy generation over the verified logits
produces "The capital of France is Paris." Generation uses a key/value cache: the
prompt is processed once, its per-layer rotary keys and values are stored, and
each new token is computed as a single position attending over the cache, which
is bit-identical to a full recompute and removes the per-token cost of
reprocessing the prefix. `scripts/chat.py` drives an interactive session against
either model: it applies the model's chat template, sends the token ids to the
runner, and decodes the generated ids, so each reply is produced entirely by the
verified operators.
`theories/Llama.v` also names the composition itself, so the layer, the stack
and the forward pass are Rocq functions rather than only an order the runner
follows: `f32_llama_layer` is RMSNorm, the projections, rotary embedding,
grouped-query causal attention, the output projection and SwiGLU inside their
two residuals, and `f32_llama_forward` embeds, runs the stack and applies the
final norm. `runners/llama_ref.ml` executes exactly that against the inductive
extraction, where no floating-point boundary is trusted at all.

`f32_sin` and `f32_cos` are compositions of the four arithmetic primitives, so
the native build extracts them structurally rather than calling the host libm:
it runs the same argument reduction and the same Taylor polynomial the
definitions specify, on the same trusted hardware-float boundary as the rest of
the development, with no library substitution anywhere in the forward.

## Inference on a hybrid SSM model

Qwen3.5 does not follow the Llama decoder pattern, and running it verified
needed a new family of primitives rather than a change of dimensions. Of its
twenty-four layers, eighteen are gated DeltaNet blocks: linear attention
carrying a per-head state matrix through the sequence, decayed and corrected at
every token, with no softmax anywhere. The remaining six are full attention,
gated on the output and rotating only a quarter of each head.

`theories/Qwen.v` adds what that requires. The DeltaNet decay term is
`-exp(A_log) * softplus(a + dt_bias)`, so a logarithm is needed, and it is the
one genuinely new transcendental in the development. Written in the stable form
`softplus x = max x 0 + log (1 + exp (-|x|))`, the logarithm is only ever taken
on (1, 2], where the arctanh series `log m = 2 * artanh ((m-1)/(m+1))`
converges quickly with no range reduction, which keeps the new function
branch-free. Alongside it are Euclidean normalisation for the query and key,
the two RMSNorm variants Qwen uses that Llama does not (one storing its weight
zero-centred and applying `1 + w`, one multiplying by `silu` of a separate
gate), the depthwise causal convolution the DeltaNet input passes through, the
gated delta rule itself, and partial rotary embedding. For text-only input the
interleaved multimodal RoPE collapses to ordinary RoPE, because the three
positional axes carry identical indices, so rotating the prefix is the whole
story.

The runner streams one layer of weights at a time and never holds the
embedding, decoding its rows on demand for the lookup and streaming them for
the logit projection, so a three-gigabyte model runs in about four gigabytes of
memory. On the chat prompt for "What is the capital of France?" the verified
forward returns the same eight highest-scoring next tokens as the PyTorch
reference, in the same order, with logits agreeing to four decimal places:
token 760 at 25.6649, then 57590, 3733, 332, 61445, 7732, 16 and 47358.

Greedy generation carries three caches, so a decode step costs one token of
arithmetic rather than a re-run of the prefix: the keys and values of the
full-attention layers, the recurrent state matrix of each DeltaNet head, and
the trailing convolution window. Over eight tokens it emits
`760, 6511, 314, 9338, 369, 2972, 57590, 159034`, which is byte-identical to
the reference and decodes to "The capital of France is **Paris**." The prompt
and those eight tokens together take five and a half minutes.

`runners/qwen_ref.ml` runs the same composition against the inductive
extraction on a small configuration, calling `f32_qwen_forward` directly rather
than composing the primitives itself, so the definitions the error bound is
stated about are the ones executed.

The primitives are defined in Rocq with their shape lemmas and extracted, so
the arithmetic that runs is the arithmetic the definitions specify.
`theories/Float_error.v` carries the propagation relation over every one of
them and on to the logits: the logarithm and softplus, both RMSNorm variants,
the depthwise convolution and its window, the gated delta step and the scan
that threads its state, partial rotary embedding, SwiGLU, and the query
preparation. `theories/Qwen.v` then assembles the layers those primitives make
up, and the error file bounds the two mixers, the residual pair they sit in,
the alternating stack and the tied-embedding projection, so the composed bound
reaches the Qwen logits the way it reaches the GPT-2 logits.

## Proof-carrying receipts

A generation can emit a receipt that binds the result to the exact weights (by
checksum), the prompt, the full output token sequence, and the IEEE-754
semantics. `theories/Receipt.v` defines the receipt, a checker `verify_receipt`, and its
soundness and completeness theorems; the prompt-preservation guarantee follows
from the proven generation property `gpt2_generation_preserves_prompt`. A receipt
is checked without trusting the producer: recompute the weight checksum from the
file, re-run the deterministic verified generation, and compare both against the
receipt. `scripts/receipt.py emit` writes a receipt for an answer and
`scripts/receipt.py verify` recomputes and re-runs to confirm it. Both the
SmolLM2 and the Qwen3.5 runner print the checksum of the weight file they
loaded, so either model can be receipted. Because the
forward is a pure verified function, the regeneration is reproducible, so anyone
holding the weights and the receipt can confirm that the recorded output is
exactly what the model produces under the proven semantics.

## Two extraction modes

The development supports two extraction modes from the same source.

The default mode, from `theories/Phases1_15_complete.v`, keeps `binary32` as Flocq's
inductive `binary_float` and keeps `Z` and `positive` as their Coq inductive
datatypes, so every integer operation, and therefore every float operation built
on it, is the computational content of its proof. `Z` is not mapped to native
machine `int`, because that mapping is unsound when a mantissa-alignment shift or
an intermediate product exceeds the representable range, in which case binary32
addition of operands with a large exponent gap produces an incorrect result.
`nat`, used only for indices, dimensions, and token identifiers and always
small, extracts to native OCaml `int`, and `ascii` extracts to OCaml `char` with
a destructuring matcher. This mode introduces no trusted floating-point
boundary; the arithmetic is exactly Flocq's. Its bignum arithmetic runs at
microseconds per operation, so a full GPT-2 forward pass takes tens of minutes.

The native mode, from `theories/Extract.v`, re-extracts the same development with
`binary32` mapped to the host's hardware float, in the manner CompCert extracts
its verified floats and treats the IEEE-754 agreement as a trusted boundary at
the OCaml level. Each operation is the binary64 result rounded to binary32
through an `Int32` bit round-trip. That the second rounding is harmless is
proved in `theories/Float_error.v`: for operands representable in binary32, rounding the
exact real result of `+`, `-`, `*`, `/` or `sqrt` to binary64 and then to
binary32 gives the same value as rounding it straight to binary32, so the
extracted value is the binary32 value Flocq specifies. The proofs instantiate
Flocq's double-rounding development at the two formats, where the width
conditions hold because binary64's 53 bits of significand exceed binary32's 24
by the required margin. Decoding reads the four little-endian bytes directly to
a binary32 with `Int32.float_of_bits`. What this mode assumes rather than proves
is the OCaml boundary itself, that the host's `float` is IEEE-754 binary64 with
round-to-nearest-ties-even and that `Int32.bits_of_float` rounds to nearest
binary32. It runs approximately three hundred times faster.

The two modes were compared directly. On a small fixture their logits are
bit-identical to nine digits. On the full 124-million-parameter GPT-2 they
produce identical token identifiers and logits that agree to four decimals, the
inductive mode in tens of minutes and the native mode in approximately seven
seconds.

## What is proved

The development proves:

- Serialization round trips: `i32`, and the binary32 and binary16 bit-pattern
  encode/decode identities.
- Storage round trips on every input: chunked split and reassembly, run-length
  compression at the value, tensor, and network level, and network sharding
  under a byte budget.
- Dimension preservation through the stack: dense, attention (per-head and the
  masked path), softmax, layer norm, the transformer blocks, the RNN family, and
  the float linear-algebra and attention primitives.
- Soundness bridges from boolean checkers to propositions: shape validity, value
  bounds, network verification, and the float finiteness certificate.
- GPT-2 parameter counts and configuration validity by reflection.
- Quantization range containment, pooling bounds, lazy-tensor round trip, and
  reflexivity of network equality.
- Prompt preservation under greedy generation, for both the integer and float
  models.
- Correct rounding and a half-ULP accuracy bound for each float primitive:
  multiplication, addition, division, and square root each return the
  round-to-nearest, ties-to-even result of the exact real operation and differ
  from it by at most half a ULP (`theories/Float_error.v`).
- Shape preservation end to end for the float model: the forward pass returns
  one row per input token, and the logit matrix has one row per token and
  `vocab_size` columns (`f32_gpt2_forward_rows`, `f32_gpt2_logits_rows`,
  `f32_gpt2_logits_row_width`).
- Harmlessness of the native build's intermediate rounding: for binary32
  operands, rounding the exact real result of `+`, `-`, `*`, `/` or `sqrt` to
  binary64 and then to binary32 equals rounding it directly to binary32
  (`theories/Float_error.v`).
- A composed error bound for the dot product: under an explicit regularity
  premise, the extracted `f32_dot` differs from the exact real inner product of
  its operands by at most a running sum of per-step roundoffs
  (`theories/Float_error.v`).
- A worst-case error bound for the whole float forward pass, from the five
  primitives through layer normalization, the exponential, softmax, causal
  attention, the transformer block and the block stack, to the logits
  (`theories/Float_error.v`).
- The same bound over the Qwen3.5 primitives and up to its logits: the
  logarithm, softplus, both RMSNorm variants, the depthwise causal convolution,
  the gated delta step and scan, partial rotary embedding, SwiGLU and the query
  preparation, then the two mixers, the residual pair, the alternating stack
  and the tied-embedding projection (`theories/Float_error.v`).
- The Llama primitives under the same relation: RMSNorm, SiLU over a vector,
  and the sine and cosine Taylor polynomials. The argument reduction is where
  the two evaluations deliberately part company, because adding and subtracting
  the magic constant is the identity in exact arithmetic and the rounding step
  in binary32, so the trigonometric bounds are stated on the reduced argument
  (`theories/Float_error.v`).
- A backward-error statement for the dot product: the computed value is exactly
  the real inner product of the same operands with each product scaled by a
  factor within `(1 + u)^(n+1) - 1` of one, so the perturbation is relative and
  its size is set by the length of that one dot product rather than by the depth
  of the surrounding network (`f32_dot_backward`).

The float arithmetic is exactly Flocq's, and each operation is proved to land
within half a ULP of the exact real result, so the extracted executable is a
faithful binary32 computation with a per-operation accuracy bound.
`theories/Float_error.v` composes those per-operation facts along the dot product, the
primitive every linear layer and every attention score is built from, and bounds
the distance between the extracted `f32_dot` and the exact real inner product of
the same values. Because each step is measured against the exact sum using the
float accumulator the previous step produced, the per-step errors add rather
than compounding geometrically, and the bound is a running sum over the
intermediates the computation itself visits. The premise, that every step has a
finite accumulator and that neither the product nor the sum overflows or falls
below the smallest normal binary32 magnitude, is recorded explicitly as
`f32_dot_regular` rather than left implicit, and a witness is exhibited so the
bound is not vacuous.

The same file carries that composition through the rest of the network, to
the logits. The method is uniform because the forward pass is: every scalar the
network computes comes from one of five primitives, and everything else is list
plumbing that performs no arithmetic. So the file proves one propagation lemma
per primitive, a handful of structural lemmas for the plumbing, and then walks
the stack: dot products and linear layers, layer normalization, the
range-reduced exponential and its Taylor series, sigmoid and GELU, the MLP, the
row maximum and softmax, scaled dot-product causal attention with head splitting
and concatenation, the transformer block, the block stack, the embeddings, and
the logit projection. The relation carried is that a binary32 value is finite
and within `d` of the real number exact arithmetic would have produced; each
primitive turns `d`-close inputs into a result that is `u * M + L * d`-close,
where `M` bounds the magnitudes in play and `L` bounds how much an operation can
amplify an existing error, and a stage of arithmetic depth `k` iterates that
affine map `k` times.

The hypotheses are explicit rather than buried: magnitudes stay under `M`,
denominators and radicands stay above `m`, no intermediate falls below the
smallest normal binary32 magnitude, and the exponential's saturation is not
engaged, so the float and the real evaluation follow the same path. The bound is
worst-case, so it compounds with depth and is far larger at GPT-2 scale than the
divergence `RESULTS.md` measures; what it establishes is that the divergence is
bounded at all, by a quantity computed from the network's own dimensions.

## Use as an IEEE-754 reference

Because the extracted forward pass is deterministic and pins every rounding and
every reduction order, it is a fixed reference against which other inference
implementations can be measured. Production float32 implementations differ from
each other and across hardware because of fused multiply-add contraction, BLAS
summation order, and GPU nondeterminism; this development fixes a single
evaluation and proves it is the one the IEEE-754 semantics, as formalized by
Flocq, specify. The repository includes a differential-testing harness that runs
the same network through this reference and through a numpy float32
implementation of the identical operations and reports the divergence and any
next-token disagreements. The sweep in `RESULTS.md` moves one dimension at a
time off a base model and finds that the divergence tracks the model width: mean
absolute logit error rises by a factor of about nineteen from `d_model` 8 to 64
at fixed depth, while depth contributes mildly across one to eight layers and
sequence length and vocabulary size hardly at all, which is what accumulating
longer dot products in a different reduction order predicts. No configuration
produced a next-token disagreement. At full scale, the reference produces the
same next-token prediction as the PyTorch implementation on GPT-2 weights, as
described above.

The Llama and Qwen3.5 paths carry the same harness, against the inductive
extraction of `f32_llama_forward` and `f32_qwen_forward` rather than the native
build, so the oracle there trusts no floating-point boundary at all. Weights are
handed to both sides as raw binary32 bit patterns, so the reference and the
mirror start from identical values and the only difference is the reduction
order. The tables in `RESULTS.md` put the Llama divergence in the same range as
GPT-2's and the Qwen divergence about an order of magnitude higher, which is
what a recurrence that carries a state matrix across the sequence, on top of a
logarithm and a convolution, predicts. No configuration produced a next-token
disagreement there either.

## Building and running

Requires Rocq 9 with `coq-flocq`, and OCaml (4.14 or later). The GPT-2 fetch and
the differential harness additionally use Python with `torch`, `transformers`,
`numpy`, and `safetensors`.

```bash
# Everything Coq lives in theories/, and extraction output lands beside it.
cd theories

# Compile the development and extract the inductive OCaml
# (phases1_15_complete.{ml,mli}).
rocq compile -R . "" Phases1_15_complete.v

# The Llama and Qwen3.5 primitives.
rocq compile -R . "" Llama.v
rocq compile -R . "" Qwen.v

# The numerical semantics: correct rounding per operation, the native build's
# rounding step, and the composed error bounds up to the logits.
rocq compile -R . "" Float_error.v

# The native re-extraction. One compile emits phases1_15_native.{ml,mli},
# llama_native.{ml,mli} and qwen_native.{ml,mli}.
rocq compile -R . "" Extract.v

# Report the assumptions behind the headline theorems.
rocq compile -R . "" Audit.v
cd ..

# Build the inductive (exact) and native (fast) GPT-2 runners.
ocamlopt -rectypes -w -a -I theories theories/phases1_15_complete.mli theories/phases1_15_complete.ml runners/gpt2_talk.ml -o gpt2_talk
ocamlopt -rectypes -w -a -I theories theories/phases1_15_native.mli theories/phases1_15_native.ml runners/gpt2_talk_native.ml -o gpt2_talk_native

# Fetch GPT-2, save f32 weights with the loader's tensor names, and print the
# PyTorch reference prediction.
python scripts/gpt2_setup.py

# Predict the next token. Arguments: mode (full|next), file, n_embd, n_head,
# n_layer, n_inner, vocab, n_positions, then the comma-separated token ids.
./gpt2_talk_native next gpt2.safetensors 768 12 12 3072 50257 1024 464,2068,7586

# Llama path: build the runner against the native extraction.
ocamlopt -rectypes -w -a -I theories theories/llama_native.mli theories/llama_native.ml runners/llama_talk_native.ml -o llama_talk_native

# Fetch SmolLM2, save f32 weights and the rotary frequencies, capture the oracle.
python scripts/smollm_setup.py

# Qwen3.5 path: build the runner against the same native extraction.
ocamlopt -rectypes -w -a -I theories theories/qwen_native.mli theories/qwen_native.ml runners/qwen_talk_native.ml -o qwen_talk_native

# Fetch Qwen3.5-0.8B, save the text decoder as f32 weights, capture the oracle.
python scripts/qwen_setup.py

# Next-token logits. Arguments: file, d, n_layer, n_head, n_kv, head_dim,
# rotary_dim, ff, vocab, deltanet heads, deltanet head_dim, conv kernel, ids.
./qwen_talk_native qwen.safetensors 1024 24 8 2 256 64 3584 248320 16 128 4 <ids>

# Chat against either model. Tokenization runs in the script; the verified
# forward runs in the built runner.
python scripts/chat.py smollm "What is the capital of France?"
python scripts/chat.py qwen "What is the capital of France?"

# The inductive references for the two later architectures, which trust no
# floating-point boundary, and the differential sweep that measures a numpy
# float32 implementation of the same operations against them.
cd theories
rocq compile -R . "" Llama_inductive.v
rocq compile -R . "" Qwen_inductive.v
cd ..
ocamlopt -rectypes -w -a -I theories theories/llama_inductive.mli theories/llama_inductive.ml runners/llama_ref.ml -o llama_ref
ocamlopt -rectypes -w -a -I theories theories/qwen_inductive.mli theories/qwen_inductive.ml runners/qwen_ref.ml -o qwen_ref
python scripts/experiment_arch.py
```

The integer path has its own build. `make -C tools` compiles the development,
builds the OCaml wrapper around the extracted serializer, and writes the two
example networks to `.safetensors`, checking each file against the bytes Coq's
own `serialize_list` produces; `make -C tools verify` then reads them back with
the Python `safetensors` library and checks the values.

`runners/ref_logits.ml` is a smaller runner that uses the verified list-based loader on
toy `.safetensors` files: it reads file bytes as inductive `Z`, calls
`parse_header_size` and `parse_header_string` to split the header from the data,
calls `f32_load_model` to assemble the typed weights, and calls
`f32_gpt2_logits`.

## Repository layout

| Path | Contents |
|------|----------|
| `theories/Phases1_15_complete.v` | The development: definitions, proofs, and the inductive extraction. |
| `theories/Float_error.v` | Numerical semantics: correct rounding per operation, the native build's rounding step, and the composed error bounds for the dot product and the whole forward pass. |
| `theories/Llama.v` | Llama primitives: RMSNorm, SiLU, `f32_sin`/`f32_cos` for RoPE, rotary embedding, slicing and SwiGLU; then the layer, the stack and the forward pass. |
| `theories/Qwen.v` | Qwen3.5 primitives: the logarithm and softplus the DeltaNet decay needs, Euclidean normalisation, the two extra RMSNorm variants, the depthwise causal convolution, the gated delta rule, and partial RoPE; then the layers they assemble into, the stack and the forward pass. |
| `theories/Extract.v` | Native re-extraction mapping `binary32` to hardware float; emits the GPT-2, Llama and Qwen3.5 targets. |
| `theories/Llama_inductive.v`, `theories/Qwen_inductive.v` | Inductive extraction of the two later architectures, with no trusted float boundary; the oracle the differential harness measures against. |
| `theories/Receipt.v` | Inference receipt, the checker `verify_receipt`, and its soundness and completeness. |
| `theories/Audit.v` | `Print Assumptions` report for the headline theorems. |
| `runners/gpt2_talk.ml` | GPT-2 runner against the inductive extraction (exact). |
| `runners/gpt2_talk_native.ml` | The same runner against the native extraction (fast). |
| `runners/llama_talk_native.ml` | SmolLM2 runner: the verified Llama forward and greedy generation. |
| `runners/qwen_talk_native.ml` | Qwen3.5 runner: the verified gated-DeltaNet and gated-attention forward. |
| `runners/ref_logits.ml` | Smaller runner using the verified list-based loader on toy models. |
| `runners/llama_ref.ml`, `runners/qwen_ref.ml` | The Llama and Qwen3.5 forwards against the inductive extraction, on small models. |
| `runners/float_smoke.ml`, `runners/float_load_run.ml`, `runners/test_bplus.ml` | Small drivers for the float path. |
| `scripts/gpt2_setup.py` | Fetches GPT-2, saves f32 weights with the loader's tensor names, prints the PyTorch reference. |
| `scripts/smollm_setup.py` | Fetches SmolLM2, saves f32 weights and rotary frequencies, prints the PyTorch oracle. |
| `scripts/qwen_setup.py` | Fetches Qwen3.5-0.8B, saves the text decoder as f32 weights, prints the PyTorch oracle. |
| `scripts/models.py` | The models the runners can be driven against, and how to reach one. |
| `scripts/chat.py` | Interactive chat against either model: local tokenization, verified forward in the runner. |
| `scripts/receipt.py` | Emit and verify proof-carrying receipts for generated answers. |
| `scripts/tiny_gpt2_ref.py` | numpy reference of the identical computation, and a tiny `.safetensors` generator. |
| `scripts/experiment_gen.py`, `scripts/experiment_cmp.py` | Differential-testing harness for GPT-2: generate models, compare the reference against numpy. |
| `scripts/arch_ref.py`, `scripts/experiment_arch.py` | numpy mirrors of the Llama and Qwen3.5 forwards, and the sweep that compares them against the inductive reference. |
| `scripts/run_batch.sh`, `scripts/run_float_demo.sh` | Drive the harness and the toy-model demo. |
| `tools/` | Makefile and OCaml I/O wrapper that export the integer example networks to `.safetensors`. |

## Related work

- [Flocq](https://flocq.gitlabpages.inria.fr/) provides the IEEE-754
  formalization this development computes in.
- [CompCert's verified floating point](https://xavierleroy.org/publi/floating-point-compcert.pdf)
  is the model for extracting Flocq arithmetic to a real executable, and for the
  trusted native-float boundary in the native build.
- [MLCert](https://github.com/OUPL/MLCert) certifies generalization bounds for
  machine learning in Coq and extracts; its focus is bounds rather than a
  transformer running real weights.
- [verinncoq/converter](https://github.com/verinncoq/converter) verifies
  properties of externally trained networks.
- [Cheerios](https://github.com/uwplse/cheerios) is verified serialization for
  Coq.

## License

MIT
