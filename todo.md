# Outstanding work

## Qwen layer bounds

`theories/Float_error.v` carries the propagation relation over the Qwen3.5
primitives: `ok_f32_abs`, `ok_silu`, `ok_max2`, `ok_delta_decay_of`,
`ok_l2norm`, `ok_squares`, `ok_rms_scale`, `ok_gate_sigmoid`. It stops there.

Missing at the primitive level: `ok_log_unit`, `ok_softplus`,
`ok_rmsnorm_zc`, `ok_rmsnorm_gated`, `ok_conv_step`, `ok_causal_conv1d`,
`ok_delta_step`, `ok_delta_scan`, `ok_partial_rope`, `ok_swiglu`,
`ok_delta_prep_q`.

`ok_log_unit` is a chain of one division, seven powers, six divisions, a
six-term plus chain and a doubling, so depth `k + 17`. `ok_softplus` composes
it with `ok_max2` and `ok_exp_approx` at depth `k + 40`. `ok_delta_step` is the
only one with structure worth noting: it decays the state, reads the memory
with `ok_mat_vec_mul` over `ok_mat_transpose`, corrects by `beta`, and reads
out with the query, so its depth carries `2 * d_k` twice.

## Qwen composed bound to the logits

With the layer primitives in place, the chain that `ok_block_forward`,
`ok_blocks_forward`, `ok_gpt2_forward` and `ok_gpt2_logits_full` form for GPT-2
has a direct Qwen counterpart: a bound per DeltaNet layer, a bound per gated
attention layer, a `blocks_reg`-style stack relation over the alternating
layer types, and a top-level statement.

## Llama propagation lemmas

`theories/Llama.v` has shape lemmas and no propagation lemmas. `f32_rmsnorm`,
`f32_silu_vec`, `f32_sin` and `f32_cos` need `ok_` forms before the Llama
forward can carry a bound. `f32_sin` and `f32_cos` reduce the argument with the
add-then-subtract magic-constant trick, which is a branch on magnitude in
disguise and needs its own premise, as the exponential's saturation does.

## Differential harness beyond GPT-2

`scripts/experiment_gen.py` and `experiment_cmp.py` sweep the GPT-2 reference
against a numpy float32 implementation of the identical operations, and
`RESULTS.md` reports it. Neither the Llama nor the Qwen path has a numpy mirror
or a differential table.

## Inductive runner for Qwen

`runners/ref_logits.ml` runs the GPT-2 stack against the inductive extraction,
which is the proof artifact rather than the trusted fast path. Qwen has only a
native runner. An inductive Qwen runner would be slow enough that a toy
configuration is the only practical target.

## Qwen runner surface

`runners/qwen_talk_native.ml` takes one prompt and exits.
`runners/llama_talk_native.ml` additionally holds weights resident and answers
queries from stdin, which is what `scripts/llama_chat.py` drives. Qwen has no
serve mode and no chat driver.

## Receipts beyond SmolLM2

`theories/Receipt.v` and `scripts/llama_receipt.py` bind a generation to a
weight checksum, a prompt, an output sequence and the IEEE-754 semantics. The
script is SmolLM2-specific; the Qwen runner emits no checksum line.

## Build coverage

`make -C tools all` has not run end to end in a single invocation: no host here
carries Rocq with `coq-flocq` and OCaml together, so the Coq half and the OCaml
half were exercised on different machines. `scripts/gpt2_setup.py` and
`scripts/smollm_setup.py` have not been re-run since the `P2W_WORK` change.

## Bound tightness

The forward-pass bound is worst-case and compounds with depth, so at GPT-2
scale it is far larger than the divergence `RESULTS.md` measures. A running
bound in the style of `f32_dot_err_bound`, or a backward-error statement that
exhibits the computed result as the exact evaluation of a perturbed expression,
would both be tighter.
