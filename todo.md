# Outstanding work

## Llama propagation lemmas

`theories/Llama.v` has shape lemmas and no propagation lemmas. `f32_rmsnorm`,
`f32_silu_vec`, `f32_sin` and `f32_cos` need `ok_` forms before the Llama
forward can carry a bound. `f32_sin` and `f32_cos` reduce the argument with the
add-then-subtract magic-constant trick, which is a branch on magnitude in
disguise and needs its own premise, as the exponential's saturation does.
`ok_silu_vec` in `theories/Float_error.v` already covers `f32_silu_vec`, and
`ok_rms_scale` covers everything `f32_rmsnorm` needs except the final weight
multiply, so the remaining work is concentrated in the trigonometry.

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

## Setup scripts since the work-directory change

`scripts/gpt2_setup.py` and `scripts/smollm_setup.py` have not been re-run since
the `P2W_WORK` change. The Coq and OCaml halves of the build, including
`make -C tools all` and `make -C tools verify`, now run end to end from a fresh
clone on a single host.

## Bound tightness

The forward-pass bound is worst-case and compounds with depth, so at GPT-2
scale it is far larger than the divergence `RESULTS.md` measures. A running
bound in the style of `f32_dot_err_bound`, or a backward-error statement that
exhibits the computed result as the exact evaluation of a perturbed expression,
would both be tighter.
