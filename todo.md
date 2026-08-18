# Outstanding work

## The Llama and Qwen forward bounds, composed

`theories/Float_error.v` carries the propagation relation over every Llama and
Qwen3.5 primitive, and over the Qwen layer, stack and logits. The Llama layer
has its pieces bounded but no composed statement of its own yet: what is
missing is the counterpart of `ok_qwen_wrap` for `f32_llama_layer`, and a stack
relation over it. Everything it would rest on is already proved.

The trigonometric bounds are stated on the reduced argument, since the magic
constant reduction is the identity in exact arithmetic and the rounding step in
binary32. Carrying a bound across the reduction itself would need a relation
between the two evaluations rather than the propagation relation used here.

## Regularity witnesses

`f32_dot_regular` has a witness, so the dot-product bounds are visibly not
vacuous, and `f32_dot_backward_ones` reuses it. The larger records the layer
bounds rest on, `exp_reg`, `ln_reg`, `ca_reg`, `log_reg`, `rms_reg`,
`dstep_reg` and the rest, have none. Exhibiting one satisfying assignment per
record would put the composed statements on the same footing.

## Bound tightness

The forward bound is worst-case and compounds with depth, so at GPT-2 scale it
is far larger than the divergence `RESULTS.md` measures. `f32_dot_backward`
gives the backward-error form for the dot product, where the perturbation is
relative and sized by that one product. Carrying the same treatment up through
the linear layers and the block would replace the compounding forward bound
with a perturbation of the weights.

## Scale of the architecture sweep

`scripts/experiment_arch.py` sweeps four samples per configuration against the
inductive reference, which is slow enough that the models are small. The GPT-2
sweep runs sixteen samples over wider models. Raising the architecture sweep to
the same scale wants either the native extraction as the oracle or more patience.
