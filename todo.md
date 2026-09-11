# Outstanding work

## Bounds that survive depth

`gpt2_logits_bounded` computes, during the forward pass, a bound on each
logit's distance from the exact real network, and every annotated operation
charges its worst-case sensitivity to each operand independently. The bound is
non-vacuous on one- and two-layer models and infinite on GPT-2 small from the
first GELU on. Carrying first-order sensitivities instead of radii (affine
arithmetic, or a mixed forward-backward bound in the style of
`f32_dot_backward`) and using the scale invariance of layer normalization are
the natural next steps.

## Annotated passes for the Llama and Qwen3.5 paths

`RunErr.v` states the generic forward pass and the relational theorem for
GPT-2 only. RMSNorm, rotary embedding, SwiGLU, the logarithm, the depthwise
convolution and the gated delta step need annotated operations and generic
counterparts of `f32_llama_forward` and `f32_qwen_forward`.

## Backward error above the linear layers

`f32_dot_backward`, `f32_mat_vec_mul_backward` and `logits_backward` give the
backward-error form for the dot product, the matrix-vector product and the
tied-embedding projection of all three models. Layer normalization, the
exponential and softmax need a statement that perturbs the weights rather than
the products.
