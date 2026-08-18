(** * Inductive extraction of the Llama development

    The counterpart of Qwen_inductive.v for the Llama path: binary32 stays
    Flocq's inductive binary_float and Z stays its inductive datatype, so no
    floating-point boundary is trusted. runners/llama_ref.ml drives it, and the
    differential harness uses it as the oracle a numpy float32 implementation of
    the same operations is measured against. *)

Require Import Phases1_15_complete.
Require Import Llama.

Set Extraction Output Directory ".".

Extraction "llama_inductive.ml"
  binary32 f32_of_Z f32_zero f32_one f32_bytes_to_binary32
  f32_plus f32_minus f32_mult f32_div f32_neg f32_sqrt
  f32_dot f32_mat_vec_mul f32_vec_add f32_vec_mult
  f32_sigmoid f32_exp_approx f32_sum f32_softmax
  f32_silu f32_silu_vec f32_rmsnorm f32_sin f32_cos
  f32_slice f32_partial_rope f32_swiglu
  llama_attn_weights mk_llama_attn_weights la_q la_k la_v la_o
  llama_mlp_weights mk_llama_mlp_weights lm_gate lm_up lm_down
  f32_llama_attn f32_llama_layer f32_llama_stack f32_llama_forward f32_llama_logits
  f32_embed_tokens f32_causal_attention f32_concat_heads.
