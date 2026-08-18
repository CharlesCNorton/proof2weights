(** * Inductive extraction of the Qwen3.5 development

    Extract.v maps binary32 to the host's hardware float, which is the trusted
    fast path. This file extracts the same Qwen definitions with binary32 left
    as Flocq's inductive binary_float and Z left as its inductive datatype, so
    every float operation is the computational content of its proof and no
    floating-point boundary is trusted at all. It is the proof artifact rather
    than the fast path: bignum arithmetic runs at microseconds per operation, so
    only a toy configuration is practical, which is what runners/qwen_ref.ml
    drives.

    The directives come from Phases1_15_complete.v through the Require below;
    nothing here overrides them, which is exactly the point. *)

Require Import Phases1_15_complete.
Require Import Llama.
Require Import Qwen.

Set Extraction Output Directory ".".

Extraction "qwen_inductive.ml"
  binary32 f32_of_Z f32_zero f32_one f32_two f32_bytes_to_binary32
  f32_plus f32_minus f32_mult f32_div f32_neg f32_abs f32_sqrt
  f32_dot f32_mat_vec_mul f32_mat_transpose f32_vec_add f32_vec_mult
  f32_sigmoid f32_exp_approx f32_sum f32_softmax
  f32_silu f32_silu_vec f32_rmsnorm f32_sin f32_cos
  f32_max2 f32_log_unit f32_softplus f32_l2norm f32_slice
  f32_rmsnorm_zc f32_rmsnorm_gated
  f32_conv_window f32_conv_step f32_causal_conv1d
  f32_delta_step f32_delta_scan f32_delta_state0 f32_delta_decay f32_delta_prep_q
  f32_partial_rope f32_swiglu f32_gate_sigmoid
  qwen_delta_weights mk_qwen_delta_weights
  qd_in_qkv qd_in_z qd_in_a qd_in_b qd_conv_w qd_a_log qd_dt_bias qd_norm_w qd_out
  qwen_attn_weights mk_qwen_attn_weights
  qa_q qa_k qa_v qa_o qa_q_norm qa_k_norm
  qwen_mlp_weights mk_qwen_mlp_weights qm_gate qm_up qm_down
  f32_qwen_delta_head f32_qwen_delta_head_out f32_qwen_delta_mix
  f32_qwen_attn_mix f32_qwen_wrap f32_qwen_final f32_qwen_logits
  f32_qwen_stack f32_qwen_forward f32_qwen_next_token_logits
  f32_embed_tokens
  f32_causal_attention f32_concat_heads f32_split_into_heads.
