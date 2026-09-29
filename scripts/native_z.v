(* The native build's roots, as Extract.v names them, extracted with Z,
   positive and N left inductive, so that every operation on them the build
   performs appears in the output. native_z.py runs it and reads the result. *)
Require Import Phases1_15_complete Llama Qwen Pretransposed Loadpre.
Require Import Float_error Bound64 Annot AnnExp RunErr.
From Flocq Require Import IEEE754.BinarySingleNaN.
From Stdlib Require Import ExtrOcamlBasic ExtrOcamlNatInt.

Extract Inductive binary_float => "float"
  [ "(fun s -> if s then (-0.0) else 0.0)"
    "(fun s -> if s then neg_infinity else infinity)"
    "nan"
    "(fun _ -> failwith ""finite"")" ]
  "(fun _ _ _ _ _ -> failwith ""match"")".
Extract Constant f32_plus => "( +. )".
Extract Constant f32_mult => "( *. )".
Extract Constant f32_div  => "( /. )".
Extract Constant f32_sqrt => "sqrt".
Extract Constant f32_neg  => "(fun a -> (-. a))".
Extract Constant f32_abs  => "abs_float".
Extract Constant f32_zero => "0.0".
Extract Constant f32_one  => "1.0".
Extract Constant f32_of_Z => "(fun _ -> 0.0)".
Extract Constant f32_lt   => "( < )".
Extract Constant f32_le   => "( <= )".
Extract Constant f32_bytes_to_binary32 => "(fun _ -> 0.0)".
Extract Constant b64_plus => "( +. )".
Extract Constant b64_mult => "( *. )".
Extract Constant b64_div => "( /. )".
Extract Constant b64_sqrt => "sqrt".
Extract Constant b64_succ => "Float.succ".
Extract Constant b64_pred => "Float.pred".
Extract Constant b64_neg => "(fun a -> (-. a))".
Extract Constant b64_abs => "abs_float".
Extract Constant b64_finite => "Float.is_finite".
Extract Constant b64_lt => "( < )".
Extract Constant b64_le => "( <= )".
Extract Constant b64_of_f32 => "(fun a -> a)".
Extract Constant f32_finite => "Float.is_finite".

Extraction "nativez.ml"
  binary32 f32_bytes_to_binary32
  f32_plus f32_minus f32_mult f32_div f32_neg f32_abs f32_sqrt
  f32_dot f32_vec_add f32_vec_mult f32_mat_vec_mul f32_mat_transpose f32_add_matrices
  f32_of_Z f32_zero f32_one f32_sigmoid f32_exp_approx f32_sum f32_softmax
  f32_sin f32_cos f32_silu f32_silu_vec f32_rmsnorm
  f32_max2 f32_log_unit f32_softplus f32_l2norm
  f32_rmsnorm_zc f32_rmsnorm_gated
  f32_conv_window f32_conv_step f32_causal_conv1d
  f32_delta_step f32_delta_scan f32_delta_state0 f32_delta_decay f32_delta_prep_q
  f32_partial_rope f32_swiglu f32_gate_sigmoid
  f32_attend f32_causal_attention f32_concat_heads f32_split_into_heads
  f32_qwen_delta_mix f32_qwen_attn_mix f32_qwen_wrap
  f32_qwen_stack f32_qwen_forward f32_qwen_logits f32_embed_tokens
  f32_load_qwen f32_qwen_logits_of
  f32_slice f32_llama_layer f32_llama_stack f32_llama_forward f32_llama_logits
  f32_load_llama f32_llama_logits_of
  f32_layer_norm_2d f32_ln_eps f32_gelu_vec
  f32_gpt2_forward f32_gpt2_logits f32_gpt2_forward_pre f32_gpt2_logits_pre
  f32_load_model_pre json_tensor_offsets
  ann ann_const ann_ops ann_dot
  g_layer_norm_2d g_vec_add g_mat_vec_mul g_add_matrices
  g_split_into_heads g_causal_attention g_concat_heads g_gelu_vec.
