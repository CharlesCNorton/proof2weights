(** * Native IEEE-754 extraction

    The default extraction, in Phases1_15_complete.v, keeps binary32 as Flocq's
    inductive binary_float and computes every operation in inductive bignum
    arithmetic: faithful, but microseconds per operation, so a GPT-2 forward
    takes tens of minutes. This file re-extracts the same development with
    binary32 mapped to the host's hardware float, in the manner CompCert
    extracts its verified floats and trusts the IEEE-754 agreement at the OCaml
    boundary.

    Each operation is the binary64 operation rounded to binary32 via the Int32
    bit round-trip. Float_error.v proves that this second rounding is harmless:
    for binary32 operands, rounding the exact real result of +, -, *, / or sqrt
    to binary64 and then to binary32 equals rounding it straight to binary32, so
    the extracted result is the correctly-rounded binary32 value Flocq
    specifies. Decoding reads the four little-endian bytes straight to a
    binary32 via Int32.float_of_bits.

    f32_sin and f32_cos are not mapped to the host libm: they are compositions
    of f32_plus, f32_minus, f32_mult and f32_div, so extracting them
    structurally makes the native build run the same argument reduction and the
    same Taylor polynomial the inductive build runs, operation for operation.

    The GPT-2, Llama and Qwen3.5 runners share one set of directives here, and
    one compile emits all three extraction targets. This is the trusted fast
    path; the inductive build remains the proof artifact. *)

Require Import Phases1_15_complete.
Require Import Llama.
Require Import Qwen.
Require Import Pretransposed.
Require Import Loadpre.
From Flocq Require Import IEEE754.BinarySingleNaN.
From Stdlib Require Import ExtrOcamlBasic.
From Stdlib Require Import ExtrOcamlNatInt.
From Stdlib Require Import ExtrOcamlZInt.

(* binary32 (= Flocq binary_float) -> OCaml float. Constructors build the
   represented value; no extracted code matches binary_float in this build.
   Each arithmetic operation rounds the binary64 result to binary32 through the
   Int32 bit round-trip. *)
Extract Inductive binary_float => "float"
  [ "(fun s -> if s then (-0.0) else 0.0)"
    "(fun s -> if s then neg_infinity else infinity)"
    "nan"
    "(fun (s, m, e) -> let v = ldexp (float_of_int m) e in if s then (-. v) else v)" ]
  "(fun _ _ _ _ _ -> failwith ""binary_float match in native build"")".

(* Round a binary64 result to nearest binary32 via the Int32 bit round-trip. *)
Extract Constant f32_plus => "(fun a b -> Int32.float_of_bits (Int32.bits_of_float (a +. b)))".
Extract Constant f32_mult => "(fun a b -> Int32.float_of_bits (Int32.bits_of_float (a *. b)))".
Extract Constant f32_div  => "(fun a b -> Int32.float_of_bits (Int32.bits_of_float (a /. b)))".
Extract Constant f32_sqrt => "(fun a -> Int32.float_of_bits (Int32.bits_of_float (sqrt a)))".
Extract Constant f32_neg  => "(fun a -> (-. a))".
Extract Constant f32_abs  => "abs_float".
Extract Constant f32_zero => "0.0".
Extract Constant f32_one  => "1.0".
Extract Constant f32_of_Z => "(fun n -> Int32.float_of_bits (Int32.bits_of_float (float_of_int n)))".
Extract Constant f32_lt   => "(fun (a:float) (b:float) -> a < b)".
Extract Constant f32_le   => "(fun (a:float) (b:float) -> a <= b)".

(* Decode four little-endian bytes straight to a binary32 value. *)
Extract Constant f32_bytes_to_binary32 => "(fun bs -> match bs with b0 :: b1 :: b2 :: b3 :: _ -> Int32.float_of_bits (Int32.logor (Int32.of_int b0) (Int32.logor (Int32.shift_left (Int32.of_int b1) 8) (Int32.logor (Int32.shift_left (Int32.of_int b2) 16) (Int32.shift_left (Int32.of_int b3) 24)))) | _ -> 0.0)".

Set Extraction Output Directory ".".

Extraction "phases1_15_native.ml"
  binary32 f32_bytes_to_binary32
  f32_dot f32_vec_add f32_mat_vec_mul f32_add_matrices
  f32_layer_norm_2d f32_ln_eps
  f32_split_into_heads f32_causal_attention f32_concat_heads f32_gelu_vec
  f32_gpt2_forward f32_gpt2_logits
  f32_gpt2_forward_pre f32_gpt2_logits_pre
  f32_load_model_pre
  json_tensor_offsets.

Extraction "llama_native.ml"
  binary32 f32_bytes_to_binary32
  f32_plus f32_minus f32_mult f32_div f32_sqrt f32_one
  f32_dot f32_vec_add f32_vec_mult f32_mat_vec_mul
  f32_of_Z f32_sin f32_cos
  f32_rmsnorm f32_silu_vec
  f32_attend f32_causal_attention f32_concat_heads
  f32_slice f32_partial_rope f32_swiglu
  f32_llama_layer f32_llama_stack f32_llama_forward f32_llama_logits
  f32_embed_tokens
  f32_load_llama f32_llama_logits_of
  json_tensor_offsets.

(* The annotated forward pass of RunErr.v. Its bounds are binary64 values, which
   are OCaml floats as they stand; the outward operations are the host's
   binary64 arithmetic followed by [Float.succ] or [Float.pred]. *)
Require Import Float_error Bound64 Annot AnnExp RunErr.

Extract Constant b64_plus => "( +. )".
Extract Constant b64_mult => "( *. )".
Extract Constant b64_div => "( /. )".
Extract Constant b64_sqrt => "sqrt".
Extract Constant b64_succ => "Float.succ".
Extract Constant b64_pred => "Float.pred".
Extract Constant b64_neg => "(fun a -> (-. a))".
Extract Constant b64_abs => "abs_float".
Extract Constant b64_finite => "Float.is_finite".
Extract Constant b64_lt => "(fun (a:float) (b:float) -> a < b)".
Extract Constant b64_le => "(fun (a:float) (b:float) -> a <= b)".
Extract Constant b64_of_f32 => "(fun a -> a)".
Extract Constant f32_finite => "Float.is_finite".

Extraction "runerr_native.ml"
  binary32 f32_bytes_to_binary32 json_tensor_offsets f32_ln_eps
  ann ann_const ann_ops ann_dot
  g_layer_norm_2d g_vec_add g_mat_vec_mul g_add_matrices
  g_split_into_heads g_causal_attention g_concat_heads g_gelu_vec.

Extraction "qwen_native.ml"
  binary32 f32_bytes_to_binary32
  f32_plus f32_minus f32_mult f32_div f32_neg f32_abs f32_sqrt
  f32_dot f32_vec_add f32_vec_mult f32_mat_vec_mul f32_mat_transpose
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
  json_tensor_offsets.
