(** * Assumption report for the headline theorems

    Compile from this directory, after the files it reports on. Each
    [Print Assumptions] below prints either "Closed under the global context"
    or the axioms the proof term depends on.

    Two axiom families appear in the output. Results about real numbers carry
    the classical axioms of [Coq.Reals], which Flocq's operations inherit:
    [classic], [functional_extensionality_dep],
    [ClassicalDedekindReals.sig_forall_dec] and [sig_not_dec]. The bounds
    discharged by CoqInterval in Series.v, and the theorems composed from
    them in Truth.v and AnnExp.v, carry in addition the primitive-float and
    primitive-integer interface CoqInterval computes with: [PrimFloat],
    [FloatAxioms], [PrimInt63], [Uint63Axioms], [Sint63Axioms] and
    [PrimArray]. *)

From Stdlib Require Import String.
Require Import Phases1_15_complete.
Require Import Llama.
Require Import Qwen.
Require Import Float_error.
Require Import Narrow.
Require Import Backward.
Require Import Pretransposed.
Require Import Dot.
Require Import Cache.
Require Import Cache_attn.
Require Import Causal.
Require Import Runner.
Require Import Loader.
Require Import Loadpre.
Require Import Series.
Require Import Truth.
Require Import Witness.
Require Import RoundChk.
Require Import Bound64.
Require Import Native.
Require Import Machine.
Require Import Decode.
Require Import Annot.
Require Import AnnExp.
Require Import RunErr.
Require Enclose EncGPT2 EncExt EncLlama EncQwen ElemTable.

(** * Serialization, quantization and storage *)

Print Assumptions roundtrip_z.
Print Assumptions roundtrip_f32.
Print Assumptions roundtrip_f16.
Print Assumptions softmax_entry_range.
Print Assumptions reassemble_split_into_chunks.
Print Assumptions rle_roundtrip.
Print Assumptions decompress_compress_tensor.
Print Assumptions decompress_compress_network.
Print Assumptions unshard_shard_network.

(** * Shape preservation through the float forward pass *)

Print Assumptions f32_gpt2_forward_rows.
Print Assumptions f32_gpt2_logits_rows.
Print Assumptions f32_gpt2_logits_row_width.

(** * The native build's rounding step *)

Print Assumptions f32_double_round_plus.
Print Assumptions f32_double_round_mult.
Print Assumptions f32_double_round_div.
Print Assumptions f32_double_round_sqrt.

(** * The native build's arithmetic, at the level of floats *)

Print Assumptions narrow_widen.
Print Assumptions narrow_widen_plus.
Print Assumptions narrow_widen_minus.
Print Assumptions narrow_widen_mult.
Print Assumptions narrow_widen_div.
Print Assumptions narrow_widen_sqrt.
Print Assumptions widen_neg.
Print Assumptions widen_abs.
Print Assumptions widen_lt.
Print Assumptions widen_le.
Print Assumptions widen_finite.
Print Assumptions narrow_of_Z.
Print Assumptions f32_of_Z_constants.

(** * Backward error with underflow *)

Print Assumptions f32_dot_backward_mixed.
Print Assumptions f32_mat_vec_mul_backward_mixed.
Print Assumptions logits_backward_mixed.

(** * The forward pass over pre-transposed weights *)

Print Assumptions f32_linear_pre_correct.
Print Assumptions f32_attention_pre_correct.
Print Assumptions f32_mlp_pre_correct.
Print Assumptions f32_block_pre_correct.
Print Assumptions f32_blocks_pre_correct.
Print Assumptions f32_gpt2_forward_pre_correct.
Print Assumptions f32_gpt2_logits_pre_correct.

(** * The rounding model *)

Print Assumptions f32_round_mixed.
Print Assumptions f32_round_relz.
Print Assumptions regz_zero.
Print Assumptions round_err_le.
Print Assumptions no_overflow.

(** * The dot product *)

Print Assumptions f32_mac_step_error.
Print Assumptions f32_dot_error.
Print Assumptions f32_mac_step_mixed.
Print Assumptions f32_dot_error_mixed.
Print Assumptions f32_dot_backward.
Print Assumptions f32_dot_backward_ones.
Print Assumptions f32_dot_regular_ones.
Print Assumptions dot_regular_with_zero.
Print Assumptions dot_ok_with_zero.
Print Assumptions dotreg_with_zero.

(** * The forward bound, stage by stage up to the logits *)

Print Assumptions ok_plus.
Print Assumptions ok_mult.
Print Assumptions ok_div.
Print Assumptions ok_sqrt.
Print Assumptions ok_dot.
Print Assumptions ok_layer_norm_2d.
Print Assumptions ok_sat.
Print Assumptions ok_exp_approx.
Print Assumptions ok_sigmoid.
Print Assumptions ok_tanh.
Print Assumptions ok_gelu.
Print Assumptions ok_gelu_vec.
Print Assumptions ok_mlp_forward.
Print Assumptions ok_softmax_2d.
Print Assumptions ok_attend.
Print Assumptions ok_causal_attention.
Print Assumptions ok_attention_forward.
Print Assumptions ok_block_forward.
Print Assumptions ok_blocks_forward.
Print Assumptions ok_gpt2_forward.
Print Assumptions ok_gpt2_logits_full.

(** * The Llama primitives, and the same chain up to its logits *)

Print Assumptions ok_rmsnorm.
Print Assumptions ok_silu_vec.
Print Assumptions ok_sin.
Print Assumptions ok_cos.
Print Assumptions ok_llama_attn.
Print Assumptions ok_llama_wrap.
Print Assumptions ok_llama_stack.
Print Assumptions ok_llama_forward.
Print Assumptions ok_llama_logits_full.

(** * The Qwen3.5 primitives, and the same chain up to its logits *)

Print Assumptions ok_log_unit.
Print Assumptions ok_softplus.
Print Assumptions ok_rmsnorm_zc.
Print Assumptions ok_rmsnorm_gated.
Print Assumptions ok_conv_step.
Print Assumptions ok_causal_conv1d.
Print Assumptions ok_delta_step.
Print Assumptions ok_delta_scan.
Print Assumptions ok_partial_rope.
Print Assumptions ok_swiglu.
Print Assumptions ok_delta_prep_q.
Print Assumptions ok_qwen_delta_mix.
Print Assumptions ok_qwen_attn_mix.
Print Assumptions ok_qwen_wrap.
Print Assumptions ok_qwen_stack.
Print Assumptions ok_qwen_forward.
Print Assumptions ok_qwen_logits_full.

(** * Satisfiable premises *)

Print Assumptions amp_ok_witness.
Print Assumptions exp_reg_zero.
Print Assumptions exp_reg_masked.
Print Assumptions gelu_reg_zero.
Print Assumptions exp_at_zero.
Print Assumptions exp_at_masked.
Print Assumptions sigmoid_at_zero.
Print Assumptions gelu_at_zero.

(** * Backward error through the linear layers and the logit projection *)

Print Assumptions f32_mat_vec_mul_backward.
Print Assumptions logits_backward.
Print Assumptions f32_gpt2_logits_backward.
Print Assumptions f32_qwen_logits_backward.
Print Assumptions f32_llama_logits_backward.

(** * A forward pass that carries its own bound *)

Print Assumptions up_plus_spec.
Print Assumptions up_mult_spec.
Print Assumptions up_div_spec.
Print Assumptions dn_plus_spec.
Print Assumptions dn_mult_spec.
Print Assumptions dn_sqrt_spec.
Print Assumptions b64_of_f32_spec.
Print Assumptions round_err_out.
Print Assumptions f32_round_int_integer.
Print Assumptions aok_plus.
Print Assumptions aok_mult.
Print Assumptions aok_div.
Print Assumptions aok_sqrt.
Print Assumptions aok_max.
Print Assumptions em1_up_spec.
Print Assumptions ae_core.
Print Assumptions aok_exp.
Print Assumptions g_gpt2_logits_f32.
Print Assumptions gpt2_logits_bounded.
Print Assumptions gpt2_logit_bounded.

(** * Cached decoding equals full recomputation *)

Print Assumptions delta_scan_app.
Print Assumptions delta_scan_snoc.
Print Assumptions delta_state_app.
Print Assumptions delta_markov.
Print Assumptions conv_window_cached_correct.
Print Assumptions conv_hist_step.
Print Assumptions causal_attention_nth.
Print Assumptions causal_attention_app.
Print Assumptions causal_attention_prefix.
Print Assumptions causal_attention_snoc.
Print Assumptions transpose_involutive.

(** * Causality of the forward passes *)

Print Assumptions causal_attention_causal.
Print Assumptions causal_conv1d.
Print Assumptions causal_delta_scan.
Print Assumptions causal_llama_forward.
Print Assumptions causal_qwen_forward.
Print Assumptions causal_gpt2_forward.
Print Assumptions llama_forward_prefix.
Print Assumptions qwen_forward_prefix.
Print Assumptions gpt2_logits_prefix.

(** * One decode step per token equals evaluating the whole sequence *)

Print Assumptions decode_stream_correct.
Print Assumptions gpt2_decode_step.
Print Assumptions llama_decode_step.
Print Assumptions qwen_decode_step.

(** * Decoders that carry their caches *)

Print Assumptions run_realized.
Print Assumptions gpt2_decode_correct.
Print Assumptions llama_decode_correct.
Print Assumptions qwen_decode_correct.

(** * The transposed decode the checkpoint runners perform *)

Print Assumptions decode_transposed_correct.
Print Assumptions runner_linear_correct.
Print Assumptions runner_linear_2d_correct.

(** * What a named load returns, and validation *)

Print Assumptions f32_load_named_length.
Print Assumptions f32_load_named_nth.
Print Assumptions f32_load_named_absent.
Print Assumptions f32_load_named_checked_sound.
Print Assumptions f32_load_named_checked_rejects.
Print Assumptions validated_logits_shape.
Print Assumptions loaded_model_blocks.

(** * Loading a checkpoint through a byte reader *)

Print Assumptions f32_read_vec_load_named.
Print Assumptions read_rows_ok.
Print Assumptions read_transposed_ok.
Print Assumptions f32_load_model_pre_correct.
Print Assumptions read_llama_layer_ok.
Print Assumptions f32_load_llama_correct.
Print Assumptions read_qwen_layer_ok.
Print Assumptions f32_load_qwen_correct.

(** * The elementary functions against the mathematical functions *)

Print Assumptions sin_series_bound.
Print Assumptions cos_series_bound.
Print Assumptions log_series_bound.
Print Assumptions exp_series_bound.
Print Assumptions ln2_split_bound.
Print Assumptions Qb_correct.
Print Assumptions sin_poly_actual.
Print Assumptions cos_poly_actual.
Print Assumptions ok_sin_true.
Print Assumptions ok_cos_true.
Print Assumptions exp_core_vs_true.
Print Assumptions exp_approx_vs_true.
Print Assumptions ok_exp_true.
Print Assumptions ok_exp_true_in_range.
Print Assumptions log_unit_vs_true.
Print Assumptions ok_log_true.
Print Assumptions ok_sqrt_true.
Print Assumptions ok_reduce_2pi.
Print Assumptions reduce_vs_true.
Print Assumptions ok_sin_true_full.
Print Assumptions ok_cos_true_full.
Print Assumptions sigmoid_vs_true.
Print Assumptions tanh_vs_true.
Print Assumptions silu_vs_true.
Print Assumptions softplus_vs_true.
Print Assumptions gelu_vs_true.
Print Assumptions ok_sigmoid_true.
Print Assumptions ok_tanh_true.
Print Assumptions ok_silu_true.
Print Assumptions ok_softplus_true.
Print Assumptions ok_gelu_true.

(** The constants the series divide by, and those that are not the factorial
    they approximate. *)
Print Assumptions B2R_fac19.
Print Assumptions fac14_not_exact.
Print Assumptions fac15_not_exact.
Print Assumptions fac19_not_exact.

(** * Checking a side condition by computation *)

Print Assumptions regz_Q.
Print Assumptions u_le_half.
Print Assumptions two_eta_le_normal_lo.

(** * Rounding to the nearest integer *)

Print Assumptions round_neg_04.
Print Assumptions round_neg_06.
Print Assumptions round_neg_126.
Print Assumptions round_2p23_differs.

(** * The elementary functions over every binary32 input of their domains *)

Print Assumptions ElemTable.exp_bound.
Print Assumptions ElemTable.log_bound.
Print Assumptions ElemTable.sin_bound.
Print Assumptions ElemTable.cos_bound.
Print Assumptions ElemTable.sigmoid_bound.
Print Assumptions ElemTable.tanh_bound.
Print Assumptions ElemTable.softplus_bound.
Print Assumptions ElemTable.gelu_bound.

(** * Enclosures of the reference logits *)

Print Assumptions Enclose.enc_ops_rel.
Print Assumptions EncExt.enc_ln_in.
Print Assumptions EncExt.enc_abs_in.
Print Assumptions EncGPT2.gpt2_logits_enclosed.
Print Assumptions EncLlama.g_llama_logits_f32.
Print Assumptions EncLlama.llama_logits_enclosed.
Print Assumptions EncQwen.g_qwen_logits_f32.
Print Assumptions EncQwen.qwen_logits_enclosed.
