(** * Audit: assumption report for the headline theorems

    Compile from this directory, after the files it reports on:

      rocq compile -R . "" Phases1_15_complete.v
      rocq compile -R . "" Float_error.v
      rocq compile -R . "" Audit.v

    Each [Print Assumptions] below prints either "Closed under the global
    context" or the list of axioms the proof term depends on. The integer
    pipeline reports the former. The float results report the four classical
    axioms that [Coq.Reals] introduces and that Flocq's [binary_float]
    operations inherit; nothing in this development adds an axiom of its
    own. *)

From Stdlib Require Import String.
Require Import Phases1_15_complete.
Require Import Float_error.

(** Serialization and quantization: constructive. *)
Print Assumptions roundtrip_z.
Print Assumptions roundtrip_f32.
Print Assumptions roundtrip_f16.

(** Integer inference: constructive. *)
Print Assumptions softmax_entry_range.

(** Storage layer, chunking, compression, sharding: constructive. *)
Print Assumptions reassemble_split_into_chunks.
Print Assumptions rle_roundtrip.
Print Assumptions decompress_compress_tensor.
Print Assumptions decompress_compress_network.
Print Assumptions unshard_shard_network.

(** Float forward pass: classical, through Flocq's dependency on Reals. *)
Print Assumptions f32_gpt2_forward_rows.
Print Assumptions f32_gpt2_logits_rows.
Print Assumptions f32_gpt2_logits_row_width.

(** The native build's rounding step: classical, same inheritance. *)
Print Assumptions f32_double_round_plus.
Print Assumptions f32_double_round_mult.
Print Assumptions f32_double_round_div.
Print Assumptions f32_double_round_sqrt.

(** The composed dot-product error bound: classical, same inheritance. *)
Print Assumptions f32_mac_step_error.
Print Assumptions f32_dot_error.
Print Assumptions f32_dot_regular_ones.

(** The forward-pass bound, stage by stage up to the logits. *)
Print Assumptions ok_plus.
Print Assumptions ok_mult.
Print Assumptions ok_div.
Print Assumptions ok_sqrt.
Print Assumptions ok_dot.
Print Assumptions ok_layer_norm_2d.
Print Assumptions ok_exp_approx.
Print Assumptions ok_gelu_vec.
Print Assumptions ok_mlp_forward.
Print Assumptions ok_softmax_2d.
Print Assumptions ok_causal_attention.
Print Assumptions ok_attention_forward.
Print Assumptions ok_block_forward.
Print Assumptions ok_blocks_forward.
Print Assumptions ok_gpt2_forward.
Print Assumptions ok_gpt2_logits_full.
