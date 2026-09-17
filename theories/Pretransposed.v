(** * The GPT-2 forward pass over pre-transposed weights

    [f32_linear_forward w b x] is
    [f32_vec_add (f32_mat_vec_mul (f32_mat_transpose w) x) b], and the
    extracted [f32_mat_transpose] indexes a list of lists, so it is quadratic in
    the output dimension. That is the one reason the checkpoint runner of
    Runner.v does not call [f32_gpt2_logits]: on GPT-2 small the transposes
    alone would dominate the forward pass by four orders of magnitude.

    This file removes the reason. It gives the same forward pass over weights
    that are already transposed, which is the order the runner decodes them in,
    and proves that on the transpose of a model it returns what the original
    returns on the model. The runner can then call an extracted forward pass
    rather than rebuild the composition in OCaml, and the composition it runs
    is the one the proofs are about.

    Nothing here is a new definition of the arithmetic. Every function below
    differs from its counterpart in Phases1_15_complete.v by dropping one
    [f32_mat_transpose], and each correctness lemma is that drop undone. *)

From Stdlib Require Import List.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.

Import ListNotations.

(** * The linear layer *)

Definition f32_linear_pre (wT : list (list binary32)) (bias : list binary32)
                          (x : list binary32) : list binary32 :=
  f32_vec_add (f32_mat_vec_mul wT x) bias.

Lemma f32_linear_pre_correct : forall w b x,
  f32_linear_pre (f32_mat_transpose w) b x = f32_linear_forward w b x.
Proof. reflexivity. Qed.

Definition f32_linear_pre_2d (wT : list (list binary32)) (bias : list binary32)
                             (x : list (list binary32)) : list (list binary32) :=
  List.map (f32_linear_pre wT bias) x.

Lemma f32_linear_pre_2d_correct : forall w b x,
  f32_linear_pre_2d (f32_mat_transpose w) b x = f32_linear_forward_2d w b x.
Proof. reflexivity. Qed.

(** * Attention and the feed-forward *)

Definition f32_attention_pre (n_embd n_head : nat)
    (c_attn_wT : list (list binary32)) (c_attn_b : list binary32)
    (c_proj_wT : list (list binary32)) (c_proj_b : list binary32)
    (hidden : list (list binary32)) : list (list binary32) :=
  let qkv := f32_linear_pre_2d c_attn_wT c_attn_b hidden in
  let d := n_embd in
  let head_dim := Nat.div d n_head in
  let split_qkv row :=
    (List.firstn d row, List.firstn d (List.skipn d row), List.skipn (2 * d) row) in
  let qkv_split := List.map split_qkv qkv in
  let qs := List.map (fun '(q, _, _) => q) qkv_split in
  let ks := List.map (fun '(_, k, _) => k) qkv_split in
  let vs := List.map (fun '(_, _, v) => v) qkv_split in
  let q_heads := f32_split_into_heads n_head qs in
  let k_heads := f32_split_into_heads n_head ks in
  let v_heads := f32_split_into_heads n_head vs in
  let head_outs :=
    List.map (fun '(q, (k, v)) => f32_causal_attention q k v head_dim)
      (List.combine q_heads (List.combine k_heads v_heads)) in
  f32_linear_pre_2d c_proj_wT c_proj_b (f32_concat_heads head_outs).

Lemma f32_attention_pre_correct : forall n_embd n_head caw cab cpw cpb h,
  f32_attention_pre n_embd n_head (f32_mat_transpose caw) cab
                    (f32_mat_transpose cpw) cpb h
  = f32_attention_forward n_embd n_head caw cab cpw cpb h.
Proof. reflexivity. Qed.

Definition f32_mlp_pre
    (c_fc_wT : list (list binary32)) (c_fc_b : list binary32)
    (c_proj_wT : list (list binary32)) (c_proj_b : list binary32)
    (hidden : list (list binary32)) : list (list binary32) :=
  let h := f32_linear_pre_2d c_fc_wT c_fc_b hidden in
  let h_gelu := List.map f32_gelu_vec h in
  f32_linear_pre_2d c_proj_wT c_proj_b h_gelu.

Lemma f32_mlp_pre_correct : forall fcw fcb mpw mpb h,
  f32_mlp_pre (f32_mat_transpose fcw) fcb (f32_mat_transpose mpw) mpb h
  = f32_mlp_forward fcw fcb mpw mpb h.
Proof. reflexivity. Qed.

(** * The block, the stack and the logits

    The weight records are reused unchanged; in a pre-transposed model their
    matrix fields hold the transpose of what they hold in a model. *)

Definition transpose_attn (w : f32_attention_weights) : f32_attention_weights :=
  mk_f32_attention_weights
    (f32_mat_transpose (f32_attn_c_attn_weight w)) (f32_attn_c_attn_bias w)
    (f32_mat_transpose (f32_attn_c_proj_weight w)) (f32_attn_c_proj_bias w).

Definition transpose_mlp (w : f32_mlp_weights) : f32_mlp_weights :=
  mk_f32_mlp_weights
    (f32_mat_transpose (f32_mlp_c_fc_weight w)) (f32_mlp_c_fc_bias w)
    (f32_mat_transpose (f32_mlp_c_proj_weight w)) (f32_mlp_c_proj_bias w).

Definition transpose_block (b : f32_block_weights) : f32_block_weights :=
  mk_f32_block_weights (f32_block_ln_1 b) (transpose_attn (f32_block_attn b))
                       (f32_block_ln_2 b) (transpose_mlp (f32_block_mlp b)).

Definition transpose_model (m : f32_model_weights) : f32_model_weights :=
  mk_f32_model_weights (f32_wte m) (f32_wpe m)
                       (List.map transpose_block (f32_blocks m)) (f32_ln_f m).

Definition f32_block_pre (cfg : gpt2_inference_config) (eps : binary32)
                         (block : f32_block_weights)
                         (hidden : list (list binary32)) : list (list binary32) :=
  let ln1 := f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                               (f32_ln_bias (f32_block_ln_1 block)) eps hidden in
  let attn_out := f32_attention_pre (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                    (f32_attn_c_attn_weight (f32_block_attn block))
                    (f32_attn_c_attn_bias (f32_block_attn block))
                    (f32_attn_c_proj_weight (f32_block_attn block))
                    (f32_attn_c_proj_bias (f32_block_attn block)) ln1 in
  let hidden2 := f32_add_matrices hidden attn_out in
  let ln2 := f32_layer_norm_2d (f32_ln_weight (f32_block_ln_2 block))
                               (f32_ln_bias (f32_block_ln_2 block)) eps hidden2 in
  let mlp_out := f32_mlp_pre (f32_mlp_c_fc_weight (f32_block_mlp block))
                   (f32_mlp_c_fc_bias (f32_block_mlp block))
                   (f32_mlp_c_proj_weight (f32_block_mlp block))
                   (f32_mlp_c_proj_bias (f32_block_mlp block)) ln2 in
  f32_add_matrices hidden2 mlp_out.

Lemma f32_block_pre_correct : forall cfg eps b h,
  f32_block_pre cfg eps (transpose_block b) h = f32_block_forward cfg eps b h.
Proof. intros cfg eps [ln1 [caw cab cpw cpb] ln2 [fcw fcb mpw mpb]] h. reflexivity. Qed.

Fixpoint f32_blocks_pre (cfg : gpt2_inference_config) (eps : binary32)
                        (blocks : list f32_block_weights)
                        (hidden : list (list binary32)) : list (list binary32) :=
  match blocks with
  | [] => hidden
  | b :: rest => f32_blocks_pre cfg eps rest (f32_block_pre cfg eps b hidden)
  end.

Lemma f32_blocks_pre_correct : forall cfg eps bs h,
  f32_blocks_pre cfg eps (List.map transpose_block bs) h
  = f32_blocks_forward cfg eps bs h.
Proof.
  intros cfg eps bs. induction bs as [|b bs IH]; intros h; [reflexivity|].
  cbn [List.map f32_blocks_pre f32_blocks_forward].
  rewrite f32_block_pre_correct. apply IH.
Qed.

Definition f32_gpt2_forward_pre (cfg : gpt2_inference_config) (eps : binary32)
                                (model : f32_model_weights) (token_ids : list nat)
                                : list (list binary32) :=
  let seq_len := List.length token_ids in
  let tok := f32_embed_tokens (f32_wte model) token_ids in
  let pos := f32_embed_positions (f32_wpe model) seq_len in
  let hidden := f32_add_matrices tok pos in
  let transformed := f32_blocks_pre cfg eps (f32_blocks model) hidden in
  f32_layer_norm_2d (f32_ln_weight (f32_ln_f model))
                    (f32_ln_bias (f32_ln_f model)) eps transformed.

Theorem f32_gpt2_forward_pre_correct : forall cfg eps m ids,
  f32_gpt2_forward_pre cfg eps (transpose_model m) ids
  = f32_gpt2_forward cfg eps m ids.
Proof.
  intros cfg eps [wte wpe bs lnf] ids.
  unfold f32_gpt2_forward_pre, f32_gpt2_forward, transpose_model.
  cbn [f32_wte f32_wpe f32_blocks f32_ln_f].
  now rewrite f32_blocks_pre_correct.
Qed.

Definition f32_gpt2_logits_pre (cfg : gpt2_inference_config) (eps : binary32)
                               (model : f32_model_weights) (token_ids : list nat)
                               : list (list binary32) :=
  let hidden := f32_gpt2_forward_pre cfg eps model token_ids in
  List.map (fun h_row => List.map (fun w_row => f32_dot h_row w_row) (f32_wte model))
           hidden.

(** The statement the runner relies on: on a model whose matrices are stored
    transposed, which is the order the checkpoint runner decodes them in, the
    pre-transposed pass returns the logits [f32_gpt2_logits] returns. *)
Theorem f32_gpt2_logits_pre_correct : forall cfg eps m ids,
  f32_gpt2_logits_pre cfg eps (transpose_model m) ids
  = f32_gpt2_logits cfg eps m ids.
Proof.
  intros cfg eps m ids.
  unfold f32_gpt2_logits_pre, f32_gpt2_logits.
  rewrite f32_gpt2_forward_pre_correct.
  now destruct m.
Qed.
