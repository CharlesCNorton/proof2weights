(** * The GPT-2 pass on pre-transposed weights, and the enclosure of its logits

    The generic pass of RunErr.v transposes each weight matrix for every row
    it multiplies. Here the same pass takes its matrices already transposed,
    which is the order the checkpoint runner decodes them in, and on the
    transpose of a model it computes exactly what the generic pass computes on
    the model, in any arithmetic. Instantiated at the enclosure arithmetic of
    Enclose.v, it encloses every logit the exact real reference produces. *)

From Stdlib Require Import ZArith Reals List.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Float_error RunErr Pretransposed Enclose.

Import ListNotations.

Section Pre.

Context {T : Type} (A : ops T).

Definition g_linear_pre_2d (wT : list (list T)) (b : list T) (x : list (list T))
  : list (list T) :=
  List.map (fun row => g_vec_add A (g_mat_vec_mul A wT row) b) x.

Definition g_attention_pre (n_embd n_head : nat)
    (c_attn_wT : list (list T)) (c_attn_b : list T)
    (c_proj_wT : list (list T)) (c_proj_b : list T)
    (hidden : list (list T)) : list (list T) :=
  let qkv := g_linear_pre_2d c_attn_wT c_attn_b hidden in
  let d := n_embd in
  let head_dim := Nat.div d n_head in
  let split_qkv row :=
    (List.firstn d row, List.firstn d (List.skipn d row), List.skipn (2 * d) row) in
  let qkv_split := List.map split_qkv qkv in
  let qs := List.map (fun '(q, _, _) => q) qkv_split in
  let ks := List.map (fun '(_, k, _) => k) qkv_split in
  let vs := List.map (fun '(_, _, v) => v) qkv_split in
  let q_heads := g_split_into_heads n_head qs in
  let k_heads := g_split_into_heads n_head ks in
  let v_heads := g_split_into_heads n_head vs in
  let head_outputs := List.map (fun '(qh, (kh, vh)) => g_causal_attention A qh kh vh head_dim)
                               (List.combine q_heads (List.combine k_heads v_heads)) in
  let concat := g_concat_heads head_outputs in
  g_linear_pre_2d c_proj_wT c_proj_b concat.

Definition g_mlp_pre
    (c_fc_wT : list (list T)) (c_fc_b : list T)
    (c_proj_wT : list (list T)) (c_proj_b : list T)
    (hidden : list (list T)) : list (list T) :=
  let h := g_linear_pre_2d c_fc_wT c_fc_b hidden in
  let h_gelu := List.map (g_gelu_vec A) h in
  g_linear_pre_2d c_proj_wT c_proj_b h_gelu.

Definition g_block_pre (cfg : gpt2_inference_config) (eps : T) (bl : gblock T)
    (hidden : list (list T)) : list (list T) :=
  let ln1 := g_layer_norm_2d A (gb_ln1w bl) (gb_ln1b bl) eps hidden in
  let attn_out := g_attention_pre (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                    (gb_caw bl) (gb_cab bl) (gb_cpw bl) (gb_cpb bl) ln1 in
  let hidden2 := g_add_matrices A hidden attn_out in
  let ln2 := g_layer_norm_2d A (gb_ln2w bl) (gb_ln2b bl) eps hidden2 in
  let mlp_out := g_mlp_pre (gb_fcw bl) (gb_fcb bl) (gb_mpw bl) (gb_mpb bl) ln2 in
  g_add_matrices A hidden2 mlp_out.

Fixpoint g_blocks_pre (cfg : gpt2_inference_config) (eps : T)
    (blocks : list (gblock T)) (hidden : list (list T)) : list (list T) :=
  match blocks with
  | [] => hidden
  | b :: rest => g_blocks_pre cfg eps rest (g_block_pre cfg eps b hidden)
  end.

Definition g_gpt2_logits_pre (cfg : gpt2_inference_config) (eps : T) (m : gmodel T)
    (toks : list nat) : list (list T) :=
  let seq_len := List.length toks in
  let tok := g_embed_tokens (gm_wte m) toks in
  let pos := g_embed_positions (gm_wpe m) seq_len in
  let hidden := g_add_matrices A tok pos in
  let transformed := g_blocks_pre cfg eps (gm_blocks m) hidden in
  let normed := g_layer_norm_2d A (gm_lnfw m) (gm_lnfb m) eps transformed in
  List.map (fun h_row => List.map (fun w_row => op_dot A h_row w_row) (gm_wte m)) normed.

(** A model with its four matrices per block transposed. *)
Definition gblock_T (b : gblock T) : gblock T :=
  mk_gblock (gb_ln1w b) (gb_ln1b b)
    (g_mat_transpose A (gb_caw b)) (gb_cab b) (g_mat_transpose A (gb_cpw b)) (gb_cpb b)
    (gb_ln2w b) (gb_ln2b b)
    (g_mat_transpose A (gb_fcw b)) (gb_fcb b) (g_mat_transpose A (gb_mpw b)) (gb_mpb b).

Definition gmodel_T (m : gmodel T) : gmodel T :=
  mk_gmodel (gm_wte m) (gm_wpe m) (List.map gblock_T (gm_blocks m)) (gm_lnfw m) (gm_lnfb m).

Lemma g_block_pre_correct : forall cfg eps b h,
  g_block_pre cfg eps (gblock_T b) h = g_block_forward A cfg eps b h.
Proof. intros cfg eps b h. destruct b. reflexivity. Qed.

Lemma g_blocks_pre_correct : forall cfg eps bs h,
  g_blocks_pre cfg eps (List.map gblock_T bs) h = g_blocks_forward A cfg eps bs h.
Proof.
  intros cfg eps bs. induction bs as [|b bs IH]; intros h; [reflexivity|].
  cbn [List.map g_blocks_pre g_blocks_forward]. rewrite g_block_pre_correct. apply IH.
Qed.

Theorem g_gpt2_logits_pre_correct : forall cfg eps m toks,
  g_gpt2_logits_pre cfg eps (gmodel_T m) toks = g_gpt2_logits A cfg eps m toks.
Proof.
  intros cfg eps m toks. unfold g_gpt2_logits_pre, g_gpt2_logits, g_gpt2_forward, gmodel_T.
  cbn [gm_wte gm_wpe gm_blocks gm_lnfw gm_lnfb]. rewrite g_blocks_pre_correct. reflexivity.
Qed.

End Pre.

(** * Transposing commutes with mapping the entries *)

Lemma map_transpose : forall (T : Type) (A : ops T) (f : binary32 -> T) w,
  op_const A f32_zero = f f32_zero ->
  List.map (List.map f) (f32_mat_transpose w) = g_mat_transpose A (List.map (List.map f) w).
Proof.
  intros T A f w H. unfold f32_mat_transpose, g_mat_transpose.
  destruct w as [|row rows]; [reflexivity|].
  cbn [List.map]. rewrite List.length_map, List.map_map. apply List.map_ext. intros c.
  rewrite List.map_map. cbn [List.map]. f_equal.
  - rewrite H. symmetry. apply List.map_nth.
  - rewrite List.map_map. apply List.map_ext. intros rd. rewrite H. symmetry. apply List.map_nth.
Qed.

Lemma gmodel_map_transpose : forall (T : Type) (A : ops T) (f : binary32 -> T) m,
  op_const A f32_zero = f f32_zero ->
  gmodel_map f (to_gmodel (transpose_model m)) = gmodel_T A (gmodel_map f (to_gmodel m)).
Proof.
  intros T A f [wte wpe bs lnf] H.
  unfold gmodel_map, to_gmodel, transpose_model, gmodel_T.
  cbn [f32_wte f32_wpe f32_blocks f32_ln_f gm_wte gm_wpe gm_blocks gm_lnfw gm_lnfb].
  f_equal. rewrite !List.map_map. apply List.map_ext.
  intros [ln1 [caw cab cpw cpb] ln2 [fcw fcb mpw mpb]].
  unfold gblock_map, to_gblock, transpose_block, transpose_attn, transpose_mlp, gblock_T.
  cbn. rewrite !(map_transpose T A f) by exact H. reflexivity.
Qed.

(** * The enclosure of the checkpoint's logits

    Run on the transposed weights the checkpoint runner holds, the
    pre-transposed pass in the enclosure arithmetic returns, for every logit,
    an interval that contains the logit of the exact real reference: the same
    network on the same weights and tokens, in real arithmetic, with every
    exponential the true exponential of its argument saturated to [[-88, 88]]. *)
Theorem gpt2_logits_enclosed : forall FB prec cfg eps m toks, (149 <= FB)%Z ->
  Forall2 (Forall2 (enc_in FB))
    (g_gpt2_logits_pre (enc_ops FB prec) cfg (enc_const eps)
       (gmodel_map enc_const (to_gmodel (transpose_model m))) toks)
    (g_gpt2_logits real_ops cfg (B2R eps) (real_model m) toks).
Proof.
  intros FB prec cfg eps m toks HFB.
  rewrite (gmodel_map_transpose enc (enc_ops FB prec) enc_const m eq_refl).
  rewrite g_gpt2_logits_pre_correct.
  apply (r_gpt2_logits (enc_ops FB prec) real_ops (enc_in FB) (enc_ops_rel FB HFB prec)).
  - apply enc_const_in.
  - apply model_rel_map2. intros x. apply enc_const_in.
Qed.
