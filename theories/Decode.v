(** * Decoders for the three models

    Each model's decode step is assembled from the combinators of Machine.v:
    per layer, the key/value caches of its attention heads, and for Qwen3.5's
    DeltaNet layers the convolution window and the recurrent state of each
    head. The theorems below say that running a decode step over a token
    sequence, one token at a time, emits exactly the logit rows the model's
    forward pass computes for the whole sequence, as equalities of binary32
    values. These steps are what the checkpoint runners extract and run. *)

From Stdlib Require Import List Lia PeanoNat ZArith.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Llama Qwen Cache Cache_attn Causal.
Require Import Pretransposed Loadpre Machine.

Import ListNotations.

(** * Shared pieces *)

Definition map_step {A B : Type} (g : A -> B) (s : unit) (x : A) : unit * B := (s, g x).

Lemma realized_map_step : forall {A B : Type} (g : A -> B),
  realized tt (map_step g) (List.map g).
Proof. intros A B g. apply realized_map. Qed.

Definition attn_state : Type := (nat * list (list binary32) * list (list binary32))%type.
Definition attn0 : attn_state := (0%nat, [], []).

Lemma add_matrices_zipw : forall a b : list (list binary32),
  f32_add_matrices a b = zipw f32_vec_add a b.
Proof.
  intros a b. unfold f32_add_matrices, zipw.
  apply map_ext. intros [x y]. reflexivity.
Qed.

Lemma map_id_rows : forall xs : list (list binary32),
  List.map (fun r : list binary32 => r) xs = xs.
Proof. intros xs. apply map_id. Qed.

(** The residual sum as the layer definitions write it. *)
Lemma map_pair_zipw : forall X Y : list (list binary32),
  List.map (fun p => let '(a, b) := p in f32_vec_add a b) (List.combine X Y)
  = zipw f32_vec_add X Y.
Proof. intros X Y. unfold zipw. apply map_ext. intros [a b]. reflexivity. Qed.

(** * GPT-2 *)

Definition g2_qh (d nh h : nat) (r : list binary32) : list binary32 :=
  List.nth h (f32_split_row_into_heads nh (List.firstn d r)) [].
Definition g2_kh (d nh h : nat) (r : list binary32) : list binary32 :=
  List.nth h (f32_split_row_into_heads nh (List.firstn d (List.skipn d r))) [].
Definition g2_vh (d nh h : nat) (r : list binary32) : list binary32 :=
  List.nth h (f32_split_row_into_heads nh (List.skipn (2 * d) r)) [].

Definition g2_head_step (d nh h : nat) :=
  attn_step (fun _ => g2_qh d nh h) (fun _ => g2_kh d nh h) (fun _ => g2_vh d nh h)
            (Nat.div d nh).

Definition g2_att_step (d nh : nat) (w : f32_attention_weights) :=
  comp_step (map_step (f32_linear_pre (f32_attn_c_attn_weight w) (f32_attn_c_attn_bias w)))
    (comp_step (heads_step nh (fun _ => attn0) (g2_head_step d nh))
               (map_step (f32_linear_pre (f32_attn_c_proj_weight w) (f32_attn_c_proj_bias w)))).

Definition g2_att_init (nh : nat) :=
  (tt, (List.map (fun _ : nat => attn0) (List.seq 0 nh), tt)).

Definition g2_mlp_row (w : f32_mlp_weights) (r : list binary32) : list binary32 :=
  f32_linear_pre (f32_mlp_c_proj_weight w) (f32_mlp_c_proj_bias w)
    (f32_gelu_vec (f32_linear_pre (f32_mlp_c_fc_weight w) (f32_mlp_c_fc_bias w) r)).

Definition g2_ln (l : f32_layernorm_weights) (eps : binary32) (r : list binary32) :=
  f32_layer_norm_vec (f32_ln_weight l) (f32_ln_bias l) eps r.

Definition g2_block_step (cfg : gpt2_inference_config) (eps : binary32)
    (b : f32_block_weights) :=
  comp_step
    (zip_step (map_step (fun r : list binary32 => r))
              (comp_step (map_step (g2_ln (f32_block_ln_1 b) eps))
                         (g2_att_step (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                                      (f32_block_attn b)))
              f32_vec_add)
    (zip_step (map_step (fun r : list binary32 => r))
              (map_step (fun r => g2_mlp_row (f32_block_mlp b) (g2_ln (f32_block_ln_2 b) eps r)))
              f32_vec_add).

Definition g2_block_init (nh : nat) := ((tt, (tt, g2_att_init nh)), (tt, tt)).

Lemma g2_attention_eq : forall d nh caw cab cpw cpb hidden,
  f32_attention_pre d nh caw cab cpw cpb hidden
  = List.map (f32_linear_pre cpw cpb)
      (f32_concat_heads (List.map (fun h =>
         f32_causal_attention
           (mapi (fun _ => g2_qh d nh h) (List.map (f32_linear_pre caw cab) hidden))
           (mapi (fun _ => g2_kh d nh h) (List.map (f32_linear_pre caw cab) hidden))
           (mapi (fun _ => g2_vh d nh h) (List.map (f32_linear_pre caw cab) hidden))
           (Nat.div d nh)) (List.seq 0 nh))).
Proof.
  intros d nh caw cab cpw cpb hidden.
  unfold f32_attention_pre, f32_split_into_heads, f32_linear_pre_2d. cbv zeta.
  f_equal. f_equal.
  rewrite !combine_map_same, map_map. apply map_ext. intros h. cbn.
  rewrite <- !map_mapi, !map_map. reflexivity.
Qed.

Lemma realized_g2_att : forall d nh w, (0 < nh)%nat ->
  realized (g2_att_init nh) (g2_att_step d nh w)
    (f32_attention_pre d nh (f32_attn_c_attn_weight w) (f32_attn_c_attn_bias w)
                            (f32_attn_c_proj_weight w) (f32_attn_c_proj_bias w)).
Proof.
  intros d nh w Hnh.
  pose (QKV := f32_linear_pre (f32_attn_c_attn_weight w) (f32_attn_c_attn_bias w)).
  pose (PRJ := f32_linear_pre (f32_attn_c_proj_weight w) (f32_attn_c_proj_bias w)).
  pose (Hh := fun h (ys : list (list binary32)) =>
          f32_causal_attention (mapi (fun _ => g2_qh d nh h) ys)
            (mapi (fun _ => g2_kh d nh h) ys) (mapi (fun _ => g2_vh d nh h) ys)
            (Nat.div d nh)).
  pose (HEADS := fun ys => f32_concat_heads (List.map (fun h => Hh h ys) (List.seq 0 nh))).
  assert (R : realized (g2_att_init nh) (g2_att_step d nh w)
                (fun xs => (fun ys => List.map PRJ (HEADS ys)) (List.map QKV xs))).
  { apply (realized_comp tt (map_step QKV) (List.map QKV) _ _
             (fun ys => List.map PRJ (HEADS ys))); [apply realized_map_step|].
    apply (realized_comp _ _ HEADS tt (map_step PRJ) (List.map PRJ));
      [|apply realized_map_step].
    apply realized_heads; [exact Hnh|]. intros h _. apply realized_attn. }
  eapply realized_ext; [|exact R]. intros xs. symmetry. apply g2_attention_eq.
Qed.

Lemma realized_g2_block : forall cfg eps b, (0 < gpt2_inf_n_head cfg)%nat ->
  realized (g2_block_init (gpt2_inf_n_head cfg)) (g2_block_step cfg eps b)
           (f32_block_pre cfg eps b).
Proof.
  intros cfg eps b Hnh.
  pose (LN1 := g2_ln (f32_block_ln_1 b) eps).
  pose (ATT := f32_attention_pre (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                 (f32_attn_c_attn_weight (f32_block_attn b))
                 (f32_attn_c_attn_bias (f32_block_attn b))
                 (f32_attn_c_proj_weight (f32_block_attn b))
                 (f32_attn_c_proj_bias (f32_block_attn b))).
  pose (MLPR := fun r => g2_mlp_row (f32_block_mlp b) (g2_ln (f32_block_ln_2 b) eps r)).
  pose (ID := List.map (fun r : list binary32 => r)).
  pose (H2 := fun xs => zipw f32_vec_add (ID xs) (ATT (List.map LN1 xs))).
  pose (OUT := fun ys => zipw f32_vec_add (ID ys) (List.map MLPR ys)).
  assert (R : realized (g2_block_init (gpt2_inf_n_head cfg)) (g2_block_step cfg eps b)
                (fun xs => OUT (H2 xs))).
  { apply (realized_comp _ _ H2 _ _ OUT).
    - apply (realized_zip tt (map_step (fun r : list binary32 => r)) ID _ _
               (fun xs => ATT (List.map LN1 xs))).
      + apply realized_map_step.
      + apply (realized_comp tt (map_step LN1) (List.map LN1) _ _ ATT).
        * apply realized_map_step.
        * apply realized_g2_att. exact Hnh.
    - apply realized_zip; apply realized_map_step. }
  eapply realized_ext; [|exact R]. intros xs.
  unfold OUT, H2, ID, MLPR, LN1, ATT, f32_block_pre. cbv zeta.
  rewrite !map_id_rows, !add_matrices_zipw.
  unfold f32_layer_norm_2d, f32_mlp_pre, f32_linear_pre_2d, g2_mlp_row, g2_ln.
  rewrite !map_map. reflexivity.
Qed.

Lemma blocks_pre_stack : forall cfg eps bs h,
  f32_blocks_pre cfg eps bs h = stack_fun (List.map (f32_block_pre cfg eps) bs) h.
Proof.
  intros cfg eps bs. induction bs as [|b bs IH]; intros h; [reflexivity|].
  cbn [f32_blocks_pre List.map stack_fun]. apply IH.
Qed.

Definition g2_blocks (cfg : gpt2_inference_config) (eps : binary32)
    (m : f32_model_weights) :=
  List.map (fun b => (g2_block_init (gpt2_inf_n_head cfg), g2_block_step cfg eps b))
           (f32_blocks m).

Definition g2_emb (m : f32_model_weights) (t tok : nat) : list binary32 :=
  f32_vec_add (f32_lookup_embedding (f32_wte m) tok) (f32_lookup_embedding (f32_wpe m) t).

Definition g2_final (eps : binary32) (m : f32_model_weights) (h : list binary32)
    : list binary32 :=
  let hn := g2_ln (f32_ln_f m) eps h in
  List.map (fun w_row => f32_dot hn w_row) (f32_wte m).

(** The GPT-2 decode step, and the state it starts from. *)
Definition gpt2_init (cfg : gpt2_inference_config) (eps : binary32)
    (m : f32_model_weights) :=
  (0%nat, (List.map fst (g2_blocks cfg eps m), tt)).

Definition gpt2_step (cfg : gpt2_inference_config) (eps : binary32)
    (m : f32_model_weights) :=
  comp_step (fun (t : nat) (tok : nat) => (S t, g2_emb m t tok))
    (comp_step (stack_step (List.map snd (g2_blocks cfg eps m)))
               (map_step (g2_final eps m))).

Lemma embed_mapi : forall m toks,
  f32_add_matrices (f32_embed_tokens (f32_wte m) toks)
                   (f32_embed_positions (f32_wpe m) (List.length toks))
  = mapi (g2_emb m) toks.
Proof.
  intros m toks.
  unfold f32_add_matrices, f32_embed_tokens, f32_embed_positions, mapi, g2_emb.
  generalize 0%nat as n.
  induction toks as [|t toks IH]; intros n; [reflexivity|].
  cbn [List.length List.seq List.map List.combine]. f_equal. apply IH.
Qed.

(** Decoding GPT-2 one token at a time, with a key/value cache per head and
    layer, emits exactly the logit rows of the pre-transposed forward pass. *)
Theorem gpt2_decode_correct : forall cfg eps m toks,
  (0 < gpt2_inf_n_head cfg)%nat ->
  run (gpt2_step cfg eps m) (gpt2_init cfg eps m) toks
  = f32_gpt2_logits_pre cfg eps m toks.
Proof.
  intros cfg eps m toks Hnh. apply run_realized.
  pose (EMB := mapi (g2_emb m)).
  pose (STK := stack_fun (List.map (f32_block_pre cfg eps) (f32_blocks m))).
  pose (FIN := List.map (g2_final eps m)).
  assert (R : realized (gpt2_init cfg eps m) (gpt2_step cfg eps m)
                (fun ts => (fun hs => FIN (STK hs)) (EMB ts))).
  { apply (realized_comp 0%nat (fun (t : nat) (tok : nat) => (S t, g2_emb m t tok))
             EMB _ _ (fun hs => FIN (STK hs))).
    - apply realized_mapi.
    - apply (realized_comp _ _ STK tt (map_step (g2_final eps m)) FIN);
        [|apply realized_map_step].
      unfold STK, g2_blocks. apply realized_stack.
      induction (f32_blocks m) as [|b bs IH]; constructor; [|exact IH].
      cbn [fst snd]. apply realized_g2_block. exact Hnh. }
  eapply realized_ext; [|exact R]. intros ts.
  unfold FIN, STK, EMB, f32_gpt2_logits_pre, f32_gpt2_forward_pre. cbv zeta.
  rewrite embed_mapi, blocks_pre_stack.
  unfold f32_layer_norm_2d, g2_final, g2_ln. rewrite map_map. reflexivity.
Qed.

(** * Rotary tables as functions of the position

    The runners' tables hold these rows for the positions of the input; a
    decoder, which does not know the length ahead, forms the row of each
    position as it reaches it. *)

Definition rope_angles_at (invf : list binary32) (pos : nat) : list binary32 :=
  List.map (fun fj => f32_mult (f32_of_Z (Z.of_nat pos)) fj) invf.
Definition rope_cos_at (invf : list binary32) (pos : nat) : list binary32 :=
  let ang := rope_angles_at invf pos in List.map f32_cos ang ++ List.map f32_cos ang.
Definition rope_sin_at (invf : list binary32) (pos : nat) : list binary32 :=
  let ang := rope_angles_at invf pos in List.map f32_sin ang ++ List.map f32_sin ang.

Lemma rope_tables_at : forall invf npos t, (t < npos)%nat ->
  List.nth t (f32_rope_cos invf npos) [] = rope_cos_at invf t
  /\ List.nth t (f32_rope_sin invf npos) [] = rope_sin_at invf t.
Proof.
  intros invf npos t H. unfold f32_rope_cos, f32_rope_sin, f32_rope_angles.
  rewrite map_seq_spec, !map_map, !nth_map_seq0 by exact H. split; reflexivity.
Qed.

Lemma combine_map_r : forall {A B C : Type} (l : list A) (l' : list B) (f : B -> C),
  List.combine l (List.map f l') = List.map (fun p => (fst p, f (snd p))) (List.combine l l').
Proof.
  intros A B C l. induction l as [|a l IH]; intros l' f; [reflexivity|].
  destruct l' as [|b l']; [reflexivity|]. cbn. f_equal. apply IH.
Qed.

(** One head's stream, sliced out of a projection and rotated at its position,
    with the rotation row given by the formula. *)
Definition lm_rope (hd : nat) (invf : list binary32) (c t : nat) (r : list binary32) :=
  f32_partial_rope hd (rope_cos_at invf t) (rope_sin_at invf t) (f32_slice (c * hd) hd r).

Lemma rope_rows_mapi : forall {A : Type} hd cosv sinv invf c (xs : list A)
    (proj : A -> list binary32),
  (forall t, (t < List.length xs)%nat ->
     List.nth t cosv [] = rope_cos_at invf t /\ List.nth t sinv [] = rope_sin_at invf t) ->
  f32_llama_rope_rows hd cosv sinv c (List.map proj xs)
  = mapi (fun t x => lm_rope hd invf c t (proj x)) xs.
Proof.
  intros A hd cosv sinv invf c xs proj Ht.
  unfold f32_llama_rope_rows, mapi. rewrite length_map, combine_map_r, map_map.
  apply map_ext_in. intros [t x] Hin. cbn [fst snd].
  apply in_combine_l, in_seq in Hin.
  destruct (Ht t ltac:(lia)) as [Hc Hs]. rewrite Hc, Hs. reflexivity.
Qed.

(** * Llama *)

Definition lm_qkv (w : llama_attn_weights) (h : list binary32)
    : list binary32 * list binary32 * list binary32 :=
  (f32_mat_vec_mul (la_q w) h, f32_mat_vec_mul (la_k w) h, f32_mat_vec_mul (la_v w) h).

Definition lm_Q (hd : nat) (invf : list binary32) (hh : nat) :=
  fun t (x : list binary32 * list binary32 * list binary32) => lm_rope hd invf hh t (fst (fst x)).
Definition lm_K (hd : nat) (invf : list binary32) (c : nat) :=
  fun t (x : list binary32 * list binary32 * list binary32) => lm_rope hd invf c t (snd (fst x)).
Definition lm_V (hd : nat) (c : nat) :=
  fun (_ : nat) (x : list binary32 * list binary32 * list binary32) => f32_slice (c * hd) hd (snd x).

Definition lm_head_step (nh nkv hd : nat) (invf : list binary32) (hh : nat) :=
  let c := Nat.div hh (Nat.div nh nkv) in
  attn_step (lm_Q hd invf hh) (lm_K hd invf c) (lm_V hd c) hd.

Definition lm_att_step (nh nkv hd : nat) (invf : list binary32) (w : llama_attn_weights) :=
  comp_step (map_step (lm_qkv w))
    (comp_step (heads_step nh (fun _ => attn0) (lm_head_step nh nkv hd invf))
               (map_step (fun r => f32_mat_vec_mul (la_o w) r))).

Definition lm_att_init (nh : nat) :=
  (tt, (List.map (fun _ : nat => attn0) (List.seq 0 nh), tt)).

Definition lm_heads_f (nh nkv hd : nat) (invf : list binary32)
    (ys : list (list binary32 * list binary32 * list binary32)) :=
  f32_concat_heads (List.map (fun hh =>
    let c := Nat.div hh (Nat.div nh nkv) in
    f32_causal_attention (mapi (lm_Q hd invf hh) ys) (mapi (lm_K hd invf c) ys)
                         (mapi (lm_V hd c) ys) hd) (List.seq 0 nh)).

Definition lm_attn_f (nh nkv hd : nat) (invf : list binary32) (w : llama_attn_weights)
    (hn : list (list binary32)) : list (list binary32) :=
  List.map (fun r => f32_mat_vec_mul (la_o w) r)
           (lm_heads_f nh nkv hd invf (List.map (lm_qkv w) hn)).

Lemma realized_lm_att : forall nh nkv hd invf w, (0 < nh)%nat ->
  realized (lm_att_init nh) (lm_att_step nh nkv hd invf w) (lm_attn_f nh nkv hd invf w).
Proof.
  intros nh nkv hd invf w Hnh.
  pose (O := fun r => f32_mat_vec_mul (la_o w) r).
  pose (HEADS := lm_heads_f nh nkv hd invf).
  assert (R : realized (lm_att_init nh) (lm_att_step nh nkv hd invf w)
                (fun xs => (fun ys => List.map O (HEADS ys)) (List.map (lm_qkv w) xs))).
  { apply (realized_comp tt (map_step (lm_qkv w)) (List.map (lm_qkv w)) _ _
             (fun ys => List.map O (HEADS ys))); [apply realized_map_step|].
    apply (realized_comp _ _ HEADS tt (map_step O) (List.map O)); [|apply realized_map_step].
    apply realized_heads; [exact Hnh|]. intros hh _. apply realized_attn. }
  eapply realized_ext; [|exact R]. reflexivity.
Qed.

Lemma llama_attn_eq : forall nh nkv hd w cosv sinv invf hn,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  (forall t, (t < List.length hn)%nat ->
     List.nth t cosv [] = rope_cos_at invf t /\ List.nth t sinv [] = rope_sin_at invf t) ->
  f32_llama_attn nh nkv hd w cosv sinv hn = lm_attn_f nh nkv hd invf w hn.
Proof.
  intros nh nkv hd w cosv sinv invf hn Hnh Hnkv Hmod Ht.
  unfold f32_llama_attn, f32_llama_heads, lm_attn_f, lm_heads_f. cbv zeta.
  f_equal. f_equal. apply map_ext_in. intros hh Hh. apply in_seq in Hh.
  assert (Hc : (Nat.div hh (Nat.div nh nkv) < nkv)%nat) by (apply group_index_lt; lia).
  rewrite !nth_map_seq_lt by lia.
  assert (Ht' : forall t, (t < List.length (List.map (lm_qkv w) hn))%nat ->
     List.nth t cosv [] = rope_cos_at invf t /\ List.nth t sinv [] = rope_sin_at invf t)
    by (intros t H; rewrite length_map in H; exact (Ht t H)).
  f_equal.
  - replace (List.map (fun h => f32_mat_vec_mul (la_q w) h) hn)
      with (List.map (fun x => fst (fst x)) (List.map (lm_qkv w) hn))
      by (rewrite map_map; reflexivity).
    apply rope_rows_mapi. exact Ht'.
  - replace (List.map (fun h => f32_mat_vec_mul (la_k w) h) hn)
      with (List.map (fun x => snd (fst x)) (List.map (lm_qkv w) hn))
      by (rewrite map_map; reflexivity).
    apply rope_rows_mapi. exact Ht'.
  - unfold lm_V. rewrite <- (map_mapi (fun x => f32_slice (Nat.div hh (Nat.div nh nkv) * hd) hd (snd x))).
    rewrite !map_map. reflexivity.
Qed.

Definition lm_mlp_row (eps : binary32) (L : llama_layer_weights) (r : list binary32) :=
  f32_swiglu (lm_gate (ll_mlp L)) (lm_up (ll_mlp L)) (lm_down (ll_mlp L))
             (f32_rmsnorm (ll_ln2 L) eps r).

Definition lm_layer_step (nh nkv hd : nat) (eps : binary32) (invf : list binary32)
    (L : llama_layer_weights) :=
  comp_step
    (zip_step (map_step (fun r : list binary32 => r))
              (comp_step (map_step (fun r => f32_rmsnorm (ll_ln1 L) eps r))
                         (lm_att_step nh nkv hd invf (ll_attn L)))
              f32_vec_add)
    (zip_step (map_step (fun r : list binary32 => r)) (map_step (lm_mlp_row eps L))
              f32_vec_add).

Definition lm_layer_init (nh : nat) := ((tt, (tt, lm_att_init nh)), (tt, tt)).

(** The layer with its rotation given by the formula. *)
Definition lm_layer_f (nh nkv hd : nat) (eps : binary32) (invf : list binary32)
    (L : llama_layer_weights) (h : list (list binary32)) : list (list binary32) :=
  let h2 := zipw f32_vec_add h
              (lm_attn_f nh nkv hd invf (ll_attn L)
                 (List.map (fun r => f32_rmsnorm (ll_ln1 L) eps r) h)) in
  zipw f32_vec_add h2 (List.map (lm_mlp_row eps L) h2).

Lemma realized_lm_layer : forall nh nkv hd eps invf L, (0 < nh)%nat ->
  realized (lm_layer_init nh) (lm_layer_step nh nkv hd eps invf L)
           (lm_layer_f nh nkv hd eps invf L).
Proof.
  intros nh nkv hd eps invf L Hnh.
  pose (N1 := fun r => f32_rmsnorm (ll_ln1 L) eps r).
  pose (ATT := lm_attn_f nh nkv hd invf (ll_attn L)).
  pose (ID := List.map (fun r : list binary32 => r)).
  pose (H2 := fun xs => zipw f32_vec_add (ID xs) (ATT (List.map N1 xs))).
  pose (OUT := fun ys => zipw f32_vec_add (ID ys) (List.map (lm_mlp_row eps L) ys)).
  assert (R : realized (lm_layer_init nh) (lm_layer_step nh nkv hd eps invf L)
                (fun xs => OUT (H2 xs))).
  { apply (realized_comp _ _ H2 _ _ OUT).
    - apply (realized_zip tt (map_step (fun r : list binary32 => r)) ID _ _
               (fun xs => ATT (List.map N1 xs))); [apply realized_map_step|].
      apply (realized_comp tt (map_step N1) (List.map N1) _ _ ATT);
        [apply realized_map_step|].
      apply realized_lm_att. exact Hnh.
    - apply realized_zip; apply realized_map_step. }
  eapply realized_ext; [|exact R]. intros xs.
  unfold OUT, H2, ID, lm_layer_f. rewrite !map_id_rows. reflexivity.
Qed.

Lemma lm_layer_eq : forall nh nkv hd eps ln1 ln2 aw mw invf cosv sinv h,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  (forall t, (t < List.length h)%nat ->
     List.nth t cosv [] = rope_cos_at invf t /\ List.nth t sinv [] = rope_sin_at invf t) ->
  f32_llama_layer nh nkv hd eps ln1 ln2 aw mw cosv sinv h
  = lm_layer_f nh nkv hd eps invf (mk_llama_layer_weights ln1 ln2 aw mw) h.
Proof.
  intros nh nkv hd eps ln1 ln2 aw mw invf cosv sinv h Hnh Hnkv Hmod Ht.
  unfold f32_llama_layer, f32_llama_wrap, lm_layer_f, lm_mlp_row. cbv zeta.
  cbn [ll_ln1 ll_ln2 ll_attn ll_mlp].
  rewrite llama_attn_eq with (invf := invf)
    by (try assumption; intros t Hl; rewrite length_map in Hl; exact (Ht t Hl)).
  rewrite !map_pair_zipw, map_map. reflexivity.
Qed.

Definition lm_final (eps : binary32) (m : llama_model_weights) (h : list binary32)
    : list binary32 :=
  let hn := f32_rmsnorm (lw_norm m) eps h in
  List.map (fun wrow => f32_dot hn wrow) (lw_emb m).

Definition lm_layers (nh nkv hd : nat) (eps : binary32) (m : llama_model_weights) :=
  List.map (fun L => (lm_layer_init nh, lm_layer_step nh nkv hd eps (lw_invf m) L))
           (lw_layers m).

(** The Llama decode step, and the state it starts from. *)
Definition llama_init (nh nkv hd : nat) (eps : binary32) (m : llama_model_weights) :=
  (tt, (List.map fst (lm_layers nh nkv hd eps m), tt)).

Definition llama_step (nh nkv hd : nat) (eps : binary32) (m : llama_model_weights) :=
  comp_step (map_step (f32_lookup_embedding (lw_emb m)))
    (comp_step (stack_step (List.map snd (lm_layers nh nkv hd eps m)))
               (map_step (lm_final eps m))).

Lemma lm_layer_f_length : forall nh nkv hd eps invf L h, (0 < nh)%nat ->
  List.length (lm_layer_f nh nkv hd eps invf L h) = List.length h.
Proof.
  intros. exact (realized_length _ _ _ (realized_lm_layer nh nkv hd eps invf L H) h).
Qed.

Lemma lm_stack_eq : forall nh nkv hd eps invf cosv sinv layers h,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  (forall t, (t < List.length h)%nat ->
     List.nth t cosv [] = rope_cos_at invf t /\ List.nth t sinv [] = rope_sin_at invf t) ->
  f32_llama_stack (List.map (fun L => f32_llama_layer nh nkv hd eps (ll_ln1 L) (ll_ln2 L)
                                        (ll_attn L) (ll_mlp L) cosv sinv) layers) h
  = stack_fun (List.map (lm_layer_f nh nkv hd eps invf) layers) h.
Proof.
  intros nh nkv hd eps invf cosv sinv layers.
  induction layers as [|L layers IH]; intros h Hnh Hnkv Hmod Ht; [reflexivity|].
  cbn [List.map f32_llama_stack stack_fun].
  rewrite lm_layer_eq with (invf := invf) by assumption.
  destruct L as [ln1 ln2 aw mw]. cbn [ll_ln1 ll_ln2 ll_attn ll_mlp].
  apply IH; try assumption.
  intros t Hl. rewrite lm_layer_f_length in Hl by assumption. exact (Ht t Hl).
Qed.

(** Decoding a Llama model one token at a time, with a key/value cache per
    head and layer and each position's rotation formed as it is reached,
    emits exactly the logit rows [f32_llama_logits_of] computes. *)
Theorem llama_decode_correct : forall nh nkv hd eps m ids,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  run (llama_step nh nkv hd eps m) (llama_init nh nkv hd eps m) ids
  = f32_llama_logits_of nh nkv hd eps m ids.
Proof.
  intros nh nkv hd eps m ids Hnh Hnkv Hmod.
  pose (STK := stack_fun (List.map (lm_layer_f nh nkv hd eps (lw_invf m)) (lw_layers m))).
  pose (FIN := List.map (lm_final eps m)).
  assert (R : realized (llama_init nh nkv hd eps m) (llama_step nh nkv hd eps m)
                (fun xs => (fun hs => FIN (STK hs))
                             (List.map (f32_lookup_embedding (lw_emb m)) xs))).
  { apply (realized_comp tt (map_step (f32_lookup_embedding (lw_emb m)))
             (List.map (f32_lookup_embedding (lw_emb m))) _ _ (fun hs => FIN (STK hs)));
      [apply realized_map_step|].
    apply (realized_comp _ _ STK tt (map_step (lm_final eps m)) FIN);
      [|apply realized_map_step].
    unfold STK, lm_layers. apply realized_stack.
    induction (lw_layers m) as [|L Ls IH]; constructor; [|exact IH].
    cbn [fst snd]. apply realized_lm_layer. exact Hnh. }
  rewrite (run_realized _ _ _ R ids).
  unfold FIN, STK, f32_llama_logits_of, f32_llama_logits, f32_llama_forward, f32_embed_tokens.
  cbv zeta.
  rewrite lm_stack_eq with (invf := lw_invf m).
  - rewrite map_map. apply map_ext. intros h. reflexivity.
  - exact Hnh.
  - exact Hnkv.
  - exact Hmod.
  - intros t Hl. rewrite length_map in Hl. apply rope_tables_at. exact Hl.
Qed.

(** * Qwen3.5 *)

(** ** Pieces shared by the two layer kinds *)

Lemma zipw_pair_fst : forall {B C D : Type} (f : B -> D) (ys : list B) (zs : list C),
  List.length ys = List.length zs ->
  List.map (fun x => f (fst x)) (zipw pair ys zs) = List.map f ys.
Proof.
  intros B C D f ys. unfold zipw. induction ys as [|y ys IH]; intros zs H;
    destruct zs as [|z zs]; cbn in H; try discriminate; [reflexivity|].
  cbn. f_equal. apply IH. lia.
Qed.

Lemma zipw_pair_snd : forall {B C D : Type} (f : C -> D) (ys : list B) (zs : list C),
  List.length ys = List.length zs ->
  List.map (fun x => f (snd x)) (zipw pair ys zs) = List.map f zs.
Proof.
  intros B C D f ys. unfold zipw. induction ys as [|y ys IH]; intros zs H;
    destruct zs as [|z zs]; cbn in H; try discriminate; [reflexivity|].
  cbn. f_equal. apply IH. lia.
Qed.

Lemma map_pair_zipw_gen : forall {B C D : Type} (h : B -> C -> D) (X : list B) (Y : list C),
  List.map (fun p => let '(a, b) := p in h a b) (List.combine X Y) = zipw h X Y.
Proof. intros. unfold zipw. apply map_ext. intros [a b]. reflexivity. Qed.

Lemma zipw_length : forall {B C D : Type} (h : B -> C -> D) ys zs,
  List.length (zipw h ys zs) = Nat.min (List.length ys) (List.length zs).
Proof. intros. unfold zipw. rewrite length_map, length_combine. reflexivity. Qed.

Definition qw_mlp_row (eps : binary32) (ln2 : list binary32) (mlp : qwen_mlp_weights)
    (r : list binary32) : list binary32 :=
  f32_swiglu (qm_gate mlp) (qm_up mlp) (qm_down mlp) (f32_rmsnorm_zc ln2 eps r).

Definition qw_wrap_step {S : Type} (eps : binary32) (ln1 ln2 : list binary32)
    (mlp : qwen_mlp_weights) (mix_step : S -> list binary32 -> S * list binary32) :=
  comp_step
    (zip_step (map_step (fun r : list binary32 => r))
              (comp_step (map_step (fun r => f32_rmsnorm_zc ln1 eps r)) mix_step)
              f32_vec_add)
    (zip_step (map_step (fun r : list binary32 => r)) (map_step (qw_mlp_row eps ln2 mlp))
              f32_vec_add).

Definition qw_wrap_init {S : Type} (mix_init : S) := ((tt, (tt, mix_init)), (tt, tt)).

Lemma realized_qw_wrap : forall {S : Type} eps ln1 ln2 mlp (mix_init : S) mix_step MIX,
  realized mix_init mix_step MIX ->
  realized (qw_wrap_init mix_init) (qw_wrap_step eps ln1 ln2 mlp mix_step)
           (f32_qwen_wrap eps ln1 ln2 mlp MIX).
Proof.
  intros S eps ln1 ln2 mlp mix_init mix_step MIX HM.
  pose (N1 := fun r => f32_rmsnorm_zc ln1 eps r).
  pose (ID := List.map (fun r : list binary32 => r)).
  pose (H2 := fun xs => zipw f32_vec_add (ID xs) (MIX (List.map N1 xs))).
  pose (OUT := fun ys => zipw f32_vec_add (ID ys) (List.map (qw_mlp_row eps ln2 mlp) ys)).
  assert (R : realized (qw_wrap_init mix_init) (qw_wrap_step eps ln1 ln2 mlp mix_step)
                (fun xs => OUT (H2 xs))).
  { apply (realized_comp _ _ H2 _ _ OUT).
    - apply (realized_zip tt (map_step (fun r : list binary32 => r)) ID _ _
               (fun xs => MIX (List.map N1 xs))); [apply realized_map_step|].
      apply (realized_comp tt (map_step N1) (List.map N1) _ _ MIX);
        [apply realized_map_step | exact HM].
    - apply realized_zip; apply realized_map_step. }
  eapply realized_ext; [|exact R]. intros xs.
  unfold OUT, H2, ID, N1, f32_qwen_wrap, qw_mlp_row. cbv zeta.
  rewrite !map_id_rows, !map_pair_zipw, map_map. reflexivity.
Qed.

(** Layers of two kinds share one state type, a sum; each layer's step acts on
    its own side. *)
Definition lift_l {S1 S2 : Type} (st : S1 -> list binary32 -> S1 * list binary32)
    (s : S1 + S2) (x : list binary32) : (S1 + S2) * list binary32 :=
  match s with
  | inl s1 => let r := st s1 x in (inl (fst r), snd r)
  | inr _ => (s, x)
  end.

Definition lift_r {S1 S2 : Type} (st : S2 -> list binary32 -> S2 * list binary32)
    (s : S1 + S2) (x : list binary32) : (S1 + S2) * list binary32 :=
  match s with
  | inr s2 => let r := st s2 x in (inr (fst r), snd r)
  | inl _ => (s, x)
  end.

Lemma realized_lift_l : forall {S1 S2 : Type} (i : S1) st F,
  realized i st F -> realized (@inl S1 S2 i) (lift_l st) F.
Proof.
  intros S1 S2 i st F [F0 Fs].
  assert (A : forall xs, after (lift_l st) (@inl S1 S2 i) xs = inl (after st i xs)).
  { intros xs. induction xs as [|x xs IH] using rev_ind; [reflexivity|].
    rewrite !after_snoc, IH. reflexivity. }
  split; [exact F0|]. intros xs x. rewrite A. cbn. apply Fs.
Qed.

Lemma realized_lift_r : forall {S1 S2 : Type} (i : S2) st F,
  realized i st F -> realized (@inr S1 S2 i) (lift_r st) F.
Proof.
  intros S1 S2 i st F [F0 Fs].
  assert (A : forall xs, after (lift_r st) (@inr S1 S2 i) xs = inr (after st i xs)).
  { intros xs. induction xs as [|x xs IH] using rev_ind; [reflexivity|].
    rewrite !after_snoc, IH. reflexivity. }
  split; [exact F0|]. intros xs x. rewrite A. cbn. apply Fs.
Qed.

(** ** The gated full-attention layer *)

Definition qa_triple (w : qwen_attn_weights) (h : list binary32)
    : list binary32 * list binary32 * list binary32 :=
  (f32_mat_vec_mul (qa_q w) h, f32_mat_vec_mul (qa_k w) h, f32_mat_vec_mul (qa_v w) h).

Definition qa_Q (hd rd : nat) (eps : binary32) (w : qwen_attn_weights)
    (invf : list binary32) (hh : nat) :=
  fun t (x : list binary32 * list binary32 * list binary32) =>
    f32_partial_rope rd (rope_cos_at invf t) (rope_sin_at invf t)
      (f32_rmsnorm_zc (qa_q_norm w) eps (f32_slice (hh * hd * 2) hd (fst (fst x)))).
Definition qa_K (hd rd : nat) (eps : binary32) (w : qwen_attn_weights)
    (invf : list binary32) (c : nat) :=
  fun t (x : list binary32 * list binary32 * list binary32) =>
    f32_partial_rope rd (rope_cos_at invf t) (rope_sin_at invf t)
      (f32_rmsnorm_zc (qa_k_norm w) eps (f32_slice (c * hd) hd (snd (fst x)))).
Definition qa_V (hd c : nat) :=
  fun (_ : nat) (x : list binary32 * list binary32 * list binary32) =>
    f32_slice (c * hd) hd (snd x).
Definition qa_gate (nh hd : nat) (x : list binary32 * list binary32 * list binary32) :=
  List.concat (List.map (fun hh => f32_slice (hh * hd * 2 + hd) hd (fst (fst x)))
                        (List.seq 0 nh)).
Definition qa_out (w : qwen_attn_weights) (g o : list binary32) : list binary32 :=
  f32_mat_vec_mul (qa_o w) (f32_gate_sigmoid g o).

Definition qa_head_step (nh nkv hd rd : nat) (eps : binary32) (w : qwen_attn_weights)
    (invf : list binary32) (hh : nat) :=
  let c := Nat.div hh (Nat.div nh nkv) in
  attn_step (qa_Q hd rd eps w invf hh) (qa_K hd rd eps w invf c) (qa_V hd c) hd.

Definition qa_mix_step (nh nkv hd rd : nat) (eps : binary32) (w : qwen_attn_weights)
    (invf : list binary32) :=
  comp_step (map_step (qa_triple w))
    (zip_step (map_step (qa_gate nh hd))
              (heads_step nh (fun _ => attn0) (qa_head_step nh nkv hd rd eps w invf))
              (qa_out w)).

Definition qa_mix_init (nh : nat) := (tt, (tt, List.map (fun _ : nat => attn0) (List.seq 0 nh))).

Definition qa_heads_f (nh nkv hd rd : nat) (eps : binary32) (w : qwen_attn_weights)
    (invf : list binary32) (ys : list (list binary32 * list binary32 * list binary32)) :=
  f32_concat_heads (List.map (fun hh =>
    let c := Nat.div hh (Nat.div nh nkv) in
    f32_causal_attention (mapi (qa_Q hd rd eps w invf hh) ys)
      (mapi (qa_K hd rd eps w invf c) ys) (mapi (qa_V hd c) ys) hd) (List.seq 0 nh)).

Definition qa_mix_f (nh nkv hd rd : nat) (eps : binary32) (w : qwen_attn_weights)
    (invf : list binary32) (hn : list (list binary32)) : list (list binary32) :=
  let ys := List.map (qa_triple w) hn in
  zipw (qa_out w) (List.map (qa_gate nh hd) ys) (qa_heads_f nh nkv hd rd eps w invf ys).

Lemma realized_qa_mix : forall nh nkv hd rd eps w invf, (0 < nh)%nat ->
  realized (qa_mix_init nh) (qa_mix_step nh nkv hd rd eps w invf)
           (qa_mix_f nh nkv hd rd eps w invf).
Proof.
  intros nh nkv hd rd eps w invf Hnh.
  pose (HEADS := qa_heads_f nh nkv hd rd eps w invf).
  assert (R : realized (qa_mix_init nh) (qa_mix_step nh nkv hd rd eps w invf)
                (fun xs => (fun ys => zipw (qa_out w) (List.map (qa_gate nh hd) ys) (HEADS ys))
                             (List.map (qa_triple w) xs))).
  { apply (realized_comp tt (map_step (qa_triple w)) (List.map (qa_triple w)) _ _
             (fun ys => zipw (qa_out w) (List.map (qa_gate nh hd) ys) (HEADS ys)));
      [apply realized_map_step|].
    apply (realized_zip tt (map_step (qa_gate nh hd)) (List.map (qa_gate nh hd)) _ _ HEADS);
      [apply realized_map_step|].
    apply realized_heads; [exact Hnh|]. intros hh _. apply realized_attn. }
  eapply realized_ext; [|exact R]. reflexivity.
Qed.

Lemma qwen_attn_eq : forall nh nkv hd rd eps w cosv sinv invf hn,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  (forall t, (t < List.length hn)%nat ->
     List.nth t cosv [] = rope_cos_at invf t /\ List.nth t sinv [] = rope_sin_at invf t) ->
  f32_qwen_attn_mix nh nkv hd rd eps w cosv sinv hn = qa_mix_f nh nkv hd rd eps w invf hn.
Proof.
  intros nh nkv hd rd eps w cosv sinv invf hn Hnh Hnkv Hmod Ht.
  unfold f32_qwen_attn_mix, qa_mix_f, qa_heads_f. cbv zeta.
  rewrite map_pair_zipw_gen. unfold qa_out.
  f_equal.
  - rewrite !map_map. reflexivity.
  - f_equal. apply map_ext_in. intros hh Hh. apply in_seq in Hh.
    assert (Hc : (Nat.div hh (Nat.div nh nkv) < nkv)%nat) by (apply group_index_lt; lia).
    rewrite !nth_map_seq_lt by lia.
    f_equal.
    + unfold mapi. rewrite !length_map, !combine_map_r, !map_map.
      apply map_ext_in. intros [t x] Hin. cbn [fst snd].
      apply in_combine_l, in_seq in Hin.
      destruct (Ht t ltac:(lia)) as [Hcos Hsin]. rewrite Hcos, Hsin. reflexivity.
    + unfold mapi. rewrite !length_map, !combine_map_r, !map_map.
      apply map_ext_in. intros [t x] Hin. cbn [fst snd].
      apply in_combine_l, in_seq in Hin.
      destruct (Ht t ltac:(lia)) as [Hcos Hsin]. rewrite Hcos, Hsin. reflexivity.
    + unfold qa_V.
      rewrite <- (map_mapi (fun x => f32_slice (Nat.div hh (Nat.div nh nkv) * hd) hd (snd x))).
      rewrite !map_map. reflexivity.
Qed.

(** ** The gated DeltaNet layer

    The fused query/key/value stream passes through the convolution, whose
    state is the window of the last [ck - 1] rows; the gate and the two
    per-head rates pass beside it. Each head then carries its own recurrent
    state and normalizes its read-out against its slice of the gate. *)

Lemma map_zipw_pair_l : forall {B C D : Type} (g : B * C -> D) (f : B -> D) ys zs,
  List.length ys = List.length zs -> (forall y z, g (y, z) = f y) ->
  List.map g (zipw pair ys zs) = List.map f ys.
Proof.
  intros B C D g f ys. unfold zipw. induction ys as [|y ys IH]; intros zs H E;
    destruct zs as [|z zs]; cbn in H; try discriminate; [reflexivity|].
  cbn. rewrite E. f_equal. apply IH; [lia | exact E].
Qed.

Lemma map_zipw_pair_r : forall {B C D : Type} (g : B * C -> D) (f : C -> D) ys zs,
  List.length ys = List.length zs -> (forall y z, g (y, z) = f z) ->
  List.map g (zipw pair ys zs) = List.map f zs.
Proof.
  intros B C D g f ys. unfold zipw. induction ys as [|y ys IH]; intros zs H E;
    destruct zs as [|z zs]; cbn in H; try discriminate; [reflexivity|].
  cbn. rewrite E. f_equal. apply IH; [lia | exact E].
Qed.

Definition qd_zab (w : qwen_delta_weights) (h : list binary32)
    : list binary32 * list binary32 * list binary32 :=
  (f32_mat_vec_mul (qd_in_z w) h, f32_mat_vec_mul (qd_in_a w) h,
   f32_mat_vec_mul (qd_in_b w) h).

Definition qd_conv_step (lnh lhd ck : nat) (w : qwen_delta_weights) :=
  comp_step (map_step (fun h => f32_mat_vec_mul (qd_in_qkv w) h))
            (conv_state_step (3 * (lnh * lhd)) ck (qd_conv_w w) []).

Definition qd_C (lnh lhd ck : nat) (w : qwen_delta_weights) (hn : list (list binary32))
    : list (list binary32) :=
  f32_causal_conv1d (3 * (lnh * lhd)) ck (qd_conv_w w) []
    (List.map (fun h => f32_mat_vec_mul (qd_in_qkv w) h) hn).

(** What a head reads at a position: the convolved stream, then the gate and
    the two rates. *)
Definition qd_row : Type :=
  (list binary32 * (list binary32 * list binary32 * list binary32))%type.

Definition qd_z (x : qd_row) : list binary32 := fst (fst (snd x)).
Definition qd_beta (hh : nat) (x : qd_row) : binary32 :=
  f32_sigmoid (List.nth hh (snd (snd x)) f32_zero).
Definition qd_gamma (w : qwen_delta_weights) (hh : nat) (x : qd_row) : binary32 :=
  f32_delta_decay (List.nth hh (qd_a_log w) f32_zero) (List.nth hh (qd_dt_bias w) f32_zero)
                  (List.nth hh (snd (fst (snd x))) f32_zero).
Definition qd_q (eps : binary32) (lhd hh : nat) (x : qd_row) : list binary32 :=
  f32_delta_prep_q eps lhd (f32_slice (hh * lhd) lhd (fst x)).
Definition qd_k (eps : binary32) (ldim lhd hh : nat) (x : qd_row) : list binary32 :=
  f32_l2norm eps (f32_slice (ldim + hh * lhd) lhd (fst x)).
Definition qd_v (ldim lhd hh : nat) (x : qd_row) : list binary32 :=
  f32_slice (2 * ldim + hh * lhd) lhd (fst x).
Definition qd_gnorm (eps : binary32) (lhd : nat) (nw : list binary32) (hh : nat)
    (zr o : list binary32) : list binary32 :=
  f32_rmsnorm_gated nw eps (f32_slice (hh * lhd) lhd zr) o.

Definition qd_head_step (lnh lhd : nat) (eps : binary32) (w : qwen_delta_weights) (hh : nat) :=
  zip_step (map_step qd_z)
    (delta_state_step (qd_beta hh) (qd_gamma w hh) (qd_q eps lhd hh)
                      (qd_k eps (lnh * lhd) lhd hh) (qd_v (lnh * lhd) lhd hh))
    (qd_gnorm eps lhd (qd_norm_w w) hh).

Definition qd_head_init (lhd : nat) : unit * list (list binary32) :=
  (tt, f32_delta_state0 lhd lhd).

Definition qd_head_f (lnh lhd : nat) (eps : binary32) (w : qwen_delta_weights) (hh : nat)
    (xs : list qd_row) : list (list binary32) :=
  zipw (qd_gnorm eps lhd (qd_norm_w w) hh) (List.map qd_z xs)
    (f32_delta_scan (List.map (qd_beta hh) xs) (List.map (qd_gamma w hh) xs)
       (List.map (qd_q eps lhd hh) xs) (List.map (qd_k eps (lnh * lhd) lhd hh) xs)
       (List.map (qd_v (lnh * lhd) lhd hh) xs) (f32_delta_state0 lhd lhd)).

Definition qd_heads_f (lnh lhd : nat) (eps : binary32) (w : qwen_delta_weights)
    (xs : list qd_row) : list (list binary32) :=
  f32_concat_heads (List.map (fun hh => qd_head_f lnh lhd eps w hh xs) (List.seq 0 lnh)).

Definition qd_mix_step (lnh lhd ck : nat) (eps : binary32) (w : qwen_delta_weights) :=
  comp_step (zip_step (qd_conv_step lnh lhd ck w) (map_step (qd_zab w)) pair)
    (comp_step (heads_step lnh (fun _ => qd_head_init lhd) (qd_head_step lnh lhd eps w))
               (map_step (fun r => f32_mat_vec_mul (qd_out w) r))).

Definition qd_mix_init (lnh lhd : nat) :=
  (((tt, @nil (list binary32)), tt),
   (List.map (fun _ : nat => qd_head_init lhd) (List.seq 0 lnh), tt)).

Definition qd_mix_f (lnh lhd ck : nat) (eps : binary32) (w : qwen_delta_weights)
    (hn : list (list binary32)) : list (list binary32) :=
  List.map (fun r => f32_mat_vec_mul (qd_out w) r)
    (qd_heads_f lnh lhd eps w (zipw pair (qd_C lnh lhd ck w hn) (List.map (qd_zab w) hn))).

Lemma realized_qd_conv : forall lnh lhd ck w, (0 < ck)%nat ->
  realized (tt, @nil (list binary32)) (qd_conv_step lnh lhd ck w) (qd_C lnh lhd ck w).
Proof.
  intros lnh lhd ck w Hk. unfold qd_conv_step, qd_C.
  apply (realized_comp tt (map_step (fun h => f32_mat_vec_mul (qd_in_qkv w) h))
           (List.map (fun h => f32_mat_vec_mul (qd_in_qkv w) h)) []
           (conv_state_step (3 * (lnh * lhd)) ck (qd_conv_w w) [])
           (f32_causal_conv1d (3 * (lnh * lhd)) ck (qd_conv_w w) []));
    [apply realized_map_step | apply realized_conv; exact Hk].
Qed.

Lemma realized_qd_head : forall lnh lhd eps w hh,
  realized (qd_head_init lhd) (qd_head_step lnh lhd eps w hh) (qd_head_f lnh lhd eps w hh).
Proof.
  intros lnh lhd eps w hh. unfold qd_head_init, qd_head_step, qd_head_f.
  apply (realized_zip tt (map_step qd_z) (List.map qd_z) (f32_delta_state0 lhd lhd) _ _
           (qd_gnorm eps lhd (qd_norm_w w) hh));
    [apply realized_map_step | apply realized_delta].
Qed.

Lemma realized_qd_mix_f : forall lnh lhd ck eps w, (0 < lnh)%nat -> (0 < ck)%nat ->
  realized (qd_mix_init lnh lhd) (qd_mix_step lnh lhd ck eps w) (qd_mix_f lnh lhd ck eps w).
Proof.
  intros lnh lhd ck eps w Hn Hk.
  pose (OUT := fun r => f32_mat_vec_mul (qd_out w) r).
  pose (XS := fun hn => zipw pair (qd_C lnh lhd ck w hn) (List.map (qd_zab w) hn)).
  assert (R : realized (qd_mix_init lnh lhd) (qd_mix_step lnh lhd ck eps w)
                (fun hn => (fun xs => List.map OUT (qd_heads_f lnh lhd eps w xs)) (XS hn))).
  { unfold qd_mix_init, qd_mix_step.
    apply (realized_comp ((tt, @nil (list binary32)), tt)
             (zip_step (qd_conv_step lnh lhd ck w) (map_step (qd_zab w)) pair) XS _ _
             (fun xs => List.map OUT (qd_heads_f lnh lhd eps w xs))).
    - apply (realized_zip (tt, @nil (list binary32)) (qd_conv_step lnh lhd ck w)
               (qd_C lnh lhd ck w) tt (map_step (qd_zab w)) (List.map (qd_zab w)) pair).
      + apply realized_qd_conv. exact Hk.
      + apply realized_map_step.
    - apply (realized_comp _ _ (qd_heads_f lnh lhd eps w) tt (map_step OUT) (List.map OUT));
        [|apply realized_map_step].
      unfold qd_heads_f.
      apply realized_heads; [exact Hn|]. intros hh _. apply realized_qd_head. }
  eapply realized_ext; [|exact R]. reflexivity.
Qed.

Lemma qd_head_eq : forall lnh lhd ck eps w hn hh,
  qd_head_f lnh lhd eps w hh (zipw pair (qd_C lnh lhd ck w hn) (List.map (qd_zab w) hn))
  = f32_qwen_delta_head_out (lnh * lhd) lhd eps (qd_norm_w w) (qd_a_log w) (qd_dt_bias w) hh
      (qd_C lnh lhd ck w hn) (List.map (fun h => f32_mat_vec_mul (qd_in_a w) h) hn)
      (List.map (fun h => f32_mat_vec_mul (qd_in_b w) h) hn)
      (List.map (fun h => f32_mat_vec_mul (qd_in_z w) h) hn).
Proof.
  intros lnh lhd ck eps w hn hh.
  assert (L : List.length (qd_C lnh lhd ck w hn) = List.length (List.map (qd_zab w) hn))
    by (unfold qd_C; rewrite f32_causal_conv1d_length, !length_map; reflexivity).
  unfold qd_head_f, f32_qwen_delta_head_out, f32_qwen_delta_head. cbv zeta.
  rewrite map_pair_zipw_gen.
  rewrite (map_zipw_pair_r qd_z (fun t => fst (fst t)))
    by (first [exact L | intros; reflexivity]).
  rewrite (map_zipw_pair_r (qd_beta hh) (fun t => f32_sigmoid (List.nth hh (snd t) f32_zero)))
    by (first [exact L | intros; reflexivity]).
  rewrite (map_zipw_pair_r (qd_gamma w hh)
             (fun t => f32_delta_decay (List.nth hh (qd_a_log w) f32_zero)
                         (List.nth hh (qd_dt_bias w) f32_zero)
                         (List.nth hh (snd (fst t)) f32_zero)))
    by (first [exact L | intros; reflexivity]).
  rewrite (map_zipw_pair_l (qd_q eps lhd hh)
             (fun c => f32_delta_prep_q eps lhd (f32_slice (hh * lhd) lhd c)))
    by (first [exact L | intros; reflexivity]).
  rewrite (map_zipw_pair_l (qd_k eps (lnh * lhd) lhd hh)
             (fun c => f32_l2norm eps (f32_slice (lnh * lhd + hh * lhd) lhd c)))
    by (first [exact L | intros; reflexivity]).
  rewrite (map_zipw_pair_l (qd_v (lnh * lhd) lhd hh)
             (fun c => f32_slice (2 * (lnh * lhd) + hh * lhd) lhd c))
    by (first [exact L | intros; reflexivity]).
  rewrite !map_map. reflexivity.
Qed.

Lemma qd_mix_eq : forall lnh lhd ck eps w hn, (0 < lnh)%nat ->
  f32_qwen_delta_mix lnh lhd ck eps w hn = qd_mix_f lnh lhd ck eps w hn.
Proof.
  intros lnh lhd ck eps w hn Hn.
  unfold qd_mix_f, qd_heads_f.
  set (XS := zipw pair (qd_C lnh lhd ck w hn) (List.map (qd_zab w) hn)).
  assert (LX : List.length XS = List.length hn).
  { unfold XS. rewrite zipw_length. unfold qd_C.
    rewrite f32_causal_conv1d_length, !length_map. apply Nat.min_id. }
  assert (LH : forall hh, List.length (qd_head_f lnh lhd eps w hh XS) = List.length hn).
  { intros hh. rewrite (realized_length _ _ _ (realized_qd_head lnh lhd eps w hh) XS).
    exact LX. }
  rewrite (concat_heads_seq lnh (fun hh => qd_head_f lnh lhd eps w hh XS) Hn).
  cbv beta. rewrite LH, map_map.
  unfold f32_qwen_delta_mix. cbv zeta.
  apply map_ext. intros t. f_equal. f_equal.
  rewrite map_map. apply map_ext. intros hh. f_equal.
  unfold XS. rewrite qd_head_eq. reflexivity.
Qed.

Lemma realized_qd_mix : forall lnh lhd ck eps w, (0 < lnh)%nat -> (0 < ck)%nat ->
  realized (qd_mix_init lnh lhd) (qd_mix_step lnh lhd ck eps w)
           (f32_qwen_delta_mix lnh lhd ck eps w).
Proof.
  intros lnh lhd ck eps w Hn Hk.
  apply (realized_ext _ _ (qd_mix_f lnh lhd ck eps w)).
  - intros hn. symmetry. apply qd_mix_eq. exact Hn.
  - apply realized_qd_mix_f; assumption.
Qed.

(** ** The layers of both kinds, and the model *)

Definition qa_mix_state : Type := (unit * (unit * list attn_state))%type.
Definition qd_mix_state : Type :=
  (((unit * list (list binary32)) * unit) * (list (unit * list (list binary32)) * unit))%type.
Definition qw_wrap_state (M : Type) : Type := ((unit * (unit * M)) * (unit * unit))%type.
Definition qn_state : Type := (qw_wrap_state qa_mix_state + qw_wrap_state qd_mix_state)%type.

Definition qn_layer (nh nkv hd rd lnh lhd ck : nat) (eps : binary32) (invf : list binary32)
    (L : qwen_layer_weights)
    : qn_state * (qn_state -> list binary32 -> qn_state * list binary32) :=
  match L with
  | QAttn ln1 ln2 mlp aw =>
      (inl (qw_wrap_init (qa_mix_init nh)),
       lift_l (qw_wrap_step eps ln1 ln2 mlp (qa_mix_step nh nkv hd rd eps aw invf)))
  | QDelta ln1 ln2 mlp dw =>
      (inr (qw_wrap_init (qd_mix_init lnh lhd)),
       lift_r (qw_wrap_step eps ln1 ln2 mlp (qd_mix_step lnh lhd ck eps dw)))
  end.

(** A layer with its rotation given by the formula. *)
Definition qn_layer_f (nh nkv hd rd lnh lhd ck : nat) (eps : binary32) (invf : list binary32)
    (L : qwen_layer_weights) : list (list binary32) -> list (list binary32) :=
  match L with
  | QAttn ln1 ln2 mlp aw => f32_qwen_wrap eps ln1 ln2 mlp (qa_mix_f nh nkv hd rd eps aw invf)
  | QDelta ln1 ln2 mlp dw => f32_qwen_wrap eps ln1 ln2 mlp (f32_qwen_delta_mix lnh lhd ck eps dw)
  end.

(** A layer as the forward pass writes it, with the rotary tables. *)
Definition qn_layer_tab (nh nkv hd rd lnh lhd ck : nat) (eps : binary32)
    (cosv sinv : list (list binary32)) (L : qwen_layer_weights)
    : list (list binary32) -> list (list binary32) :=
  match L with
  | QAttn ln1 ln2 mlp aw =>
      f32_qwen_wrap eps ln1 ln2 mlp (f32_qwen_attn_mix nh nkv hd rd eps aw cosv sinv)
  | QDelta ln1 ln2 mlp dw => f32_qwen_wrap eps ln1 ln2 mlp (f32_qwen_delta_mix lnh lhd ck eps dw)
  end.

Lemma realized_qn_layer : forall nh nkv hd rd lnh lhd ck eps invf L,
  (0 < nh)%nat -> (0 < lnh)%nat -> (0 < ck)%nat ->
  realized (fst (qn_layer nh nkv hd rd lnh lhd ck eps invf L))
           (snd (qn_layer nh nkv hd rd lnh lhd ck eps invf L))
           (qn_layer_f nh nkv hd rd lnh lhd ck eps invf L).
Proof.
  intros nh nkv hd rd lnh lhd ck eps invf L Hnh Hn Hk.
  destruct L as [ln1 ln2 mlp aw | ln1 ln2 mlp dw]; cbn [qn_layer qn_layer_f fst snd].
  - apply realized_lift_l. apply realized_qw_wrap. apply realized_qa_mix. exact Hnh.
  - apply realized_lift_r. apply realized_qw_wrap. apply realized_qd_mix; assumption.
Qed.

Lemma qn_layer_f_length : forall nh nkv hd rd lnh lhd ck eps invf L h,
  (0 < nh)%nat -> (0 < lnh)%nat -> (0 < ck)%nat ->
  List.length (qn_layer_f nh nkv hd rd lnh lhd ck eps invf L h) = List.length h.
Proof.
  intros nh nkv hd rd lnh lhd ck eps invf L h Hnh Hn Hk.
  exact (realized_length _ _ _ (realized_qn_layer nh nkv hd rd lnh lhd ck eps invf L Hnh Hn Hk) h).
Qed.

Lemma qwen_stack_fun : forall fs h, f32_qwen_stack fs h = stack_fun fs h.
Proof.
  intros fs. induction fs as [|f fs IH]; intros h; [reflexivity|].
  cbn [f32_qwen_stack stack_fun]. apply IH.
Qed.

Lemma qn_stack_eq : forall nh nkv hd rd lnh lhd ck eps invf cosv sinv layers h,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  (0 < lnh)%nat -> (0 < ck)%nat ->
  (forall t, (t < List.length h)%nat ->
     List.nth t cosv [] = rope_cos_at invf t /\ List.nth t sinv [] = rope_sin_at invf t) ->
  f32_qwen_stack (List.map (qn_layer_tab nh nkv hd rd lnh lhd ck eps cosv sinv) layers) h
  = stack_fun (List.map (qn_layer_f nh nkv hd rd lnh lhd ck eps invf) layers) h.
Proof.
  intros nh nkv hd rd lnh lhd ck eps invf cosv sinv layers.
  induction layers as [|L layers IH]; intros h Hnh Hnkv Hmod Hn Hk Ht; [reflexivity|].
  cbn [List.map f32_qwen_stack stack_fun].
  assert (E : qn_layer_tab nh nkv hd rd lnh lhd ck eps cosv sinv L h
              = qn_layer_f nh nkv hd rd lnh lhd ck eps invf L h).
  { destruct L as [ln1 ln2 mlp aw | ln1 ln2 mlp dw]; cbn [qn_layer_tab qn_layer_f];
      [|reflexivity].
    unfold f32_qwen_wrap. cbv zeta.
    rewrite (qwen_attn_eq nh nkv hd rd eps aw cosv sinv invf)
      by (try assumption; intros t Hl; rewrite length_map in Hl; exact (Ht t Hl)).
    reflexivity. }
  rewrite E. apply IH; try assumption.
  intros t Hl. rewrite qn_layer_f_length in Hl by assumption. exact (Ht t Hl).
Qed.

Definition qn_final (eps : binary32) (m : qwen_model_weights) (h : list binary32)
    : list binary32 :=
  let hn := f32_rmsnorm_zc (qw_norm m) eps h in
  List.map (fun wrow => f32_dot hn wrow) (qw_emb m).

Definition qn_layers (nh nkv hd rd lnh lhd ck : nat) (eps : binary32) (m : qwen_model_weights) :=
  List.map (qn_layer nh nkv hd rd lnh lhd ck eps (qw_invf m)) (qw_layers m).

(** The Qwen3.5 decode step, and the state it starts from. *)
Definition qwen_init (nh nkv hd rd lnh lhd ck : nat) (eps : binary32) (m : qwen_model_weights) :=
  (tt, (List.map fst (qn_layers nh nkv hd rd lnh lhd ck eps m), tt)).

Definition qwen_step (nh nkv hd rd lnh lhd ck : nat) (eps : binary32) (m : qwen_model_weights) :=
  comp_step (map_step (f32_lookup_embedding (qw_emb m)))
    (comp_step (stack_step (List.map snd (qn_layers nh nkv hd rd lnh lhd ck eps m)))
               (map_step (qn_final eps m))).

(** Decoding Qwen3.5 one token at a time, with a key/value cache per head of
    each attention layer, and the convolution window and a recurrent state per
    head of each DeltaNet layer, emits exactly the logit rows
    [f32_qwen_logits_of] computes. *)
Theorem qwen_decode_correct : forall nh nkv hd rd lnh lhd ck eps m ids,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  (0 < lnh)%nat -> (0 < ck)%nat ->
  run (qwen_step nh nkv hd rd lnh lhd ck eps m) (qwen_init nh nkv hd rd lnh lhd ck eps m) ids
  = f32_qwen_logits_of nh nkv hd rd lnh lhd ck eps m ids.
Proof.
  intros nh nkv hd rd lnh lhd ck eps m ids Hnh Hnkv Hmod Hn Hk.
  pose (STK := stack_fun (List.map (qn_layer_f nh nkv hd rd lnh lhd ck eps (qw_invf m))
                                   (qw_layers m))).
  pose (FIN := List.map (qn_final eps m)).
  assert (R : realized (qwen_init nh nkv hd rd lnh lhd ck eps m)
                (qwen_step nh nkv hd rd lnh lhd ck eps m)
                (fun xs => (fun hs => FIN (STK hs))
                             (List.map (f32_lookup_embedding (qw_emb m)) xs))).
  { apply (realized_comp tt (map_step (f32_lookup_embedding (qw_emb m)))
             (List.map (f32_lookup_embedding (qw_emb m))) _ _ (fun hs => FIN (STK hs)));
      [apply realized_map_step|].
    apply (realized_comp _ _ STK tt (map_step (qn_final eps m)) FIN);
      [|apply realized_map_step].
    unfold STK, qn_layers. apply realized_stack.
    induction (qw_layers m) as [|L Ls IH]; constructor; [|exact IH].
    apply realized_qn_layer; assumption. }
  rewrite (run_realized _ _ _ R ids).
  unfold FIN, STK, f32_qwen_logits_of, f32_qwen_logits, f32_qwen_forward, f32_qwen_final,
    f32_embed_tokens.
  cbv zeta.
  rewrite <- (qn_stack_eq nh nkv hd rd lnh lhd ck eps (qw_invf m)
               (f32_rope_cos (qw_invf m) (List.length ids))
               (f32_rope_sin (qw_invf m) (List.length ids))
               (qw_layers m) (List.map (f32_lookup_embedding (qw_emb m)) ids)).
  - rewrite map_map. reflexivity.
  - exact Hnh.
  - exact Hnkv.
  - exact Hmod.
  - exact Hn.
  - exact Hk.
  - intros t Hl. rewrite length_map in Hl. apply rope_tables_at. exact Hl.
Qed.
