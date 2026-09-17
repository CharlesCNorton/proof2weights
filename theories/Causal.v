(** * Causality of the forward passes

    A sequence function is causal when it preserves length and the rows it
    computes for a sequence are the first rows it computes for every extension
    of that sequence. Row-wise maps, positional maps, causal attention, the
    depthwise convolution and the gated delta scan are causal, causality
    survives composition, zipping and stacking, and so the forward passes of
    all three architectures are causal: running a model on a longer sequence
    reproduces, bit for bit, every row it produced for the shorter one.

    Together with [causal_attention_snoc], [delta_scan_snoc] and
    [conv_window_cached_correct], this is what makes a decode step that reuses
    cached keys, values, recurrent state and convolution window compute exactly
    the row a full recomputation computes. *)

From Stdlib Require Import List.
From Stdlib Require Import Lia.
From Stdlib Require Import PeanoNat.
From Stdlib Require Import ZArith.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Llama.
Require Import Qwen.
Require Import Cache.
Require Import Cache_attn.

Import ListNotations.

Definition causal {A B : Type} (F : list A -> list B) : Prop :=
  (forall xs, List.length (F xs) = List.length xs)
  /\ (forall xs ys, List.firstn (List.length xs) (F (xs ++ ys)) = F xs).

(** * Consequences of the definition *)

Lemma causal_split : forall {A B : Type} (F : list A -> list B) xs ys,
  causal F -> F (xs ++ ys) = F xs ++ List.skipn (List.length xs) (F (xs ++ ys)).
Proof.
  intros A B F xs ys [Hl Hp].
  rewrite <- (Hp xs ys). symmetry. apply List.firstn_skipn.
Qed.

Lemma causal_nth : forall {A B : Type} (F : list A -> list B) xs ys i d,
  causal F -> (i < List.length xs)%nat ->
  List.nth i (F (xs ++ ys)) d = List.nth i (F xs) d.
Proof.
  intros A B F xs ys i d HF Hi.
  rewrite (causal_split F xs ys HF).
  apply List.app_nth1. destruct HF as [Hl _]. rewrite Hl. exact Hi.
Qed.

Lemma causal_ext : forall {A B : Type} (F G : list A -> list B),
  (forall xs, F xs = G xs) -> causal F -> causal G.
Proof.
  intros A B F G Heq [Hl Hp]. split.
  - intros xs. rewrite <- Heq. apply Hl.
  - intros xs ys. rewrite <- !Heq. apply Hp.
Qed.

(** * Combinators *)

Lemma causal_id : forall {A : Type}, causal (fun xs : list A => xs).
Proof.
  intros A. split; [reflexivity|].
  intros xs ys. rewrite List.firstn_app, Nat.sub_diag, List.firstn_0, List.app_nil_r.
  apply List.firstn_all.
Qed.

Lemma causal_map : forall {A B : Type} (g : A -> B), causal (List.map g).
Proof.
  intros A B g. split.
  - intros xs. apply List.length_map.
  - intros xs ys. rewrite List.map_app, List.firstn_app, List.length_map, Nat.sub_diag,
      List.firstn_0, List.app_nil_r.
    rewrite <- (List.length_map g xs). apply List.firstn_all.
Qed.

Lemma causal_comp : forall {A B C : Type} (F : list A -> list B) (G : list B -> list C),
  causal F -> causal G -> causal (fun xs => G (F xs)).
Proof.
  intros A B C F G HF HG.
  pose proof HF as [Hfl Hfp]. pose proof HG as [Hgl Hgp]. split.
  - intros xs. rewrite Hgl, Hfl. reflexivity.
  - intros xs ys. rewrite (causal_split F xs ys HF).
    rewrite <- (Hfl xs) at 1. apply Hgp.
Qed.

Lemma causal_zip : forall {A B C D : Type} (F : list A -> list B) (G : list A -> list C)
                          (h : B * C -> D),
  causal F -> causal G -> causal (fun xs => List.map h (List.combine (F xs) (G xs))).
Proof.
  intros A B C D F G h HF HG.
  pose proof HF as [Hfl Hfp]. pose proof HG as [Hgl Hgp]. split.
  - intros xs. rewrite List.length_map, List.length_combine, Hfl, Hgl. apply Nat.min_id.
  - intros xs ys. rewrite (causal_split F xs ys HF), (causal_split G xs ys HG).
    rewrite combine_app_eq by (rewrite Hfl, Hgl; reflexivity).
    rewrite List.map_app, List.firstn_app, List.length_map, List.length_combine,
      Hfl, Hgl, Nat.min_id, Nat.sub_diag, List.firstn_0, List.app_nil_r.
    apply List.firstn_all2. rewrite List.length_map, List.length_combine, Hfl, Hgl. lia.
Qed.

(** A map that also sees each row's position. *)
Lemma causal_positional : forall {A B : Type} (g : nat * A -> B),
  causal (fun xs => List.map g (List.combine (List.seq 0 (List.length xs)) xs)).
Proof.
  intros A B g. split.
  - intros xs. rewrite List.length_map, List.length_combine, List.length_seq. apply Nat.min_id.
  - intros xs ys.
    rewrite List.length_app, List.seq_app.
    rewrite combine_app_eq by (rewrite List.length_seq; reflexivity).
    rewrite List.map_app, List.firstn_app, List.length_map, List.length_combine,
      List.length_seq, Nat.min_id, Nat.sub_diag, List.firstn_0, List.app_nil_r.
    apply List.firstn_all2.
    rewrite List.length_map, List.length_combine, List.length_seq. lia.
Qed.

(** A map over positions whose value at position [t] depends on the sequence
    only through its first [t + 1] rows. *)
Lemma causal_seqmap : forall {A B : Type} (g : nat -> list A -> B),
  (forall t xs ys, (t < List.length xs)%nat -> g t (xs ++ ys) = g t xs) ->
  causal (fun xs => List.map (fun t => g t xs) (List.seq 0 (List.length xs))).
Proof.
  intros A B g Hg. split.
  - intros xs. rewrite List.length_map, List.length_seq. reflexivity.
  - intros xs ys.
    rewrite List.length_app, List.seq_app, List.map_app, List.firstn_app,
      List.length_map, List.length_seq, Nat.sub_diag, List.firstn_0, List.app_nil_r.
    rewrite List.firstn_all2 by (rewrite List.length_map, List.length_seq; lia).
    apply List.map_ext_in. intros t Ht.
    apply List.in_seq in Ht. apply Hg. lia.
Qed.

Lemma causal_positions : forall {A B : Type} (g : nat -> B),
  causal (fun xs : list A => List.map g (List.seq 0 (List.length xs))).
Proof.
  intros A B g. apply (causal_seqmap (fun t _ => g t)). reflexivity.
Qed.

(** The heads of a multi-head stage, regrouped by position. *)
Lemma concat_heads_seq : forall nh (H : nat -> list (list binary32)),
  (0 < nh)%nat ->
  f32_concat_heads (List.map H (List.seq 0 nh))
  = List.map (fun si => List.concat (List.map (fun hh => List.nth si (H hh) []) (List.seq 0 nh)))
             (List.seq 0 (List.length (H 0%nat))).
Proof.
  intros nh H Hn. destruct nh as [|n]; [lia|].
  cbn [List.seq List.map f32_concat_heads].
  apply List.map_ext. intros si. f_equal. cbn [List.map]. f_equal.
  rewrite List.map_map. reflexivity.
Qed.

Lemma causal_concat_heads : forall {A : Type} nh
    (H : nat -> list A -> list (list binary32)),
  (0 < nh)%nat ->
  (forall hh, (hh < nh)%nat -> causal (H hh)) ->
  causal (fun xs => f32_concat_heads (List.map (fun hh => H hh xs) (List.seq 0 nh))).
Proof.
  intros A nh H Hn HH.
  assert (H0 : causal (H 0%nat)) by (apply HH; exact Hn).
  apply (causal_ext (fun xs =>
           List.map (fun si => List.concat
                       (List.map (fun hh => List.nth si (H hh xs) []) (List.seq 0 nh)))
                    (List.seq 0 (List.length xs)))).
  - intros xs. rewrite (concat_heads_seq nh (fun hh => H hh xs) Hn).
    destruct H0 as [Hl _]. rewrite Hl. reflexivity.
  - apply (causal_seqmap (fun si xs =>
             List.concat (List.map (fun hh => List.nth si (H hh xs) []) (List.seq 0 nh)))).
    intros t xs ys Ht. f_equal. apply List.map_ext_in. intros hh Hh.
    apply List.in_seq in Hh. apply causal_nth; [apply HH; lia | exact Ht].
Qed.

Lemma combine_map_same : forall {A B C : Type} (f : A -> B) (g : A -> C) (l : list A),
  List.combine (List.map f l) (List.map g l) = List.map (fun x => (f x, g x)) l.
Proof.
  intros A B C f g l. induction l as [|x l IH]; [reflexivity|].
  cbn [List.map List.combine]. f_equal. exact IH.
Qed.

(** * Causal attention *)

Lemma causal_attention_causal : forall {A : Type}
    (Q K V : list A -> list (list binary32)) d_k,
  causal Q -> causal K -> causal V ->
  causal (fun xs => f32_causal_attention (Q xs) (K xs) (V xs) d_k).
Proof.
  intros A Q K V d_k HQ HK HV.
  pose proof HQ as [Hql _]. pose proof HK as [Hkl _]. pose proof HV as [Hvl _]. split.
  - intros xs. pose proof (f32_causal_attention_rows (Q xs) (K xs) (V xs) d_k) as H.
    unfold f32_mat_rows in H. rewrite H. apply Hql.
  - intros xs ys.
    rewrite (causal_split Q xs ys HQ), (causal_split K xs ys HK), (causal_split V xs ys HV).
    rewrite <- (Hql xs) at 1.
    apply causal_attention_prefix; [rewrite Hkl, Hql | rewrite Hvl, Hql]; reflexivity.
Qed.

(** * Stacks *)

Lemma causal_llama_stack : forall fs,
  Forall causal fs -> causal (f32_llama_stack fs).
Proof.
  induction fs as [|f fs IH]; intros Hfs.
  - apply (causal_ext (fun xs => xs)); [reflexivity | apply causal_id].
  - inversion Hfs as [|? ? Hf Hrest]; subst.
    apply (causal_ext (fun xs => f32_llama_stack fs (f xs))); [reflexivity|].
    apply causal_comp; [exact Hf | apply IH, Hrest].
Qed.

Lemma causal_qwen_stack : forall fs,
  Forall causal fs -> causal (f32_qwen_stack fs).
Proof.
  induction fs as [|f fs IH]; intros Hfs.
  - apply (causal_ext (fun xs => xs)); [reflexivity | apply causal_id].
  - inversion Hfs as [|? ? Hf Hrest]; subst.
    apply (causal_ext (fun xs => f32_qwen_stack fs (f xs))); [reflexivity|].
    apply causal_comp; [exact Hf | apply IH, Hrest].
Qed.

(** * Shared stages *)

Lemma nth_map_seq_lt : forall {B : Type} (f : nat -> B) n i d,
  (i < n)%nat -> List.nth i (List.map f (List.seq 0 n)) d = f i.
Proof.
  intros B f n i d Hi.
  rewrite (List.nth_indep _ d (f 0%nat)) by (rewrite List.length_map, List.length_seq; exact Hi).
  rewrite List.map_nth, List.seq_nth by exact Hi. reflexivity.
Qed.

(** Grouped-query attention sends head [hh] to key/value head [hh / group],
    which exists when the key/value heads divide the query heads. *)
Lemma group_index_lt : forall nh nkv hh,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat -> (hh < nh)%nat ->
  (Nat.div hh (Nat.div nh nkv) < nkv)%nat.
Proof.
  intros nh nkv hh Hnh Hnkv Hmod Hh.
  pose proof (Nat.div_mod nh nkv ltac:(lia)) as Hdm.
  rewrite Hmod, Nat.add_0_r in Hdm.
  apply Nat.Div0.div_lt_upper_bound.
  rewrite Nat.mul_comm, <- Hdm. exact Hh.
Qed.

(** * The Llama forward pass *)

Lemma causal_llama_rope_rows : forall {A : Type} hd cosv sinv c
    (P : list A -> list (list binary32)),
  causal P -> causal (fun xs => f32_llama_rope_rows hd cosv sinv c (P xs)).
Proof.
  intros A hd cosv sinv c P HP.
  apply (causal_comp P (f32_llama_rope_rows hd cosv sinv c)); [exact HP|].
  unfold f32_llama_rope_rows. apply causal_positional.
Qed.

Lemma causal_llama_attn : forall nh nkv hd w cosv sinv,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  causal (f32_llama_attn nh nkv hd w cosv sinv).
Proof.
  intros nh nkv hd w cosv sinv Hnh Hnkv Hmod.
  set (PQ := List.map (fun h => f32_mat_vec_mul (la_q w) h)).
  set (PK := List.map (fun h => f32_mat_vec_mul (la_k w) h)).
  set (PV := List.map (fun h => f32_mat_vec_mul (la_v w) h)).
  set (Hd := fun hh (hn : list (list binary32)) =>
    f32_causal_attention
      (f32_llama_rope_rows hd cosv sinv hh (PQ hn))
      (f32_llama_rope_rows hd cosv sinv (Nat.div hh (Nat.div nh nkv)) (PK hn))
      (List.map (fun r => f32_slice (Nat.div hh (Nat.div nh nkv) * hd) hd r) (PV hn))
      hd).
  apply (causal_ext (fun hn => List.map (fun r => f32_mat_vec_mul (la_o w) r)
                                 (f32_concat_heads (List.map (fun hh => Hd hh hn)
                                                             (List.seq 0 nh))))).
  - intros hn. unfold f32_llama_attn, f32_llama_heads. cbv zeta.
    f_equal. f_equal.
    apply List.map_ext_in. intros hh Hin. apply List.in_seq in Hin.
    pose proof (group_index_lt nh nkv hh Hnh Hnkv Hmod ltac:(lia)) as Hg.
    unfold Hd, PQ, PK, PV.
    rewrite !nth_map_seq_lt by lia. reflexivity.
  - apply (causal_comp (fun hn => f32_concat_heads (List.map (fun hh => Hd hh hn)
                                                             (List.seq 0 nh)))
                       (List.map (fun r => f32_mat_vec_mul (la_o w) r)));
      [| apply causal_map].
    apply causal_concat_heads; [exact Hnh|]. intros hh Hh. unfold Hd.
    apply causal_attention_causal.
    + apply causal_llama_rope_rows, causal_map.
    + apply causal_llama_rope_rows, causal_map.
    + apply (causal_comp PV); apply causal_map.
Qed.

Lemma causal_llama_wrap : forall eps ln1 ln2 mw mix,
  causal mix -> causal (f32_llama_wrap eps ln1 ln2 mw mix).
Proof.
  intros eps ln1 ln2 mw mix Hmix.
  set (N1 := List.map (fun row => f32_rmsnorm ln1 eps row)).
  set (N2 := List.map (fun row => f32_rmsnorm ln2 eps row)).
  set (FF := List.map (fun row => f32_swiglu (lm_gate mw) (lm_up mw) (lm_down mw) row)).
  set (add := fun p : list binary32 * list binary32 => let '(a, b) := p in f32_vec_add a b).
  set (Hid2 := fun h => List.map add (List.combine h (mix (N1 h)))).
  assert (H2 : causal Hid2).
  { unfold Hid2. apply (causal_zip (fun h => h) (fun h => mix (N1 h)) add).
    - apply causal_id.
    - apply causal_comp; [apply causal_map | exact Hmix]. }
  apply (causal_ext (fun h => List.map add (List.combine (Hid2 h) (FF (N2 (Hid2 h)))))).
  - intros h. unfold f32_llama_wrap. cbv zeta. reflexivity.
  - apply (causal_zip Hid2 (fun h => FF (N2 (Hid2 h))) add); [exact H2|].
    apply (causal_comp Hid2 (fun h2 => FF (N2 h2))); [exact H2|].
    apply (causal_comp N2 FF); apply causal_map.
Qed.

Lemma causal_llama_layer : forall nh nkv hd eps ln1 ln2 aw mw cosv sinv,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  causal (f32_llama_layer nh nkv hd eps ln1 ln2 aw mw cosv sinv).
Proof.
  intros. unfold f32_llama_layer.
  apply causal_llama_wrap, causal_llama_attn; assumption.
Qed.

Theorem causal_llama_forward : forall eps normw fs emb,
  Forall causal fs -> causal (f32_llama_forward eps normw fs emb).
Proof.
  intros eps normw fs emb Hfs.
  apply (causal_ext (fun ids => List.map (fun row => f32_rmsnorm normw eps row)
                                  (f32_llama_stack fs (List.map (f32_lookup_embedding emb) ids)))).
  - intros ids. reflexivity.
  - apply (causal_comp (fun ids => f32_llama_stack fs (List.map (f32_lookup_embedding emb) ids))
                       (List.map (fun row => f32_rmsnorm normw eps row)));
      [| apply causal_map].
    apply (causal_comp (List.map (f32_lookup_embedding emb)) (f32_llama_stack fs));
      [apply causal_map | apply causal_llama_stack, Hfs].
Qed.

(** Running the Llama forward pass on an extended token sequence reproduces
    every row it computes for the original sequence. *)
Corollary llama_forward_prefix : forall eps normw fs emb ids1 ids2,
  Forall causal fs ->
  List.firstn (List.length ids1) (f32_llama_forward eps normw fs emb (ids1 ++ ids2))
  = f32_llama_forward eps normw fs emb ids1.
Proof.
  intros eps normw fs emb ids1 ids2 Hfs.
  apply (proj2 (causal_llama_forward eps normw fs emb Hfs)).
Qed.

(** * The Qwen3.5 forward pass *)

Lemma causal_conv1d : forall chans k w b, causal (f32_causal_conv1d chans k w b).
Proof.
  intros chans k w b. unfold f32_causal_conv1d.
  apply (causal_seqmap (fun t xs => f32_conv_step w b (f32_conv_window chans k xs t))).
  intros t xs ys Ht. f_equal. unfold f32_conv_window.
  rewrite List.firstn_app. replace (S t - List.length xs)%nat with 0%nat by lia.
  rewrite List.firstn_0, List.app_nil_r. reflexivity.
Qed.

Lemma causal_delta_scan : forall {A : Type} (Bt Gs : list A -> list binary32)
    (Qs Ks Vs : list A -> list (list binary32)) st,
  causal Bt -> causal Gs -> causal Qs -> causal Ks -> causal Vs ->
  causal (fun xs => f32_delta_scan (Bt xs) (Gs xs) (Qs xs) (Ks xs) (Vs xs) st).
Proof.
  intros A Bt Gs Qs Ks Vs st HB HG HQ HK HV.
  pose proof HB as [Hbl _]. pose proof HG as [Hgl _]. pose proof HQ as [Hql _].
  pose proof HK as [Hkl _]. pose proof HV as [Hvl _].
  split.
  - intros xs. rewrite f32_delta_scan_length, Hbl, Hgl, Hql, Hkl, Hvl, !Nat.min_id.
    reflexivity.
  - intros xs ys.
    rewrite (causal_split Bt xs ys HB), (causal_split Gs xs ys HG),
      (causal_split Qs xs ys HQ), (causal_split Ks xs ys HK), (causal_split Vs xs ys HV).
    rewrite delta_scan_app by (rewrite ?Hbl, ?Hgl, ?Hql, ?Hkl, ?Hvl; reflexivity).
    rewrite List.firstn_app, f32_delta_scan_length, Hbl, Hgl, Hql, Hkl, Hvl, !Nat.min_id,
      Nat.sub_diag, List.firstn_0, List.app_nil_r.
    apply List.firstn_all2.
    rewrite f32_delta_scan_length, Hbl, Hgl, Hql, Hkl, Hvl, !Nat.min_id. lia.
Qed.

Lemma causal_qwen_delta_head_out : forall {A : Type} ldim lhd eps nw alog dtb hh
    (C AV BV ZS : list A -> list (list binary32)),
  causal C -> causal AV -> causal BV -> causal ZS ->
  causal (fun xs => f32_qwen_delta_head_out ldim lhd eps nw alog dtb hh
                      (C xs) (AV xs) (BV xs) (ZS xs)).
Proof.
  intros A ldim lhd eps nw alog dtb hh C AV BV ZS HC HA HB HZ.
  unfold f32_qwen_delta_head_out, f32_qwen_delta_head.
  apply (causal_zip ZS (fun xs =>
    f32_delta_scan
      (List.map (fun r => f32_sigmoid (List.nth hh r f32_zero)) (BV xs))
      (List.map (fun r => f32_delta_decay (List.nth hh alog f32_zero)
                            (List.nth hh dtb f32_zero) (List.nth hh r f32_zero)) (AV xs))
      (List.map (fun r => f32_delta_prep_q eps lhd (f32_slice (hh * lhd) lhd r)) (C xs))
      (List.map (fun r => f32_l2norm eps (f32_slice (ldim + hh * lhd) lhd r)) (C xs))
      (List.map (fun r => f32_slice (2 * ldim + hh * lhd) lhd r) (C xs))
      (f32_delta_state0 lhd lhd))); [exact HZ|].
  apply causal_delta_scan;
    (apply causal_comp; [assumption | apply causal_map]).
Qed.

Lemma causal_qwen_delta_mix : forall lnh lhd ck eps w,
  causal (f32_qwen_delta_mix lnh lhd ck eps w).
Proof.
  intros lnh lhd ck eps w.
  set (ldim := (lnh * lhd)%nat).
  set (C := fun hn => f32_causal_conv1d (3 * ldim) ck (qd_conv_w w) []
                        (List.map (fun h => f32_mat_vec_mul (qd_in_qkv w) h) hn)).
  set (ZS := List.map (fun h => f32_mat_vec_mul (qd_in_z w) h)).
  set (AV := List.map (fun h => f32_mat_vec_mul (qd_in_a w) h)).
  set (BV := List.map (fun h => f32_mat_vec_mul (qd_in_b w) h)).
  set (HO := fun hh hn => f32_qwen_delta_head_out ldim lhd eps (qd_norm_w w) (qd_a_log w)
                            (qd_dt_bias w) hh (C hn) (AV hn) (BV hn) (ZS hn)).
  assert (HHO : forall hh, causal (HO hh)).
  { intros hh. unfold HO. apply causal_qwen_delta_head_out.
    - unfold C. apply (causal_comp (List.map (fun h => f32_mat_vec_mul (qd_in_qkv w) h))
                                   (f32_causal_conv1d (3 * ldim) ck (qd_conv_w w) []));
        [apply causal_map | apply causal_conv1d].
    - apply causal_map.
    - apply causal_map.
    - apply causal_map. }
  apply (causal_ext (fun hn => List.map (fun t => f32_mat_vec_mul (qd_out w)
                                  (List.concat (List.map (fun hh => List.nth t (HO hh hn) [])
                                                         (List.seq 0 lnh))))
                                (List.seq 0 (List.length hn)))).
  - intros hn. unfold f32_qwen_delta_mix. cbv zeta.
    apply List.map_ext. intros t. f_equal. f_equal.
    rewrite List.map_map. reflexivity.
  - apply (causal_seqmap (fun t hn => f32_mat_vec_mul (qd_out w)
             (List.concat (List.map (fun hh => List.nth t (HO hh hn) []) (List.seq 0 lnh))))).
    intros t xs ys Ht. f_equal. f_equal.
    apply List.map_ext. intros hh. apply causal_nth; [apply HHO | exact Ht].
Qed.

Lemma causal_qwen_attn_mix : forall nh nkv hd rd eps w cosv sinv,
  (0 < nh)%nat -> (0 < nkv)%nat -> Nat.modulo nh nkv = 0%nat ->
  causal (f32_qwen_attn_mix nh nkv hd rd eps w cosv sinv).
Proof.
  intros nh nkv hd rd eps w cosv sinv Hnh Hnkv Hmod.
  set (PQ := List.map (fun h => f32_mat_vec_mul (qa_q w) h)).
  set (PK := List.map (fun h => f32_mat_vec_mul (qa_k w) h)).
  set (PV := List.map (fun h => f32_mat_vec_mul (qa_v w) h)).
  set (rot := fun (nw : list binary32) (s : nat) (m : list (list binary32)) =>
    List.map (fun p => let '(pos, r) := p in
                f32_partial_rope rd (List.nth pos cosv []) (List.nth pos sinv [])
                  (f32_rmsnorm_zc nw eps (f32_slice s hd r)))
             (List.combine (List.seq 0 (List.length m)) m)).
  set (Hd := fun (hh : nat) (hn : list (list binary32)) =>
    f32_causal_attention (rot (qa_q_norm w) (hh * hd * 2)%nat (PQ hn))
      (rot (qa_k_norm w) (Nat.div hh (Nat.div nh nkv) * hd)%nat (PK hn))
      (List.map (fun r => f32_slice (Nat.div hh (Nat.div nh nkv) * hd) hd r) (PV hn)) hd).
  set (gatef := fun r : list binary32 =>
    List.concat (List.map (fun hh => f32_slice (hh * hd * 2 + hd) hd r) (List.seq 0 nh))).
  set (outf := fun p : list binary32 * list binary32 =>
    let '(g, o) := p in f32_mat_vec_mul (qa_o w) (f32_gate_sigmoid g o)).
  assert (Hrot : forall nw s, causal (rot nw s)).
  { intros nw s. unfold rot. apply causal_positional. }
  apply (causal_ext (fun hn => List.map outf
           (List.combine (List.map gatef (PQ hn))
                         (f32_concat_heads (List.map (fun hh => Hd hh hn) (List.seq 0 nh)))))).
  - intros hn. unfold f32_qwen_attn_mix. cbv zeta.
    f_equal. f_equal. f_equal.
    apply List.map_ext_in. intros hh Hin. apply List.in_seq in Hin.
    pose proof (group_index_lt nh nkv hh Hnh Hnkv Hmod ltac:(lia)) as Hg.
    unfold Hd, rot, PQ, PK, PV.
    rewrite !nth_map_seq_lt by lia. reflexivity.
  - apply (causal_zip (fun hn => List.map gatef (PQ hn))
                      (fun hn => f32_concat_heads (List.map (fun hh => Hd hh hn)
                                                            (List.seq 0 nh))) outf).
    + apply (causal_comp PQ (List.map gatef)); apply causal_map.
    + apply causal_concat_heads; [exact Hnh|]. intros hh Hh. unfold Hd.
      apply causal_attention_causal.
      * apply (causal_comp PQ (rot (qa_q_norm w) (hh * hd * 2)%nat));
          [apply causal_map | apply Hrot].
      * apply (causal_comp PK (rot (qa_k_norm w) (Nat.div hh (Nat.div nh nkv) * hd)%nat));
          [apply causal_map | apply Hrot].
      * apply (causal_comp PV); apply causal_map.
Qed.

Lemma causal_qwen_wrap : forall eps ln1 ln2 mlp mix,
  causal mix -> causal (f32_qwen_wrap eps ln1 ln2 mlp mix).
Proof.
  intros eps ln1 ln2 mlp mix Hmix.
  set (N1 := List.map (fun row => f32_rmsnorm_zc ln1 eps row)).
  set (N2 := List.map (fun row => f32_rmsnorm_zc ln2 eps row)).
  set (FF := List.map (fun row => f32_swiglu (qm_gate mlp) (qm_up mlp) (qm_down mlp) row)).
  set (add := fun p : list binary32 * list binary32 => let '(a, b) := p in f32_vec_add a b).
  set (Hid2 := fun h => List.map add (List.combine h (mix (N1 h)))).
  assert (H2 : causal Hid2).
  { unfold Hid2. apply (causal_zip (fun h => h) (fun h => mix (N1 h)) add).
    - apply causal_id.
    - apply causal_comp; [apply causal_map | exact Hmix]. }
  apply (causal_ext (fun h => List.map add (List.combine (Hid2 h) (FF (N2 (Hid2 h)))))).
  - intros h. unfold f32_qwen_wrap. cbv zeta. reflexivity.
  - apply (causal_zip Hid2 (fun h => FF (N2 (Hid2 h))) add); [exact H2|].
    apply (causal_comp Hid2 (fun h2 => FF (N2 h2))); [exact H2|].
    apply (causal_comp N2 FF); apply causal_map.
Qed.

Theorem causal_qwen_forward : forall eps normw fs emb,
  Forall causal fs -> causal (f32_qwen_forward eps normw fs emb).
Proof.
  intros eps normw fs emb Hfs.
  apply (causal_ext (fun ids => List.map (fun row => f32_rmsnorm_zc normw eps row)
                                  (f32_qwen_stack fs (List.map (f32_lookup_embedding emb) ids)))).
  - intros ids. reflexivity.
  - apply (causal_comp (fun ids => f32_qwen_stack fs (List.map (f32_lookup_embedding emb) ids))
                       (List.map (fun row => f32_rmsnorm_zc normw eps row)));
      [| apply causal_map].
    apply (causal_comp (List.map (f32_lookup_embedding emb)) (f32_qwen_stack fs));
      [apply causal_map | apply causal_qwen_stack, Hfs].
Qed.

Corollary qwen_forward_prefix : forall eps normw fs emb ids1 ids2,
  Forall causal fs ->
  List.firstn (List.length ids1) (f32_qwen_forward eps normw fs emb (ids1 ++ ids2))
  = f32_qwen_forward eps normw fs emb ids1.
Proof.
  intros eps normw fs emb ids1 ids2 Hfs.
  apply (proj2 (causal_qwen_forward eps normw fs emb Hfs)).
Qed.

(** * The GPT-2 forward pass *)

Lemma causal_gpt2_attention : forall n_embd n_head caw cab cpw cpb,
  (0 < n_head)%nat ->
  causal (f32_attention_forward n_embd n_head caw cab cpw cpb).
Proof.
  intros n_embd n_head caw cab cpw cpb Hnh.
  set (QKV := f32_linear_forward_2d caw cab).
  set (split := fun row : list binary32 =>
    (List.firstn n_embd row, List.firstn n_embd (List.skipn n_embd row),
     List.skipn (2 * n_embd) row)).
  set (QS := fun hidden => List.map (fun '(q, _, _) => q) (List.map split (QKV hidden))).
  set (KS := fun hidden => List.map (fun '(_, k, _) => k) (List.map split (QKV hidden))).
  set (VS := fun hidden => List.map (fun '(_, _, v) => v) (List.map split (QKV hidden))).
  set (heads := fun (m : list (list binary32)) (h : nat) =>
    List.map (fun row_heads => List.nth h row_heads [])
             (List.map (f32_split_row_into_heads n_head) m)).
  set (Hd := fun (h : nat) hidden =>
    f32_causal_attention (heads (QS hidden) h) (heads (KS hidden) h) (heads (VS hidden) h)
                         (Nat.div n_embd n_head)).
  assert (HQKV : causal QKV) by (unfold QKV, f32_linear_forward_2d; apply causal_map).
  assert (Hheads : forall (S : list (list binary32) -> list (list binary32)) h,
            causal S -> causal (fun hidden => heads (S hidden) h)).
  { intros S h HS. unfold heads.
    apply (causal_comp S (fun m => List.map (fun row_heads => List.nth h row_heads [])
                                             (List.map (f32_split_row_into_heads n_head) m)));
      [exact HS|].
    apply (causal_comp (List.map (f32_split_row_into_heads n_head))
                       (List.map (fun row_heads => List.nth h row_heads [])));
      apply causal_map. }
  apply (causal_ext (fun hidden => f32_linear_forward_2d cpw cpb
           (f32_concat_heads (List.map (fun h => Hd h hidden) (List.seq 0 n_head))))).
  - intros hidden. unfold f32_attention_forward, f32_split_into_heads. cbv zeta.
    f_equal. f_equal.
    rewrite !combine_map_same, List.map_map. reflexivity.
  - apply (causal_comp (fun hidden => f32_concat_heads (List.map (fun h => Hd h hidden)
                                                                 (List.seq 0 n_head)))
                       (f32_linear_forward_2d cpw cpb));
      [| unfold f32_linear_forward_2d; apply causal_map].
    apply causal_concat_heads; [exact Hnh|]. intros h Hh. unfold Hd.
    apply causal_attention_causal; apply Hheads.
    + unfold QS. apply (causal_comp QKV (fun x => List.map (fun '(q, _, _) => q) (List.map split x)));
        [exact HQKV|].
      apply (causal_comp (List.map split)); apply causal_map.
    + unfold KS. apply (causal_comp QKV (fun x => List.map (fun '(_, k, _) => k) (List.map split x)));
        [exact HQKV|].
      apply (causal_comp (List.map split)); apply causal_map.
    + unfold VS. apply (causal_comp QKV (fun x => List.map (fun '(_, _, v) => v) (List.map split x)));
        [exact HQKV|].
      apply (causal_comp (List.map split)); apply causal_map.
Qed.

Lemma causal_gpt2_block : forall cfg eps block,
  (0 < gpt2_inf_n_head cfg)%nat -> causal (f32_block_forward cfg eps block).
Proof.
  intros cfg eps block Hnh.
  set (LN1 := f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                                (f32_ln_bias (f32_block_ln_1 block)) eps).
  set (ATT := f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                (f32_attn_c_attn_weight (f32_block_attn block))
                (f32_attn_c_attn_bias (f32_block_attn block))
                (f32_attn_c_proj_weight (f32_block_attn block))
                (f32_attn_c_proj_bias (f32_block_attn block))).
  set (LN2 := f32_layer_norm_2d (f32_ln_weight (f32_block_ln_2 block))
                                (f32_ln_bias (f32_block_ln_2 block)) eps).
  set (MLP := f32_mlp_forward (f32_mlp_c_fc_weight (f32_block_mlp block))
                (f32_mlp_c_fc_bias (f32_block_mlp block))
                (f32_mlp_c_proj_weight (f32_block_mlp block))
                (f32_mlp_c_proj_bias (f32_block_mlp block))).
  set (H2 := fun h => f32_add_matrices h (ATT (LN1 h))).
  assert (HLN1 : causal LN1) by (unfold LN1, f32_layer_norm_2d; apply causal_map).
  assert (HLN2 : causal LN2) by (unfold LN2, f32_layer_norm_2d; apply causal_map).
  assert (HMLP : causal MLP).
  { unfold MLP, f32_mlp_forward, f32_linear_forward_2d. cbv zeta.
    apply (causal_comp
             (fun h => List.map f32_gelu_vec
                (List.map (f32_linear_forward (f32_mlp_c_fc_weight (f32_block_mlp block))
                                              (f32_mlp_c_fc_bias (f32_block_mlp block))) h))
             (List.map (f32_linear_forward (f32_mlp_c_proj_weight (f32_block_mlp block))
                                           (f32_mlp_c_proj_bias (f32_block_mlp block)))));
      [| apply causal_map].
    apply (causal_comp (List.map (f32_linear_forward (f32_mlp_c_fc_weight (f32_block_mlp block))
                                                     (f32_mlp_c_fc_bias (f32_block_mlp block))))
                       (List.map f32_gelu_vec)); apply causal_map. }
  assert (HH2 : causal H2).
  { unfold H2, f32_add_matrices.
    apply (causal_zip (fun h => h) (fun h => ATT (LN1 h))); [apply causal_id|].
    apply (causal_comp LN1 ATT); [exact HLN1 | apply causal_gpt2_attention; exact Hnh]. }
  apply (causal_ext (fun h => f32_add_matrices (H2 h) (MLP (LN2 (H2 h))))).
  - intros h. reflexivity.
  - unfold f32_add_matrices.
    apply (causal_zip H2 (fun h => MLP (LN2 (H2 h)))); [exact HH2|].
    apply (causal_comp H2 (fun h2 => MLP (LN2 h2))); [exact HH2|].
    apply (causal_comp LN2 MLP); assumption.
Qed.

Lemma causal_gpt2_blocks : forall cfg eps blocks,
  (0 < gpt2_inf_n_head cfg)%nat -> causal (f32_blocks_forward cfg eps blocks).
Proof.
  intros cfg eps blocks Hnh. induction blocks as [|b bs IH].
  - apply (causal_ext (fun xs => xs)); [reflexivity | apply causal_id].
  - apply (causal_ext (fun xs => f32_blocks_forward cfg eps bs (f32_block_forward cfg eps b xs)));
      [reflexivity|].
    apply causal_comp; [apply causal_gpt2_block; exact Hnh | exact IH].
Qed.

Theorem causal_gpt2_forward : forall cfg eps model,
  (0 < gpt2_inf_n_head cfg)%nat -> causal (f32_gpt2_forward cfg eps model).
Proof.
  intros cfg eps model Hnh.
  set (EMB := fun toks => f32_add_matrices (f32_embed_tokens (f32_wte model) toks)
                            (f32_embed_positions (f32_wpe model) (List.length toks))).
  assert (HE : causal EMB).
  { unfold EMB, f32_add_matrices, f32_embed_tokens, f32_embed_positions.
    apply (causal_zip (List.map (f32_lookup_embedding (f32_wte model)))
             (fun toks : list nat => List.map (f32_lookup_embedding (f32_wpe model))
                                              (List.seq 0 (List.length toks)))).
    - apply causal_map.
    - apply causal_positions. }
  set (LNF := f32_layer_norm_2d (f32_ln_weight (f32_ln_f model)) (f32_ln_bias (f32_ln_f model)) eps).
  apply (causal_ext (fun toks => LNF (f32_blocks_forward cfg eps (f32_blocks model) (EMB toks)))).
  - intros toks. reflexivity.
  - apply (causal_comp (fun toks => f32_blocks_forward cfg eps (f32_blocks model) (EMB toks)) LNF).
    + apply (causal_comp EMB (f32_blocks_forward cfg eps (f32_blocks model)));
        [exact HE | apply causal_gpt2_blocks; exact Hnh].
    + unfold LNF, f32_layer_norm_2d. apply causal_map.
Qed.

Lemma causal_gpt2_logits : forall cfg eps model,
  (0 < gpt2_inf_n_head cfg)%nat -> causal (f32_gpt2_logits cfg eps model).
Proof.
  intros cfg eps model Hnh.
  apply (causal_ext (fun toks => List.map (fun h_row =>
                       List.map (fun w_row => f32_dot h_row w_row) (f32_wte model))
                       (f32_gpt2_forward cfg eps model toks))); [reflexivity|].
  apply (causal_comp (f32_gpt2_forward cfg eps model));
    [apply causal_gpt2_forward; exact Hnh | apply causal_map].
Qed.

(** Running GPT-2 on an extended token sequence reproduces every logit row it
    computes for the original sequence. *)
Corollary gpt2_logits_prefix : forall cfg eps model toks1 toks2,
  (0 < gpt2_inf_n_head cfg)%nat ->
  List.firstn (List.length toks1) (f32_gpt2_logits cfg eps model (toks1 ++ toks2))
  = f32_gpt2_logits cfg eps model toks1.
Proof.
  intros cfg eps model toks1 toks2 Hnh.
  apply (proj2 (causal_gpt2_logits cfg eps model Hnh)).
Qed.

(** * Decoding one token at a time

    A decoder does not evaluate the sequence function on the whole sequence. It
    takes one step per token and keeps the last row of each, which is the row it
    emits. [decode_stream] is that decoder written over any sequence function,
    and the theorem below is that its output is the one the whole-sequence
    evaluation produces, row for row. Causality is what the proof runs on. *)

Definition decode_stream {A B : Type} (F : list A -> list B) (d : B)
                         (xs : list A) : list B :=
  List.map (fun i => List.last (F (List.firstn (S i) xs)) d)
           (List.seq 0 (List.length xs)).

Lemma last_nth : forall {A : Type} (l : list A) (d : A),
  List.last l d = List.nth (List.length l - 1) l d.
Proof.
  intros A l d. induction l as [|x l IH]; [reflexivity|].
  destruct l as [|y l']; [reflexivity|].
  change (List.last (x :: y :: l') d) with (List.last (y :: l') d).
  rewrite IH. cbn [List.length]. rewrite !Nat.sub_succ, !Nat.sub_0_r.
  reflexivity.
Qed.

Theorem decode_stream_correct : forall {A B : Type} (F : list A -> list B) d xs,
  causal F -> decode_stream F d xs = F xs.
Proof.
  intros A B F d xs HF. unfold decode_stream.
  rewrite <- (map_nth_seq d (F xs)), (proj1 HF).
  apply List.map_ext_in. intros i Hi.
  apply List.in_seq in Hi. destruct Hi as [_ Hi]. cbn in Hi.
  assert (Hlen : List.length (List.firstn (S i) xs) = S i)
    by (rewrite List.length_firstn; lia).
  rewrite last_nth, (proj1 HF), Hlen.
  replace (S i - 1)%nat with i by lia.
  rewrite <- (List.firstn_skipn (S i) xs) at 2.
  rewrite (causal_nth F (List.firstn (S i) xs) (List.skipn (S i) xs) i d HF)
    by (rewrite Hlen; lia).
  reflexivity.
Qed.

(** One decode step per token, on each of the three models, produces the rows
    a single evaluation of the whole sequence produces. *)

Corollary gpt2_decode_step : forall cfg eps model toks d,
  (0 < gpt2_inf_n_head cfg)%nat ->
  decode_stream (f32_gpt2_logits cfg eps model) d toks
  = f32_gpt2_logits cfg eps model toks.
Proof.
  intros cfg eps model toks d Hnh.
  apply decode_stream_correct, causal_gpt2_logits, Hnh.
Qed.

Corollary llama_decode_step : forall eps normw fs emb ids d,
  List.Forall causal fs ->
  decode_stream (f32_llama_forward eps normw fs emb) d ids
  = f32_llama_forward eps normw fs emb ids.
Proof.
  intros eps normw fs emb ids d Hfs.
  apply decode_stream_correct, causal_llama_forward, Hfs.
Qed.

Corollary qwen_decode_step : forall eps normw fs emb ids d,
  List.Forall causal fs ->
  decode_stream (f32_qwen_forward eps normw fs emb) d ids
  = f32_qwen_forward eps normw fs emb ids.
Proof.
  intros eps normw fs emb ids d Hfs.
  apply decode_stream_correct, causal_qwen_forward, Hfs.
Qed.
