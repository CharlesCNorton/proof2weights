(** * Causal attention and the key/value cache

    Row [i] of [f32_causal_attention] attends its query over the first [i + 1]
    key and value rows. A row therefore depends on the sequence only through
    that prefix: extending the sequence leaves every existing row unchanged,
    and a decode step appends exactly one row, the new query attending over
    the whole cache. The theorems below are equalities of binary32 values.

    The second half concerns transposition, which the attention output and
    the runners' transposed decode both go through. *)

From Stdlib Require Import List.
From Stdlib Require Import Lia.
From Stdlib Require Import ZArith.
From Stdlib Require Import PeanoNat.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.

Import ListNotations.

(** * Rows of causal attention *)

Lemma causal_attention_nth : forall q k v d_k i,
  (i < List.length q)%nat ->
  List.nth i (f32_causal_attention q k v d_k) []
  = f32_attend (List.nth i q []) (List.firstn (S i) k) (List.firstn (S i) v) d_k.
Proof.
  intros q k v d_k i Hi. unfold f32_causal_attention.
  set (f := fun p : nat * list binary32 => let '(j, qrow) := p in
              f32_attend qrow (List.firstn (S j) k) (List.firstn (S j) v) d_k).
  rewrite (List.nth_indep _ [] (f (i, [])))
    by (rewrite List.length_map, List.length_combine, List.length_seq; lia).
  rewrite List.map_nth.
  rewrite List.combine_nth by (rewrite List.length_seq; reflexivity).
  rewrite List.seq_nth by exact Hi.
  reflexivity.
Qed.

Lemma combine_app_eq : forall {A B : Type} (l1 l2 : list A) (r1 r2 : list B),
  List.length l1 = List.length r1 ->
  List.combine (l1 ++ l2) (r1 ++ r2) = List.combine l1 r1 ++ List.combine l2 r2.
Proof.
  intros A B l1. induction l1 as [|x l1 IH]; intros l2 r1 r2 H.
  - destruct r1; [reflexivity | discriminate].
  - destruct r1 as [|y r1]; [discriminate|].
    cbn [app List.combine]. f_equal. apply IH. cbn in H. lia.
Qed.

(** Extending the sequence leaves the rows already computed unchanged, and
    the new rows attend over prefixes of the extended keys and values. *)
Theorem causal_attention_app : forall q1 q2 k1 k2 v1 v2 d_k,
  List.length k1 = List.length q1 -> List.length v1 = List.length q1 ->
  f32_causal_attention (q1 ++ q2) (k1 ++ k2) (v1 ++ v2) d_k
  = f32_causal_attention q1 k1 v1 d_k
    ++ List.map (fun p => let '(i, qrow) := p in
                   f32_attend qrow (List.firstn (S i) (k1 ++ k2))
                                   (List.firstn (S i) (v1 ++ v2)) d_k)
                (List.combine (List.seq (List.length q1) (List.length q2)) q2).
Proof.
  intros q1 q2 k1 k2 v1 v2 d_k Hk Hv.
  unfold f32_causal_attention.
  rewrite List.length_app, List.seq_app.
  rewrite combine_app_eq by (rewrite List.length_seq; reflexivity).
  rewrite List.map_app. f_equal.
  apply List.map_ext_in. intros [i qrow] Hin.
  apply List.in_combine_l, List.in_seq in Hin.
  rewrite !List.firstn_app.
  replace (S i - List.length k1)%nat with 0%nat by lia.
  replace (S i - List.length v1)%nat with 0%nat by lia.
  rewrite !List.firstn_0, !List.app_nil_r. reflexivity.
Qed.

(** A prefix of the sequence computes exactly the corresponding rows. *)
Corollary causal_attention_prefix : forall q1 q2 k1 k2 v1 v2 d_k,
  List.length k1 = List.length q1 -> List.length v1 = List.length q1 ->
  List.firstn (List.length q1) (f32_causal_attention (q1 ++ q2) (k1 ++ k2) (v1 ++ v2) d_k)
  = f32_causal_attention q1 k1 v1 d_k.
Proof.
  intros q1 q2 k1 k2 v1 v2 d_k Hk Hv.
  pose proof (f32_causal_attention_rows q1 k1 v1 d_k) as Hr.
  unfold f32_mat_rows in Hr.
  rewrite causal_attention_app by assumption.
  rewrite List.firstn_app, Hr, Nat.sub_diag, List.firstn_0, List.app_nil_r.
  apply List.firstn_all2. lia.
Qed.

(** The decode step: one more position appends exactly the new query attending
    over every cached key and value, and changes nothing already emitted. *)
Corollary causal_attention_snoc : forall q k v qn kn vn d_k,
  List.length k = List.length q -> List.length v = List.length q ->
  f32_causal_attention (q ++ [qn]) (k ++ [kn]) (v ++ [vn]) d_k
  = f32_causal_attention q k v d_k ++ [f32_attend qn (k ++ [kn]) (v ++ [vn]) d_k].
Proof.
  intros q k v qn kn vn d_k Hk Hv.
  rewrite causal_attention_app by assumption.
  cbn [List.length List.seq List.combine List.map].
  rewrite !List.firstn_all2 by (rewrite List.length_app; cbn [List.length]; lia).
  reflexivity.
Qed.

(** * Rectangular matrices and transposition *)

Definition rect (m : list (list binary32)) (c : nat) : Prop :=
  Forall (fun row => List.length row = c) m.

Lemma map_nth_seq : forall {A : Type} (d : A) (l : list A),
  List.map (fun i => List.nth i l d) (List.seq 0 (List.length l)) = l.
Proof.
  intros A d l. induction l as [|x l IH]; [reflexivity|].
  cbn [List.length List.seq List.map List.nth].
  f_equal.
  rewrite <- List.seq_shift, List.map_map. cbn [List.nth]. exact IH.
Qed.

Lemma transpose_length : forall m c,
  m <> [] -> rect m c -> List.length (f32_mat_transpose m) = c.
Proof.
  intros m c Hne Hr. destruct m as [|row m]; [contradiction|].
  cbn [f32_mat_transpose].
  rewrite List.length_map, List.length_seq.
  apply (List.Forall_inv Hr).
Qed.

Lemma transpose_rect : forall m c,
  m <> [] -> rect m c -> rect (f32_mat_transpose m) (List.length m).
Proof.
  intros m c Hne Hr. destruct m as [|row m]; [contradiction|].
  cbn [f32_mat_transpose]. unfold rect.
  apply List.Forall_forall. intros x Hx.
  apply List.in_map_iff in Hx. destruct Hx as [i [Hi _]]. subst x.
  apply List.length_map.
Qed.

Lemma transpose_nth : forall m c i,
  m <> [] -> rect m c -> (i < c)%nat ->
  List.nth i (f32_mat_transpose m) []
  = List.map (fun r => List.nth i r f32_zero) m.
Proof.
  intros m c i Hne Hr Hi. destruct m as [|row m]; [contradiction|].
  cbn [f32_mat_transpose].
  assert (Hrow : List.length row = c) by (apply (List.Forall_inv Hr)).
  rewrite <- Hrow in Hi.
  rewrite (List.nth_indep _ [] (List.map (fun r => List.nth 0 r f32_zero) (row :: m)))
    by (rewrite List.length_map, List.length_seq; exact Hi).
  rewrite (List.map_nth (fun j => List.map (fun r => List.nth j r f32_zero) (row :: m))).
  rewrite List.seq_nth by exact Hi. reflexivity.
Qed.

Lemma nth_map_nth : forall (m : list (list binary32)) i j,
  List.nth i (List.map (fun r => List.nth j r f32_zero) m) f32_zero
  = List.nth j (List.nth i m []) f32_zero.
Proof.
  induction m as [|r m IH]; intros i j.
  - destruct i, j; reflexivity.
  - destruct i; [reflexivity | cbn [List.map List.nth]; apply IH].
Qed.

Lemma map_over_transpose : forall (T : list (list binary32)) i,
  List.map (fun rd => List.nth i rd f32_zero) T
  = List.map (fun j => List.nth i (List.nth j T []) f32_zero)
             (List.seq 0 (List.length T)).
Proof.
  intros T i.
  transitivity (List.map (fun rd => List.nth i rd f32_zero)
                  (List.map (fun j => List.nth j T []) (List.seq 0 (List.length T)))).
  - rewrite map_nth_seq. reflexivity.
  - rewrite List.map_map. reflexivity.
Qed.

Lemma transpose_unfold : forall T c,
  T <> [] -> rect T c ->
  f32_mat_transpose T
  = List.map (fun i => List.map (fun r => List.nth i r f32_zero) T)
             (List.seq 0 c).
Proof.
  intros T c Hne Hr. destruct T as [|row T]; [contradiction|].
  cbn [f32_mat_transpose]. rewrite (List.Forall_inv Hr). reflexivity.
Qed.

Theorem transpose_involutive : forall m c,
  m <> [] -> (0 < c)%nat -> rect m c ->
  f32_mat_transpose (f32_mat_transpose m) = m.
Proof.
  intros m c Hne Hc Hr.
  assert (HTlen : List.length (f32_mat_transpose m) = c)
    by (eapply transpose_length; eassumption).
  assert (HT : f32_mat_transpose m <> []).
  { intro H0. rewrite H0 in HTlen. cbn in HTlen. lia. }
  assert (HTr : rect (f32_mat_transpose m) (List.length m))
    by (eapply transpose_rect; eassumption).
  rewrite (transpose_unfold (f32_mat_transpose m) (List.length m) HT HTr).
  transitivity (List.map (fun i => List.nth i m []) (List.seq 0 (List.length m))).
  2: apply map_nth_seq.
  apply List.map_ext_in. intros i Hi.
  apply List.in_seq in Hi. destruct Hi as [_ Hi]. cbn in Hi.
  assert (Hlen : List.length (List.nth i m []) = c).
  { unfold rect in Hr. rewrite List.Forall_forall in Hr.
    apply Hr, List.nth_In. lia. }
  transitivity (List.map (fun j => List.nth j (List.nth i m []) f32_zero)
                  (List.seq 0 (List.length (List.nth i m [])))).
  2: apply map_nth_seq.
  rewrite Hlen, <- HTlen.
  rewrite (map_over_transpose (f32_mat_transpose m) i).
  apply List.map_ext_in. intros j Hj.
  apply List.in_seq in Hj. destruct Hj as [_ Hj].
  rewrite HTlen in Hj. cbn in Hj.
  rewrite (transpose_nth m c j Hne Hr Hj).
  apply nth_map_nth.
Qed.
