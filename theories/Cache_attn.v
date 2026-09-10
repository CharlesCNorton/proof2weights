(** * Cached attention equals the last row of full causal attention

    A decode step attends a single query over a key/value cache. The
    definition instead recomputes causal attention over the whole sequence and
    would take the last row. This file proves the two agree exactly, and
    isolates the two places where the agreement is conditional.

    The first is signed zero. Full attention adds the mask entry to every
    score, and on the last row every mask entry is [+0]; the cached path adds
    nothing. Under round-to-nearest, [x + (+0) = x] for every binary32 [x]
    except [-0], where it returns [+0]. The theorem therefore carries the
    hypothesis that no score is negative zero, which is exactly the condition
    under which the two paths coincide.

    The second is that the statement is about the last row and not about
    earlier ones. On row [i < n-1] the full computation exponentiates the
    masked columns as well. Those contribute [exp(-88)], approximately
    [6.05e-39], which is subnormal but not zero, and they enter the softmax
    denominator. A prefix-cached evaluation omits them. The two therefore
    agree only up to whether that subnormal mass survives rounding, which is
    not an identity. Decoding only ever computes the last row, so the theorem
    covers the case that runs. *)

From Stdlib Require Import List.
From Stdlib Require Import Lia.
From Stdlib Require Import ZArith.
From Stdlib Require Import PeanoNat.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.

Import ListNotations.

(** * Adding positive zero

    The only binary32 value [+0] does not fix is [-0]. *)

Lemma f32_plus_zero_r : forall x : binary32,
  x <> B754_zero true -> f32_plus x f32_zero = x.
Proof.
  intros x Hx. destruct x as [s|s| |s m e H]; try reflexivity.
  - destruct s; [contradiction | reflexivity].
Qed.

(** And it does not fix it: the two paths genuinely differ there. *)
Example f32_plus_zero_neg : f32_plus (B754_zero true) f32_zero = B754_zero false.
Proof. reflexivity. Qed.

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

(** * The last row of the causal mask is all zeros *)

Lemma causal_mask_last_row : forall n,
  List.nth n (f32_causal_mask (S n)) []
  = List.repeat f32_zero (S n).
Proof.
  intros n. unfold f32_causal_mask.
  rewrite (List.nth_indep _ []
    (List.map (fun col => f32_causal_mask_entry 0 col) (List.seq 0 (S n))))
    by (rewrite List.length_map, List.length_seq; lia).
  rewrite (List.map_nth (fun row =>
    List.map (fun col => f32_causal_mask_entry row col) (List.seq 0 (S n)))).
  rewrite List.seq_nth by lia. cbn [Nat.add].
  rewrite <- (List.length_seq (S n) 0) at 2.
  rewrite <- List.map_const.
  apply List.map_ext_in. intros col Hcol.
  apply List.in_seq in Hcol. destruct Hcol as [_ Hcol]. cbn in Hcol.
  unfold f32_causal_mask_entry.
  destruct (Nat.leb col n) eqn:E; [reflexivity|].
  apply Nat.leb_gt in E. lia.
Qed.

(** Zipping a row against an all-zero mask row is the identity, away from
    negative zero. *)
Lemma apply_mask_zero_row : forall row n,
  List.length row = n ->
  Forall (fun x => x <> B754_zero true) row ->
  List.map (fun p => let '(s, mv) := p in f32_plus s mv)
           (List.combine row (List.repeat f32_zero n)) = row.
Proof.
  induction row as [|x row IH]; intros n Hlen Hall.
  - destruct n; reflexivity.
  - destruct n as [|n]; [discriminate|].
    cbn [List.repeat List.combine List.map].
    inversion Hall; subst.
    rewrite f32_plus_zero_r by assumption.
    f_equal. apply IH; [cbn in Hlen; lia | assumption].
Qed.

(** * The cached decode step *)

Definition f32_attn_scale (d_k : nat) : binary32 :=
  f32_div f32_one (f32_sqrt (f32_of_Z (Z.of_nat d_k))).

(** [nth] commutes with [map] when the function sends the default to the
    default, which is the case for every row-wise stage below. *)
Lemma nth_map_nil2 : forall {A B : Type} (f : A -> list B) (l : list A) (d : A) n,
  f d = [] ->
  List.nth n (List.map f l) [] = f (List.nth n l d).
Proof.
  intros A B f l d n Hf. revert n.
  induction l as [|x l IH]; intros n.
  - destruct n; cbn [List.map List.nth]; symmetry; exact Hf.
  - destruct n; [reflexivity | cbn [List.map List.nth]; apply IH].
Qed.

Lemma scale_scores_unfold : forall m d_k,
  f32_scale_scores m d_k
  = List.map (fun row => List.map (fun x => f32_mult x (f32_attn_scale d_k)) row) m.
Proof. reflexivity. Qed.

Definition f32_attend_cached (qrow : list binary32)
    (kc vc : list (list binary32)) (d_k : nat) : list binary32 :=
  let scores := List.map (fun kj => f32_mult (f32_dot qrow kj) (f32_attn_scale d_k)) kc in
  let w := f32_softmax scores in
  List.map (fun vcol => f32_dot w vcol) (f32_mat_transpose vc).

(** * The theorem

    Attending a single query over the cache is exactly the last row of full
    causal attention over the same sequence. *)

Theorem attention_last_row_cached :
  forall q k v d_k n c,
  List.length q = S n ->
  List.length k = S n ->
  rect k c -> (0 < c)%nat ->
  Forall (fun x => x <> B754_zero true)
    (List.map (fun kj => f32_mult (f32_dot (List.nth n q []) kj) (f32_attn_scale d_k)) k) ->
  List.nth n (f32_causal_attention q k v d_k) []
  = f32_attend_cached (List.nth n q []) k v d_k.
Proof.
  intros q k v d_k n c Hq Hk Hr Hc Hnz.
  assert (Hkne : k <> []).
  { intro H0. rewrite H0 in Hk. cbn in Hk. discriminate. }
  unfold f32_causal_attention, f32_attend_cached. cbv zeta.
  (* the outer matrix product: extract row n *)
  unfold f32_mat_mul at 1.
  rewrite (List.nth_indep _ []
    (List.map (fun b_col => f32_dot [] b_col) (f32_mat_transpose v)))
    by (rewrite List.length_map;
        unfold f32_softmax_2d, f32_apply_mask, f32_scale_scores;
        rewrite !List.length_map, List.length_combine, !List.length_map;
        unfold f32_mat_mul; rewrite List.length_map;
        unfold f32_causal_mask; rewrite List.length_map, List.length_seq;
        rewrite Hq; unfold f32_mat_rows; rewrite Hq; lia).
  rewrite (List.map_nth (fun a_row =>
    List.map (fun b_col => f32_dot a_row b_col) (f32_mat_transpose v))).
  f_equal.
  (* the softmax row *)
  unfold f32_softmax_2d.
  rewrite (nth_map_nil2 _ _ (@nil binary32)) by reflexivity.
  f_equal.
  (* the masked score row *)
  unfold f32_apply_mask.
  rewrite (nth_map_nil2 _ _ (@nil binary32, @nil binary32)) by reflexivity.
  rewrite List.combine_nth by
    (unfold f32_scale_scores, f32_mat_mul, f32_causal_mask, f32_mat_rows;
     rewrite !List.length_map, List.length_seq, Hq; reflexivity).
  cbn [fst snd].
  (* the mask row *)
  unfold f32_mat_rows. rewrite Hq, causal_mask_last_row.
  (* the scaled score row *)
  rewrite scale_scores_unfold.
  rewrite (nth_map_nil2 _ _ (@nil binary32)) by reflexivity.
  unfold f32_mat_mul.
  rewrite (List.nth_indep _ []
    (List.map (fun b_col => f32_dot [] b_col)
              (f32_mat_transpose (f32_mat_transpose k))))
    by (rewrite List.length_map, Hq; lia).
  rewrite (List.map_nth (fun a_row =>
    List.map (fun b_col => f32_dot a_row b_col)
             (f32_mat_transpose (f32_mat_transpose k)))).
  rewrite (transpose_involutive k c Hkne Hc Hr).
  rewrite List.map_map.
  assert (Hrow :
    List.map (fun p => let '(s, mv) := p in f32_plus s mv)
      (List.combine
         (List.map (fun x : list binary32 =>
            f32_mult (f32_dot (List.nth n q []) x) (f32_attn_scale d_k)) k)
         (List.repeat f32_zero (S n)))
    = List.map (fun x : list binary32 =>
        f32_mult (f32_dot (List.nth n q []) x) (f32_attn_scale d_k)) k).
  { apply apply_mask_zero_row.
    - rewrite List.length_map. exact Hk.
    - exact Hnz. }
  rewrite Hrow. reflexivity.
Qed.
