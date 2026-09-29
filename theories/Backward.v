(** * Backward error of the dot product, with underflow

    [f32_dot_backward] in Float_error.v says the computed inner product is the
    exact inner product of the operands with each product perturbed by a
    relative factor, which is the classical statement. It holds under
    [f32_dot_regular], whose [relz] clauses require every product and every
    partial sum to be zero or at least [2^-126] in magnitude, so that rounding
    is relative there. That hypothesis is false where the networks actually
    compute: the [exp] of an attention score more than 87.4 below its row
    maximum is subnormal, and so are the products the attention's weighted sum
    of values then accumulates.

    This file drops it. The mixed rounding model of Float_error.v,
    [f32_round z = z (1 + d) + e] with [|d| <= u], [|e| <= eta] and [d e = 0],
    holds for every real with no hypothesis, so the same induction goes through
    with an absolute term carried alongside the relative one. The result is the
    mixed backward-error statement: the computed dot product is the exact inner
    product with relatively perturbed terms, plus an absolute displacement of
    at most [2 n eta (1 + u)^n], where the only hypothesis left is that nothing
    overflows.

    The relative factor is unchanged, so the classical statement is the special
    case where the absolute term is known to vanish. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Lia.
From Stdlib Require Import List.
From Flocq Require Import Core.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Float_error.

Import ListNotations.
Open Scope R_scope.

(** * Overflow is the only hypothesis

    [f32_dot_fin] is [f32_dot_regular] with the two [relz] clauses removed: it
    says the accumulator and each product are finite and that neither rounding
    overflows, and nothing about how small anything is. *)

Inductive f32_dot_fin : list binary32 -> list binary32 -> binary32 -> Prop :=
| dfin_nil_l : forall ys a, f32_dot_fin nil ys a
| dfin_nil_r : forall xs a, f32_dot_fin xs nil a
| dfin_cons : forall x xs y ys a,
    is_finite a = true ->
    is_finite (f32_mult x y) = true ->
    Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true ->
    Rlt_bool (Rabs (f32_round (B2R a + B2R (f32_mult x y))))
             (bpow radix2 emax32) = true ->
    f32_dot_fin xs ys (f32_plus a (f32_mult x y)) ->
    f32_dot_fin (x :: xs) (y :: ys) a.

(** Every regular dot product is a finite one. *)
Lemma f32_dot_regular_fin : forall xs ys a,
  f32_dot_regular xs ys a -> f32_dot_fin xs ys a.
Proof.
  intros xs ys a H.
  induction H as [ys a | xs a | x xs y ys a Hfa Hfm Hb1 Hb2 _ _ _ IH].
  - apply dfin_nil_l.
  - apply dfin_nil_r.
  - now apply dfin_cons.
Qed.

(** * The absolute term *)

Definition dot_eta (n : nat) : R := 2 * INR n * f32_eta * (1 + f32_u) ^ n.

Lemma dot_eta_nonneg : forall n, 0 <= dot_eta n.
Proof.
  intros n. unfold dot_eta.
  assert (H1 : 0 <= INR n) by apply pos_INR.
  assert (H2 : 0 <= f32_eta) by (pose proof f32_eta_pos; lra).
  assert (H3 : 0 <= (1 + f32_u) ^ n) by (pose proof (pow1u_ge_1 n); lra).
  apply Rmult_le_pos;
    [apply Rmult_le_pos; [apply Rmult_le_pos; [lra | exact H1] | exact H2]
    | exact H3].
Qed.

(** One step of accumulation adds at most [eta (1 + u)] from the product's
    rounding and [eta] from the sum's, and the budget has room for both. *)
Lemma dot_eta_step : forall n,
  (f32_eta * (1 + f32_u) + f32_eta) * (1 + f32_u) ^ n + dot_eta n
  <= dot_eta (S n).
Proof.
  intros n. unfold dot_eta.
  pose proof f32_eta_pos as He.
  pose proof f32_u_pos as Hu.
  pose proof (pow1u_ge_1 n) as Hp.
  pose proof (pos_INR n) as Hn.
  rewrite S_INR. cbn [pow].
  set (P := (1 + f32_u) ^ n) in *.
  set (N := INR n) in *.
  (* the difference is eta * P * u * (2 N + 1), which is nonnegative *)
  cut (0 <= 2 * (N + 1) * f32_eta * ((1 + f32_u) * P)
            - ((f32_eta * (1 + f32_u) + f32_eta) * P + 2 * N * f32_eta * P)).
  { lra. }
  replace (2 * (N + 1) * f32_eta * ((1 + f32_u) * P)
           - ((f32_eta * (1 + f32_u) + f32_eta) * P + 2 * N * f32_eta * P))
    with (f32_eta * P * (f32_u * (2 * N + 1))) by ring.
  assert (HE : 0 <= f32_eta) by lra.
  assert (HP : 0 <= P) by lra.
  assert (HU : 0 <= f32_u) by lra.
  assert (HN : 0 <= 2 * N + 1) by lra.
  apply Rmult_le_pos;
    [ apply Rmult_le_pos; [exact HE | exact HP]
    | apply Rmult_le_pos; [exact HU | exact HN] ].
Qed.

(** * The induction *)

Lemma f32_dot_aux_backward_mixed : forall xs ys a,
  f32_dot_fin xs ys a ->
  exists alpha ts delta,
    Rabs (alpha - 1) <= (1 + f32_u) ^ (List.length xs) - 1
    /\ List.length ts = List.length xs
    /\ Forall (fun t => Rabs (t - 1)
                        <= (1 + f32_u) ^ (S (List.length xs)) - 1) ts
    /\ Rabs delta <= dot_eta (List.length xs)
    /\ B2R (f32_dot_aux xs ys a)
       = B2R a * alpha + Rdot_terms xs ys ts + delta.
Proof.
  intros xs ys a Hfin.
  induction Hfin as [ys a | xs a | x xs y ys a Hfa Hfm Hb1 Hb2 Hfin IH].
  - exists 1, nil, 0. cbn [f32_dot_aux List.length Rdot_terms].
    repeat apply conj.
    + rewrite Rminus_diag_eq by reflexivity. rewrite Rabs_R0. simpl. lra.
    + reflexivity.
    + constructor.
    + rewrite Rabs_R0. apply dot_eta_nonneg.
    + ring.
  - exists 1, (List.repeat 1 (List.length xs)), 0.
    destruct xs as [|x xs]; cbn [f32_dot_aux].
    + repeat apply conj.
      * rewrite Rminus_diag_eq by reflexivity. rewrite Rabs_R0.
        cbn [List.length]. simpl. lra.
      * cbn [List.repeat List.length]. reflexivity.
      * constructor.
      * rewrite Rabs_R0. apply dot_eta_nonneg.
      * cbn [Rdot_terms]. ring.
    + repeat apply conj.
      * rewrite Rminus_diag_eq by reflexivity. rewrite Rabs_R0.
        pose proof (pow1u_ge_1 (List.length (x :: xs))). lra.
      * apply List.repeat_length.
      * apply Forall_forall. intros t Ht.
        apply List.repeat_spec in Ht. subst t.
        rewrite Rminus_diag_eq by reflexivity. rewrite Rabs_R0.
        pose proof (pow1u_ge_1 (S (List.length (x :: xs)))). lra.
      * rewrite Rabs_R0. apply dot_eta_nonneg.
      * rewrite Rdot_terms_ones. ring.
  - destruct IH as [alpha' [ts' [delta' (Hal & Hlen & Hts & Hdl & Heq)]]].
    pose proof (f32_mult_correct x y Hb1) as HM.
    pose proof (f32_plus_correct a (f32_mult x y) Hfa Hfm Hb2) as HP.
    destruct (f32_round_mixed (B2R x * B2R y)) as [m [em (Hm & Hem & _ & Hrm)]].
    destruct (f32_round_mixed (B2R a + B2R (f32_mult x y)))
      as [e [ee (He & Hee & _ & Hre)]].
    exists ((1 + e) * alpha'),
           ((1 + m) * ((1 + e) * alpha') :: ts'),
           ((em * (1 + e) + ee) * alpha' + delta').
    cbn [f32_dot_aux List.length Rdot_terms].
    assert (Hstep : B2R (f32_plus a (f32_mult x y))
                    = (B2R a + B2R x * B2R y * (1 + m)) * (1 + e)
                      + (em * (1 + e) + ee)).
    { rewrite HP, Hre, HM, Hrm. ring. }
    repeat apply conj.
    + apply pert_step; assumption.
    + simpl. f_equal. exact Hlen.
    + constructor.
      * apply pert_step; [apply pert_step; assumption | exact Hm].
      * eapply Forall_pert_weaken; [| exact Hts]. lia.
    + (* the absolute term *)
      eapply Rle_trans; [apply Rabs_triang|].
      eapply Rle_trans with
        ((f32_eta * (1 + f32_u) + f32_eta) * (1 + f32_u) ^ (List.length xs)
         + dot_eta (List.length xs)); [| apply dot_eta_step].
      apply Rplus_le_compat; [| exact Hdl].
      rewrite Rabs_mult.
      apply Rmult_le_compat.
      * apply Rabs_pos.
      * apply Rabs_pos.
      * eapply Rle_trans; [apply Rabs_triang|].
        apply Rplus_le_compat; [| exact Hee].
        rewrite Rabs_mult.
        apply Rmult_le_compat; try apply Rabs_pos; [exact Hem|].
        pose proof f32_u_pos.
        apply Rabs_le. apply Rabs_le_inv in He. lra.
      * apply pert_abs. exact Hal.
    + rewrite Heq, Hstep. ring.
Qed.

(** * The statement

    The computed dot product is the exact inner product of its operands with
    each product perturbed by a relative factor of at most [(1 + u)^(n+1) - 1],
    displaced by at most [2 n eta (1 + u)^n]. The relative factor is set by the
    length of the one dot product and not by the depth of the network, and the
    displacement is a multiple of the smallest positive binary32, so it matters
    only where the exact inner product is itself of that size. *)

Theorem f32_dot_backward_mixed : forall xs ys,
  f32_dot_fin xs ys f32_zero ->
  exists ts delta,
    List.length ts = List.length xs
    /\ Forall (fun t => Rabs (t - 1)
                        <= (1 + f32_u) ^ (S (List.length xs)) - 1) ts
    /\ Rabs delta <= dot_eta (List.length xs)
    /\ B2R (f32_dot xs ys) = Rdot_terms xs ys ts + delta.
Proof.
  intros xs ys Hfin.
  destruct (f32_dot_aux_backward_mixed xs ys f32_zero Hfin)
    as [alpha [ts [delta (Hal & Hlen & Hts & Hdl & Heq)]]].
  exists ts, delta. repeat apply conj; try assumption.
  unfold f32_dot. rewrite Heq, B2R_f32_zero. ring.
Qed.

(** The classical statement is the case where the operands keep rounding
    relative, which is what [f32_dot_regular] asserts. *)
Corollary f32_dot_backward_mixed_of_regular : forall xs ys,
  f32_dot_regular xs ys f32_zero ->
  exists ts delta,
    List.length ts = List.length xs
    /\ Forall (fun t => Rabs (t - 1)
                        <= (1 + f32_u) ^ (S (List.length xs)) - 1) ts
    /\ Rabs delta <= dot_eta (List.length xs)
    /\ B2R (f32_dot xs ys) = Rdot_terms xs ys ts + delta.
Proof.
  intros xs ys H.
  apply f32_dot_backward_mixed, f32_dot_regular_fin, H.
Qed.

(** * Through the linear layers

    Every scalar a linear layer or a logit projection produces is a dot
    product, so the statement lifts to them the way the relative one does. A
    row of length at most [kk] carries at most [dot_eta kk], because the
    displacement grows with the length. *)

Lemma dot_eta_mono : forall n k, (n <= k)%nat -> dot_eta n <= dot_eta k.
Proof.
  intros n k Hle. unfold dot_eta.
  assert (Hpow : (1 + f32_u) ^ n <= (1 + f32_u) ^ k).
  { apply Rle_pow; [pose proof f32_u_pos; lra | exact Hle]. }
  assert (HN : INR n <= INR k) by (apply le_INR; exact Hle).
  assert (H0 : 0 <= INR n) by apply pos_INR.
  assert (HE : 0 <= f32_eta) by (pose proof f32_eta_pos; lra).
  assert (HP : 0 <= (1 + f32_u) ^ n) by (pose proof (pow1u_ge_1 n); lra).
  apply Rmult_le_compat; try assumption.
  - apply Rmult_le_pos; [apply Rmult_le_pos; [lra | exact H0] | exact HE].
  - apply Rmult_le_compat_r; [exact HE | lra].
Qed.

Lemma f32_dot_map_backward_mixed : forall kk rows v,
  Forall (fun row => f32_dot_fin row v f32_zero
                     /\ (List.length row <= kk)%nat) rows ->
  Forall2 (fun row out =>
     exists ts delta, List.length ts = List.length row
       /\ Forall (fun t => Rabs (t - 1) <= (1 + f32_u) ^ (S kk) - 1) ts
       /\ Rabs delta <= dot_eta kk
       /\ B2R out = Rdot_terms row v ts + delta)
    rows (List.map (fun row => f32_dot row v) rows).
Proof.
  intros kk rows v H.
  induction H as [|row rows' [Hfin Hlen] H IH]; cbn [List.map]; constructor.
  - destruct (f32_dot_backward_mixed row v Hfin) as [ts [d (Hl & Ht & Hd & He)]].
    exists ts, d. repeat apply conj; try assumption.
    + eapply Forall_pert_weaken; [| exact Ht]. lia.
    + eapply Rle_trans; [exact Hd | apply dot_eta_mono, Hlen].
  - exact IH.
Qed.

Lemma f32_dot_map_backward_mixed_r : forall kk h rows,
  (List.length h <= kk)%nat ->
  Forall (fun row => f32_dot_fin h row f32_zero) rows ->
  Forall2 (fun row out =>
     exists ts delta, List.length ts = List.length h
       /\ Forall (fun t => Rabs (t - 1) <= (1 + f32_u) ^ (S kk) - 1) ts
       /\ Rabs delta <= dot_eta kk
       /\ B2R out = Rdot_terms h row ts + delta)
    rows (List.map (fun row => f32_dot h row) rows).
Proof.
  intros kk h rows Hlen H.
  induction H as [|row rows' Hfin H IH]; cbn [List.map]; constructor.
  - destruct (f32_dot_backward_mixed h row Hfin) as [ts [d (Hl & Ht & Hd & He)]].
    exists ts, d. repeat apply conj; try assumption.
    + eapply Forall_pert_weaken; [| exact Ht]. lia.
    + eapply Rle_trans; [exact Hd | apply dot_eta_mono, Hlen].
  - exact IH.
Qed.

(** The matrix-vector product: one perturbed and displaced inner product per
    row. *)
Corollary f32_mat_vec_mul_backward_mixed : forall kk mm v,
  Forall (fun row => f32_dot_fin row v f32_zero
                     /\ (List.length row <= kk)%nat) mm ->
  Forall2 (fun row out =>
     exists ts delta, List.length ts = List.length row
       /\ Forall (fun t => Rabs (t - 1) <= (1 + f32_u) ^ (S kk) - 1) ts
       /\ Rabs delta <= dot_eta kk
       /\ B2R out = Rdot_terms row v ts + delta)
    mm (f32_mat_vec_mul mm v).
Proof.
  intros kk mm v H. unfold f32_mat_vec_mul.
  apply f32_dot_map_backward_mixed, H.
Qed.

(** The tied-embedding projection of all three models. *)
Definition logits_backward_mixed_of (kk : nat)
    (emb h logits : list (list binary32)) : Prop :=
  Forall2 (fun hrow outrow =>
     Forall2 (fun wrow out =>
        exists ts delta, List.length ts = List.length hrow
          /\ Forall (fun t => Rabs (t - 1) <= (1 + f32_u) ^ (S kk) - 1) ts
          /\ Rabs delta <= dot_eta kk
          /\ B2R out = Rdot_terms hrow wrow ts + delta)
       emb outrow)
    h logits.

Lemma logits_backward_mixed : forall kk emb h,
  Forall (fun hrow => (List.length hrow <= kk)%nat
                      /\ Forall (fun wrow => f32_dot_fin hrow wrow f32_zero) emb) h ->
  logits_backward_mixed_of kk emb h
    (List.map (fun hrow => List.map (fun wrow => f32_dot hrow wrow) emb) h).
Proof.
  intros kk emb h H. unfold logits_backward_mixed_of.
  induction H as [|hrow h' [Hlen Hfin] H IH]; cbn [List.map]; constructor.
  - apply f32_dot_map_backward_mixed_r; assumption.
  - exact IH.
Qed.
