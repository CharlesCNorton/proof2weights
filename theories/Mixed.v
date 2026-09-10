(** * The mixed error model, and why the old one is unsatisfiable

    Float_error.v propagates error through the relation [regz M z], which
    requires every exact intermediate to satisfy [2^-126 <= |z|]. That is the
    condition under which rounding is a pure relative perturbation. It is also
    false at zero, and the forward pass produces exact zeros structurally: the
    gated DeltaNet recurrence starts from an all-zero state matrix, the
    depthwise convolution zero-pads its window, and any zero weight or
    activation zeroes a product inside a dot product.

    Flocq supplies the standard repair. [error_N_FLT] holds for every real
    number with no range hypothesis at all: rounding is [z * (1 + d) + e] with
    [|d| <= u], [|e| <= eta], and [d * e = 0], so in the normal range it is
    relative and near zero it is absolute. This file instantiates that at
    binary32, proves the old relation unsatisfiable at zero, and gives the
    propagation lemmas for addition and multiplication in the mixed model,
    which carry no lower bound and therefore admit zeros and subnormals.

    This is a demonstration of the repair on the primitives, not a rebuild of
    the whole composition. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Lia.
From Stdlib Require Import List.
From Flocq Require Import Core.
From Flocq Require Import Relative.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Float_error.

Import ListNotations.

Open Scope R_scope.

(** * The absolute term

    [f32_eta] is half the smallest positive binary32, the largest absolute
    error rounding can introduce where the relative model breaks down. *)

Definition f32_eta : R := / 2 * bpow radix2 f32_emin.

Lemma f32_eta_pos : 0 < f32_eta.
Proof.
  unfold f32_eta. apply Rmult_lt_0_compat; [lra | apply bpow_gt_0].
Qed.

Lemma f32_normal_lo_pos : 0 < f32_normal_lo.
Proof. unfold f32_normal_lo. apply bpow_gt_0. Qed.

(** * Rounding, unconditionally

    No hypothesis on [z]. This is the statement [regz] was standing in for. *)

Theorem f32_round_mixed : forall z : R,
  exists d e,
    Rabs d <= f32_u /\ Rabs e <= f32_eta /\ d * e = 0
    /\ f32_round z = z * (1 + d) + e.
Proof.
  intros z. unfold f32_round, f32_u, f32_eta.
  change f32_fexp with (FLT_exp f32_emin prec32).
  change (round_mode mode_NE) with (Znearest (fun n => negb (Z.even n))).
  destruct (error_N_FLT radix2 f32_emin prec32 prec32_gt_0
              (fun n => negb (Z.even n)) z)
    as [d [e (Hd & He & Hde & Hr)]].
  exists d, e. repeat split; assumption.
Qed.

(** * The old relation is unsatisfiable at zero

    Not merely hard to establish: false, for every budget [M]. *)

Theorem regz_excludes_zero : forall M : R, ~ regz M 0.
Proof.
  intros M [Hlo _]. rewrite Rabs_R0 in Hlo.
  pose proof f32_normal_lo_pos. lra.
Qed.

(** A dot product with an exactly zero product cannot satisfy the regularity
    premise, so every bound resting on [dotreg] is vacuous there. A zero
    weight or a zero activation suffices. *)
Theorem dotreg_excludes_zero_product : forall M x xs y ys a,
  B2R x * B2R y = 0 -> ~ dotreg M (x :: xs) (y :: ys) a.
Proof.
  intros M x xs y ys a Hz Hd.
  inversion Hd as [| |x' xs' y' ys' a' Hbx Hz1 Hz2 Hrest]; subst.
  rewrite Hz in Hz1. apply (regz_excludes_zero M Hz1).
Qed.

(** The same for the backward-error premise. *)
Theorem f32_dot_regular_excludes_zero_product : forall x xs y ys a,
  B2R x * B2R y = 0 -> ~ f32_dot_regular (x :: xs) (y :: ys) a.
Proof.
  intros x xs y ys a Hz Hd.
  inversion Hd as [| |x' xs' y' ys' a' Hfa Hfm Hb1 Hb2 Hu1 Hu2 Hrest]; subst.
  rewrite Hz, Rabs_R0 in Hu1.
  pose proof f32_normal_lo_pos. lra.
Qed.

(** * Propagation in the mixed model

    The relation carried is the same [ok d x r]. What changes is the step: the
    per-operation cost becomes [u * M + eta] instead of [u * M], and in
    exchange the lower-bound hypothesis disappears. Overflow and finiteness
    conditions remain, because those are genuine. *)

Lemma ok_plus_mixed : forall M d x y rx ry,
  ok d x rx -> ok d y ry ->
  Rabs (B2R x + B2R y) <= M ->
  is_finite x = true -> is_finite y = true ->
  Rlt_bool (Rabs (f32_round (B2R x + B2R y))) (bpow radix2 emax32) = true ->
  is_finite (f32_plus x y) = true ->
  ok (f32_u * M + f32_eta + 2 * d) (f32_plus x y) (rx + ry).
Proof.
  intros M d x y rx ry [Hfx Hdx] [Hfy Hdy] HM Hx Hy Hb Hfin.
  split; [exact Hfin|].
  rewrite (f32_plus_correct x y Hx Hy Hb).
  destruct (f32_round_mixed (B2R x + B2R y)) as [dl [el (Hdl & Hel & _ & Hr)]].
  rewrite Hr.
  replace ((B2R x + B2R y) * (1 + dl) + el - (rx + ry))
    with ((B2R x + B2R y) * dl + el + ((B2R x - rx) + (B2R y - ry))) by ring.
  eapply Rle_trans; [apply Rabs_triang|].
  assert (H1 : Rabs ((B2R x + B2R y) * dl + el) <= f32_u * M + f32_eta).
  { eapply Rle_trans; [apply Rabs_triang|].
    apply Rplus_le_compat; [|exact Hel].
    rewrite Rabs_mult, Rmult_comm.
    apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  assert (H2 : Rabs ((B2R x - rx) + (B2R y - ry)) <= 2 * d).
  { eapply Rle_trans; [apply Rabs_triang | lra]. }
  lra.
Qed.

Lemma ok_mult_mixed : forall M d x y rx ry,
  ok d x rx -> ok d y ry ->
  Rabs (B2R x) <= M -> Rabs ry <= M ->
  Rabs (B2R x * B2R y) <= M ->
  Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true ->
  is_finite (f32_mult x y) = true ->
  ok (f32_u * M + f32_eta + 2 * M * d) (f32_mult x y) (rx * ry).
Proof.
  intros M d x y rx ry [Hfx Hdx] [Hfy Hdy] Hbx Hby HM Hb Hfin.
  split; [exact Hfin|].
  rewrite (f32_mult_correct x y Hb).
  destruct (f32_round_mixed (B2R x * B2R y)) as [dl [el (Hdl & Hel & _ & Hr)]].
  rewrite Hr.
  replace (B2R x * B2R y * (1 + dl) + el - rx * ry)
    with (B2R x * B2R y * dl + el
          + (B2R x * (B2R y - ry) + (B2R x - rx) * ry)) by ring.
  eapply Rle_trans; [apply Rabs_triang|].
  assert (H1 : Rabs (B2R x * B2R y * dl + el) <= f32_u * M + f32_eta).
  { eapply Rle_trans; [apply Rabs_triang|].
    apply Rplus_le_compat; [|exact Hel].
    rewrite Rabs_mult, Rmult_comm.
    apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  assert (H2 : Rabs (B2R x * (B2R y - ry)) <= M * d).
  { rewrite Rabs_mult. apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  assert (H3 : Rabs ((B2R x - rx) * ry) <= d * M).
  { rewrite Rabs_mult. apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  assert (H4 : Rabs (B2R x * (B2R y - ry) + (B2R x - rx) * ry) <= M * d + d * M).
  { eapply Rle_trans; [apply Rabs_triang | lra]. }
  lra.
Qed.

(** The mixed step applies where the old one cannot: at an exactly zero
    product, which [regz] rejects outright. *)
Corollary ok_mult_mixed_at_zero : forall M d x y rx ry,
  ok d x rx -> ok d y ry ->
  Rabs (B2R x) <= M -> Rabs ry <= M ->
  B2R x * B2R y = 0 ->
  0 <= M ->
  Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true ->
  is_finite (f32_mult x y) = true ->
  ok (f32_u * M + f32_eta + 2 * M * d) (f32_mult x y) (rx * ry).
Proof.
  intros M d x y rx ry Hx Hy Hbx Hby Hz HM Hb Hfin.
  eapply ok_mult_mixed; try eassumption.
  rewrite Hz, Rabs_R0. exact HM.
Qed.

(** * The cost of the repair

    One [eta] per operation, which is [2^-150], against a bound that is
    otherwise measured in multiples of [u * M]. The absolute term is
    negligible beside the relative one whenever [M] is at all large, so
    admitting zeros costs essentially nothing. *)

Lemma f32_eta_small : f32_eta < f32_u.
Proof.
  unfold f32_eta, f32_u, f32_emin, prec32, emax32.
  apply Rmult_lt_compat_l; [lra|].
  apply bpow_lt. lia.
Qed.

(** * The dot product, without a lower bound

    [f32_dot_regular] carries two conditions the mixed model does not need:
    that the product and the running sum are both at least [2^-126] in
    magnitude. Dropping them is what admits zeros. What remains is finiteness
    of the accumulator and absence of overflow, which are genuine. *)

Inductive dot_ok : list binary32 -> list binary32 -> binary32 -> Prop :=
| dok_nil_l : forall ys a, dot_ok nil ys a
| dok_nil_r : forall xs a, dot_ok xs nil a
| dok_cons : forall x xs y ys a,
    is_finite a = true ->
    is_finite (f32_mult x y) = true ->
    Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true ->
    Rlt_bool (Rabs (f32_round (B2R a + B2R (f32_mult x y)))) (bpow radix2 emax32) = true ->
    dot_ok xs ys (f32_plus a (f32_mult x y)) ->
    dot_ok (x :: xs) (y :: ys) a.

(** One multiply-accumulate, in the mixed model: the same two relative terms
    as before, plus one absolute term per rounding. *)
Theorem f32_mac_step_mixed : forall a x y : binary32,
  is_finite a = true ->
  is_finite (f32_mult x y) = true ->
  Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true ->
  Rlt_bool (Rabs (f32_round (B2R a + B2R (f32_mult x y)))) (bpow radix2 emax32) = true ->
  Rabs (B2R (f32_plus a (f32_mult x y)) - (B2R a + B2R x * B2R y))
    <= f32_u * Rabs (B2R a + B2R x * B2R y)
       + f32_u * (1 + f32_u) * Rabs (B2R x * B2R y)
       + f32_eta * (2 + f32_u).
Proof.
  intros a x y Hfa Hfm Hb1 Hb2.
  pose proof (f32_mult_correct x y Hb1) as HM.
  destruct (f32_round_mixed (B2R x * B2R y)) as [d1 [e1 (Hd1 & He1 & _ & Hr1)]].
  destruct (f32_round_mixed (B2R a + B2R (f32_mult x y)))
    as [d2 [e2 (Hd2 & He2 & _ & Hr2)]].
  rewrite (f32_plus_correct a (f32_mult x y) Hfa Hfm Hb2), Hr2, HM, Hr1.
  replace ((B2R a + (B2R x * B2R y * (1 + d1) + e1)) * (1 + d2) + e2
           - (B2R a + B2R x * B2R y))
    with ((B2R a + B2R x * B2R y) * d2
          + B2R x * B2R y * (d1 * (1 + d2))
          + e1 * (1 + d2) + e2) by ring.
  pose proof f32_u_pos as Hup.
  assert (T1 : Rabs ((B2R a + B2R x * B2R y) * d2)
               <= f32_u * Rabs (B2R a + B2R x * B2R y))
    by (apply abs_mul_le, Hd2).
  assert (T2 : Rabs (B2R x * B2R y * (d1 * (1 + d2)))
               <= f32_u * (1 + f32_u) * Rabs (B2R x * B2R y)).
  { apply abs_mul_le. rewrite Rabs_mult.
    apply Rmult_le_compat;
      [apply Rabs_pos | apply Rabs_pos | exact Hd1 | apply Rabs_1_plus, Hd2]. }
  assert (T3 : Rabs (e1 * (1 + d2)) <= f32_eta * (1 + f32_u)).
  { rewrite Rabs_mult.
    apply Rmult_le_compat;
      [apply Rabs_pos | apply Rabs_pos | exact He1 | apply Rabs_1_plus, Hd2]. }
  eapply Rle_trans; [apply Rabs_triang|].
  eapply Rle_trans; [apply Rplus_le_compat_r, Rabs_triang|].
  eapply Rle_trans; [apply Rplus_le_compat_r, Rplus_le_compat_r, Rabs_triang|].
  lra.
Qed.

Fixpoint dot_err_mixed (xs ys : list binary32) (a : binary32) : R :=
  match xs, ys with
  | x :: xs', y :: ys' =>
      f32_u * Rabs (B2R a + B2R x * B2R y)
      + f32_u * (1 + f32_u) * Rabs (B2R x * B2R y)
      + f32_eta * (2 + f32_u)
      + dot_err_mixed xs' ys' (f32_plus a (f32_mult x y))
  | _, _ => 0
  end.

Theorem f32_dot_aux_error_mixed : forall xs ys a,
  dot_ok xs ys a ->
  Rabs (B2R (f32_dot_aux xs ys a) - Rdot_aux xs ys (B2R a))
    <= dot_err_mixed xs ys a.
Proof.
  intros xs ys a Hreg.
  induction Hreg as [ys a | xs a | x xs y ys a Hfa Hfm Hb1 Hb2 Hreg IH].
  - simpl. rewrite Rminus_diag_eq by reflexivity. rewrite Rabs_R0. apply Rle_refl.
  - destruct xs as [|x xs]; simpl;
      rewrite Rminus_diag_eq by reflexivity; rewrite Rabs_R0; apply Rle_refl.
  - simpl.
    set (a' := f32_plus a (f32_mult x y)) in *.
    replace (B2R (f32_dot_aux xs ys a') - Rdot_aux xs ys (B2R a + B2R x * B2R y))
      with ((B2R (f32_dot_aux xs ys a') - Rdot_aux xs ys (B2R a'))
            + (Rdot_aux xs ys (B2R a') - Rdot_aux xs ys (B2R a + B2R x * B2R y)))
      by ring.
    eapply Rle_trans; [apply Rabs_triang|].
    rewrite (Rdot_aux_shift xs ys (B2R a')) in IH |- *.
    rewrite (Rdot_aux_shift xs ys (B2R a + B2R x * B2R y)).
    replace (B2R a' + Rdot_aux xs ys 0
             - (B2R a + B2R x * B2R y + Rdot_aux xs ys 0))
      with (B2R a' - (B2R a + B2R x * B2R y)) by ring.
    pose proof (f32_mac_step_mixed a x y Hfa Hfm Hb1 Hb2) as Hstep.
    fold a' in Hstep. lra.
Qed.

Corollary f32_dot_error_mixed : forall xs ys,
  dot_ok xs ys f32_zero ->
  Rabs (B2R (f32_dot xs ys) - Rdot_aux xs ys 0) <= dot_err_mixed xs ys f32_zero.
Proof.
  intros xs ys Hreg.
  pose proof (f32_dot_aux_error_mixed xs ys f32_zero Hreg) as H.
  rewrite B2R_f32_zero in H. exact H.
Qed.

(** * Non-vacuity where the old model was vacuous

    A dot product one of whose entries is an exact zero. [f32_dot_regular]
    rejects it outright by [dotreg_excludes_zero_product]; the mixed premise
    holds. *)

Lemma round_zero_in_range :
  Rlt_bool (Rabs (f32_round 0)) (bpow radix2 emax32) = true.
Proof.
  unfold f32_round. rewrite round_0 by apply valid_rnd_round_mode.
  rewrite Rabs_R0. apply Rlt_bool_true, bpow_gt_0.
Qed.

(** Rounding a value that is already a binary32 cannot overflow. *)
Lemma round_B2R_in_range : forall x : binary32,
  Rlt_bool (Rabs (f32_round (B2R x))) (bpow radix2 emax32) = true.
Proof.
  intros x. unfold f32_round.
  rewrite round_generic;
    [apply Rlt_bool_true, abs_B2R_lt_emax
    | apply valid_rnd_round_mode
    | apply generic_format_B2R].
Qed.

Example dot_ok_with_zero :
  dot_ok (f32_one :: f32_zero :: nil) (f32_one :: f32_one :: nil) f32_zero.
Proof.
  assert (Hp : B2R f32_one * B2R f32_one = 1)
    by (rewrite !f32_one_correct; ring).
  assert (Hs : B2R f32_zero + B2R (f32_mult f32_one f32_one) = 1)
    by (rewrite B2R_f32_zero, B2R_mult_one; ring).
  apply dok_cons.
  - reflexivity.
  - exact mult_one_finite.
  - rewrite Hp, f32_round_1, Rabs_R1. apply Rlt_bool_true, one_lt_emax.
  - rewrite Hs, f32_round_1, Rabs_R1. apply Rlt_bool_true, one_lt_emax.
  - assert (Hz : B2R f32_zero * B2R f32_one = 0)
      by (rewrite B2R_f32_zero; ring).
    apply dok_cons.
    + vm_compute. reflexivity.
    + vm_compute. reflexivity.
    + rewrite Hz. apply round_zero_in_range.
    + assert (Hm : B2R (f32_mult f32_zero f32_one) = 0) by (vm_compute; reflexivity).
      rewrite Hm, Rplus_0_r. apply round_B2R_in_range.
    + apply dok_nil_l.
Qed.

(** And the old premise indeed fails on the same input. *)
Example dot_regular_rejects_zero :
  ~ dotreg 1 (f32_zero :: nil) (f32_one :: nil) f32_zero.
Proof.
  apply dotreg_excludes_zero_product.
  rewrite B2R_f32_zero. ring.
Qed.
