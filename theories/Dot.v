(** * The dot product's running bound, and its non-vacuity

    Float_error.v carries the forward bound through the network as an affine
    map on a single error budget. This file gives the sharper statement for the
    one primitive every linear layer and every attention score is built from:
    the dot product, bounded by a running sum over the intermediates the
    computation itself visits, rather than by a worst case iterated over depth.

    The premise is finiteness of the accumulator and absence of overflow. It
    imposes no lower bound on any intermediate, so it holds where a weight or
    an activation is exactly zero: the gated DeltaNet recurrence starts from an
    all-zero state matrix, the depthwise convolution zero-pads its window, and
    an attention score far below its row maximum has a subnormal softmax weight. Witnesses
    at an exact zero close the file, for this bound, for the backward-error
    statement and for the composed one. *)

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

Lemma f32_normal_lo_pos : 0 < f32_normal_lo.
Proof. unfold f32_normal_lo. apply bpow_gt_0. Qed.

(** The absolute term costs one eta per operation, against a bound otherwise
    measured in multiples of [u * M], so admitting zeros is nearly free. *)
Lemma f32_eta_small : f32_eta < f32_u.
Proof.
  unfold f32_eta, f32_u, f32_emin, prec32, emax32.
  apply Rmult_lt_compat_l; [lra|].
  apply bpow_lt. lia.
Qed.

(** * The premise

    Finiteness and absence of overflow at every step, and nothing else. *)

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

(** One multiply-accumulate: the two relative terms the roundings contribute,
    and one absolute term per rounding. *)
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

(** The running bound: one step's contribution plus the rest, taken at the
    accumulator the computation actually reaches. *)
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

(** * Non-vacuity at an exact zero *)

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

(** A dot product one of whose products is an exact zero satisfies the running
    bound's premise. *)
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

(** So does the premise of the backward-error statement, where rounding must be
    relative: the zero product rounds to itself. *)
Example dot_regular_with_zero :
  f32_dot_regular (f32_one :: f32_zero :: nil) (f32_one :: f32_one :: nil) f32_zero.
Proof.
  assert (Hp : B2R f32_one * B2R f32_one = 1)
    by (rewrite !f32_one_correct; ring).
  assert (Hs : B2R f32_zero + B2R (f32_mult f32_one f32_one) = 1)
    by (rewrite B2R_f32_zero, B2R_mult_one; ring).
  assert (Hpl : B2R (f32_plus f32_zero (f32_mult f32_one f32_one)) = 1).
  { rewrite (f32_plus_correct f32_zero (f32_mult f32_one f32_one));
      [ rewrite B2R_f32_zero, B2R_mult_one, Rplus_0_l; apply f32_round_1
      | reflexivity
      | exact mult_one_finite
      | rewrite B2R_f32_zero, B2R_mult_one, Rplus_0_l, f32_round_1, Rabs_R1;
        apply Rlt_bool_true, one_lt_emax ]. }
  apply dot_reg_cons.
  - reflexivity.
  - exact mult_one_finite.
  - rewrite Hp, f32_round_1, Rabs_R1. apply Rlt_bool_true, one_lt_emax.
  - rewrite Hs, f32_round_1, Rabs_R1. apply Rlt_bool_true, one_lt_emax.
  - left. rewrite Hp, Rabs_R1. apply normal_lo_le_1.
  - left. rewrite Hs, Rabs_R1. apply normal_lo_le_1.
  - assert (Hz : B2R f32_zero * B2R f32_one = 0)
      by (rewrite B2R_f32_zero; ring).
    assert (Hm : B2R (f32_mult f32_zero f32_one) = 0) by (vm_compute; reflexivity).
    apply dot_reg_cons.
    + vm_compute. reflexivity.
    + vm_compute. reflexivity.
    + rewrite Hz. apply round_zero_in_range.
    + rewrite Hm, Rplus_0_r. apply round_B2R_in_range.
    + right. exact Hz.
    + left. rewrite Hm, Rplus_0_r, Hpl, Rabs_R1. apply normal_lo_le_1.
    + apply dot_reg_nil_l.
Qed.

(** And so does the premise of the composed forward bound. *)
Example dotreg_with_zero :
  dotreg 4 (f32_one :: f32_zero :: nil) (f32_one :: f32_one :: nil) f32_zero.
Proof.
  assert (Hu : f32_u <= / 2) by apply u_le_half.
  assert (Het : 2 * f32_eta <= f32_normal_lo) by apply two_eta_le_normal_lo.
  assert (Hnl : f32_normal_lo <= 1) by apply normal_lo_le_1.
  assert (Hep : 0 < f32_eta) by apply f32_eta_pos.
  assert (Hpl : B2R (f32_plus f32_zero (f32_mult f32_one f32_one)) = 1).
  { rewrite (f32_plus_correct f32_zero (f32_mult f32_one f32_one));
      [ rewrite B2R_f32_zero, B2R_mult_one, Rplus_0_l; apply f32_round_1
      | reflexivity
      | exact mult_one_finite
      | rewrite B2R_f32_zero, B2R_mult_one, Rplus_0_l, f32_round_1, Rabs_R1;
        apply Rlt_bool_true, one_lt_emax ]. }
  apply dotreg_cons.
  - rewrite f32_one_correct, Rabs_R1. lra.
  - unfold regz. rewrite !f32_one_correct.
    replace (1 * 1) with 1 by ring. rewrite Rabs_R1. lra.
  - unfold regz. rewrite B2R_f32_zero, B2R_mult_one.
    replace (0 + 1) with 1 by ring. rewrite Rabs_R1. lra.
  - apply dotreg_cons.
    + rewrite B2R_f32_zero, Rabs_R0. lra.
    + unfold regz. rewrite B2R_f32_zero, f32_one_correct.
      replace (0 * 1) with 0 by ring. rewrite Rabs_R0. lra.
    + unfold regz.
      assert (Hm : B2R (f32_mult f32_zero f32_one) = 0) by (vm_compute; reflexivity).
      rewrite Hm, Hpl, Rplus_0_r, Rabs_R1. lra.
    + apply dotreg_nil_l.
Qed.
