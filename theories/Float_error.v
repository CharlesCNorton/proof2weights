(** * Numerical semantics of the extracted float arithmetic

    The shape and serialization theorems say nothing about numerical behaviour.
    This file says what the extracted floating-point computation actually
    computes, in four steps that build on one another.

    - Per operation. Each extracted primitive returns the round-to-nearest,
      ties-to-even result of the exact real operation on its operands, and
      differs from that exact result by at most half a unit in the last place.

    - The native build's rounding step. For binary32 operands, rounding the
      exact real result of [+], [-], [*], [/] or [sqrt] to binary64 and then to
      binary32 gives the same value as rounding it straight to binary32, which
      is what makes Extract.v's hardware-float path compute the binary32 value
      Flocq specifies.

    - The dot product. Composing the per-operation facts along [f32_dot] bounds
      its distance from the exact real inner product by a running sum over the
      intermediates the computation itself visits.

    - The forward pass. The same composition carried through every remaining
      stage, to [f32_gpt2_logits]. Every scalar the network computes comes from
      one of five primitives and everything else is list plumbing that performs
      no arithmetic, so one propagation lemma per primitive plus the structural
      lemmas reaches the top.

    Every hypothesis the bounds rest on is explicit: magnitudes stay under [M],
    denominators and radicands stay above [m], no intermediate falls below the
    smallest normal binary32 magnitude, and the exponential's saturation is not
    engaged, so the float and the real evaluation follow the same path. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Lia.
From Stdlib Require Import List.
From Flocq Require Import Core.
From Flocq Require Import Relative.
From Flocq Require Import Double_rounding.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Llama.
Require Import Qwen.

Import ListNotations.
Open Scope R_scope.

(** * Correct rounding, per operation *)

(** The binary32 exponent function and round-to-nearest-even rounding, as used
    by Flocq's correctness lemmas. *)
Definition f32_fexp := SpecFloat.fexp prec32 emax32.
Definition f32_round (z : R) : R := round radix2 f32_fexp (round_mode mode_NE) z.

(** Multiplication is correctly rounded when the rounded product is in range. *)
Theorem f32_mult_correct : forall x y : binary32,
  Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true ->
  B2R (f32_mult x y) = f32_round (B2R x * B2R y).
Proof.
  intros x y Hb. unfold f32_mult. unfold f32_round, f32_fexp in *.
  pose proof (Bmult_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y) as H.
  rewrite Hb in H. destruct H as [HB _]. exact HB.
Qed.

(** Addition is correctly rounded when the rounded sum is in range. *)
Theorem f32_plus_correct : forall x y : binary32,
  is_finite x = true -> is_finite y = true ->
  Rlt_bool (Rabs (f32_round (B2R x + B2R y))) (bpow radix2 emax32) = true ->
  B2R (f32_plus x y) = f32_round (B2R x + B2R y).
Proof.
  intros x y Hx Hy Hb. unfold f32_plus. unfold f32_round, f32_fexp in *.
  pose proof (Bplus_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y Hx Hy) as H.
  rewrite Hb in H. destruct H as [HB _]. exact HB.
Qed.

(** Division is correctly rounded for a nonzero divisor when the rounded
    quotient is in range. *)
Theorem f32_div_correct : forall x y : binary32,
  B2R y <> 0 ->
  Rlt_bool (Rabs (f32_round (B2R x / B2R y))) (bpow radix2 emax32) = true ->
  B2R (f32_div x y) = f32_round (B2R x / B2R y).
Proof.
  intros x y Hy Hb. unfold f32_div. unfold f32_round, f32_fexp in *.
  pose proof (Bdiv_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y Hy) as H.
  rewrite Hb in H. destruct H as [HB _]. exact HB.
Qed.

(** Square root is correctly rounded (it cannot overflow). *)
Theorem f32_sqrt_correct : forall x : binary32,
  B2R (f32_sqrt x) = f32_round (sqrt (B2R x)).
Proof.
  intros x. unfold f32_sqrt. unfold f32_round, f32_fexp.
  pose proof (Bsqrt_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x) as H.
  destruct H as [HB _]. exact HB.
Qed.

(** * Numerical accuracy: each primitive is within half a ULP of the exact result.

    Correct rounding is qualitative; this is the quantitative form. The
    round-to-nearest result differs from the exact real value by at most half a
    unit in the last place, so every operation in the forward pass lands within
    half a ULP of the exact real operation. This is the per-operation numerical
    error bound; the forward pass is their composition. *)

Definition f32_ulp (z : R) : R := ulp radix2 f32_fexp z.

Lemma f32_fexp_valid : Valid_exp f32_fexp.
Proof.
  unfold f32_fexp, SpecFloat.fexp. apply FLT_exp_valid. unfold Prec_gt_0, prec32. lia.
Qed.

Theorem f32_mult_error : forall x y : binary32,
  Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true ->
  Rabs (B2R (f32_mult x y) - B2R x * B2R y) <= / 2 * f32_ulp (B2R x * B2R y).
Proof.
  intros x y H. rewrite (f32_mult_correct x y H). unfold f32_round, f32_ulp.
  apply error_le_half_ulp. apply f32_fexp_valid.
Qed.

Theorem f32_plus_error : forall x y : binary32,
  is_finite x = true -> is_finite y = true ->
  Rlt_bool (Rabs (f32_round (B2R x + B2R y))) (bpow radix2 emax32) = true ->
  Rabs (B2R (f32_plus x y) - (B2R x + B2R y)) <= / 2 * f32_ulp (B2R x + B2R y).
Proof.
  intros x y Hx Hy H. rewrite (f32_plus_correct x y Hx Hy H). unfold f32_round, f32_ulp.
  apply error_le_half_ulp. apply f32_fexp_valid.
Qed.

Theorem f32_div_error : forall x y : binary32,
  B2R y <> 0 ->
  Rlt_bool (Rabs (f32_round (B2R x / B2R y))) (bpow radix2 emax32) = true ->
  Rabs (B2R (f32_div x y) - B2R x / B2R y) <= / 2 * f32_ulp (B2R x / B2R y).
Proof.
  intros x y Hy H. rewrite (f32_div_correct x y Hy H). unfold f32_round, f32_ulp.
  apply error_le_half_ulp. apply f32_fexp_valid.
Qed.

Theorem f32_sqrt_error : forall x : binary32,
  Rabs (B2R (f32_sqrt x) - sqrt (B2R x)) <= / 2 * f32_ulp (sqrt (B2R x)).
Proof.
  intros x. rewrite (f32_sqrt_correct x). unfold f32_round, f32_ulp.
  apply error_le_half_ulp. apply f32_fexp_valid.
Qed.

(** * The composed bound for the dot product *)


(** * The rounding model

    [f32_u] is the unit roundoff of binary32, and [f32_normal_lo] is the
    smallest magnitude at which round-to-nearest is guaranteed to be a relative
    perturbation, that is, the bottom of the normal range. *)

Definition f32_emin : Z := (3 - emax32 - prec32)%Z.
Definition f32_u : R := / 2 * bpow radix2 (- prec32 + 1).
Definition f32_normal_lo : R := bpow radix2 (f32_emin + prec32 - 1).

Lemma f32_u_pos : 0 < f32_u.
Proof.
  unfold f32_u. apply Rmult_lt_0_compat; [lra | apply bpow_gt_0].
Qed.

(** Rounding a normal-range value is multiplication by [1 + e] with
    [|e| <= f32_u]. *)
Lemma f32_round_rel : forall z : R,
  f32_normal_lo <= Rabs z ->
  exists e, Rabs e <= f32_u /\ f32_round z = z * (1 + e).
Proof.
  intros z Hz. unfold f32_round, f32_u, f32_normal_lo in *.
  change f32_fexp with (FLT_exp f32_emin prec32).
  change (round_mode mode_NE) with (Znearest (fun n => negb (Z.even n))).
  apply (relative_error_N_FLT_ex radix2 f32_emin prec32 prec32_gt_0 _ z Hz).
Qed.

Lemma Rabs_1_plus : forall e, Rabs e <= f32_u -> Rabs (1 + e) <= 1 + f32_u.
Proof.
  intros e He.
  eapply Rle_trans; [apply Rabs_triang|]. rewrite Rabs_R1. lra.
Qed.

Lemma abs_mul_le : forall z e c, Rabs e <= c -> Rabs (z * e) <= c * Rabs z.
Proof.
  intros z e c He. rewrite Rabs_mult, Rmult_comm.
  apply Rmult_le_compat_r; [apply Rabs_pos | exact He].
Qed.

(** * One multiply-accumulate step

    The two roundings a step performs move the result away from
    [B2R a + B2R x * B2R y] by at most one unit roundoff of the sum plus one,
    slightly inflated, of the product. *)

Theorem f32_mac_step_error : forall a x y : binary32,
  is_finite a = true ->
  is_finite (f32_mult x y) = true ->
  Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true ->
  Rlt_bool (Rabs (f32_round (B2R a + B2R (f32_mult x y)))) (bpow radix2 emax32) = true ->
  f32_normal_lo <= Rabs (B2R x * B2R y) ->
  f32_normal_lo <= Rabs (B2R a + B2R (f32_mult x y)) ->
  Rabs (B2R (f32_plus a (f32_mult x y)) - (B2R a + B2R x * B2R y))
    <= f32_u * Rabs (B2R a + B2R x * B2R y)
       + f32_u * (1 + f32_u) * Rabs (B2R x * B2R y).
Proof.
  intros a x y Hfa Hfm Hb1 Hb2 Hu1 Hu2.
  pose proof (f32_mult_correct x y Hb1) as HM.
  destruct (f32_round_rel _ Hu1) as [e1 [He1 Hr1]].
  destruct (f32_round_rel _ Hu2) as [e2 [He2 Hr2]].
  rewrite (f32_plus_correct a (f32_mult x y) Hfa Hfm Hb2), Hr2, HM, Hr1.
  replace ((B2R a + B2R x * B2R y * (1 + e1)) * (1 + e2)
             - (B2R a + B2R x * B2R y))
    with ((B2R a + B2R x * B2R y) * e2
            + (B2R x * B2R y) * (e1 * (1 + e2))) by ring.
  eapply Rle_trans; [apply Rabs_triang|].
  apply Rplus_le_compat.
  - apply abs_mul_le, He2.
  - apply abs_mul_le.
    rewrite Rabs_mult.
    apply Rmult_le_compat;
      [apply Rabs_pos | apply Rabs_pos | exact He1 | apply Rabs_1_plus, He2].
Qed.

(** * The exact inner product, and the regularity premise *)

(** The real accumulation [f32_dot_aux] approximates, in the same order. *)
Fixpoint Rdot_aux (xs ys : list binary32) (acc : R) : R :=
  match xs, ys with
  | x :: xs', y :: ys' => Rdot_aux xs' ys' (acc + B2R x * B2R y)
  | _, _ => acc
  end.

Lemma Rdot_aux_shift : forall xs ys c,
  Rdot_aux xs ys c = c + Rdot_aux xs ys 0.
Proof.
  induction xs as [|x xs IH]; intros ys c.
  - simpl. ring.
  - destruct ys as [|y ys]; simpl.
    + ring.
    + rewrite (IH ys (c + B2R x * B2R y)), (IH ys (0 + B2R x * B2R y)). ring.
Qed.

(** Every step runs where the correctness lemmas apply: finite accumulator,
    no overflow in either operation, and no underflow in either. *)
Inductive f32_dot_regular : list binary32 -> list binary32 -> binary32 -> Prop :=
| dot_reg_nil_l : forall ys a, f32_dot_regular nil ys a
| dot_reg_nil_r : forall xs a, f32_dot_regular xs nil a
| dot_reg_cons : forall x xs y ys a,
    is_finite a = true ->
    is_finite (f32_mult x y) = true ->
    Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true ->
    Rlt_bool (Rabs (f32_round (B2R a + B2R (f32_mult x y)))) (bpow radix2 emax32) = true ->
    f32_normal_lo <= Rabs (B2R x * B2R y) ->
    f32_normal_lo <= Rabs (B2R a + B2R (f32_mult x y)) ->
    f32_dot_regular xs ys (f32_plus a (f32_mult x y)) ->
    f32_dot_regular (x :: xs) (y :: ys) a.

(** The running bound: one step's contribution plus the rest, taken at the
    accumulator the computation actually reaches. *)
Fixpoint f32_dot_err_bound (xs ys : list binary32) (a : binary32) : R :=
  match xs, ys with
  | x :: xs', y :: ys' =>
      f32_u * Rabs (B2R a + B2R x * B2R y)
      + f32_u * (1 + f32_u) * Rabs (B2R x * B2R y)
      + f32_dot_err_bound xs' ys' (f32_plus a (f32_mult x y))
  | _, _ => 0
  end.

Lemma f32_dot_err_bound_nonneg : forall xs ys a,
  0 <= f32_dot_err_bound xs ys a.
Proof.
  pose proof f32_u_pos as Hu.
  induction xs as [|x xs IH]; intros ys a.
  - simpl. lra.
  - destruct ys as [|y ys]; simpl; [lra|].
    specialize (IH ys (f32_plus a (f32_mult x y))).
    pose proof (Rabs_pos (B2R a + B2R x * B2R y)).
    pose proof (Rabs_pos (B2R x * B2R y)).
    assert (0 <= f32_u * Rabs (B2R a + B2R x * B2R y)) by (apply Rmult_le_pos; lra).
    assert (0 <= f32_u * (1 + f32_u) * Rabs (B2R x * B2R y))
      by (apply Rmult_le_pos; [apply Rmult_le_pos|]; lra).
    lra.
Qed.

(** * The composed bound *)

Theorem f32_dot_aux_error : forall xs ys a,
  f32_dot_regular xs ys a ->
  Rabs (B2R (f32_dot_aux xs ys a) - Rdot_aux xs ys (B2R a))
    <= f32_dot_err_bound xs ys a.
Proof.
  intros xs ys a Hreg.
  induction Hreg as [ys a | xs a | x xs y ys a Hfa Hfm Hb1 Hb2 Hu1 Hu2 Hreg IH].
  - simpl. rewrite Rminus_diag_eq by reflexivity. rewrite Rabs_R0. apply Rle_refl.
  - destruct xs as [|x xs]; simpl;
      rewrite Rminus_diag_eq by reflexivity; rewrite Rabs_R0; apply Rle_refl.
  - simpl.
    set (a' := f32_plus a (f32_mult x y)) in *.
    (* Split into the tail's error at a', and the shift from a' to the exact
       partial sum, which is exactly this step's error. *)
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
    pose proof (f32_mac_step_error a x y Hfa Hfm Hb1 Hb2 Hu1 Hu2) as Hstep.
    fold a' in Hstep.
    lra.
Qed.

Lemma B2R_f32_zero : B2R f32_zero = 0.
Proof. reflexivity. Qed.

(** [f32_dot] starts from zero, so the exact reference is the plain inner
    product of the operands' real values. *)
Corollary f32_dot_error : forall xs ys,
  f32_dot_regular xs ys f32_zero ->
  Rabs (B2R (f32_dot xs ys) - Rdot_aux xs ys 0)
    <= f32_dot_err_bound xs ys f32_zero.
Proof.
  intros xs ys Hreg.
  pose proof (f32_dot_aux_error xs ys f32_zero Hreg) as H.
  rewrite B2R_f32_zero in H. exact H.
Qed.

(** * The premise is satisfiable

    A bound guarded by an unsatisfiable hypothesis says nothing, so a witness
    is exhibited: the one-term dot product of ones runs entirely inside the
    regular regime. *)

Lemma f32_round_1 : f32_round 1 = 1.
Proof.
  unfold f32_round.
  replace 1 with (bpow radix2 0) by reflexivity.
  apply round_generic; [typeclasses eauto|].
  apply generic_format_bpow.
  unfold f32_fexp, SpecFloat.fexp, SpecFloat.emin, prec32, emax32. lia.
Qed.

Lemma one_lt_emax : 1 < bpow radix2 emax32.
Proof.
  replace 1 with (bpow radix2 0) by reflexivity.
  apply bpow_lt. unfold emax32. lia.
Qed.

Lemma normal_lo_le_1 : f32_normal_lo <= 1.
Proof.
  unfold f32_normal_lo.
  replace 1 with (bpow radix2 0) by reflexivity.
  apply bpow_le. unfold f32_emin, prec32, emax32. lia.
Qed.

Lemma mult_one_finite : is_finite (f32_mult f32_one f32_one) = true.
Proof. vm_compute. reflexivity. Qed.

Lemma B2R_mult_one : B2R (f32_mult f32_one f32_one) = 1.
Proof.
  rewrite (f32_mult_correct f32_one f32_one).
  - rewrite !f32_one_correct, Rmult_1_l. apply f32_round_1.
  - rewrite !f32_one_correct, Rmult_1_l, f32_round_1, Rabs_R1.
    apply Rlt_bool_true, one_lt_emax.
Qed.

Example f32_dot_regular_ones :
  f32_dot_regular (f32_one :: nil) (f32_one :: nil) f32_zero.
Proof.
  assert (Hprod : B2R f32_one * B2R f32_one = 1)
    by (rewrite !f32_one_correct; ring).
  assert (Hsum : B2R f32_zero + B2R (f32_mult f32_one f32_one) = 1)
    by (rewrite B2R_f32_zero, B2R_mult_one; ring).
  apply dot_reg_cons.
  - reflexivity.
  - exact mult_one_finite.
  - rewrite Hprod, f32_round_1, Rabs_R1. apply Rlt_bool_true, one_lt_emax.
  - rewrite Hsum, f32_round_1, Rabs_R1. apply Rlt_bool_true, one_lt_emax.
  - rewrite Hprod, Rabs_R1. apply normal_lo_le_1.
  - rewrite Hsum, Rabs_R1. apply normal_lo_le_1.
  - apply dot_reg_nil_l.
Qed.

(** * Double rounding through binary64 is harmless

    Stated over exponents, so this section works in [Z_scope]. *)

Open Scope Z_scope.


(** * The two formats

    [f32_emin] is the exponent floor [BinarySingleNaN] uses for [binary32], so
    that [generic_format_B2R] lands in exactly this format. *)

Definition prec64 : Z := 53.
Definition emax64 : Z := 1024.
Definition emin64 : Z := 3 - emax64 - prec64.

(** The parameters the conditions below are checked against. *)
Example prec32_value : prec32 = 24 := eq_refl.
Example f32_emin_value : f32_emin = -149 := eq_refl.
Example prec64_value : prec64 = 53 := eq_refl.
Example emin64_value : emin64 = -1074 := eq_refl.

#[local] Instance prec32_gt_0_inst : Prec_gt_0 prec32.
Proof. exact prec32_gt_0. Qed.

#[local] Instance prec64_gt_0_inst : Prec_gt_0 prec64.
Proof. unfold Prec_gt_0, prec64. lia. Qed.

Notation b32 := (FLT_exp f32_emin prec32).
Notation b64 := (FLT_exp emin64 prec64).

(** Every side condition Flocq's double-rounding theorems raise is either a
    positivity fact about a precision, the radix being even, or an inequality
    between the exponent parameters. *)
Ltac dr_side :=
  solve [ assumption
        | exact prec32_gt_0_inst
        | exact prec64_gt_0_inst
        | exists 1%Z; reflexivity
        | left; unfold f32_emin, emin64, prec32, prec64, emax32, emax64; lia
        | unfold f32_emin, emin64, prec32, prec64, emax32, emax64; lia ].

(** Every binary32 value is representable in the binary32 format. *)
Lemma FLT_format_B2R : forall x : binary32,
  FLT_format radix2 f32_emin prec32 (B2R x).
Proof.
  intros x.
  apply FLT_format_generic; solve [ apply generic_format_B2R | dr_side ].
Qed.

(** * Double rounding is harmless at these two formats

    binary64 carries 53 bits of significand against binary32's 24, which
    satisfies every width condition Flocq's theorems require: [2*24+1 <= 53]
    for addition, [2*24 <= 53] for multiplication and division, and
    [2*24+2 <= 53] for square root. The exponent conditions hold because
    [emin64 = -1074] sits far below [f32_emin = -149]. *)

Theorem double_round_plus : forall choice1 choice2 x y,
  FLT_format radix2 f32_emin prec32 x ->
  FLT_format radix2 f32_emin prec32 y ->
  round radix2 b32 (Znearest choice1) (round radix2 b64 (Znearest choice2) (x + y))
  = round radix2 b32 (Znearest choice1) (x + y).
Proof.
  intros choice1 choice2 x y Fx Fy.
  apply round_round_plus_FLT; dr_side.
Qed.

Theorem double_round_minus : forall choice1 choice2 x y,
  FLT_format radix2 f32_emin prec32 x ->
  FLT_format radix2 f32_emin prec32 y ->
  round radix2 b32 (Znearest choice1) (round radix2 b64 (Znearest choice2) (x - y))
  = round radix2 b32 (Znearest choice1) (x - y).
Proof.
  intros choice1 choice2 x y Fx Fy.
  apply round_round_minus_FLT; dr_side.
Qed.

Theorem double_round_mult : forall (rnd : R -> Z) {Vr : Valid_rnd rnd} x y,
  FLT_format radix2 f32_emin prec32 x ->
  FLT_format radix2 f32_emin prec32 y ->
  round radix2 b32 rnd (round radix2 b64 rnd (x * y))
  = round radix2 b32 rnd (x * y).
Proof.
  intros rnd Vr x y Fx Fy.
  apply round_round_mult_FLT; dr_side.
Qed.

Theorem double_round_div : forall choice1 choice2 x y,
  y <> 0%R ->
  FLT_format radix2 f32_emin prec32 x ->
  FLT_format radix2 f32_emin prec32 y ->
  round radix2 b32 (Znearest choice1) (round radix2 b64 (Znearest choice2) (x / y))
  = round radix2 b32 (Znearest choice1) (x / y).
Proof.
  intros choice1 choice2 x y Hy Fx Fy.
  apply round_round_div_FLT; dr_side.
Qed.

Theorem double_round_sqrt : forall choice1 choice2 x,
  FLT_format radix2 f32_emin prec32 x ->
  round radix2 b32 (Znearest choice1) (round radix2 b64 (Znearest choice2) (sqrt x))
  = round radix2 b32 (Znearest choice1) (sqrt x).
Proof.
  intros choice1 choice2 x Fx.
  apply round_round_sqrt_FLT; dr_side.
Qed.

(** * The same statements on the development's own values

    The operands the native build feeds to a rounding step are [binary32]
    values, so these are the forms that apply directly. *)

Corollary f32_double_round_plus : forall choice1 choice2 (x y : binary32),
  round radix2 b32 (Znearest choice1)
    (round radix2 b64 (Znearest choice2) (B2R x + B2R y))
  = round radix2 b32 (Znearest choice1) (B2R x + B2R y).
Proof.
  intros. apply double_round_plus; apply FLT_format_B2R.
Qed.

Corollary f32_double_round_minus : forall choice1 choice2 (x y : binary32),
  round radix2 b32 (Znearest choice1)
    (round radix2 b64 (Znearest choice2) (B2R x - B2R y))
  = round radix2 b32 (Znearest choice1) (B2R x - B2R y).
Proof.
  intros. apply double_round_minus; apply FLT_format_B2R.
Qed.

Corollary f32_double_round_mult : forall choice (x y : binary32),
  round radix2 b32 (Znearest choice)
    (round radix2 b64 (Znearest choice) (B2R x * B2R y))
  = round radix2 b32 (Znearest choice) (B2R x * B2R y).
Proof.
  intros.
  apply double_round_mult; solve [ apply FLT_format_B2R | apply valid_rnd_N ].
Qed.

Corollary f32_double_round_div : forall choice1 choice2 (x y : binary32),
  B2R y <> 0%R ->
  round radix2 b32 (Znearest choice1)
    (round radix2 b64 (Znearest choice2) (B2R x / B2R y))
  = round radix2 b32 (Znearest choice1) (B2R x / B2R y).
Proof.
  intros choice1 choice2 x y Hy.
  apply double_round_div; try assumption; apply FLT_format_B2R.
Qed.

Corollary f32_double_round_sqrt : forall choice1 choice2 (x : binary32),
  round radix2 b32 (Znearest choice1)
    (round radix2 b64 (Znearest choice2) (sqrt (B2R x)))
  = round radix2 b32 (Znearest choice1) (sqrt (B2R x)).
Proof.
  intros. apply double_round_sqrt. apply FLT_format_B2R.
Qed.

Close Scope Z_scope.

(** * The composed bound for the whole forward pass *)


(** * Rounding, in the form the propagation needs *)

(** A rounded value is no larger than the exact one, up to one roundoff. *)
Lemma round_abs_le : forall z,
  f32_normal_lo <= Rabs z -> Rabs (f32_round z) <= Rabs z * (1 + f32_u).
Proof.
  intros z Hz. destruct (f32_round_rel z Hz) as [e [He Hr]].
  rewrite Hr, Rabs_mult.
  apply Rmult_le_compat_l; [apply Rabs_pos | apply Rabs_1_plus, He].
Qed.

(** [regz M z]: the exact result [z] of an operation is in the normal range and
    small enough that rounding it cannot overflow. *)
Definition regz (M z : R) : Prop :=
  f32_normal_lo <= Rabs z /\ Rabs z * (1 + f32_u) <= M.

Lemma regz_abs_le : forall M z, regz M z -> Rabs z <= M.
Proof.
  intros M z [_ Hb]. pose proof f32_u_pos.
  pose proof (Rabs_pos z). nra.
Qed.

Lemma no_overflow : forall M z,
  M < bpow radix2 emax32 -> regz M z ->
  Rlt_bool (Rabs (f32_round z)) (bpow radix2 emax32) = true.
Proof.
  intros M z HM [Hn Hb]. apply Rlt_bool_true.
  eapply Rle_lt_trans; [apply round_abs_le; exact Hn | lra].
Qed.

Lemma round_err_le : forall M z,
  regz M z -> Rabs (f32_round z - z) <= f32_u * M.
Proof.
  intros M z Hr. destruct Hr as [Hn Hb].
  destruct (f32_round_rel z Hn) as [e [He Hq]].
  rewrite Hq. replace (z * (1 + e) - z) with (z * e) by ring.
  rewrite Rabs_mult, Rmult_comm.
  apply Rmult_le_compat; try apply Rabs_pos; [exact He|].
  pose proof f32_u_pos. pose proof (Rabs_pos z). nra.
Qed.

(** * The propagation relation *)

Definition ok (d : R) (x : binary32) (r : R) : Prop :=
  is_finite x = true /\ Rabs (B2R x - r) <= d.

Lemma ok_weaken : forall d d' x r, ok d x r -> d <= d' -> ok d' x r.
Proof. intros d d' x r [Hf Hd] Hle. split; [exact Hf | lra]. Qed.

Lemma ok_exact : forall x, is_finite x = true -> ok 0 x (B2R x).
Proof.
  intros x Hf. split; [exact Hf|].
  replace (B2R x - B2R x) with 0 by ring. rewrite Rabs_R0. apply Rle_refl.
Qed.

(** [ok] carries a magnitude bound on the real reference back to the float. *)
Lemma ok_abs : forall d x r M,
  ok d x r -> Rabs r <= M -> Rabs (B2R x) <= M + d.
Proof.
  intros d x r M [_ Hd] Hr.
  replace (B2R x) with ((B2R x - r) + r) by ring.
  eapply Rle_trans; [apply Rabs_triang | lra].
Qed.

(** * The amplification budget

    [L] bounds how much each primitive can magnify an incoming error: two for
    addition, [2 * M] for multiplication, and the derivative bounds for
    division and square root, which is where the lower bound [m] on
    denominators and radicands enters. *)

Definition amp_ok (M m L : R) : Prop :=
  0 < m /\ 1 <= M /\
  2 <= L /\
  2 * M <= L /\
  1 / m + M / (m * m) <= L /\
  1 / (2 * sqrt m) <= L.

Lemma amp_L_pos : forall M m L, amp_ok M m L -> 1 <= L.
Proof. intros M m L (_ & _ & H & _). lra. Qed.

(** * Primitive propagation *)

Lemma ok_plus : forall M m L d x y rx ry,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok d x rx -> ok d y ry ->
  regz M (B2R x + B2R y) ->
  ok (f32_u * M + L * d) (f32_plus x y) (rx + ry).
Proof.
  intros M m L d x y rx ry HM Hamp Hx Hy Hz.
  assert (HL : 2 <= L) by (destruct Hamp as (_ & _ & H & _); exact H).
  destruct Hx as [Hfx Hdx]. destruct Hy as [Hfy Hdy].
  assert (Hbool : Rlt_bool (Rabs (f32_round (B2R x + B2R y))) (bpow radix2 emax32) = true)
    by (eapply no_overflow; eassumption).
  assert (Hd0 : 0 <= d) by (pose proof (Rabs_pos (B2R x - rx)); lra).
  split.
  - unfold f32_plus.
    pose proof (Bplus_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32
                  mode_NE x y Hfx Hfy) as H.
    unfold f32_round, f32_fexp in Hbool. rewrite Hbool in H.
    destruct H as [_ [Hf _]]. exact Hf.
  - rewrite (f32_plus_correct x y Hfx Hfy Hbool).
    replace (f32_round (B2R x + B2R y) - (rx + ry))
      with ((f32_round (B2R x + B2R y) - (B2R x + B2R y))
            + ((B2R x - rx) + (B2R y - ry))) by ring.
    eapply Rle_trans; [apply Rabs_triang|].
    assert (H1 : Rabs (f32_round (B2R x + B2R y) - (B2R x + B2R y)) <= f32_u * M)
      by (eapply round_err_le; eassumption).
    assert (H2 : Rabs ((B2R x - rx) + (B2R y - ry)) <= d + d)
      by (eapply Rle_trans; [apply Rabs_triang | lra]).
    nra.
Qed.

Lemma ok_neg : forall d x r, ok d x r -> ok d (f32_neg x) (- r).
Proof.
  intros d x r [Hf Hd]. split.
  - unfold f32_neg. rewrite is_finite_Bopp. exact Hf.
  - unfold f32_neg. rewrite B2R_Bopp.
    replace (- B2R x - - r) with (- (B2R x - r)) by ring.
    rewrite Rabs_Ropp. exact Hd.
Qed.

Lemma ok_minus : forall M m L d x y rx ry,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok d x rx -> ok d y ry ->
  regz M (B2R x + B2R (f32_neg y)) ->
  ok (f32_u * M + L * d) (f32_minus x y) (rx - ry).
Proof.
  intros M m L d x y rx ry HM Hamp Hx Hy Hz.
  unfold f32_minus.
  replace (rx - ry) with (rx + - ry) by ring.
  eapply ok_plus; try eassumption. apply ok_neg, Hy.
Qed.

Lemma ok_mult : forall M m L d x y rx ry,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok d x rx -> ok d y ry ->
  Rabs (B2R x) <= M -> Rabs ry <= M ->
  regz M (B2R x * B2R y) ->
  ok (f32_u * M + L * d) (f32_mult x y) (rx * ry).
Proof.
  intros M m L d x y rx ry HM Hamp Hx Hy Hbx Hby Hz.
  assert (HL : 2 * M <= L) by (destruct Hamp as (_ & _ & _ & H & _); exact H).
  destruct Hx as [Hfx Hdx]. destruct Hy as [Hfy Hdy].
  assert (Hbool : Rlt_bool (Rabs (f32_round (B2R x * B2R y))) (bpow radix2 emax32) = true)
    by (eapply no_overflow; eassumption).
  assert (Hd0 : 0 <= d) by (pose proof (Rabs_pos (B2R x - rx)); lra).
  split.
  - unfold f32_mult.
    pose proof (Bmult_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32
                  mode_NE x y) as H.
    unfold f32_round, f32_fexp in Hbool. rewrite Hbool in H.
    destruct H as [_ [Hf _]]. rewrite Hfx, Hfy in Hf. simpl in Hf. exact Hf.
  - rewrite (f32_mult_correct x y Hbool).
    replace (f32_round (B2R x * B2R y) - rx * ry)
      with ((f32_round (B2R x * B2R y) - B2R x * B2R y)
            + (B2R x * (B2R y - ry) + (B2R x - rx) * ry)) by ring.
    eapply Rle_trans; [apply Rabs_triang|].
    assert (H1 : Rabs (f32_round (B2R x * B2R y) - B2R x * B2R y) <= f32_u * M)
      by (eapply round_err_le; eassumption).
    assert (H2 : Rabs (B2R x * (B2R y - ry)) <= M * d).
    { rewrite Rabs_mult. apply Rmult_le_compat; try apply Rabs_pos; assumption. }
    assert (H3 : Rabs ((B2R x - rx) * ry) <= d * M).
    { rewrite Rabs_mult. apply Rmult_le_compat; try apply Rabs_pos; assumption. }
    assert (H4 : Rabs (B2R x * (B2R y - ry) + (B2R x - rx) * ry) <= M * d + d * M)
      by (eapply Rle_trans; [apply Rabs_triang | lra]).
    nra.
Qed.

Lemma abs_div_le : forall a b c k,
  0 < k -> k <= Rabs b -> Rabs a <= c -> Rabs (a / b) <= c / k.
Proof.
  intros a b c k Hk Hb Ha.
  assert (Hb0 : b <> 0).
  { intro Hz. subst b. rewrite Rabs_R0 in Hb. lra. }
  assert (Hbp : 0 < Rabs b) by lra.
  unfold Rdiv. rewrite Rabs_mult, Rabs_inv. fold (Rabs a / Rabs b).
  apply Rle_trans with (c / Rabs b).
  - unfold Rdiv. apply Rmult_le_compat_r; [left; apply Rinv_0_lt_compat; lra | exact Ha].
  - unfold Rdiv. apply Rmult_le_compat_l.
    + eapply Rle_trans; [apply Rabs_pos | exact Ha].
    + apply Rinv_le_contravar; lra.
Qed.

Lemma ok_div : forall M m L d x y rx ry,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok d x rx -> ok d y ry ->
  Rabs rx <= M ->
  m <= Rabs (B2R y) -> m <= Rabs ry ->
  regz M (B2R x / B2R y) ->
  ok (f32_u * M + L * d) (f32_div x y) (rx / ry).
Proof.
  intros M m L d x y rx ry HM Hamp Hx Hy Hbx Hmy Hmry Hz.
  assert (Hm : 0 < m) by (destruct Hamp as (H & _); exact H).
  assert (HL : 1 / m + M / (m * m) <= L)
    by (destruct Hamp as (_ & _ & _ & _ & H & _); exact H).
  destruct Hx as [Hfx Hdx]. destruct Hy as [Hfy Hdy].
  assert (Hd0 : 0 <= d) by (pose proof (Rabs_pos (B2R x - rx)); lra).
  assert (Hy0 : B2R y <> 0).
  { intro Hz0. rewrite Hz0, Rabs_R0 in Hmy. lra. }
  assert (Hry0 : ry <> 0).
  { intro Hz0. rewrite Hz0, Rabs_R0 in Hmry. lra. }
  assert (Hbool : Rlt_bool (Rabs (f32_round (B2R x / B2R y))) (bpow radix2 emax32) = true)
    by (eapply no_overflow; eassumption).
  split.
  - unfold f32_div.
    pose proof (Bdiv_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32
                  mode_NE x y Hy0) as H.
    unfold f32_round, f32_fexp in Hbool. rewrite Hbool in H.
    destruct H as [_ [Hf _]]. rewrite Hfx in Hf. exact Hf.
  - rewrite (f32_div_correct x y Hy0 Hbool).
    replace (f32_round (B2R x / B2R y) - rx / ry)
      with ((f32_round (B2R x / B2R y) - B2R x / B2R y)
            + ((B2R x - rx) / B2R y
               + (rx * (ry - B2R y)) / (B2R y * ry))) by (field; split; assumption).
    eapply Rle_trans; [apply Rabs_triang|].
    assert (H1 : Rabs (f32_round (B2R x / B2R y) - B2R x / B2R y) <= f32_u * M)
      by (eapply round_err_le; eassumption).
    assert (H2 : Rabs ((B2R x - rx) / B2R y) <= d / m)
      by (apply abs_div_le with (k := m); assumption).
    assert (H3 : Rabs ((rx * (ry - B2R y)) / (B2R y * ry)) <= (M * d) / (m * m)).
    { apply abs_div_le with (k := m * m).
      - nra.
      - rewrite Rabs_mult. nra.
      - rewrite Rabs_mult.
        apply Rmult_le_compat; try apply Rabs_pos; [exact Hbx|].
        replace (ry - B2R y) with (- (B2R y - ry)) by ring.
        rewrite Rabs_Ropp. exact Hdy. }
    assert (H4 : Rabs ((B2R x - rx) / B2R y
                       + (rx * (ry - B2R y)) / (B2R y * ry)) <= d / m + (M * d) / (m * m))
      by (eapply Rle_trans; [apply Rabs_triang | lra]).
    assert (H5 : d / m + (M * d) / (m * m) = d * (1 / m + M / (m * m)))
      by (field; lra).
    assert (H6 : d * (1 / m + M / (m * m)) <= d * L)
      by (apply Rmult_le_compat_l; assumption).
    nra.
Qed.

(** Square root of a positive finite value is finite. *)
Lemma f32_sqrt_finite : forall x,
  is_finite x = true -> 0 < B2R x -> is_finite (f32_sqrt x) = true.
Proof.
  intros x Hf Hp. unfold f32_sqrt.
  pose proof (Bsqrt_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x) as H.
  destruct H as [_ [Hfin _]].
  destruct x as [s|s| |s mx ex Hmx]; simpl in Hp; try discriminate.
  - lra.
  - destruct s.
    + exfalso.
      assert (@F2R radix2 {| Fnum := SpecFloat.cond_Zopp true (Z.pos mx); Fexp := ex |} < 0)
        by (apply F2R_lt_0; reflexivity).
      lra.
    + exact Hfin.
Qed.

Lemma ok_sqrt : forall M m L d x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok d x rx ->
  m <= B2R x -> m <= rx ->
  regz M (sqrt (B2R x)) ->
  ok (f32_u * M + L * d) (f32_sqrt x) (sqrt rx).
Proof.
  intros M m L d x rx HM Hamp Hx Hmx Hmrx Hz.
  assert (Hm : 0 < m) by (destruct Hamp as (H & _); exact H).
  assert (HL : 1 / (2 * sqrt m) <= L)
    by (destruct Hamp as (_ & _ & _ & _ & _ & H); exact H).
  destruct Hx as [Hfx Hdx].
  assert (Hd0 : 0 <= d) by (pose proof (Rabs_pos (B2R x - rx)); lra).
  assert (Hsm : 0 < sqrt m) by (apply sqrt_lt_R0; exact Hm).
  assert (Hsx : sqrt m <= sqrt (B2R x)) by (apply sqrt_le_1; lra).
  assert (Hsr : sqrt m <= sqrt rx) by (apply sqrt_le_1; lra).
  split.
  - apply f32_sqrt_finite; [exact Hfx | lra].
  - rewrite f32_sqrt_correct.
    replace (f32_round (sqrt (B2R x)) - sqrt rx)
      with ((f32_round (sqrt (B2R x)) - sqrt (B2R x)) + (sqrt (B2R x) - sqrt rx)) by ring.
    eapply Rle_trans; [apply Rabs_triang|].
    assert (H1 : Rabs (f32_round (sqrt (B2R x)) - sqrt (B2R x)) <= f32_u * M)
      by (eapply round_err_le; eassumption).
    assert (Hprod : (sqrt (B2R x) - sqrt rx) * (sqrt (B2R x) + sqrt rx) = B2R x - rx).
    { replace ((sqrt (B2R x) - sqrt rx) * (sqrt (B2R x) + sqrt rx))
        with (sqrt (B2R x) * sqrt (B2R x) - sqrt rx * sqrt rx) by ring.
      rewrite !sqrt_sqrt by lra. reflexivity. }
    assert (H2 : Rabs (sqrt (B2R x) - sqrt rx) * (2 * sqrt m) <= d).
    { eapply Rle_trans with (Rabs (sqrt (B2R x) - sqrt rx) * (sqrt (B2R x) + sqrt rx)).
      - apply Rmult_le_compat_l; [apply Rabs_pos | lra].
      - rewrite <- (Rabs_pos_eq (sqrt (B2R x) + sqrt rx)) by lra.
        rewrite <- Rabs_mult, Hprod. exact Hdx. }
    assert (H3 : Rabs (sqrt (B2R x) - sqrt rx) <= d * (1 / (2 * sqrt m))).
    { assert (0 < 2 * sqrt m) by lra.
      apply Rmult_le_reg_r with (2 * sqrt m); [lra|].
      replace (d * (1 / (2 * sqrt m)) * (2 * sqrt m)) with d by (field; lra).
      exact H2. }
    assert (H4 : d * (1 / (2 * sqrt m)) <= d * L)
      by (apply Rmult_le_compat_l; assumption).
    nra.
Qed.

(** * Iterating the affine step

    A stage of arithmetic depth [k] carries a bound of [errN n] to
    [errN (n + k)]. Because [errN] is monotone in [n], a stage may always be
    charged more depth than it uses, which is what lets the stages compose
    without tracking exact depths. *)

Fixpoint errN (M L : R) (n : nat) : R :=
  match n with
  | O => 0
  | S k => f32_u * M + L * errN M L k
  end.

Lemma errN_nonneg : forall M L n, 0 <= M -> 1 <= L -> 0 <= errN M L n.
Proof.
  intros M L n HM HL. pose proof f32_u_pos.
  induction n as [|k IH]; simpl; [lra | nra].
Qed.

Lemma errN_mono_S : forall M L n, 0 <= M -> 1 <= L -> errN M L n <= errN M L (S n).
Proof.
  intros M L n HM HL. pose proof f32_u_pos.
  pose proof (errN_nonneg M L n HM HL).
  simpl. nra.
Qed.

Lemma errN_mono : forall M L n n',
  0 <= M -> 1 <= L -> (n <= n')%nat -> errN M L n <= errN M L n'.
Proof.
  intros M L n n' HM HL Hle.
  induction Hle as [|p Hp IH]; [apply Rle_refl|].
  eapply Rle_trans; [exact IH | apply errN_mono_S; assumption].
Qed.

Lemma ok_errN_mono : forall M L n n' x r,
  0 <= M -> 1 <= L -> (n <= n')%nat ->
  ok (errN M L n) x r -> ok (errN M L n') x r.
Proof.
  intros M L n n' x r HM HL Hle H.
  eapply ok_weaken; [exact H | apply errN_mono; assumption].
Qed.

(** * Structural lemmas

    The list plumbing performs no arithmetic, so it transports the relation
    unchanged. *)

Lemma Forall2_map2 : forall (A B C D : Type)
    (P : A -> B -> Prop) (Q : C -> D -> Prop) (f : A -> C) (g : B -> D) xs rs,
  Forall2 P xs rs -> (forall a b, P a b -> Q (f a) (g b)) ->
  Forall2 Q (List.map f xs) (List.map g rs).
Proof.
  intros A B C D P Q f g xs rs H Hf.
  induction H as [|a b xs rs Hab H IH]; simpl; constructor; auto.
Qed.

Lemma Forall2_map_combine : forall (A B C D E F : Type)
    (P : A -> D -> Prop) (Q : B -> E -> Prop) (S : C -> F -> Prop)
    (f : A * B -> C) (g : D * E -> F) xs rs ys ss,
  Forall2 P xs rs -> Forall2 Q ys ss ->
  (forall a d b e, P a d -> Q b e -> S (f (a, b)) (g (d, e))) ->
  Forall2 S (List.map f (List.combine xs ys)) (List.map g (List.combine rs ss)).
Proof.
  intros A B C D E F P Q S f g xs rs ys ss H1.
  revert ys ss.
  induction H1 as [|a d xs rs Had H1 IH]; intros ys ss H2 Hf; simpl.
  - constructor.
  - destruct H2 as [|b e ys ss Hbe H2]; simpl; constructor; auto.
Qed.

Lemma Forall2_map_seq : forall (A B : Type) (Q : A -> B -> Prop)
    (f : nat -> A) (g : nat -> B) n s,
  (forall i, Q (f i) (g i)) -> Forall2 Q (List.map f (List.seq s n)) (List.map g (List.seq s n)).
Proof.
  intros A B Q f g n. induction n as [|k IH]; intros s H; simpl; constructor; auto.
Qed.

Lemma Forall2_nth : forall (A B : Type) (P : A -> B -> Prop) xs rs i da db,
  Forall2 P xs rs -> P da db -> P (List.nth i xs da) (List.nth i rs db).
Proof.
  intros A B P xs rs i da db H. revert i.
  induction H as [|a b xs rs Hab H IH]; intros i Hd; destruct i; simpl; auto.
Qed.

Lemma Forall2_firstn : forall (A B : Type) (P : A -> B -> Prop) n xs rs,
  Forall2 P xs rs -> Forall2 P (List.firstn n xs) (List.firstn n rs).
Proof.
  intros A B P n. induction n as [|k IH]; intros xs rs H; simpl; [constructor|].
  destruct H as [|a b xs rs Hab H]; simpl; constructor; auto.
Qed.

Lemma Forall2_skipn : forall (A B : Type) (P : A -> B -> Prop) n xs rs,
  Forall2 P xs rs -> Forall2 P (List.skipn n xs) (List.skipn n rs).
Proof.
  intros A B P n. induction n as [|k IH]; intros xs rs H; simpl; [exact H|].
  destruct H as [|a b xs rs Hab H]; simpl; [constructor | auto].
Qed.

Lemma Forall2_app : forall (A B : Type) (P : A -> B -> Prop) xs rs ys ss,
  Forall2 P xs rs -> Forall2 P ys ss -> Forall2 P (xs ++ ys) (rs ++ ss).
Proof.
  intros A B P xs rs ys ss H1 H2.
  induction H1 as [|a b xs rs Hab H1 IH]; simpl; [exact H2 | constructor; auto].
Qed.

Lemma Forall2_concat : forall (A B : Type) (P : A -> B -> Prop) xs rs,
  Forall2 (Forall2 P) xs rs -> Forall2 P (List.concat xs) (List.concat rs).
Proof.
  intros A B P xs rs H.
  induction H as [|a b xs rs Hab H IH]; simpl; [constructor | apply Forall2_app; auto].
Qed.

Lemma Forall2_length : forall (A B : Type) (P : A -> B -> Prop) xs rs,
  Forall2 P xs rs -> List.length xs = List.length rs.
Proof.
  intros A B P xs rs H. induction H; simpl; auto.
Qed.

(** * Vectors and matrices *)

Definition okv (d : R) (xs : list binary32) (rs : list R) : Prop :=
  Forall2 (ok d) xs rs.

Definition okm (d : R) (xs : list (list binary32)) (rs : list (list R)) : Prop :=
  Forall2 (okv d) xs rs.

Lemma okv_weaken : forall d d' xs rs, okv d xs rs -> d <= d' -> okv d' xs rs.
Proof.
  intros d d' xs rs H Hle. eapply Forall2_map2 with (f := fun x => x) (g := fun r => r) in H.
  - rewrite !List.map_id in H. exact H.
  - intros a b Hab. eapply ok_weaken; eassumption.
Qed.

Lemma okm_weaken : forall d d' xs rs, okm d xs rs -> d <= d' -> okm d' xs rs.
Proof.
  intros d d' xs rs H Hle.
  eapply Forall2_map2 with (f := fun x => x) (g := fun r => r) in H.
  - rewrite !List.map_id in H. exact H.
  - intros a b Hab. eapply okv_weaken; eassumption.
Qed.

(** * Dot product

    The reference is the same accumulation performed in exact real arithmetic,
    in the same order. *)

Fixpoint Rdotr (xs ys : list R) (acc : R) : R :=
  match xs, ys with
  | x :: xs', y :: ys' => Rdotr xs' ys' (acc + x * y)
  | _, _ => acc
  end.

Definition Rdot (xs ys : list R) : R := Rdotr xs ys 0.

(** Each multiply-accumulate along the way stays in the regular regime. *)
Inductive dotreg (M : R) : list binary32 -> list binary32 -> binary32 -> Prop :=
| dotreg_nil_l : forall ys a, dotreg M nil ys a
| dotreg_nil_r : forall xs a, dotreg M xs nil a
| dotreg_cons : forall x xs y ys a,
    Rabs (B2R x) <= M ->
    regz M (B2R x * B2R y) ->
    regz M (B2R a + B2R (f32_mult x y)) ->
    dotreg M xs ys (f32_plus a (f32_mult x y)) ->
    dotreg M (x :: xs) (y :: ys) a.

Lemma ok_dot_aux : forall M m L xs ys a,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  dotreg M xs ys a ->
  forall n rs ss ra,
  okv (errN M L n) xs rs -> okv (errN M L n) ys ss ->
  Forall (fun r => Rabs r <= M) ss ->
  ok (errN M L n) a ra ->
  ok (errN M L (n + 2 * List.length xs)) (f32_dot_aux xs ys a) (Rdotr rs ss ra).
Proof.
  intros M m L xs ys a HM Hamp Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  induction Hreg as [ys a | xs a | x xs y ys a Hbx Hz1 Hz2 Hreg IH];
    intros n rs ss ra Hx Hy Hss Ha.
  - inversion Hx; subst. simpl. rewrite Nat.add_0_r. exact Ha.
  - inversion Hy; subst.
    destruct rs as [|r rs']; destruct xs as [|x xs]; simpl;
      (eapply ok_errN_mono with (n := n); [exact HM0 | exact HL1 | lia | exact Ha]).
  - inversion Hx as [|x' rx xs' rs' Hxr Hxs]; subst.
    inversion Hy as [|y' ry ys' ss' Hyr Hys]; subst.
    inversion Hss as [|ry' ss'' Hbry Hss']; subst.
    simpl.
    eapply ok_errN_mono with (n := (S (S n) + 2 * List.length xs)%nat);
      [exact HM0 | exact HL1 | lia | ].
    apply IH.
    + eapply okv_weaken; [exact Hxs | apply errN_mono; auto; lia].
    + eapply okv_weaken; [exact Hys | apply errN_mono; auto; lia].
    + exact Hss'.
    + change (errN M L (S (S n))) with (f32_u * M + L * errN M L (S n)).
      eapply ok_plus with (m := m).
      * exact HM.
      * exact Hamp.
      * eapply ok_errN_mono with (n := n); [exact HM0 | exact HL1 | lia | exact Ha].
      * change (errN M L (S n)) with (f32_u * M + L * errN M L n).
        eapply ok_mult with (m := m); try eassumption.
      * exact Hz2.
Qed.

Lemma ok_f32_zero : forall M L, ok (errN M L 0) f32_zero 0.
Proof.
  intros M L. split; [reflexivity|].
  change (errN M L 0) with 0.
  rewrite B2R_f32_zero.
  replace (0 - 0) with 0 by ring. rewrite Rabs_R0. apply Rle_refl.
Qed.

Lemma ok_dot : forall M m L xs ys rs ss,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  dotreg M xs ys f32_zero ->
  okv (errN M L 0) xs rs -> okv (errN M L 0) ys ss ->
  Forall (fun r => Rabs r <= M) ss ->
  ok (errN M L (2 * List.length xs)) (f32_dot xs ys) (Rdot rs ss).
Proof.
  intros M m L xs ys rs ss HM Hamp Hreg Hx Hy Hss.
  unfold f32_dot, Rdot.
  pose proof (ok_dot_aux M m L xs ys f32_zero HM Hamp Hreg 0 rs ss 0
                Hx Hy Hss (ok_f32_zero M L)) as H.
  simpl in H. exact H.
Qed.

Lemma ok_zero_any : forall d, 0 <= d -> ok d f32_zero 0.
Proof.
  intros d Hd. split; [reflexivity|].
  rewrite B2R_f32_zero. replace (0 - 0) with 0 by ring.
  rewrite Rabs_R0. exact Hd.
Qed.

Lemma ok_dot_n : forall M m L n xs ys rs ss,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  dotreg M xs ys f32_zero ->
  okv (errN M L n) xs rs -> okv (errN M L n) ys ss ->
  Forall (fun r => Rabs r <= M) ss ->
  ok (errN M L (n + 2 * List.length xs)) (f32_dot xs ys) (Rdot rs ss).
Proof.
  intros M m L n xs ys rs ss HM Hamp Hreg Hx Hy Hss.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  unfold f32_dot, Rdot.
  eapply ok_dot_aux; try eassumption.
  apply ok_zero_any, errN_nonneg; assumption.
Qed.

(** * Vector and matrix stages *)

Definition Rvec_add (xs ys : list R) : list R :=
  List.map (fun '(x, y) => x + y) (List.combine xs ys).

Definition Rmat_transpose (m : list (list R)) : list (list R) :=
  match m with
  | [] => []
  | row :: _ =>
      List.map (fun ci => List.map (fun rd => List.nth ci rd 0) m)
               (List.seq 0 (List.length row))
  end.

Definition Rmat_vec_mul (m : list (list R)) (v : list R) : list R :=
  List.map (fun row => Rdot row v) m.

Definition Rmat_mul (a b : list (list R)) : list (list R) :=
  let bt := Rmat_transpose b in
  List.map (fun ar => List.map (fun bc => Rdot ar bc) bt) a.

Definition Radd_matrices (a b : list (list R)) : list (list R) :=
  List.map (fun '(ra, rb) => Rvec_add ra rb) (List.combine a b).

Definition Rlinear_forward (w : list (list R)) (b : list R) (x : list R) : list R :=
  Rvec_add (Rmat_vec_mul (Rmat_transpose w) x) b.

Definition Rlinear_forward_2d (w : list (list R)) (b : list R)
                              (x : list (list R)) : list (list R) :=
  List.map (Rlinear_forward w b) x.

(** Transpose moves values without arithmetic. *)
Lemma ok_mat_transpose : forall d mm rm,
  0 <= d -> okm d mm rm -> okm d (f32_mat_transpose mm) (Rmat_transpose rm).
Proof.
  intros d mm rm Hd H.
  destruct H as [|row rrow mm' rm' Hrow H]; simpl; [constructor|].
  rewrite (Forall2_length _ _ _ _ _ Hrow).
  apply Forall2_map_seq. intros i.
  unfold okv. constructor.
  - eapply Forall2_nth; [exact Hrow | apply ok_zero_any, Hd].
  - apply Forall2_map2 with (P := okv d); [exact H|].
    intros a b Hab. eapply Forall2_nth; [exact Hab | apply ok_zero_any, Hd].
Qed.

Lemma ok_vec_add : forall M m L n xs ys rs ss,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L n) xs rs -> okv (errN M L n) ys ss ->
  Forall2 (fun x y => regz M (B2R x + B2R y)) xs ys ->
  okv (errN M L (S n)) (f32_vec_add xs ys) (Rvec_add rs ss).
Proof.
  intros M m L n xs ys rs ss HM Hamp Hx Hy Hreg.
  unfold f32_vec_add, Rvec_add, okv.
  revert rs ss Hx Hy.
  induction Hreg as [|x y xs ys Hxy Hreg IH]; intros rs ss Hx Hy.
  - inversion Hx; subst. simpl. constructor.
  - inversion Hx as [|x' rx xs' rs' Hxr Hxs]; subst.
    inversion Hy as [|y' ry ys' ss' Hyr Hys]; subst.
    simpl. constructor.
    + change (errN M L (S n)) with (f32_u * M + L * errN M L n).
      eapply ok_plus with (m := m); eassumption.
    + apply IH; assumption.
Qed.

(** [k] is any bound on the row length, so a stage may be charged more depth
    than it uses. *)
Lemma ok_mat_vec_mul : forall M m L n k mm rm v rv,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) mm rm ->
  okv (errN M L n) v rv ->
  Forall (fun r => Rabs r <= M) rv ->
  Forall (fun row => dotreg M row v f32_zero /\ (List.length row <= k)%nat) mm ->
  okv (errN M L (n + 2 * k)) (f32_mat_vec_mul mm v) (Rmat_vec_mul rm rv).
Proof.
  intros M m L n k mm rm v rv HM Hamp Hmm Hv Hrv Hrows.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  unfold f32_mat_vec_mul, Rmat_vec_mul, okv.
  revert Hrows.
  induction Hmm as [|row rrow mm' rm' Hrow Hmm IH]; intros Hrows; simpl;
    [constructor|].
  pose proof (Forall_inv Hrows) as [Hreg Hlen].
  pose proof (Forall_inv_tail Hrows) as Hrows'.
  constructor.
  - eapply ok_errN_mono with (n := (n + 2 * List.length row)%nat);
      [exact HM0 | exact HL1 | lia |].
    eapply ok_dot_n; eassumption.
  - apply IH; exact Hrows'.
Qed.

Lemma ok_add_matrices : forall M m L n aa ra bb rb,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) aa ra -> okm (errN M L n) bb rb ->
  Forall2 (fun r1 r2 => Forall2 (fun x y => regz M (B2R x + B2R y)) r1 r2) aa bb ->
  okm (errN M L (S n)) (f32_add_matrices aa bb) (Radd_matrices ra rb).
Proof.
  intros M m L n aa ra bb rb HM Hamp Ha Hb Hreg.
  unfold f32_add_matrices, Radd_matrices, okm.
  revert ra rb Ha Hb.
  induction Hreg as [|r1 r2 aa' bb' Hr Hreg IH]; intros ra rb Ha Hb.
  - inversion Ha; subst. simpl. constructor.
  - inversion Ha as [|a1 ra1 aa'' ra' Har Has]; subst.
    inversion Hb as [|b1 rb1 bb'' rb' Hbr Hbs]; subst.
    simpl. constructor.
    + eapply ok_vec_add; eassumption.
    + apply IH; assumption.
Qed.

Lemma ok_linear_forward : forall M m L n k w rw b rb x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) w rw ->
  okv (errN M L n) b rb ->
  okv (errN M L n) x rx ->
  Forall (fun r => Rabs r <= M) rx ->
  Forall (fun row => dotreg M row x f32_zero /\ (List.length row <= k)%nat)
         (f32_mat_transpose w) ->
  Forall2 (fun p q => regz M (B2R p + B2R q))
          (f32_mat_vec_mul (f32_mat_transpose w) x) b ->
  okv (errN M L (S (n + 2 * k))) (f32_linear_forward w b x) (Rlinear_forward rw rb rx).
Proof.
  intros M m L n k w rw b rb x rx HM Hamp Hw Hb Hx Hrx Hrows Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  unfold f32_linear_forward, Rlinear_forward.
  eapply ok_vec_add with (m := m); try eassumption.
  - eapply ok_mat_vec_mul with (m := m); try eassumption.
    apply ok_mat_transpose; [apply errN_nonneg; assumption | exact Hw].
  - eapply okv_weaken; [exact Hb | apply errN_mono; auto; lia].
Qed.

Lemma ok_linear_forward_2d : forall M m L n k w rw b rb x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) w rw ->
  okv (errN M L n) b rb ->
  okm (errN M L n) x rx ->
  Forall (fun row => Forall (fun r => Rabs r <= M) row) rx ->
  Forall (fun row =>
            Forall (fun r => dotreg M r row f32_zero /\ (List.length r <= k)%nat)
                   (f32_mat_transpose w)
            /\ Forall2 (fun p q => regz M (B2R p + B2R q))
                       (f32_mat_vec_mul (f32_mat_transpose w) row) b) x ->
  okm (errN M L (S (n + 2 * k))) (f32_linear_forward_2d w b x)
      (Rlinear_forward_2d rw rb rx).
Proof.
  intros M m L n k w rw b rb x rx HM Hamp Hw Hb Hx Hbnd Hreg.
  unfold f32_linear_forward_2d, Rlinear_forward_2d, okm.
  revert Hbnd Hreg.
  induction Hx as [|row rrow x' rx' Hrow Hx IH]; intros Hbnd Hreg; simpl;
    [constructor|].
  pose proof (Forall_inv Hbnd) as Hb1. pose proof (Forall_inv_tail Hbnd) as Hb2.
  pose proof (Forall_inv Hreg) as [Hr1 Hr2]. pose proof (Forall_inv_tail Hreg) as Hr3.
  constructor.
  - eapply ok_linear_forward with (m := m); eassumption.
  - apply IH; assumption.
Qed.

(** * Sums, mean, variance

    Program constants keep their own real value as reference, so the mirror
    divides by exactly the constant the code divides by. *)

Definition Rsum (rs : list R) : R := List.fold_left Rplus rs 0.

Inductive sumreg (M : R) : list binary32 -> binary32 -> Prop :=
| sumreg_nil : forall a, sumreg M nil a
| sumreg_cons : forall x xs a,
    regz M (B2R a + B2R x) ->
    sumreg M xs (f32_plus a x) ->
    sumreg M (x :: xs) a.

Lemma ok_sum_aux : forall M m L xs a,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  sumreg M xs a ->
  forall n rs ra,
  okv (errN M L n) xs rs -> ok (errN M L n) a ra ->
  ok (errN M L (n + List.length xs))
     (List.fold_left f32_plus xs a) (List.fold_left Rplus rs ra).
Proof.
  intros M m L xs a HM Hamp Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  induction Hreg as [a | x xs a Hz Hreg IH]; intros n rs ra Hx Ha.
  - inversion Hx; subst. simpl.
    eapply ok_errN_mono with (n := n); [exact HM0 | exact HL1 | lia | exact Ha].
  - inversion Hx as [|x' rx xs' rs' Hxr Hxs]; subst.
    simpl.
    eapply ok_errN_mono with (n := (S n + List.length xs)%nat);
      [exact HM0 | exact HL1 | lia |].
    apply IH.
    + eapply okv_weaken; [exact Hxs | apply errN_mono; auto; lia].
    + change (errN M L (S n)) with (f32_u * M + L * errN M L n).
      eapply ok_plus with (m := m); eassumption.
Qed.

Lemma ok_sum : forall M m L n xs rs,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  sumreg M xs f32_zero ->
  okv (errN M L n) xs rs ->
  ok (errN M L (n + List.length xs)) (f32_sum xs) (Rsum rs).
Proof.
  intros M m L n xs rs HM Hamp Hreg Hx.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  unfold f32_sum, Rsum.
  eapply ok_sum_aux; try eassumption.
  apply ok_zero_any, errN_nonneg; assumption.
Qed.

Definition Rmean (rs : list R) (n : nat) : R :=
  Rsum rs / B2R (f32_of_Z (Z.of_nat n)).

Lemma ok_mean : forall M m L n xs rs,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  sumreg M xs f32_zero ->
  okv (errN M L n) xs rs ->
  is_finite (f32_of_Z (Z.of_nat (List.length xs))) = true ->
  m <= Rabs (B2R (f32_of_Z (Z.of_nat (List.length xs)))) ->
  Rabs (Rsum rs) <= M ->
  regz M (B2R (f32_sum xs) / B2R (f32_of_Z (Z.of_nat (List.length xs)))) ->
  ok (errN M L (S (n + List.length xs))) (f32_mean xs) (Rmean rs (List.length xs)).
Proof.
  intros M m L n xs rs HM Hamp Hsr Hx Hfin Hlo Hbnd Hz.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  unfold f32_mean, Rmean.
  change (errN M L (S (n + List.length xs)))
    with (f32_u * M + L * errN M L (n + List.length xs)).
  eapply ok_div with (m := m); try eassumption.
  - eapply ok_sum; eassumption.
  - eapply ok_weaken; [apply ok_exact; exact Hfin | apply errN_nonneg; assumption].
Qed.

Definition Rvariance (rs : list R) (rmu : R) (n : nat) : R :=
  Rsum (List.map (fun r => (r - rmu) * (r - rmu)) rs)
  / B2R (f32_of_Z (Z.of_nat n)).


(** The squared deviations, elementwise. *)
Lemma ok_sq_devs : forall M m L n xs rs mu rmu,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L n) xs rs -> ok (errN M L n) mu rmu ->
  Forall2 (fun x r => regz M (B2R x + B2R (f32_neg mu))
                      /\ regz M (B2R (f32_minus x mu) * B2R (f32_minus x mu))
                      /\ Rabs (B2R (f32_minus x mu)) <= M
                      /\ Rabs (r - rmu) <= M) xs rs ->
  okv (errN M L (S (S n)))
      (List.map (fun x => let d := f32_minus x mu in f32_mult d d) xs)
      (List.map (fun r => (r - rmu) * (r - rmu)) rs).
Proof.
  intros M m L n xs rs mu rmu HM Hamp Hx Hmu Hreg.
  unfold okv in *. revert Hx.
  induction Hreg as [|x r xs' rs' (Hz1 & Hz2 & Hb1 & Hb2) Hreg IH];
    intros Hx; simpl; [constructor|].
  inversion Hx as [|x' rx xs'' rs'' Hxr Hxs]; subst.
  constructor.
  - assert (Hd : ok (errN M L (S n)) (f32_minus x mu) (r - rmu)).
    { change (errN M L (S n)) with (f32_u * M + L * errN M L n).
      eapply ok_minus with (m := m); eassumption. }
    change (errN M L (S (S n))) with (f32_u * M + L * errN M L (S n)).
    eapply ok_mult with (m := m); eassumption.
  - apply IH; exact Hxs.
Qed.

Lemma ok_variance : forall M m L n xs rs mu rmu,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L n) xs rs -> ok (errN M L n) mu rmu ->
  Forall2 (fun x r => regz M (B2R x + B2R (f32_neg mu))
                      /\ regz M (B2R (f32_minus x mu) * B2R (f32_minus x mu))
                      /\ Rabs (B2R (f32_minus x mu)) <= M
                      /\ Rabs (r - rmu) <= M) xs rs ->
  sumreg M (List.map (fun x => let d := f32_minus x mu in f32_mult d d) xs) f32_zero ->
  is_finite (f32_of_Z (Z.of_nat (List.length xs))) = true ->
  m <= Rabs (B2R (f32_of_Z (Z.of_nat (List.length xs)))) ->
  Rabs (Rsum (List.map (fun r => (r - rmu) * (r - rmu)) rs)) <= M ->
  regz M (B2R (f32_sum (List.map (fun x => let d := f32_minus x mu in f32_mult d d) xs))
          / B2R (f32_of_Z (Z.of_nat (List.length xs)))) ->
  ok (errN M L (S (S (S n) + List.length xs)))
     (f32_variance xs mu) (Rvariance rs rmu (List.length xs)).
Proof.
  intros M m L n xs rs mu rmu HM Hamp Hx Hmu Hreg Hsr Hfin Hlo Hbnd Hz.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  assert (Hlen : List.length (List.map (fun x => let d := f32_minus x mu in f32_mult d d) xs)
                 = List.length xs) by apply List.length_map.
  unfold f32_variance, Rvariance.
  change (errN M L (S (S (S n) + List.length xs)))
    with (f32_u * M + L * errN M L (S (S n) + List.length xs)).
  eapply ok_div with (m := m); try eassumption.
  - pose proof (ok_sum M m L (S (S n))
                  (List.map (fun x => let d := f32_minus x mu in f32_mult d d) xs)
                  (List.map (fun r => (r - rmu) * (r - rmu)) rs)
                  HM Hamp Hsr (ok_sq_devs M m L n xs rs mu rmu HM Hamp Hx Hmu Hreg)) as H.
    rewrite Hlen in H. exact H.
  - eapply ok_weaken; [apply ok_exact; exact Hfin | apply errN_nonneg; assumption].
Qed.

Lemma Forall2_conj : forall (A B : Type) (P Q : A -> B -> Prop) xs rs,
  Forall2 P xs rs -> Forall2 Q xs rs -> Forall2 (fun a b => P a b /\ Q a b) xs rs.
Proof.
  intros A B P Q xs rs H1. induction H1 as [|a b xs rs Hab H1 IH]; intros H2.
  - constructor.
  - inversion H2 as [|a' b' xs' rs' Hab2 H2']; subst.
    constructor; [split; assumption | apply IH; exact H2'].
Qed.

Lemma Forall2_combine_seq : forall (A B : Type) (P : A -> B -> Prop) xs rs s,
  Forall2 P xs rs ->
  Forall2 (fun p q => fst p = fst q /\ P (snd p) (snd q))
          (List.combine (List.seq s (List.length xs)) xs)
          (List.combine (List.seq s (List.length rs)) rs).
Proof.
  intros A B P xs rs s H. revert s.
  induction H as [|a b xs rs Hab H IH]; intros s; simpl; constructor.
  - split; [reflexivity | exact Hab].
  - apply IH.
Qed.

Lemma ok_one : forall d, 0 <= d -> ok d f32_one 1.
Proof.
  intros d Hd. split; [exact f32_one_finite|].
  rewrite f32_one_correct. replace (1 - 1) with 0 by ring.
  rewrite Rabs_R0. exact Hd.
Qed.

(** * Layer normalization

    [mu], [denom], [gamma], [beta] and the input are all required at one common
    depth [k]; the caller raises them with [errN] monotonicity. Each output
    entry then costs four operations. *)

Definition Rlayer_norm_elem (rgamma rbeta : list R) (rmu rdenom : R)
                            (p : nat * R) : R :=
  let (idx, xi) := p in
  List.nth idx rgamma 1 * ((xi - rmu) / rdenom) + List.nth idx rbeta 0.

Definition f32_layer_norm_elem (gamma beta : list binary32) (mu denom : binary32)
                               (p : nat * binary32) : binary32 :=
  let (idx, xi) := p in
  f32_plus (f32_mult (List.nth idx gamma f32_one)
                     (f32_div (f32_minus xi mu) denom))
           (List.nth idx beta f32_zero).

Lemma ok_layer_norm_elems : forall M m L k len gamma rgamma beta rbeta
                                   mu rmu denom rdenom xs rs,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  List.length xs = len -> List.length rs = len ->
  okv (errN M L k) gamma rgamma -> okv (errN M L k) beta rbeta ->
  ok (errN M L k) mu rmu -> ok (errN M L k) denom rdenom ->
  okv (errN M L k) xs rs ->
  m <= Rabs (B2R denom) -> m <= Rabs rdenom ->
  Forall2 (fun p q =>
    regz M (B2R (snd p) + B2R (f32_neg mu))
    /\ Rabs (snd q - rmu) <= M
    /\ regz M (B2R (f32_minus (snd p) mu) / B2R denom)
    /\ Rabs (B2R (List.nth (fst p) gamma f32_one)) <= M
    /\ Rabs ((snd q - rmu) / rdenom) <= M
    /\ regz M (B2R (List.nth (fst p) gamma f32_one)
               * B2R (f32_div (f32_minus (snd p) mu) denom))
    /\ regz M (B2R (f32_mult (List.nth (fst p) gamma f32_one)
                             (f32_div (f32_minus (snd p) mu) denom))
               + B2R (List.nth (fst p) beta f32_zero)))
    (List.combine (List.seq 0 len) xs)
    (List.combine (List.seq 0 len) rs) ->
  okv (errN M L (S (S (S (S k)))))
      (List.map (f32_layer_norm_elem gamma beta mu denom)
                (List.combine (List.seq 0 len) xs))
      (List.map (Rlayer_norm_elem rgamma rbeta rmu rdenom)
                (List.combine (List.seq 0 len) rs)).
Proof.
  intros M m L k len gamma rgamma beta rbeta mu rmu denom rdenom xs rs
         HM Hamp Hlx Hlr Hg Hb Hmu Hden Hx Hlo1 Hlo2 Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  assert (Hd0 : 0 <= errN M L k) by (apply errN_nonneg; assumption).
  unfold okv in *.
  pose proof (Forall2_combine_seq _ _ (ok (errN M L k)) xs rs 0 Hx) as Hpairs.
  rewrite Hlx, Hlr in Hpairs.
  pose proof (Forall2_conj _ _ _ _ _ _ Hpairs Hreg) as Hall.
  eapply Forall2_map2; [exact Hall|].
  intros p q [[Hidx Hval] (Hz1 & Hb2 & Hz2 & Hb3 & Hb4 & Hz3 & Hz4)].
  destruct p as [i xi]. destruct q as [j ri]. simpl in *. subst j.
  unfold f32_layer_norm_elem, Rlayer_norm_elem.
  assert (Hsub : ok (errN M L (S k)) (f32_minus xi mu) (ri - rmu)).
  { change (errN M L (S k)) with (f32_u * M + L * errN M L k).
    eapply ok_minus with (m := m); eassumption. }
  assert (Hnorm : ok (errN M L (S (S k))) (f32_div (f32_minus xi mu) denom)
                     ((ri - rmu) / rdenom)).
  { change (errN M L (S (S k))) with (f32_u * M + L * errN M L (S k)).
    eapply ok_div with (m := m); try eassumption.
    eapply ok_errN_mono with (n := k); [exact HM0 | exact HL1 | lia | exact Hden]. }
  assert (Hgi : ok (errN M L (S (S k))) (List.nth i gamma f32_one) (List.nth i rgamma 1)).
  { eapply ok_errN_mono with (n := k); [exact HM0 | exact HL1 | lia |].
    eapply Forall2_nth; [exact Hg | apply ok_one; exact Hd0]. }
  assert (Hbi : ok (errN M L (S (S (S k)))) (List.nth i beta f32_zero) (List.nth i rbeta 0)).
  { eapply ok_errN_mono with (n := k); [exact HM0 | exact HL1 | lia |].
    eapply Forall2_nth; [exact Hb | apply ok_zero_any; exact Hd0]. }
  assert (Hmul : ok (errN M L (S (S (S k))))
                    (f32_mult (List.nth i gamma f32_one)
                              (f32_div (f32_minus xi mu) denom))
                    (List.nth i rgamma 1 * ((ri - rmu) / rdenom))).
  { change (errN M L (S (S (S k)))) with (f32_u * M + L * errN M L (S (S k))).
    eapply ok_mult with (m := m); eassumption. }
  change (errN M L (S (S (S (S k))))) with (f32_u * M + L * errN M L (S (S (S k)))).
  eapply ok_plus with (m := m); eassumption.
Qed.

(** The layer-norm row, with every length written as the float input's. *)
Definition Rlayer_norm_vec (rgamma rbeta : list R) (reps : R) (len : nat)
                           (rx : list R) : list R :=
  let mu := Rmean rx len in
  let var := Rvariance rx mu len in
  let denom := sqrt (var + reps) in
  List.map (Rlayer_norm_elem rgamma rbeta mu denom)
           (List.combine (List.seq 0 len) rx).

(** The side conditions of one layer-norm row: the mean, the variance, the
    shifted variance, its square root, and the four operations per entry. *)
Record ln_reg (M m : R) (gamma beta : list binary32) (eps : binary32)
              (x : list binary32) (rgamma rbeta : list R) (reps : R) (rx : list R)
              : Prop := {
  lnr_len : List.length rx = List.length x;
  lnr_sum : sumreg M x f32_zero;
  lnr_fin_n : is_finite (f32_of_Z (Z.of_nat (List.length x))) = true;
  lnr_lo_n : m <= Rabs (B2R (f32_of_Z (Z.of_nat (List.length x))));
  lnr_bnd_sum : Rabs (Rsum rx) <= M;
  lnr_z_mean : regz M (B2R (f32_sum x) / B2R (f32_of_Z (Z.of_nat (List.length x))));
  lnr_dev :
    Forall2 (fun xi ri =>
      regz M (B2R xi + B2R (f32_neg (f32_mean x)))
      /\ regz M (B2R (f32_minus xi (f32_mean x)) * B2R (f32_minus xi (f32_mean x)))
      /\ Rabs (B2R (f32_minus xi (f32_mean x))) <= M
      /\ Rabs (ri - Rmean rx (List.length x)) <= M) x rx;
  lnr_sum2 :
    sumreg M (List.map (fun xi => let d := f32_minus xi (f32_mean x) in f32_mult d d) x)
           f32_zero;
  lnr_bnd_sum2 :
    Rabs (Rsum (List.map (fun r => (r - Rmean rx (List.length x))
                                   * (r - Rmean rx (List.length x))) rx)) <= M;
  lnr_z_var :
    regz M (B2R (f32_sum (List.map (fun xi => let d := f32_minus xi (f32_mean x) in
                                              f32_mult d d) x))
            / B2R (f32_of_Z (Z.of_nat (List.length x))));
  lnr_z_shift : regz M (B2R (f32_variance x (f32_mean x)) + B2R eps);
  lnr_rad_lo : m <= B2R (f32_plus (f32_variance x (f32_mean x)) eps);
  lnr_rad_lo_r :
    m <= Rvariance rx (Rmean rx (List.length x)) (List.length x) + reps;
  lnr_z_sqrt : regz M (sqrt (B2R (f32_plus (f32_variance x (f32_mean x)) eps)));
  lnr_den_lo : m <= Rabs (B2R (f32_sqrt (f32_plus (f32_variance x (f32_mean x)) eps)));
  lnr_den_lo_r :
    m <= Rabs (sqrt (Rvariance rx (Rmean rx (List.length x)) (List.length x) + reps));
  lnr_elems :
    Forall2 (fun p q =>
      regz M (B2R (snd p) + B2R (f32_neg (f32_mean x)))
      /\ Rabs (snd q - Rmean rx (List.length x)) <= M
      /\ regz M (B2R (f32_minus (snd p) (f32_mean x))
                 / B2R (f32_sqrt (f32_plus (f32_variance x (f32_mean x)) eps)))
      /\ Rabs (B2R (List.nth (fst p) gamma f32_one)) <= M
      /\ Rabs ((snd q - Rmean rx (List.length x))
               / sqrt (Rvariance rx (Rmean rx (List.length x)) (List.length x) + reps)) <= M
      /\ regz M (B2R (List.nth (fst p) gamma f32_one)
                 * B2R (f32_div (f32_minus (snd p) (f32_mean x))
                                (f32_sqrt (f32_plus (f32_variance x (f32_mean x)) eps))))
      /\ regz M (B2R (f32_mult (List.nth (fst p) gamma f32_one)
                        (f32_div (f32_minus (snd p) (f32_mean x))
                                 (f32_sqrt (f32_plus (f32_variance x (f32_mean x)) eps))))
                 + B2R (List.nth (fst p) beta f32_zero)))
      (List.combine (List.seq 0 (List.length x)) x)
      (List.combine (List.seq 0 (List.length x)) rx)
}.

Definition ln_depth (n len : nat) : nat :=
  let n1 := S (n + len) in
  let n2 := S (S (S n1) + len) in
  S (S (S (S (S (S n2))))).

Lemma ok_layer_norm_vec : forall M m L n gamma rgamma beta rbeta eps reps x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L n) gamma rgamma -> okv (errN M L n) beta rbeta ->
  ok (errN M L n) eps reps ->
  okv (errN M L n) x rx ->
  ln_reg M m gamma beta eps x rgamma rbeta reps rx ->
  okv (errN M L (ln_depth n (List.length x)))
      (f32_layer_norm_vec gamma beta eps x)
      (Rlayer_norm_vec rgamma rbeta reps (List.length x) rx).
Proof.
  intros M m L n gamma rgamma beta rbeta eps reps x rx HM Hamp Hg Hb Heps Hx Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  destruct Hreg as [Hlen0 Hsum0 Hfin0 Hlo0 Hbsum0 Hzmean0 Hdev0 Hsum20 Hbsum20
                    Hzvar0 Hzshift0 Hradlo0 Hradlor0 Hzsqrt0 Hdenlo0 Hdenlor0 Helems0].
  set (len := List.length x) in *.
  set (n1 := S (n + len)).
  assert (Hmu : ok (errN M L n1) (f32_mean x) (Rmean rx len)).
  { unfold n1. eapply ok_mean with (m := m); eassumption. }
  set (n2 := S (S (S n1) + len)).
  assert (Hvar : ok (errN M L n2) (f32_variance x (f32_mean x))
                    (Rvariance rx (Rmean rx len) len)).
  { unfold n2. eapply ok_variance with (m := m); try eassumption.
    eapply okv_weaken; [exact Hx | apply errN_mono; auto; unfold n1; lia]. }
  set (k := S (S n2)).
  assert (Hshift : ok (errN M L (S n2)) (f32_plus (f32_variance x (f32_mean x)) eps)
                      (Rvariance rx (Rmean rx len) len + reps)).
  { change (errN M L (S n2)) with (f32_u * M + L * errN M L n2).
    eapply ok_plus with (m := m); try eassumption.
    eapply ok_errN_mono with (n := n); [exact HM0 | exact HL1 | unfold n2, n1; lia | exact Heps]. }
  assert (Hden : ok (errN M L k) (f32_sqrt (f32_plus (f32_variance x (f32_mean x)) eps))
                    (sqrt (Rvariance rx (Rmean rx len) len + reps))).
  { unfold k. change (errN M L (S (S n2))) with (f32_u * M + L * errN M L (S n2)).
    eapply ok_sqrt with (m := m); eassumption. }
  unfold f32_layer_norm_vec, Rlayer_norm_vec.
  eapply okv_weaken.
  - eapply ok_layer_norm_elems with (m := m) (k := k) (len := len); try eassumption.
    + reflexivity.
    + eapply okv_weaken; [exact Hg | apply errN_mono; auto; unfold k, n2, n1; lia].
    + eapply okv_weaken; [exact Hb | apply errN_mono; auto; unfold k, n2, n1; lia].
    + eapply ok_errN_mono with (n := n1);
        [exact HM0 | exact HL1 | unfold k, n2, n1; lia | exact Hmu].
    + eapply okv_weaken; [exact Hx | apply errN_mono; auto; unfold k, n2, n1; lia].
  - apply errN_mono; [exact HM0 | exact HL1 |].
    unfold ln_depth, k, n2, n1. lia.
Qed.

Definition Rlayer_norm_2d (rgamma rbeta : list R) (reps : R) (len : nat)
                          (rm : list (list R)) : list (list R) :=
  List.map (Rlayer_norm_vec rgamma rbeta reps len) rm.

Lemma ok_layer_norm_2d : forall M m L n len gamma rgamma beta rbeta eps reps mm rm,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L n) gamma rgamma -> okv (errN M L n) beta rbeta ->
  ok (errN M L n) eps reps ->
  okm (errN M L n) mm rm ->
  Forall (fun row => List.length row = len) mm ->
  Forall2 (fun row rrow => ln_reg M m gamma beta eps row rgamma rbeta reps rrow) mm rm ->
  okm (errN M L (ln_depth n len)) (f32_layer_norm_2d gamma beta eps mm)
      (Rlayer_norm_2d rgamma rbeta reps len rm).
Proof.
  intros M m L n len gamma rgamma beta rbeta eps reps mm rm
         HM Hamp Hg Hb Heps Hmm Hlens Hregs.
  unfold f32_layer_norm_2d, Rlayer_norm_2d, okm in *.
  revert Hlens Hmm.
  induction Hregs as [|row rrow mm' rm' Hr Hregs IH]; intros Hlens Hmm; simpl;
    [constructor|].
  inversion Hmm as [|row' rrow' mm'' rm'' Hrow Hmms]; subst.
  pose proof (Forall_inv Hlens) as Hl1. pose proof (Forall_inv_tail Hlens) as Hl2.
  constructor.
  - rewrite <- Hl1. eapply ok_layer_norm_vec with (m := m); eassumption.
  - apply IH; assumption.
Qed.

(** * Depth-indexed forms of the primitives

    Every remaining stage is a chain of these, so stating them at [errN] level
    removes the arithmetic bookkeeping from the proofs that follow. *)

Lemma ok_plus_S : forall M m L k x y rx ry,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx -> ok (errN M L k) y ry ->
  regz M (B2R x + B2R y) ->
  ok (errN M L (S k)) (f32_plus x y) (rx + ry).
Proof.
  intros. change (errN M L (S k)) with (f32_u * M + L * errN M L k).
  eapply ok_plus; eassumption.
Qed.

Lemma ok_minus_S : forall M m L k x y rx ry,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx -> ok (errN M L k) y ry ->
  regz M (B2R x + B2R (f32_neg y)) ->
  ok (errN M L (S k)) (f32_minus x y) (rx - ry).
Proof.
  intros. change (errN M L (S k)) with (f32_u * M + L * errN M L k).
  eapply ok_minus; eassumption.
Qed.

Lemma ok_mult_S : forall M m L k x y rx ry,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx -> ok (errN M L k) y ry ->
  Rabs (B2R x) <= M -> Rabs ry <= M ->
  regz M (B2R x * B2R y) ->
  ok (errN M L (S k)) (f32_mult x y) (rx * ry).
Proof.
  intros. change (errN M L (S k)) with (f32_u * M + L * errN M L k).
  eapply ok_mult; eassumption.
Qed.

Lemma ok_div_S : forall M m L k x y rx ry,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx -> ok (errN M L k) y ry ->
  Rabs rx <= M -> m <= Rabs (B2R y) -> m <= Rabs ry ->
  regz M (B2R x / B2R y) ->
  ok (errN M L (S k)) (f32_div x y) (rx / ry).
Proof.
  intros. change (errN M L (S k)) with (f32_u * M + L * errN M L k).
  eapply ok_div; eassumption.
Qed.

Lemma ok_sqrt_S : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx ->
  m <= B2R x -> m <= rx ->
  regz M (sqrt (B2R x)) ->
  ok (errN M L (S k)) (f32_sqrt x) (sqrt rx).
Proof.
  intros. change (errN M L (S k)) with (f32_u * M + L * errN M L k).
  eapply ok_sqrt; eassumption.
Qed.

(** Constants of the program keep their own real value, so they are exact at
    any depth. *)
Lemma ok_const : forall M L k (c : binary32),
  0 <= M -> 1 <= L -> is_finite c = true -> ok (errN M L k) c (B2R c).
Proof.
  intros M L k c HM HL Hf.
  eapply ok_weaken; [apply ok_exact; exact Hf | apply errN_nonneg; assumption].
Qed.

(** * The exponential

    [f32_exp_approx] saturates its argument, divides by 256, evaluates a
    six-term Taylor series and squares eight times. The saturation is the only
    branch in the forward pass; the bound below is stated for arguments the
    saturation leaves alone, so the float and the real evaluation follow the
    same path. *)

Definition e_r  (x : binary32) : binary32 := f32_div x f32_exp_div.
Definition e_r2 (x : binary32) : binary32 := f32_mult (e_r x) (e_r x).
Definition e_r3 (x : binary32) : binary32 := f32_mult (e_r2 x) (e_r x).
Definition e_r4 (x : binary32) : binary32 := f32_mult (e_r3 x) (e_r x).
Definition e_r5 (x : binary32) : binary32 := f32_mult (e_r4 x) (e_r x).
Definition e_r6 (x : binary32) : binary32 := f32_mult (e_r5 x) (e_r x).

Definition e_d2 (x : binary32) : binary32 := f32_div (e_r2 x) f32_two.
Definition e_d3 (x : binary32) : binary32 := f32_div (e_r3 x) f32_six.
Definition e_d4 (x : binary32) : binary32 := f32_div (e_r4 x) f32_twenty_four.
Definition e_d5 (x : binary32) : binary32 := f32_div (e_r5 x) f32_one_twenty.
Definition e_d6 (x : binary32) : binary32 := f32_div (e_r6 x) f32_seven_twenty.

Definition e_p1 (x : binary32) : binary32 := f32_plus (e_d5 x) (e_d6 x).
Definition e_p2 (x : binary32) : binary32 := f32_plus (e_d4 x) (e_p1 x).
Definition e_p3 (x : binary32) : binary32 := f32_plus (e_d3 x) (e_p2 x).
Definition e_p4 (x : binary32) : binary32 := f32_plus (e_d2 x) (e_p3 x).
Definition e_p5 (x : binary32) : binary32 := f32_plus (e_r x) (e_p4 x).
Definition e_poly (x : binary32) : binary32 := f32_plus f32_one (e_p5 x).

Definition e_sq (s : binary32) : binary32 := f32_mult s s.

Definition sq8 (s0 : binary32) : binary32 :=
  let s := e_sq s0 in let s := e_sq s in let s := e_sq s in let s := e_sq s in
  let s := e_sq s in let s := e_sq s in let s := e_sq s in let s := e_sq s in s.

Lemma f32_exp_approx_unclamped : forall x,
  f32_lt f32_exp_hi x = false -> f32_lt x f32_exp_lo = false ->
  f32_exp_approx x = sq8 (e_poly x).
Proof.
  intros x H1 H2.
  unfold f32_exp_approx, sq8, e_sq, e_poly, e_p5, e_p4, e_p3, e_p2, e_p1,
         e_d6, e_d5, e_d4, e_d3, e_d2, e_r6, e_r5, e_r4, e_r3, e_r2, e_r.
  rewrite H1, H2. reflexivity.
Qed.

Lemma sq8_iter : forall s, sq8 s = Nat.iter 8 e_sq s.
Proof. intros s. reflexivity. Qed.

Definition Re_r  (rx : R) : R := rx / B2R f32_exp_div.
Definition Re_r2 (rx : R) : R := Re_r rx * Re_r rx.
Definition Re_r3 (rx : R) : R := Re_r2 rx * Re_r rx.
Definition Re_r4 (rx : R) : R := Re_r3 rx * Re_r rx.
Definition Re_r5 (rx : R) : R := Re_r4 rx * Re_r rx.
Definition Re_r6 (rx : R) : R := Re_r5 rx * Re_r rx.

Definition Re_d2 (rx : R) : R := Re_r2 rx / B2R f32_two.
Definition Re_d3 (rx : R) : R := Re_r3 rx / B2R f32_six.
Definition Re_d4 (rx : R) : R := Re_r4 rx / B2R f32_twenty_four.
Definition Re_d5 (rx : R) : R := Re_r5 rx / B2R f32_one_twenty.
Definition Re_d6 (rx : R) : R := Re_r6 rx / B2R f32_seven_twenty.

Definition Re_p1 (rx : R) : R := Re_d5 rx + Re_d6 rx.
Definition Re_p2 (rx : R) : R := Re_d4 rx + Re_p1 rx.
Definition Re_p3 (rx : R) : R := Re_d3 rx + Re_p2 rx.
Definition Re_p4 (rx : R) : R := Re_d2 rx + Re_p3 rx.
Definition Re_p5 (rx : R) : R := Re_r rx + Re_p4 rx.
Definition Re_poly (rx : R) : R := 1 + Re_p5 rx.

Definition Re_sq (s : R) : R := s * s.
Definition Re_exp_approx (rx : R) : R := Nat.iter 8 Re_sq (Re_poly rx).

(** Iterated squaring, with the magnitude conditions supplied per step. *)
Lemma ok_iter_sq : forall M m L j k s rs,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) s rs ->
  (forall i, (i < j)%nat ->
     Rabs (B2R (Nat.iter i e_sq s)) <= M
     /\ Rabs (Nat.iter i Re_sq rs) <= M
     /\ regz M (B2R (Nat.iter i e_sq s) * B2R (Nat.iter i e_sq s))) ->
  ok (errN M L (k + j)) (Nat.iter j e_sq s) (Nat.iter j Re_sq rs).
Proof.
  intros M m L j. induction j as [|j IH]; intros k s rs HM Hamp Hs Hcond.
  - simpl. rewrite Nat.add_0_r. exact Hs.
  - simpl. replace (k + S j)%nat with (S (k + j))%nat by lia.
    unfold e_sq at 1. unfold Re_sq at 1.
    eapply ok_mult_S with (m := m); try eassumption.
    + apply IH; try assumption. intros i Hi. apply Hcond. lia.
    + apply IH; try assumption. intros i Hi. apply Hcond. lia.
    + apply (Hcond j). lia.
    + apply (Hcond j). lia.
    + apply (Hcond j). lia.
Qed.

Lemma fin_exp_div : is_finite f32_exp_div = true.
Proof. vm_compute. reflexivity. Qed.
Lemma fin_two : is_finite f32_two = true.
Proof. vm_compute. reflexivity. Qed.
Lemma fin_six : is_finite f32_six = true.
Proof. vm_compute. reflexivity. Qed.
Lemma fin_twenty_four : is_finite f32_twenty_four = true.
Proof. vm_compute. reflexivity. Qed.
Lemma fin_one_twenty : is_finite f32_one_twenty = true.
Proof. vm_compute. reflexivity. Qed.
Lemma fin_seven_twenty : is_finite f32_seven_twenty = true.
Proof. vm_compute. reflexivity. Qed.

(** The side conditions of one exponential. *)
Record exp_reg (M m : R) (x : binary32) (rx : R) : Prop := {
  exr_hi : f32_lt f32_exp_hi x = false;
  exr_lo : f32_lt x f32_exp_lo = false;
  exr_bx : Rabs rx <= M;
  exr_mdiv : m <= Rabs (B2R f32_exp_div);
  exr_zr : regz M (B2R x / B2R f32_exp_div);
  exr_br : Rabs (B2R (e_r x)) <= M;
  exr_brr : Rabs (Re_r rx) <= M;
  exr_z2 : regz M (B2R (e_r x) * B2R (e_r x));
  exr_b2 : Rabs (B2R (e_r2 x)) <= M;
  exr_b2r : Rabs (Re_r2 rx) <= M;
  exr_z3 : regz M (B2R (e_r2 x) * B2R (e_r x));
  exr_b3 : Rabs (B2R (e_r3 x)) <= M;
  exr_b3r : Rabs (Re_r3 rx) <= M;
  exr_z4 : regz M (B2R (e_r3 x) * B2R (e_r x));
  exr_b4 : Rabs (B2R (e_r4 x)) <= M;
  exr_b4r : Rabs (Re_r4 rx) <= M;
  exr_z5 : regz M (B2R (e_r4 x) * B2R (e_r x));
  exr_b5 : Rabs (B2R (e_r5 x)) <= M;
  exr_b5r : Rabs (Re_r5 rx) <= M;
  exr_z6 : regz M (B2R (e_r5 x) * B2R (e_r x));
  exr_b6r : Rabs (Re_r6 rx) <= M;
  exr_m2 : m <= Rabs (B2R f32_two);
  exr_m6 : m <= Rabs (B2R f32_six);
  exr_m24 : m <= Rabs (B2R f32_twenty_four);
  exr_m120 : m <= Rabs (B2R f32_one_twenty);
  exr_m720 : m <= Rabs (B2R f32_seven_twenty);
  exr_zd2 : regz M (B2R (e_r2 x) / B2R f32_two);
  exr_zd3 : regz M (B2R (e_r3 x) / B2R f32_six);
  exr_zd4 : regz M (B2R (e_r4 x) / B2R f32_twenty_four);
  exr_zd5 : regz M (B2R (e_r5 x) / B2R f32_one_twenty);
  exr_zd6 : regz M (B2R (e_r6 x) / B2R f32_seven_twenty);
  exr_zp1 : regz M (B2R (e_d5 x) + B2R (e_d6 x));
  exr_zp2 : regz M (B2R (e_d4 x) + B2R (e_p1 x));
  exr_zp3 : regz M (B2R (e_d3 x) + B2R (e_p2 x));
  exr_zp4 : regz M (B2R (e_d2 x) + B2R (e_p3 x));
  exr_zp5 : regz M (B2R (e_r x) + B2R (e_p4 x));
  exr_zpoly : regz M (B2R f32_one + B2R (e_p5 x));
  exr_sq : forall i, (i < 8)%nat ->
     Rabs (B2R (Nat.iter i e_sq (e_poly x))) <= M
     /\ Rabs (Nat.iter i Re_sq (Re_poly rx)) <= M
     /\ regz M (B2R (Nat.iter i e_sq (e_poly x)) * B2R (Nat.iter i e_sq (e_poly x)))
}.

Lemma ok_exp_poly : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx ->
  exp_reg M m x rx ->
  ok (errN M L (k + 13)) (e_poly x) (Re_poly rx).
Proof.
  intros M m L k x rx HM Hamp Hx Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  destruct Hreg.
  assert (Hr : ok (errN M L (S k)) (e_r x) (Re_r rx)).
  { unfold e_r, Re_r. eapply ok_div_S with (m := m); try eassumption.
    apply ok_const; [exact HM0 | exact HL1 | exact fin_exp_div]. }
  assert (Hr2 : ok (errN M L (S (S k))) (e_r2 x) (Re_r2 rx)).
  { unfold e_r2, Re_r2. eapply ok_mult_S with (m := m); eassumption. }
  assert (Hr3 : ok (errN M L (S (S (S k)))) (e_r3 x) (Re_r3 rx)).
  { unfold e_r3, Re_r3. eapply ok_mult_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := (S k)%nat); [exact HM0 | exact HL1 | lia | exact Hr]. }
  assert (Hr4 : ok (errN M L (S (S (S (S k))))) (e_r4 x) (Re_r4 rx)).
  { unfold e_r4, Re_r4. eapply ok_mult_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := (S k)%nat); [exact HM0 | exact HL1 | lia | exact Hr]. }
  assert (Hr5 : ok (errN M L (S (S (S (S (S k)))))) (e_r5 x) (Re_r5 rx)).
  { unfold e_r5, Re_r5. eapply ok_mult_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := (S k)%nat); [exact HM0 | exact HL1 | lia | exact Hr]. }
  assert (Hr6 : ok (errN M L (S (S (S (S (S (S k))))))) (e_r6 x) (Re_r6 rx)).
  { unfold e_r6, Re_r6. eapply ok_mult_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := (S k)%nat); [exact HM0 | exact HL1 | lia | exact Hr]. }
  assert (Hd2 : ok (errN M L (k + 7)) (e_d2 x) (Re_d2 rx)).
  { unfold e_d2, Re_d2.
    replace (k + 7)%nat with (S (k + 6))%nat by lia.
    eapply ok_div_S with (m := m); try eassumption.
    - eapply ok_errN_mono with (n := (S (S k))%nat); [exact HM0 | exact HL1 | lia | exact Hr2].
    - apply ok_const; [exact HM0 | exact HL1 | exact fin_two]. }
  assert (Hd3 : ok (errN M L (k + 7)) (e_d3 x) (Re_d3 rx)).
  { unfold e_d3, Re_d3.
    replace (k + 7)%nat with (S (k + 6))%nat by lia.
    eapply ok_div_S with (m := m); try eassumption.
    - eapply ok_errN_mono with (n := (S (S (S k)))%nat); [exact HM0 | exact HL1 | lia | exact Hr3].
    - apply ok_const; [exact HM0 | exact HL1 | exact fin_six]. }
  assert (Hd4 : ok (errN M L (k + 7)) (e_d4 x) (Re_d4 rx)).
  { unfold e_d4, Re_d4.
    replace (k + 7)%nat with (S (k + 6))%nat by lia.
    eapply ok_div_S with (m := m); try eassumption.
    - eapply ok_errN_mono with (n := (S (S (S (S k))))%nat); [exact HM0 | exact HL1 | lia | exact Hr4].
    - apply ok_const; [exact HM0 | exact HL1 | exact fin_twenty_four]. }
  assert (Hd5 : ok (errN M L (k + 7)) (e_d5 x) (Re_d5 rx)).
  { unfold e_d5, Re_d5.
    replace (k + 7)%nat with (S (k + 6))%nat by lia.
    eapply ok_div_S with (m := m); try eassumption.
    - eapply ok_errN_mono with (n := (S (S (S (S (S k)))))%nat); [exact HM0 | exact HL1 | lia | exact Hr5].
    - apply ok_const; [exact HM0 | exact HL1 | exact fin_one_twenty]. }
  assert (Hd6 : ok (errN M L (k + 7)) (e_d6 x) (Re_d6 rx)).
  { unfold e_d6, Re_d6.
    replace (k + 7)%nat with (S (k + 6))%nat by lia.
    eapply ok_div_S with (m := m); try eassumption.
    - eapply ok_errN_mono with (n := (S (S (S (S (S (S k))))))%nat);
        [exact HM0 | exact HL1 | lia | exact Hr6].
    - apply ok_const; [exact HM0 | exact HL1 | exact fin_seven_twenty]. }
  assert (Hp1 : ok (errN M L (k + 8)) (e_p1 x) (Re_p1 rx)).
  { unfold e_p1, Re_p1. replace (k + 8)%nat with (S (k + 7))%nat by lia.
    eapply ok_plus_S with (m := m); eassumption. }
  assert (Hp2 : ok (errN M L (k + 9)) (e_p2 x) (Re_p2 rx)).
  { unfold e_p2, Re_p2. replace (k + 9)%nat with (S (k + 8))%nat by lia.
    eapply ok_plus_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := (k + 7)%nat); [exact HM0 | exact HL1 | lia | exact Hd4]. }
  assert (Hp3 : ok (errN M L (k + 10)) (e_p3 x) (Re_p3 rx)).
  { unfold e_p3, Re_p3. replace (k + 10)%nat with (S (k + 9))%nat by lia.
    eapply ok_plus_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := (k + 7)%nat); [exact HM0 | exact HL1 | lia | exact Hd3]. }
  assert (Hp4 : ok (errN M L (k + 11)) (e_p4 x) (Re_p4 rx)).
  { unfold e_p4, Re_p4. replace (k + 11)%nat with (S (k + 10))%nat by lia.
    eapply ok_plus_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := (k + 7)%nat); [exact HM0 | exact HL1 | lia | exact Hd2]. }
  assert (Hp5 : ok (errN M L (k + 12)) (e_p5 x) (Re_p5 rx)).
  { unfold e_p5, Re_p5. replace (k + 12)%nat with (S (k + 11))%nat by lia.
    eapply ok_plus_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := (S k)%nat); [exact HM0 | exact HL1 | lia | exact Hr]. }
  unfold e_poly, Re_poly. replace (k + 13)%nat with (S (k + 12))%nat by lia.
  eapply ok_plus_S with (m := m); try eassumption.
  replace 1 with (B2R f32_one) by apply f32_one_correct.
  apply ok_const; [exact HM0 | exact HL1 | exact f32_one_finite].
Qed.

Lemma ok_exp_approx : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx ->
  exp_reg M m x rx ->
  ok (errN M L (k + 21)) (f32_exp_approx x) (Re_exp_approx rx).
Proof.
  intros M m L k x rx HM Hamp Hx Hreg.
  rewrite (f32_exp_approx_unclamped x (exr_hi _ _ _ _ Hreg) (exr_lo _ _ _ _ Hreg)).
  rewrite sq8_iter. unfold Re_exp_approx.
  replace (k + 21)%nat with ((k + 13) + 8)%nat by lia.
  eapply ok_iter_sq with (m := m); try eassumption.
  - eapply ok_exp_poly with (m := m); eassumption.
  - exact (exr_sq _ _ _ _ Hreg).
Qed.

(** * Sigmoid and GELU *)

Definition Rsigmoid (rx : R) : R := 1 / (1 + Re_exp_approx (- rx)).

Record sig_reg (M m : R) (x : binary32) (rx : R) : Prop := {
  sgr_exp : exp_reg M m (f32_neg x) (- rx);
  sgr_z : regz M (B2R f32_one + B2R (f32_exp_approx (f32_neg x)));
  sgr_num : Rabs 1 <= M;
  sgr_lo : m <= Rabs (B2R (f32_plus f32_one (f32_exp_approx (f32_neg x))));
  sgr_lo_r : m <= Rabs (1 + Re_exp_approx (- rx));
  sgr_zd : regz M (B2R f32_one
                   / B2R (f32_plus f32_one (f32_exp_approx (f32_neg x))))
}.

Lemma ok_sigmoid : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx ->
  sig_reg M m x rx ->
  ok (errN M L (k + 23)) (f32_sigmoid x) (Rsigmoid rx).
Proof.
  intros M m L k x rx HM Hamp Hx Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  destruct Hreg as [Hexp Hz Hnum Hlo Hlor Hzd].
  assert (Hone : ok (errN M L (k + 22)) f32_one 1).
  { replace 1 with (B2R f32_one) by apply f32_one_correct.
    apply ok_const; [exact HM0 | exact HL1 | exact f32_one_finite]. }
  assert (He : ok (errN M L (k + 21)) (f32_exp_approx (f32_neg x))
                  (Re_exp_approx (- rx))).
  { eapply ok_exp_approx with (m := m); try eassumption. apply ok_neg, Hx. }
  assert (Hden : ok (errN M L (k + 22))
                    (f32_plus f32_one (f32_exp_approx (f32_neg x)))
                    (1 + Re_exp_approx (- rx))).
  { replace (k + 22)%nat with (S (k + 21))%nat by lia.
    eapply ok_plus_S with (m := m); try eassumption.
    replace 1 with (B2R f32_one) by apply f32_one_correct.
    apply ok_const; [exact HM0 | exact HL1 | exact f32_one_finite]. }
  unfold f32_sigmoid, Rsigmoid.
  replace (k + 23)%nat with (S (k + 22))%nat by lia.
  eapply ok_div_S with (m := m); eassumption.
Qed.

Definition Rgelu_c : R := B2R (f32_div f32_gelu_coeff f32_gelu_scale).

Definition Rgelu (rx : R) : R := rx * Rsigmoid (Rgelu_c * rx).

Record gelu_reg (M m : R) (x : binary32) (rx : R) : Prop := {
  glr_fin : is_finite (f32_div f32_gelu_coeff f32_gelu_scale) = true;
  glr_bc : Rabs (B2R (f32_div f32_gelu_coeff f32_gelu_scale)) <= M;
  glr_brx : Rabs rx <= M;
  glr_zc : regz M (B2R (f32_div f32_gelu_coeff f32_gelu_scale) * B2R x);
  glr_sig : sig_reg M m (f32_mult (f32_div f32_gelu_coeff f32_gelu_scale) x)
                    (Rgelu_c * rx);
  glr_bx : Rabs (B2R x) <= M;
  glr_bs : Rabs (Rsigmoid (Rgelu_c * rx)) <= M;
  glr_zm : regz M (B2R x
                   * B2R (f32_sigmoid (f32_mult
                            (f32_div f32_gelu_coeff f32_gelu_scale) x)))
}.

Lemma ok_gelu : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx ->
  gelu_reg M m x rx ->
  ok (errN M L (k + 25)) (f32_gelu x) (Rgelu rx).
Proof.
  intros M m L k x rx HM Hamp Hx Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  destruct Hreg as [Hfin Hbc Hbrx Hzc Hsig Hbx Hbs Hzm].
  assert (Hc : ok (errN M L k) (f32_div f32_gelu_coeff f32_gelu_scale) Rgelu_c)
    by (unfold Rgelu_c; apply ok_const; assumption).
  assert (Hcx : ok (errN M L (S k))
                   (f32_mult (f32_div f32_gelu_coeff f32_gelu_scale) x)
                   (Rgelu_c * rx))
    by (eapply ok_mult_S with (m := m); eassumption).
  assert (Hs : ok (errN M L (S k + 23))
                  (f32_sigmoid (f32_mult (f32_div f32_gelu_coeff f32_gelu_scale) x))
                  (Rsigmoid (Rgelu_c * rx)))
    by (eapply ok_sigmoid with (m := m); eassumption).
  unfold f32_gelu, Rgelu.
  replace (k + 25)%nat with (S (S k + 23))%nat by lia.
  eapply ok_mult_S with (m := m); try eassumption.
  eapply ok_errN_mono with (n := k); [exact HM0 | exact HL1 | lia | exact Hx].
Qed.

Lemma ok_gelu_vec : forall M m L k xs rs,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L k) xs rs ->
  Forall2 (gelu_reg M m) xs rs ->
  okv (errN M L (k + 25)) (f32_gelu_vec xs) (List.map Rgelu rs).
Proof.
  intros M m L k xs rs HM Hamp Hx Hreg.
  unfold f32_gelu_vec, okv in *.
  revert Hx. induction Hreg as [|x r xs' rs' Hr Hreg IH]; intros Hx; simpl;
    [constructor|].
  inversion Hx as [|x' rx xs'' rs'' Hxr Hxs]; subst.
  constructor.
  - eapply ok_gelu with (m := m); eassumption.
  - apply IH; exact Hxs.
Qed.

(** * The MLP *)

Definition lin_reg (M : R) (k : nat) (w : list (list binary32)) (b : list binary32)
                   (x : list (list binary32)) (rx : list (list R)) : Prop :=
  Forall (fun row => Forall (fun r => Rabs r <= M) row) rx
  /\ Forall (fun row =>
        Forall (fun r => dotreg M r row f32_zero /\ (List.length r <= k)%nat)
               (f32_mat_transpose w)
        /\ Forall2 (fun p q => regz M (B2R p + B2R q))
                   (f32_mat_vec_mul (f32_mat_transpose w) row) b) x.

Lemma ok_linear2d : forall M m L n k w rw b rb x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) w rw -> okv (errN M L n) b rb -> okm (errN M L n) x rx ->
  lin_reg M k w b x rx ->
  okm (errN M L (S (n + 2 * k))) (f32_linear_forward_2d w b x)
      (Rlinear_forward_2d rw rb rx).
Proof.
  intros M m L n k w rw b rb x rx HM Hamp Hw Hb Hx [H1 H2].
  eapply ok_linear_forward_2d with (m := m); eassumption.
Qed.

Definition Rmlp_forward (rcfw : list (list R)) (rcfb : list R)
                        (rcpw : list (list R)) (rcpb : list R)
                        (rh : list (list R)) : list (list R) :=
  let h := Rlinear_forward_2d rcfw rcfb rh in
  let hg := List.map (List.map Rgelu) h in
  Rlinear_forward_2d rcpw rcpb hg.

Lemma ok_mlp_forward : forall M m L n k1 k2 cfw rcfw cfb rcfb cpw rcpw cpb rcpb h rh,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) cfw rcfw -> okv (errN M L n) cfb rcfb ->
  okm (errN M L n) cpw rcpw -> okv (errN M L n) cpb rcpb ->
  okm (errN M L n) h rh ->
  lin_reg M k1 cfw cfb h rh ->
  Forall2 (Forall2 (gelu_reg M m))
          (f32_linear_forward_2d cfw cfb h) (Rlinear_forward_2d rcfw rcfb rh) ->
  lin_reg M k2 cpw cpb
          (List.map f32_gelu_vec (f32_linear_forward_2d cfw cfb h))
          (List.map (List.map Rgelu) (Rlinear_forward_2d rcfw rcfb rh)) ->
  okm (errN M L (S (S (n + 2 * k1) + 25 + 2 * k2)))
      (f32_mlp_forward cfw cfb cpw cpb h)
      (Rmlp_forward rcfw rcfb rcpw rcpb rh).
Proof.
  intros M m L n k1 k2 cfw rcfw cfb rcfb cpw rcpw cpb rcpb h rh
         HM Hamp Hcfw Hcfb Hcpw Hcpb Hh Hlin1 Hgel Hlin2.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  assert (Hstep1 : okm (errN M L (S (n + 2 * k1))) (f32_linear_forward_2d cfw cfb h)
                       (Rlinear_forward_2d rcfw rcfb rh))
    by (eapply ok_linear2d with (m := m); eassumption).
  assert (Hstep2 : okm (errN M L (S (n + 2 * k1) + 25))
                       (List.map f32_gelu_vec (f32_linear_forward_2d cfw cfb h))
                       (List.map (List.map Rgelu) (Rlinear_forward_2d rcfw rcfb rh))).
  { clear Hlin1 Hlin2. unfold okm in *.
    revert Hstep1.
    induction Hgel as [|r rr hs rhs Hrow Hgel IH]; intros Hstep1;
      cbn [List.map]; [constructor|].
    inversion Hstep1 as [|r' rr' hs' rhs' Hr Hrest]; subst.
    constructor.
    - eapply ok_gelu_vec with (m := m); eassumption.
    - apply IH; exact Hrest. }
  unfold f32_mlp_forward, Rmlp_forward.
  eapply ok_linear2d with (m := m); try eassumption.
  - eapply okm_weaken; [exact Hcpw | apply errN_mono; auto; lia].
  - eapply okv_weaken; [exact Hcpb | apply errN_mono; auto; lia].
Qed.

(** * Comparison, and the maximum a softmax subtracts *)

Lemma f32_lt_correct : forall x y,
  is_finite x = true -> is_finite y = true ->
  f32_lt x y = true <-> B2R x < B2R y.
Proof.
  intros x y Hx Hy. unfold f32_lt, f32_compare.
  rewrite (Bcompare_correct prec32 emax32 x y Hx Hy).
  destruct (Rcompare_spec (B2R x) (B2R y)) as [H|H|H]; split; intro Hc;
    try discriminate; try lra; try reflexivity.
Qed.

Definition Rmax_vec (rs : list R) : R :=
  match rs with
  | [] => 0
  | r :: rest => List.fold_left (fun acc y => Rmax acc y) rest r
  end.

Lemma ok_max_fold : forall d xs rs a ra,
  Forall (fun x => is_finite x = true) xs ->
  is_finite a = true ->
  Forall2 (ok d) xs rs -> ok d a ra ->
  ok d (List.fold_left (fun acc y => if f32_lt acc y then y else acc) xs a)
       (List.fold_left (fun acc y => Rmax acc y) rs ra).
Proof.
  intros d xs rs a ra Hfin Hfa H. revert a ra Hfa.
  induction H as [|x rx xs' rs' Hxr H IH]; intros a ra Hfa Ha; simpl; [exact Ha|].
  pose proof (Forall_inv Hfin) as Hfx. pose proof (Forall_inv_tail Hfin) as Hfin'.
  apply IH; try assumption.
  - destruct (f32_lt a x) eqn:E; assumption.
  - destruct Hxr as [_ Hdx]. destruct Ha as [_ Hda].
    destruct (f32_lt a x) eqn:E.
    + rewrite (f32_lt_correct a x Hfa Hfx) in E.
      split; [exact Hfx|].
      unfold Rmax. destruct (Rle_dec ra rx) as [Hle|Hlt].
      * exact Hdx.
      * apply Rabs_le. apply Rabs_le_inv in Hdx. apply Rabs_le_inv in Hda. lra.
    + assert (Hge : ~ (B2R a < B2R x)).
      { intro Hc. rewrite <- (f32_lt_correct a x Hfa Hfx) in Hc.
        rewrite Hc in E. discriminate. }
      split; [exact Hfa|].
      unfold Rmax. destruct (Rle_dec ra rx) as [Hle|Hlt].
      * apply Rabs_le. apply Rabs_le_inv in Hdx. apply Rabs_le_inv in Hda. lra.
      * exact Hda.
Qed.

Lemma ok_max_vec : forall d xs rs,
  Forall (fun x => is_finite x = true) xs ->
  Forall2 (ok d) xs rs -> 0 <= d ->
  ok d (f32_max_vec xs) (Rmax_vec rs).
Proof.
  intros d xs rs Hfin H Hd.
  destruct H as [|x rx xs' rs' Hxr H]; simpl.
  - apply ok_zero_any, Hd.
  - apply ok_max_fold; try assumption.
    + exact (Forall_inv_tail Hfin).
    + destruct Hxr; assumption.
Qed.

(** * Softmax *)

Definition Rsoftmax (rs : list R) : list R :=
  let mx := Rmax_vec rs in
  let shifted := List.map (fun r => r - mx) rs in
  let exps := List.map Re_exp_approx shifted in
  let se := Rsum exps in
  List.map (fun e => e / se) exps.

Record sm_reg (M m : R) (xs : list binary32) (rs : list R) : Prop := {
  smr_fin : Forall (fun x => is_finite x = true) xs;
  smr_shift :
    Forall2 (fun x r => regz M (B2R x + B2R (f32_neg (f32_max_vec xs)))) xs rs;
  smr_exp :
    Forall2 (fun x r => exp_reg M m (f32_minus x (f32_max_vec xs)) (r - Rmax_vec rs))
            xs rs;
  smr_sum :
    sumreg M (f32_exp_vec (List.map (fun x => f32_minus x (f32_max_vec xs)) xs))
           f32_zero;
  smr_div :
    Forall2 (fun e re =>
      Rabs re <= M
      /\ m <= Rabs (B2R (f32_sum (f32_exp_vec
                          (List.map (fun x => f32_minus x (f32_max_vec xs)) xs))))
      /\ m <= Rabs (Rsum (List.map Re_exp_approx
                          (List.map (fun r => r - (Rmax_vec rs)) rs)))
      /\ regz M (B2R e / B2R (f32_sum (f32_exp_vec
                    (List.map (fun x => f32_minus x (f32_max_vec xs)) xs)))))
      (f32_exp_vec (List.map (fun x => f32_minus x (f32_max_vec xs)) xs))
      (List.map Re_exp_approx (List.map (fun r => r - (Rmax_vec rs)) rs))
}.

Definition sm_depth (k len : nat) : nat := S (k + 22 + len).

Lemma ok_softmax : forall M m L k xs rs,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L k) xs rs ->
  sm_reg M m xs rs ->
  okv (errN M L (sm_depth k (List.length xs))) (f32_softmax xs) (Rsoftmax rs).
Proof.
  intros M m L k xs rs HM Hamp Hx Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  assert (Hd0 : 0 <= errN M L k) by (apply errN_nonneg; assumption).
  destruct Hreg as [Hfin Hshift Hexp Hsum Hdiv].
  assert (Hmx : ok (errN M L k) (f32_max_vec xs) (Rmax_vec rs))
    by (apply ok_max_vec; assumption).
  (* the maximum is fixed data from here on *)
  unfold f32_softmax, Rsoftmax, sm_depth.
  remember (f32_max_vec xs) as MX eqn:EMX.
  remember (Rmax_vec rs) as RMX eqn:ERMX.
  clear EMX ERMX Hfin.
  (* shifted *)
  assert (Hsh : okv (errN M L (S k))
                    (List.map (fun x => f32_minus x MX) xs)
                    (List.map (fun r => r - RMX) rs)).
  { unfold okv in *. clear Hexp Hsum Hdiv.
    revert Hx. induction Hshift as [|x r xs' rs' Hz Hshift IH]; intros Hx;
      cbn [List.map]; [constructor|].
    inversion Hx as [|x' rx xs'' rs'' Hxr Hxs]; subst.
    constructor.
    - eapply ok_minus_S with (m := m); eassumption.
    - apply IH; exact Hxs. }
  (* exponentials *)
  assert (Hex : okv (errN M L (S k + 21))
                    (f32_exp_vec (List.map (fun x => f32_minus x MX) xs))
                    (List.map Re_exp_approx (List.map (fun r => r - RMX) rs))).
  { unfold f32_exp_vec, okv in *. clear Hsum Hdiv Hx Hshift.
    revert Hsh. induction Hexp as [|x r xs' rs' He Hexp IH]; intros Hsh;
      cbn [List.map]; [constructor|].
    inversion Hsh as [|a ra as' ras' Har Hars]; subst.
    constructor.
    - eapply ok_exp_approx with (m := m); eassumption.
    - apply IH; exact Hars. }
  (* sum *)
  assert (Hlen : List.length (f32_exp_vec
                   (List.map (fun x => f32_minus x MX) xs))
                 = List.length xs)
    by (unfold f32_exp_vec; rewrite !List.length_map; reflexivity).
  assert (Hse : ok (errN M L (S k + 21 + List.length xs))
                   (f32_sum (f32_exp_vec
                      (List.map (fun x => f32_minus x MX) xs)))
                   (Rsum (List.map Re_exp_approx
                      (List.map (fun r => r - RMX) rs)))).
  { pose proof (ok_sum M m L (S k + 21) _ _ HM Hamp Hsum Hex) as H.
    rewrite Hlen in H. exact H. }
  (* divide *)
  replace (S (k + 22 + List.length xs))%nat
    with (S (S k + 21 + List.length xs))%nat by lia.
  unfold okv in *.
  assert (Hexw : Forall2 (ok (errN M L (S k + 21 + List.length xs)))
                   (f32_exp_vec (List.map (fun x => f32_minus x MX) xs))
                   (List.map Re_exp_approx (List.map (fun r => r - RMX) rs))).
  { eapply okv_weaken; [exact Hex | apply errN_mono; auto; lia]. }
  clear Hex Hsh Hx Hshift Hexp Hsum Hlen Hmx.
  remember (f32_sum (f32_exp_vec
              (List.map (fun x => f32_minus x MX) xs))) as SE eqn:ESE.
  remember (Rsum (List.map Re_exp_approx
              (List.map (fun r => r - RMX) rs))) as RSE eqn:ERSE.
  clear ESE ERSE.
  revert Hexw.
  induction Hdiv as [|e re es res (Hb & Hlo & Hlor & Hz) Hdiv IH]; intros Hexw;
    cbn [List.map]; [constructor|].
  inversion Hexw as [|e' re' es' res' Her Hers]; subst.
  constructor.
  - eapply ok_div_S with (m := m); eassumption.
  - apply IH; exact Hers.
Qed.

Definition Rsoftmax_2d (rm : list (list R)) : list (list R) := List.map Rsoftmax rm.

Lemma ok_softmax_2d : forall M m L k len mm rm,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L k) mm rm ->
  Forall (fun row => List.length row = len) mm ->
  Forall2 (sm_reg M m) mm rm ->
  okm (errN M L (sm_depth k len)) (f32_softmax_2d mm) (Rsoftmax_2d rm).
Proof.
  intros M m L k len mm rm HM Hamp Hmm Hlens Hregs.
  unfold f32_softmax_2d, Rsoftmax_2d, okm in *.
  revert Hlens Hmm.
  induction Hregs as [|row rrow mm' rm' Hr Hregs IH]; intros Hlens Hmm;
    cbn [List.map]; [constructor|].
  inversion Hmm as [|a b as' bs' Ha Has]; subst.
  pose proof (Forall_inv Hlens) as Hl1. pose proof (Forall_inv_tail Hlens) as Hl2.
  constructor.
  - rewrite <- Hl1. eapply ok_softmax with (m := m); eassumption.
  - apply IH; assumption.
Qed.

(** * Matrix product *)

Lemma ok_mat_mul : forall M m L n kk aa ra bb rb,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) aa ra -> okm (errN M L n) bb rb ->
  Forall (fun col => Forall (fun r => Rabs r <= M) col) (Rmat_transpose rb) ->
  Forall (fun ar => Forall (fun bc => dotreg M ar bc f32_zero
                                      /\ (List.length ar <= kk)%nat)
                           (f32_mat_transpose bb)) aa ->
  okm (errN M L (n + 2 * kk)) (f32_mat_mul aa bb) (Rmat_mul ra rb).
Proof.
  intros M m L n kk aa ra bb rb HM Hamp Ha Hb Hbnd Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  assert (Hbt : okm (errN M L n) (f32_mat_transpose bb) (Rmat_transpose rb))
    by (apply ok_mat_transpose; [apply errN_nonneg; assumption | exact Hb]).
  unfold f32_mat_mul, Rmat_mul, okm in *.
  clear Hb.
  revert Hreg.
  induction Ha as [|ar rar aa' ra' Har Ha IH]; intros Hreg; cbn [List.map];
    [constructor|].
  pose proof (Forall_inv Hreg) as Hr1. pose proof (Forall_inv_tail Hreg) as Hr2.
  constructor.
  - clear IH Ha Hreg Hr2.
    revert Hbnd Hr1.
    induction Hbt as [|bc rbc bt' rbt' Hbc Hbt IH]; intros Hbnd Hr1;
      cbn [List.map]; [constructor|].
    pose proof (Forall_inv Hbnd) as Hb1. pose proof (Forall_inv_tail Hbnd) as Hb2.
    pose proof (Forall_inv Hr1) as [Hd1 Hd2]. pose proof (Forall_inv_tail Hr1) as Hr1'.
    constructor.
    + eapply ok_errN_mono with (n := (n + 2 * List.length ar)%nat);
        [exact HM0 | exact HL1 | lia |].
      eapply ok_dot_n with (m := m); eassumption.
    + apply IH; assumption.
  - apply IH; exact Hr2.
Qed.

(** * Scaling, masking, and causal attention *)

Definition Rscale (d_k : nat) : R :=
  B2R (f32_div f32_one (f32_sqrt (f32_of_Z (Z.of_nat d_k)))).

Definition Rscale_scores (rm : list (list R)) (d_k : nat) : list (list R) :=
  List.map (fun row => List.map (fun x => x * Rscale d_k) row) rm.

Lemma ok_scale_scores : forall M m L k mm rm d_k,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L k) mm rm ->
  is_finite (f32_div f32_one (f32_sqrt (f32_of_Z (Z.of_nat d_k)))) = true ->
  Rabs (Rscale d_k) <= M ->
  Forall2 (fun row rrow =>
             Forall2 (fun x rx => Rabs (B2R x) <= M
                                  /\ regz M (B2R x
                                       * B2R (f32_div f32_one
                                                (f32_sqrt (f32_of_Z (Z.of_nat d_k))))))
                     row rrow) mm rm ->
  okm (errN M L (S k)) (f32_scale_scores mm d_k) (Rscale_scores rm d_k).
Proof.
  intros M m L k mm rm d_k HM Hamp Hmm Hfin Hbs Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  assert (Hs : ok (errN M L k) (f32_div f32_one (f32_sqrt (f32_of_Z (Z.of_nat d_k))))
                  (Rscale d_k))
    by (unfold Rscale; apply ok_const; assumption).
  unfold f32_scale_scores, Rscale_scores, okm in *.
  revert Hmm.
  induction Hreg as [|row rrow mm' rm' Hr Hreg IH]; intros Hmm;
    cbn [List.map]; [constructor|].
  inversion Hmm as [|a b as' bs' Ha Has]; subst.
  constructor.
  - clear IH Hreg Has Hmm.
    unfold okv in *.
    revert Ha. induction Hr as [|x rx row' rrow' [Hbx Hz] Hr IH]; intros Ha;
      cbn [List.map]; [constructor|].
    inversion Ha as [|x' rx' r1 r2 Hxr Hxs]; subst.
    constructor.
    + eapply ok_mult_S with (m := m); eassumption.
    + apply IH; exact Hxs.
  - apply IH; exact Has.
Qed.

Definition Rcausal_mask (n : nat) : list (list R) :=
  List.map (fun row => List.map (fun col => B2R (f32_causal_mask_entry row col))
                                (List.seq 0 n))
           (List.seq 0 n).

Lemma fin_mask_entry : forall r c, is_finite (f32_causal_mask_entry r c) = true.
Proof.
  intros r c. unfold f32_causal_mask_entry.
  destruct (Nat.leb c r); vm_compute; reflexivity.
Qed.

Lemma ok_causal_mask : forall d n, 0 <= d -> okm d (f32_causal_mask n) (Rcausal_mask n).
Proof.
  intros d n Hd. unfold f32_causal_mask, Rcausal_mask, okm.
  apply Forall2_map_seq. intros i.
  unfold okv. apply Forall2_map_seq. intros j.
  eapply ok_weaken; [apply ok_exact, fin_mask_entry | exact Hd].
Qed.

Definition Rapply_mask (rs rmask : list (list R)) : list (list R) :=
  List.map (fun '(sr, mr) => List.map (fun '(s, mv) => s + mv) (List.combine sr mr))
           (List.combine rs rmask).

Lemma ok_apply_mask : forall M m L k ss rss mk rmk,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L k) ss rss -> okm (errN M L k) mk rmk ->
  Forall2 (fun sr mr => Forall2 (fun s mv => regz M (B2R s + B2R mv)) sr mr) ss mk ->
  okm (errN M L (S k)) (f32_apply_mask ss mk) (Rapply_mask rss rmk).
Proof.
  intros M m L k ss rss mk rmk HM Hamp Hs Hm Hreg.
  unfold f32_apply_mask, Rapply_mask, okm in *.
  revert rss rmk Hs Hm.
  induction Hreg as [|sr mr ss' mk' Hrow Hreg IH]; intros rss rmk Hs Hm;
    cbn [List.map List.combine].
  - inversion Hs; subst. cbn [List.combine List.map]. constructor.
  - inversion Hs as [|a ra ss'' rss' Ha Has]; subst.
    inversion Hm as [|b rb mk'' rmk' Hb Hbs]; subst.
    cbn [List.combine List.map]. constructor.
    + clear IH Hreg Has Hbs Hs Hm.
      unfold okv in *.
      revert ra rb Ha Hb.
      induction Hrow as [|s mv sr' mr' Hz Hrow IH]; intros ra rb Ha Hb;
        cbn [List.combine List.map].
      * inversion Ha; subst. cbn [List.combine List.map]. constructor.
      * inversion Ha as [|s' rs1 sr'' ra' Har Has']; subst.
        inversion Hb as [|m' rm1 mr'' rb' Hbr Hbs']; subst.
        cbn [List.combine List.map]. constructor.
        -- eapply ok_plus_S with (m := m); eassumption.
        -- apply IH; assumption.
    + apply IH; assumption.
Qed.

Definition Rcausal_attention (rq rk rv : list (list R)) (d_k : nat) : list (list R) :=
  let scores := Rscale_scores (Rmat_mul rq (Rmat_transpose rk)) d_k in
  let masked := Rapply_mask scores (Rcausal_mask (List.length rq)) in
  Rmat_mul (Rsoftmax_2d masked) rv.

(** * Head splitting and concatenation

    These move values between layouts and perform no arithmetic, so they
    transport the relation unchanged. *)

Definition Rsplit_row_into_heads (num_heads : nat) (row : list R) : list (list R) :=
  let head_dim := Nat.div (List.length row) num_heads in
  List.map (fun h => List.firstn head_dim (List.skipn (h * head_dim) row))
           (List.seq 0 num_heads).

Definition Rsplit_into_heads (num_heads : nat) (rm : list (list R))
                             : list (list (list R)) :=
  let rows_split := List.map (Rsplit_row_into_heads num_heads) rm in
  List.map (fun h => List.map (fun row_heads => List.nth h row_heads []) rows_split)
           (List.seq 0 num_heads).

Definition Rconcat_heads (heads : list (list (list R))) : list (list R) :=
  match heads with
  | [] => []
  | first_head :: _ =>
      List.map (fun si => List.concat (List.map (fun head => List.nth si head []) heads))
               (List.seq 0 (List.length first_head))
  end.

Lemma ok_split_row : forall d nh row rrow,
  okv d row rrow -> List.length row = List.length rrow ->
  Forall2 (okv d) (f32_split_row_into_heads nh row) (Rsplit_row_into_heads nh rrow).
Proof.
  intros d nh row rrow H Hlen.
  unfold f32_split_row_into_heads, Rsplit_row_into_heads.
  rewrite Hlen.
  apply Forall2_map_seq. intros i.
  unfold okv. apply Forall2_firstn, Forall2_skipn, H.
Qed.

Lemma ok_split_heads : forall d nh mm rm,
  okm d mm rm ->
  Forall2 (okm d) (f32_split_into_heads nh mm) (Rsplit_into_heads nh rm).
Proof.
  intros d nh mm rm H.
  unfold f32_split_into_heads, Rsplit_into_heads.
  apply Forall2_map_seq. intros i.
  unfold okm. apply Forall2_map2 with (P := Forall2 (okv d)).
  - eapply Forall2_map2; [exact H|].
    intros a b Hab. apply ok_split_row; [exact Hab | eapply Forall2_length; exact Hab].
  - intros a b Hab. eapply Forall2_nth; [exact Hab | constructor].
Qed.

Lemma ok_concat_heads : forall d hs rhs,
  Forall2 (okm d) hs rhs ->
  okm d (f32_concat_heads hs) (Rconcat_heads rhs).
Proof.
  intros d hs rhs H.
  unfold f32_concat_heads, Rconcat_heads.
  destruct H as [|h rh hs' rhs' Hh H]; [constructor|].
  rewrite (Forall2_length _ _ _ _ _ Hh).
  unfold okm. apply Forall2_map_seq. intros i.
  unfold okv. apply Forall2_concat.
  constructor.
  - eapply Forall2_nth; [exact Hh | constructor].
  - eapply Forall2_map2; [exact H|].
    intros a b Hab. eapply Forall2_nth; [exact Hab | constructor].
Qed.

Definition ca_s1 (q k : list (list binary32)) : list (list binary32) :=
  f32_mat_mul q (f32_mat_transpose k).
Definition ca_s2 (q k : list (list binary32)) (d_k : nat) : list (list binary32) :=
  f32_scale_scores (ca_s1 q k) d_k.
Definition ca_s3 (q k : list (list binary32)) (d_k : nat) : list (list binary32) :=
  f32_apply_mask (ca_s2 q k d_k) (f32_causal_mask (f32_mat_rows q)).
Definition ca_s4 (q k : list (list binary32)) (d_k : nat) : list (list binary32) :=
  f32_softmax_2d (ca_s3 q k d_k).

Definition Rca_s1 (rq rk : list (list R)) : list (list R) :=
  Rmat_mul rq (Rmat_transpose rk).
Definition Rca_s2 (rq rk : list (list R)) (d_k : nat) : list (list R) :=
  Rscale_scores (Rca_s1 rq rk) d_k.
Definition Rca_s3 (rq rk : list (list R)) (d_k len : nat) : list (list R) :=
  Rapply_mask (Rca_s2 rq rk d_k) (Rcausal_mask len).
Definition Rca_s4 (rq rk : list (list R)) (d_k len : nat) : list (list R) :=
  Rsoftmax_2d (Rca_s3 rq rk d_k len).

Definition Rcausal_attention' (rq rk rv : list (list R)) (d_k len : nat)
                              : list (list R) :=
  Rmat_mul (Rca_s4 rq rk d_k len) rv.

Record ca_reg (M m : R) (q k v : list (list binary32)) (rq rk rv : list (list R))
              (d_k kq kv len : nat) : Prop := {
  car_rows : f32_mat_rows q = len;
  car_mm1_bnd : Forall (fun col => Forall (fun r => Rabs r <= M) col)
                       (Rmat_transpose (Rmat_transpose rk));
  car_mm1 : Forall (fun ar => Forall (fun bc => dotreg M ar bc f32_zero
                                                /\ (List.length ar <= kq)%nat)
                                     (f32_mat_transpose (f32_mat_transpose k))) q;
  car_sc_fin : is_finite (f32_div f32_one (f32_sqrt (f32_of_Z (Z.of_nat d_k)))) = true;
  car_sc_bnd : Rabs (Rscale d_k) <= M;
  car_sc : Forall2 (fun row rrow =>
             Forall2 (fun x rx => Rabs (B2R x) <= M
                        /\ regz M (B2R x * B2R (f32_div f32_one
                                     (f32_sqrt (f32_of_Z (Z.of_nat d_k)))))) row rrow)
             (ca_s1 q k) (Rca_s1 rq rk);
  car_mask : Forall2 (fun sr mr => Forall2 (fun s mv => regz M (B2R s + B2R mv)) sr mr)
                     (ca_s2 q k d_k) (f32_causal_mask len);
  car_sm_len : Forall (fun row => List.length row = len) (ca_s3 q k d_k);
  car_sm : Forall2 (sm_reg M m) (ca_s3 q k d_k) (Rca_s3 rq rk d_k len);
  car_mm2_bnd : Forall (fun col => Forall (fun r => Rabs r <= M) col)
                       (Rmat_transpose rv);
  car_mm2 : Forall (fun ar => Forall (fun bc => dotreg M ar bc f32_zero
                                                /\ (List.length ar <= kv)%nat)
                                     (f32_mat_transpose v)) (ca_s4 q k d_k)
}.

Definition ca_depth (n kq kv len : nat) : nat := sm_depth (n + 2 * kq + 2) len + 2 * kv.

Lemma ok_causal_attention : forall M m L n q rq k rk v rv d_k kq kv len,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) q rq -> okm (errN M L n) k rk -> okm (errN M L n) v rv ->
  ca_reg M m q k v rq rk rv d_k kq kv len ->
  okm (errN M L (ca_depth n kq kv len))
      (f32_causal_attention q k v d_k) (Rcausal_attention' rq rk rv d_k len).
Proof.
  intros M m L n q rq k rk v rv d_k kq kv len HM Hamp Hq Hk Hv Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  destruct Hreg as [Hrows Hb1 Hd1 Hfin Hbs Hsc Hmask Hsmlen Hsm Hb2 Hd2].
  assert (H1 : okm (errN M L (n + 2 * kq)) (ca_s1 q k) (Rca_s1 rq rk)).
  { unfold ca_s1, Rca_s1. eapply ok_mat_mul with (m := m); try eassumption.
    apply ok_mat_transpose; [apply errN_nonneg; assumption | exact Hk]. }
  assert (H2 : okm (errN M L (S (n + 2 * kq))) (ca_s2 q k d_k) (Rca_s2 rq rk d_k)).
  { unfold ca_s2, Rca_s2. eapply ok_scale_scores with (m := m); eassumption. }
  assert (H3 : okm (errN M L (S (S (n + 2 * kq)))) (ca_s3 q k d_k)
                   (Rca_s3 rq rk d_k len)).
  { unfold ca_s3, Rca_s3. rewrite Hrows.
    eapply ok_apply_mask with (m := m); try eassumption.
    apply ok_causal_mask, errN_nonneg; assumption. }
  assert (H4 : okm (errN M L (sm_depth (n + 2 * kq + 2) len)) (ca_s4 q k d_k)
                   (Rca_s4 rq rk d_k len)).
  { unfold ca_s4, Rca_s4.
    replace (n + 2 * kq + 2)%nat with (S (S (n + 2 * kq)))%nat by lia.
    eapply ok_softmax_2d with (m := m); eassumption. }
  unfold f32_causal_attention, Rcausal_attention', ca_depth.
  replace (f32_mat_mul (f32_softmax_2d
             (f32_apply_mask (f32_scale_scores (f32_mat_mul q (f32_mat_transpose k)) d_k)
                (f32_causal_mask (f32_mat_rows q)))) v)
    with (f32_mat_mul (ca_s4 q k d_k) v) by reflexivity.
  eapply ok_mat_mul with (m := m); try eassumption.
  eapply okm_weaken; [exact Hv | apply errN_mono; auto; unfold sm_depth; lia].
Qed.

(** * Multi-head attention *)

Definition Rattn_qkv (rcaw : list (list R)) (rcab : list R)
                     (rh : list (list R)) : list (list R) :=
  Rlinear_forward_2d rcaw rcab rh.

Definition f32_attn_split (d : nat) (qkv : list (list binary32))
                          : list (list binary32) * list (list binary32)
                            * list (list binary32) :=
  (List.map (fun row => List.firstn d row) qkv,
   List.map (fun row => List.firstn d (List.skipn d row)) qkv,
   List.map (fun row => List.skipn (2 * d) row) qkv).

Definition Rattn_split (d : nat) (qkv : list (list R))
                       : list (list R) * list (list R) * list (list R) :=
  (List.map (fun row => List.firstn d row) qkv,
   List.map (fun row => List.firstn d (List.skipn d row)) qkv,
   List.map (fun row => List.skipn (2 * d) row) qkv).

Lemma attn_split_eq : forall (A : Type) (d : nat) (qkv : list (list A)),
  List.map (fun '(q, _, _) => q)
    (List.map (fun row =>
       (List.firstn d row, List.firstn d (List.skipn d row), List.skipn (2 * d) row))
       qkv)
  = List.map (fun row => List.firstn d row) qkv
  /\ List.map (fun '(_, k, _) => k)
       (List.map (fun row =>
          (List.firstn d row, List.firstn d (List.skipn d row), List.skipn (2 * d) row))
          qkv)
     = List.map (fun row => List.firstn d (List.skipn d row)) qkv
  /\ List.map (fun '(_, _, v) => v)
       (List.map (fun row =>
          (List.firstn d row, List.firstn d (List.skipn d row), List.skipn (2 * d) row))
          qkv)
     = List.map (fun row => List.skipn (2 * d) row) qkv.
Proof.
  intros A d qkv. repeat split; rewrite List.map_map; reflexivity.
Qed.

Lemma ok_firstn_map : forall d n mm rm,
  okm d mm rm -> okm d (List.map (fun row => List.firstn n row) mm)
                       (List.map (fun row => List.firstn n row) rm).
Proof.
  intros d n mm rm H. unfold okm in *.
  eapply Forall2_map2; [exact H|]. intros a b Hab. apply Forall2_firstn, Hab.
Qed.

Lemma ok_midslice_map : forall d n mm rm,
  okm d mm rm ->
  okm d (List.map (fun row => List.firstn n (List.skipn n row)) mm)
        (List.map (fun row => List.firstn n (List.skipn n row)) rm).
Proof.
  intros d n mm rm H. unfold okm in *.
  eapply Forall2_map2; [exact H|].
  intros a b Hab. apply Forall2_firstn, Forall2_skipn, Hab.
Qed.

Lemma ok_skipn_map : forall d n mm rm,
  okm d mm rm -> okm d (List.map (fun row => List.skipn n row) mm)
                       (List.map (fun row => List.skipn n row) rm).
Proof.
  intros d n mm rm H. unfold okm in *.
  eapply Forall2_map2; [exact H|]. intros a b Hab. apply Forall2_skipn, Hab.
Qed.

Lemma Forall2_combine : forall (A B C D : Type) (P : A -> C -> Prop) (Q : B -> D -> Prop)
    xs rs ys ss,
  Forall2 P xs rs -> Forall2 Q ys ss ->
  Forall2 (fun p q => P (fst p) (fst q) /\ Q (snd p) (snd q))
          (List.combine xs ys) (List.combine rs ss).
Proof.
  intros A B C D P Q xs rs ys ss H1. revert ys ss.
  induction H1 as [|a c xs rs Hac H1 IH]; intros ys ss H2;
    cbn [List.combine]; [constructor|].
  destruct H2 as [|b d ys ss Hbd H2]; cbn [List.combine]; constructor.
  - split; assumption.
  - apply IH; exact H2.
Qed.

(** The three projections of the fused QKV row. *)
Definition af_q (d : nat) (qkv : list (list binary32)) : list (list binary32) :=
  List.map (fun row => List.firstn d row) qkv.
Definition af_k (d : nat) (qkv : list (list binary32)) : list (list binary32) :=
  List.map (fun row => List.firstn d (List.skipn d row)) qkv.
Definition af_v (d : nat) (qkv : list (list binary32)) : list (list binary32) :=
  List.map (fun row => List.skipn (2 * d) row) qkv.

Definition Raf_q (d : nat) (qkv : list (list R)) : list (list R) :=
  List.map (fun row => List.firstn d row) qkv.
Definition Raf_k (d : nat) (qkv : list (list R)) : list (list R) :=
  List.map (fun row => List.firstn d (List.skipn d row)) qkv.
Definition Raf_v (d : nat) (qkv : list (list R)) : list (list R) :=
  List.map (fun row => List.skipn (2 * d) row) qkv.

Definition af_triples (nh d : nat) (qkv : list (list binary32)) :=
  List.combine (f32_split_into_heads nh (af_q d qkv))
    (List.combine (f32_split_into_heads nh (af_k d qkv))
                  (f32_split_into_heads nh (af_v d qkv))).

Definition Raf_triples (nh d : nat) (qkv : list (list R)) :=
  List.combine (Rsplit_into_heads nh (Raf_q d qkv))
    (List.combine (Rsplit_into_heads nh (Raf_k d qkv))
                  (Rsplit_into_heads nh (Raf_v d qkv))).

Definition af_outs (nh d hd : nat) (qkv : list (list binary32)) :=
  List.map (fun '(qh, (kh, vh)) => f32_causal_attention qh kh vh hd)
           (af_triples nh d qkv).

Definition Raf_outs (nh d hd len : nat) (qkv : list (list R)) :=
  List.map (fun '(qh, (kh, vh)) => Rcausal_attention' qh kh vh hd len)
           (Raf_triples nh d qkv).

Definition Rattention_forward (n_embd n_head : nat)
    (rcaw : list (list R)) (rcab : list R)
    (rcpw : list (list R)) (rcpb : list R)
    (rhidden : list (list R)) (len : nat) : list (list R) :=
  let qkv := Rlinear_forward_2d rcaw rcab rhidden in
  Rlinear_forward_2d rcpw rcpb
    (Rconcat_heads (Raf_outs n_head n_embd (Nat.div n_embd n_head) len qkv)).

Definition af_depth (n k1 kq kv len k2 : nat) : nat :=
  S (ca_depth (S (n + 2 * k1)) kq kv len + 2 * k2).

Lemma ok_attention_forward :
  forall M m L n n_embd n_head k1 k2 kq kv len
         caw rcaw cab rcab cpw rcpw cpb rcpb hidden rhidden,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) caw rcaw -> okv (errN M L n) cab rcab ->
  okm (errN M L n) cpw rcpw -> okv (errN M L n) cpb rcpb ->
  okm (errN M L n) hidden rhidden ->
  lin_reg M k1 caw cab hidden rhidden ->
  Forall2 (fun p rp =>
      ca_reg M m (fst p) (fst (snd p)) (snd (snd p))
             (fst rp) (fst (snd rp)) (snd (snd rp))
             (Nat.div n_embd n_head) kq kv len)
    (af_triples n_head n_embd (f32_linear_forward_2d caw cab hidden))
    (Raf_triples n_head n_embd (Rlinear_forward_2d rcaw rcab rhidden)) ->
  lin_reg M k2 cpw cpb
    (f32_concat_heads (af_outs n_head n_embd (Nat.div n_embd n_head)
                               (f32_linear_forward_2d caw cab hidden)))
    (Rconcat_heads (Raf_outs n_head n_embd (Nat.div n_embd n_head) len
                             (Rlinear_forward_2d rcaw rcab rhidden))) ->
  okm (errN M L (af_depth n k1 kq kv len k2))
      (f32_attention_forward n_embd n_head caw cab cpw cpb hidden)
      (Rattention_forward n_embd n_head rcaw rcab rcpw rcpb rhidden len).
Proof.
  intros M m L n n_embd n_head k1 k2 kq kv len
         caw rcaw cab rcab cpw rcpw cpb rcpb hidden rhidden
         HM Hamp Hcaw Hcab Hcpw Hcpb Hh Hlin1 Hheads Hlin2.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  set (n1 := S (n + 2 * k1)).
  assert (Hqkv : okm (errN M L n1) (f32_linear_forward_2d caw cab hidden)
                     (Rlinear_forward_2d rcaw rcab rhidden))
    by (unfold n1; eapply ok_linear2d with (m := m); eassumption).
  assert (Hq : Forall2 (okm (errN M L n1))
                 (f32_split_into_heads n_head
                    (af_q n_embd (f32_linear_forward_2d caw cab hidden)))
                 (Rsplit_into_heads n_head
                    (Raf_q n_embd (Rlinear_forward_2d rcaw rcab rhidden))))
    by (apply ok_split_heads, ok_firstn_map, Hqkv).
  assert (Hk : Forall2 (okm (errN M L n1))
                 (f32_split_into_heads n_head
                    (af_k n_embd (f32_linear_forward_2d caw cab hidden)))
                 (Rsplit_into_heads n_head
                    (Raf_k n_embd (Rlinear_forward_2d rcaw rcab rhidden))))
    by (apply ok_split_heads, ok_midslice_map, Hqkv).
  assert (Hv : Forall2 (okm (errN M L n1))
                 (f32_split_into_heads n_head
                    (af_v n_embd (f32_linear_forward_2d caw cab hidden)))
                 (Rsplit_into_heads n_head
                    (Raf_v n_embd (Rlinear_forward_2d rcaw rcab rhidden))))
    by (apply ok_split_heads, ok_skipn_map, Hqkv).
  pose proof (Forall2_combine _ _ _ _ _ _ _ _ _ _ Hk Hv) as Hkv.
  pose proof (Forall2_combine _ _ _ _ _ _ _ _ _ _ Hq Hkv) as Hqkvh.
  pose proof (Forall2_conj _ _ _ _ _ _ Hqkvh Hheads) as Hall.
  assert (Houts : Forall2 (okm (errN M L (ca_depth n1 kq kv len)))
    (af_outs n_head n_embd (Nat.div n_embd n_head)
             (f32_linear_forward_2d caw cab hidden))
    (Raf_outs n_head n_embd (Nat.div n_embd n_head) len
              (Rlinear_forward_2d rcaw rcab rhidden))).
  { unfold af_outs, Raf_outs.
    eapply Forall2_map2; [exact Hall|].
    intros p rp [[Hp1 [Hp2 Hp3]] Hca].
    destruct p as [qh [kh vh]]. destruct rp as [rqh [rkh rvh]].
    cbn [fst snd] in Hp1, Hp2, Hp3, Hca |- *.
    eapply ok_causal_attention with (m := m); eassumption. }
  assert (Hcat : okm (errN M L (ca_depth n1 kq kv len))
    (f32_concat_heads (af_outs n_head n_embd (Nat.div n_embd n_head)
                               (f32_linear_forward_2d caw cab hidden)))
    (Rconcat_heads (Raf_outs n_head n_embd (Nat.div n_embd n_head) len
                             (Rlinear_forward_2d rcaw rcab rhidden))))
    by (apply ok_concat_heads, Houts).
  assert (Heq : f32_attention_forward n_embd n_head caw cab cpw cpb hidden
                = f32_linear_forward_2d cpw cpb
                    (f32_concat_heads
                       (af_outs n_head n_embd (Nat.div n_embd n_head)
                                (f32_linear_forward_2d caw cab hidden)))).
  { unfold f32_attention_forward, af_outs, af_triples, af_q, af_k, af_v.
    cbv zeta. rewrite !List.map_map. reflexivity. }
  rewrite Heq. unfold Rattention_forward, af_depth.
  eapply ok_linear2d with (m := m); try eassumption.
  - eapply okm_weaken; [exact Hcpw
    | apply errN_mono; auto; unfold ca_depth, sm_depth, n1; lia].
  - eapply okv_weaken; [exact Hcpb
    | apply errN_mono; auto; unfold ca_depth, sm_depth, n1; lia].
Qed.

(** * The transformer block

    The real-side weights are taken as plain lists rather than mirrored
    records, so the block lemma states exactly which real matrix each float
    matrix is being compared against. *)

Definition Rblock_forward
    (rln1w rln1b : list R) (rcaw : list (list R)) (rcab : list R)
    (rcpw : list (list R)) (rcpb : list R)
    (rln2w rln2b : list R) (rcfw : list (list R)) (rcfb : list R)
    (rmpw : list (list R)) (rmpb : list R)
    (reps : R) (n_embd n_head len : nat) (rhidden : list (list R))
    : list (list R) :=
  let ln1 := Rlayer_norm_2d rln1w rln1b reps len rhidden in
  let attn := Rattention_forward n_embd n_head rcaw rcab rcpw rcpb ln1 len in
  let hidden2 := Radd_matrices rhidden attn in
  let ln2 := Rlayer_norm_2d rln2w rln2b reps len hidden2 in
  let mlp := Rmlp_forward rcfw rcfb rmpw rmpb ln2 in
  Radd_matrices hidden2 mlp.

Definition blk_depth (n len k1 kq kv k2 km1 km2 : nat) : nat :=
  let d1 := ln_depth n len in
  let d2 := af_depth d1 k1 kq kv len k2 in
  let d3 := S d2 in
  let d4 := ln_depth d3 len in
  let d5 := S (S (d4 + 2 * km1) + 25 + 2 * km2) in
  S d5.

Lemma ok_block_forward :
  forall M m L n cfg eps reps block hidden rhidden
         rln1w rln1b rcaw rcab rcpw rcpb rln2w rln2b rcfw rcfb rmpw rmpb
         len k1 kq kv k2 km1 km2,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L n) eps reps ->
  okm (errN M L n) hidden rhidden ->
  okv (errN M L n) (f32_ln_weight (f32_block_ln_1 block)) rln1w ->
  okv (errN M L n) (f32_ln_bias (f32_block_ln_1 block)) rln1b ->
  okm (errN M L n) (f32_attn_c_attn_weight (f32_block_attn block)) rcaw ->
  okv (errN M L n) (f32_attn_c_attn_bias (f32_block_attn block)) rcab ->
  okm (errN M L n) (f32_attn_c_proj_weight (f32_block_attn block)) rcpw ->
  okv (errN M L n) (f32_attn_c_proj_bias (f32_block_attn block)) rcpb ->
  okv (errN M L n) (f32_ln_weight (f32_block_ln_2 block)) rln2w ->
  okv (errN M L n) (f32_ln_bias (f32_block_ln_2 block)) rln2b ->
  okm (errN M L n) (f32_mlp_c_fc_weight (f32_block_mlp block)) rcfw ->
  okv (errN M L n) (f32_mlp_c_fc_bias (f32_block_mlp block)) rcfb ->
  okm (errN M L n) (f32_mlp_c_proj_weight (f32_block_mlp block)) rmpw ->
  okv (errN M L n) (f32_mlp_c_proj_bias (f32_block_mlp block)) rmpb ->
  (* layer norm 1 *)
  Forall (fun row => List.length row = len) hidden ->
  Forall2 (fun row rrow =>
             ln_reg M m (f32_ln_weight (f32_block_ln_1 block))
                    (f32_ln_bias (f32_block_ln_1 block)) eps row
                    rln1w rln1b reps rrow) hidden rhidden ->
  (* attention *)
  lin_reg M k1 (f32_attn_c_attn_weight (f32_block_attn block))
               (f32_attn_c_attn_bias (f32_block_attn block))
               (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                                  (f32_ln_bias (f32_block_ln_1 block)) eps hidden)
               (Rlayer_norm_2d rln1w rln1b reps len rhidden) ->
  Forall2 (fun p rp =>
      ca_reg M m (fst p) (fst (snd p)) (snd (snd p))
             (fst rp) (fst (snd rp)) (snd (snd rp))
             (Nat.div (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)) kq kv len)
    (af_triples (gpt2_inf_n_head cfg) (gpt2_inf_n_embd cfg)
       (f32_linear_forward_2d (f32_attn_c_attn_weight (f32_block_attn block))
          (f32_attn_c_attn_bias (f32_block_attn block))
          (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                             (f32_ln_bias (f32_block_ln_1 block)) eps hidden)))
    (Raf_triples (gpt2_inf_n_head cfg) (gpt2_inf_n_embd cfg)
       (Rlinear_forward_2d rcaw rcab (Rlayer_norm_2d rln1w rln1b reps len rhidden))) ->
  lin_reg M k2 (f32_attn_c_proj_weight (f32_block_attn block))
               (f32_attn_c_proj_bias (f32_block_attn block))
    (f32_concat_heads (af_outs (gpt2_inf_n_head cfg) (gpt2_inf_n_embd cfg)
       (Nat.div (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg))
       (f32_linear_forward_2d (f32_attn_c_attn_weight (f32_block_attn block))
          (f32_attn_c_attn_bias (f32_block_attn block))
          (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                             (f32_ln_bias (f32_block_ln_1 block)) eps hidden))))
    (Rconcat_heads (Raf_outs (gpt2_inf_n_head cfg) (gpt2_inf_n_embd cfg)
       (Nat.div (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)) len
       (Rlinear_forward_2d rcaw rcab (Rlayer_norm_2d rln1w rln1b reps len rhidden)))) ->
  (* first residual *)
  Forall2 (fun r1 r2 => Forall2 (fun x y => regz M (B2R x + B2R y)) r1 r2)
    hidden (f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
              (f32_attn_c_attn_weight (f32_block_attn block))
              (f32_attn_c_attn_bias (f32_block_attn block))
              (f32_attn_c_proj_weight (f32_block_attn block))
              (f32_attn_c_proj_bias (f32_block_attn block))
              (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                                 (f32_ln_bias (f32_block_ln_1 block)) eps hidden)) ->
  (* layer norm 2 *)
  Forall (fun row => List.length row = len)
    (f32_add_matrices hidden
       (f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
          (f32_attn_c_attn_weight (f32_block_attn block))
          (f32_attn_c_attn_bias (f32_block_attn block))
          (f32_attn_c_proj_weight (f32_block_attn block))
          (f32_attn_c_proj_bias (f32_block_attn block))
          (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                             (f32_ln_bias (f32_block_ln_1 block)) eps hidden))) ->
  Forall2 (fun row rrow =>
             ln_reg M m (f32_ln_weight (f32_block_ln_2 block))
                    (f32_ln_bias (f32_block_ln_2 block)) eps row
                    rln2w rln2b reps rrow)
    (f32_add_matrices hidden
       (f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
          (f32_attn_c_attn_weight (f32_block_attn block))
          (f32_attn_c_attn_bias (f32_block_attn block))
          (f32_attn_c_proj_weight (f32_block_attn block))
          (f32_attn_c_proj_bias (f32_block_attn block))
          (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                             (f32_ln_bias (f32_block_ln_1 block)) eps hidden)))
    (Radd_matrices rhidden
       (Rattention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
          rcaw rcab rcpw rcpb (Rlayer_norm_2d rln1w rln1b reps len rhidden) len)) ->
  (* MLP *)
  lin_reg M km1 (f32_mlp_c_fc_weight (f32_block_mlp block))
                (f32_mlp_c_fc_bias (f32_block_mlp block))
    (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_2 block))
       (f32_ln_bias (f32_block_ln_2 block)) eps
       (f32_add_matrices hidden
          (f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
             (f32_attn_c_attn_weight (f32_block_attn block))
             (f32_attn_c_attn_bias (f32_block_attn block))
             (f32_attn_c_proj_weight (f32_block_attn block))
             (f32_attn_c_proj_bias (f32_block_attn block))
             (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                                (f32_ln_bias (f32_block_ln_1 block)) eps hidden))))
    (Rlayer_norm_2d rln2w rln2b reps len
       (Radd_matrices rhidden
          (Rattention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
             rcaw rcab rcpw rcpb (Rlayer_norm_2d rln1w rln1b reps len rhidden) len))) ->
  Forall2 (Forall2 (gelu_reg M m))
    (f32_linear_forward_2d (f32_mlp_c_fc_weight (f32_block_mlp block))
       (f32_mlp_c_fc_bias (f32_block_mlp block))
       (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_2 block))
          (f32_ln_bias (f32_block_ln_2 block)) eps
          (f32_add_matrices hidden
             (f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                (f32_attn_c_attn_weight (f32_block_attn block))
                (f32_attn_c_attn_bias (f32_block_attn block))
                (f32_attn_c_proj_weight (f32_block_attn block))
                (f32_attn_c_proj_bias (f32_block_attn block))
                (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                                   (f32_ln_bias (f32_block_ln_1 block)) eps hidden)))))
    (Rlinear_forward_2d rcfw rcfb
       (Rlayer_norm_2d rln2w rln2b reps len
          (Radd_matrices rhidden
             (Rattention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                rcaw rcab rcpw rcpb (Rlayer_norm_2d rln1w rln1b reps len rhidden) len)))) ->
  lin_reg M km2 (f32_mlp_c_proj_weight (f32_block_mlp block))
                (f32_mlp_c_proj_bias (f32_block_mlp block))
    (List.map f32_gelu_vec
       (f32_linear_forward_2d (f32_mlp_c_fc_weight (f32_block_mlp block))
          (f32_mlp_c_fc_bias (f32_block_mlp block))
          (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_2 block))
             (f32_ln_bias (f32_block_ln_2 block)) eps
             (f32_add_matrices hidden
                (f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                   (f32_attn_c_attn_weight (f32_block_attn block))
                   (f32_attn_c_attn_bias (f32_block_attn block))
                   (f32_attn_c_proj_weight (f32_block_attn block))
                   (f32_attn_c_proj_bias (f32_block_attn block))
                   (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                                      (f32_ln_bias (f32_block_ln_1 block)) eps hidden))))))
    (List.map (List.map Rgelu)
       (Rlinear_forward_2d rcfw rcfb
          (Rlayer_norm_2d rln2w rln2b reps len
             (Radd_matrices rhidden
                (Rattention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                   rcaw rcab rcpw rcpb
                   (Rlayer_norm_2d rln1w rln1b reps len rhidden) len))))) ->
  (* second residual *)
  Forall2 (fun r1 r2 => Forall2 (fun x y => regz M (B2R x + B2R y)) r1 r2)
    (f32_add_matrices hidden
       (f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
          (f32_attn_c_attn_weight (f32_block_attn block))
          (f32_attn_c_attn_bias (f32_block_attn block))
          (f32_attn_c_proj_weight (f32_block_attn block))
          (f32_attn_c_proj_bias (f32_block_attn block))
          (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                             (f32_ln_bias (f32_block_ln_1 block)) eps hidden)))
    (f32_mlp_forward (f32_mlp_c_fc_weight (f32_block_mlp block))
       (f32_mlp_c_fc_bias (f32_block_mlp block))
       (f32_mlp_c_proj_weight (f32_block_mlp block))
       (f32_mlp_c_proj_bias (f32_block_mlp block))
       (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_2 block))
          (f32_ln_bias (f32_block_ln_2 block)) eps
          (f32_add_matrices hidden
             (f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                (f32_attn_c_attn_weight (f32_block_attn block))
                (f32_attn_c_attn_bias (f32_block_attn block))
                (f32_attn_c_proj_weight (f32_block_attn block))
                (f32_attn_c_proj_bias (f32_block_attn block))
                (f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                                   (f32_ln_bias (f32_block_ln_1 block)) eps hidden))))) ->
  okm (errN M L (blk_depth n len k1 kq kv k2 km1 km2))
      (f32_block_forward cfg eps block hidden)
      (Rblock_forward rln1w rln1b rcaw rcab rcpw rcpb rln2w rln2b rcfw rcfb rmpw rmpb
                      reps (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg) len rhidden).
Proof.
  intros M m L n cfg eps reps block hidden rhidden
         rln1w rln1b rcaw rcab rcpw rcpb rln2w rln2b rcfw rcfb rmpw rmpb
         len k1 kq kv k2 km1 km2
         HM Hamp Heps Hh H1w H1b Hcaw Hcab Hcpw Hcpb H2w H2b Hcfw Hcfb Hmpw Hmpb
         Hlen1 Hln1 Hlin1 Hheads Hlin2 Hres1 Hlen2 Hln2 Hmlin1 Hgel Hmlin2 Hres2.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  unfold f32_block_forward, Rblock_forward, blk_depth. cbv zeta.
  set (LN1 := f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                (f32_ln_bias (f32_block_ln_1 block)) eps hidden) in *.
  set (RLN1 := Rlayer_norm_2d rln1w rln1b reps len rhidden) in *.
  set (ATT := f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                (f32_attn_c_attn_weight (f32_block_attn block))
                (f32_attn_c_attn_bias (f32_block_attn block))
                (f32_attn_c_proj_weight (f32_block_attn block))
                (f32_attn_c_proj_bias (f32_block_attn block)) LN1) in *.
  set (RATT := Rattention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                 rcaw rcab rcpw rcpb RLN1 len) in *.
  set (HH2 := f32_add_matrices hidden ATT) in *.
  set (RHH2 := Radd_matrices rhidden RATT) in *.
  set (LN2 := f32_layer_norm_2d (f32_ln_weight (f32_block_ln_2 block))
                (f32_ln_bias (f32_block_ln_2 block)) eps HH2) in *.
  set (RLN2 := Rlayer_norm_2d rln2w rln2b reps len RHH2) in *.
  set (d1 := ln_depth n len).
  assert (A1 : okm (errN M L d1) LN1 RLN1)
    by (unfold d1, LN1, RLN1; eapply ok_layer_norm_2d with (m := m); eassumption).
  set (d2 := af_depth d1 k1 kq kv len k2).
  assert (A2 : okm (errN M L d2) ATT RATT).
  { unfold d2, ATT, RATT. eapply ok_attention_forward with (m := m); try eassumption;
      (eapply okm_weaken; [eassumption | apply errN_mono; auto; unfold d1, ln_depth; lia])
      || (eapply okv_weaken; [eassumption | apply errN_mono; auto; unfold d1, ln_depth; lia]). }
  set (d3 := S d2).
  assert (A3 : okm (errN M L d3) HH2 RHH2).
  { unfold d3, HH2, RHH2. eapply ok_add_matrices with (m := m); try eassumption.
    eapply okm_weaken; [exact Hh | apply errN_mono; auto;
      unfold d2, af_depth, ca_depth, sm_depth, d1, ln_depth; lia]. }
  set (d4 := ln_depth d3 len).
  assert (A4 : okm (errN M L d4) LN2 RLN2).
  { unfold d4, LN2, RLN2. eapply ok_layer_norm_2d with (m := m); try eassumption;
      (eapply okv_weaken; [eassumption | apply errN_mono; auto;
         unfold d3, d2, af_depth, ca_depth, sm_depth, d1, ln_depth; lia])
      || (eapply ok_errN_mono with (n := n); [exact HM0 | exact HL1 |
            unfold d3, d2, af_depth, ca_depth, sm_depth, d1, ln_depth; lia | eassumption]). }
  set (d5 := S (S (d4 + 2 * km1) + 25 + 2 * km2)).
  assert (A5 : okm (errN M L d5)
                 (f32_mlp_forward (f32_mlp_c_fc_weight (f32_block_mlp block))
                    (f32_mlp_c_fc_bias (f32_block_mlp block))
                    (f32_mlp_c_proj_weight (f32_block_mlp block))
                    (f32_mlp_c_proj_bias (f32_block_mlp block)) LN2)
                 (Rmlp_forward rcfw rcfb rmpw rmpb RLN2)).
  { unfold d5. eapply ok_mlp_forward with (m := m); try eassumption;
      (eapply okm_weaken; [eassumption | apply errN_mono; auto;
         unfold d4, d3, d2, af_depth, ca_depth, sm_depth, d1, ln_depth; lia])
      || (eapply okv_weaken; [eassumption | apply errN_mono; auto;
         unfold d4, d3, d2, af_depth, ca_depth, sm_depth, d1, ln_depth; lia]). }
  eapply ok_add_matrices with (m := m); try eassumption.
  eapply okm_weaken; [exact A3 | apply errN_mono; auto;
    unfold d5, d4, d3, ln_depth; lia].
Qed.

(** * The block stack

    [blocks_reg] records that every block in the stack meets its own bound, at
    a depth that only grows. [ok_block_forward] is what discharges each step. *)

Inductive blocks_reg (M L : R) (cfg : gpt2_inference_config) (eps : binary32)
  : nat -> list f32_block_weights -> list (list binary32) -> list (list R)
    -> nat -> list (list R) -> Prop :=
| breg_nil : forall n h rh,
    okm (errN M L n) h rh ->
    blocks_reg M L cfg eps n [] h rh n rh
| breg_cons : forall n b bs h rh n' rh' nfin rhfin,
    (n <= n')%nat ->
    okm (errN M L n') (f32_block_forward cfg eps b h) rh' ->
    blocks_reg M L cfg eps n' bs (f32_block_forward cfg eps b h) rh' nfin rhfin ->
    blocks_reg M L cfg eps n (b :: bs) h rh nfin rhfin.

Lemma ok_blocks_forward : forall M L cfg eps blocks h rh n nfin rhfin,
  blocks_reg M L cfg eps n blocks h rh nfin rhfin ->
  okm (errN M L nfin) (f32_blocks_forward cfg eps blocks h) rhfin.
Proof.
  intros M L cfg eps blocks h rh n nfin rhfin Hreg.
  induction Hreg as [n h rh Hh | n b bs h rh n' rh' nfin rhfin Hle Hb Hreg IH];
    cbn [f32_blocks_forward]; [exact Hh | exact IH].
Qed.

(** * Embeddings

    A lookup moves a row without arithmetic. *)

Lemma Forall2_map_same : forall (A B C : Type) (Q : B -> C -> Prop)
    (f : A -> B) (g : A -> C) l,
  (forall a, Q (f a) (g a)) -> Forall2 Q (List.map f l) (List.map g l).
Proof.
  intros A B C Q f g l H. induction l as [|a l IH]; cbn [List.map]; constructor; auto.
Qed.

Definition Rlookup (re : list (list R)) (i : nat) : list R := List.nth i re [].

Definition Rembed_tokens (re : list (list R)) (ids : list nat) : list (list R) :=
  List.map (Rlookup re) ids.

Definition Rembed_positions (re : list (list R)) (n : nat) : list (list R) :=
  List.map (Rlookup re) (List.seq 0 n).

Lemma ok_embed_tokens : forall d e re ids,
  okm d e re -> okm d (f32_embed_tokens e ids) (Rembed_tokens re ids).
Proof.
  intros d e re ids H. unfold f32_embed_tokens, Rembed_tokens, okm.
  apply Forall2_map_same. intros i.
  unfold f32_lookup_embedding, Rlookup.
  eapply Forall2_nth; [exact H | constructor].
Qed.

Lemma ok_embed_positions : forall d e re n,
  okm d e re -> okm d (f32_embed_positions e n) (Rembed_positions re n).
Proof.
  intros d e re n H. unfold f32_embed_positions, Rembed_positions, okm.
  apply Forall2_map_same. intros i.
  unfold f32_lookup_embedding, Rlookup.
  eapply Forall2_nth; [exact H | constructor].
Qed.

(** * The full forward pass *)

Definition Rgpt2_forward (rwte rwpe : list (list R)) (rlnfw rlnfb : list R)
                         (reps : R) (len : nat) (toks : list nat)
                         (rtransformed : list (list R)) : list (list R) :=
  Rlayer_norm_2d rlnfw rlnfb reps len rtransformed.

Lemma ok_gpt2_forward :
  forall M m L n cfg eps reps model toks rwte rwpe rlnfw rlnfb
         nfin rtransformed len,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L nfin) eps reps ->
  okm (errN M L n) (f32_wte model) rwte ->
  okm (errN M L n) (f32_wpe model) rwpe ->
  okv (errN M L nfin) (f32_ln_weight (f32_ln_f model)) rlnfw ->
  okv (errN M L nfin) (f32_ln_bias (f32_ln_f model)) rlnfb ->
  Forall2 (fun r1 r2 => Forall2 (fun x y => regz M (B2R x + B2R y)) r1 r2)
    (f32_embed_tokens (f32_wte model) toks)
    (f32_embed_positions (f32_wpe model) (List.length toks)) ->
  blocks_reg M L cfg eps (S n) (f32_blocks model)
    (f32_add_matrices (f32_embed_tokens (f32_wte model) toks)
                      (f32_embed_positions (f32_wpe model) (List.length toks)))
    (Radd_matrices (Rembed_tokens rwte toks)
                   (Rembed_positions rwpe (List.length toks)))
    nfin rtransformed ->
  Forall (fun row => List.length row = len)
    (f32_blocks_forward cfg eps (f32_blocks model)
       (f32_add_matrices (f32_embed_tokens (f32_wte model) toks)
                         (f32_embed_positions (f32_wpe model) (List.length toks)))) ->
  Forall2 (fun row rrow =>
      ln_reg M m (f32_ln_weight (f32_ln_f model)) (f32_ln_bias (f32_ln_f model))
             eps row rlnfw rlnfb reps rrow)
    (f32_blocks_forward cfg eps (f32_blocks model)
       (f32_add_matrices (f32_embed_tokens (f32_wte model) toks)
                         (f32_embed_positions (f32_wpe model) (List.length toks))))
    rtransformed ->
  okm (errN M L (ln_depth nfin len))
      (f32_gpt2_forward cfg eps model toks)
      (Rgpt2_forward rwte rwpe rlnfw rlnfb reps len toks rtransformed).
Proof.
  intros M m L n cfg eps reps model toks rwte rwpe rlnfw rlnfb
         nfin rtransformed len
         HM Hamp Heps Hwte Hwpe Hlnw Hlnb Hres Hblocks Hlens Hln.
  assert (Htr : okm (errN M L nfin)
                  (f32_blocks_forward cfg eps (f32_blocks model)
                     (f32_add_matrices (f32_embed_tokens (f32_wte model) toks)
                        (f32_embed_positions (f32_wpe model) (List.length toks))))
                  rtransformed)
    by (eapply ok_blocks_forward; exact Hblocks).
  unfold f32_gpt2_forward, Rgpt2_forward.
  eapply ok_layer_norm_2d with (m := m); eassumption.
Qed.

(** * Logits *)

Definition Rgpt2_logits (rwte : list (list R)) (rhidden : list (list R))
                        : list (list R) :=
  List.map (fun h_row => List.map (fun w_row => Rdot h_row w_row) rwte) rhidden.

Lemma ok_gpt2_logits : forall M m L n kk wte rwte hidden rhidden,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L n) wte rwte ->
  okm (errN M L n) hidden rhidden ->
  Forall (fun col => Forall (fun r => Rabs r <= M) col) rwte ->
  Forall (fun h_row => Forall (fun w_row => dotreg M h_row w_row f32_zero
                                            /\ (List.length h_row <= kk)%nat) wte)
         hidden ->
  okm (errN M L (n + 2 * kk))
      (List.map (fun h_row => List.map (fun w_row => f32_dot h_row w_row) wte) hidden)
      (Rgpt2_logits rwte rhidden).
Proof.
  intros M m L n kk wte rwte hidden rhidden HM Hamp Hwte Hh Hbnd Hreg.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  unfold Rgpt2_logits, okm in *.
  revert Hreg.
  induction Hh as [|hr rhr hs rhs Hhr Hh IH]; intros Hreg; cbn [List.map];
    [constructor|].
  pose proof (Forall_inv Hreg) as Hr1. pose proof (Forall_inv_tail Hreg) as Hr2.
  constructor.
  - clear IH Hh Hreg Hr2.
    unfold okv in *. revert Hbnd Hr1.
    induction Hwte as [|w rw ws rws Hw Hwte IH]; intros Hbnd Hr1; cbn [List.map];
      [constructor|].
    pose proof (Forall_inv Hbnd) as Hb1. pose proof (Forall_inv_tail Hbnd) as Hb2.
    pose proof (Forall_inv Hr1) as [Hd1 Hd2]. pose proof (Forall_inv_tail Hr1) as Hr1'.
    constructor.
    + eapply ok_errN_mono with (n := (n + 2 * List.length hr)%nat);
        [exact HM0 | exact HL1 | lia |].
      eapply ok_dot_n with (m := m); eassumption.
    + apply IH; assumption.
  - apply IH; exact Hr2.
Qed.

(** The top-level statement: every logit the extracted network produces is
    within [errN M L (D + 2 * kk)] of the logit exact real arithmetic would
    produce from the same weights, where [D] is the depth reached by the
    forward pass. *)
Corollary ok_gpt2_logits_full :
  forall M m L D kk cfg eps model toks rwte rhidden,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okm (errN M L D) (f32_wte model) rwte ->
  okm (errN M L D) (f32_gpt2_forward cfg eps model toks) rhidden ->
  Forall (fun col => Forall (fun r => Rabs r <= M) col) rwte ->
  Forall (fun h_row => Forall (fun w_row => dotreg M h_row w_row f32_zero
                                            /\ (List.length h_row <= kk)%nat)
                              (f32_wte model))
         (f32_gpt2_forward cfg eps model toks) ->
  okm (errN M L (D + 2 * kk))
      (f32_gpt2_logits cfg eps model toks)
      (Rgpt2_logits rwte rhidden).
Proof.
  intros M m L D kk cfg eps model toks rwte rhidden HM Hamp Hwte Hh Hbnd Hreg.
  unfold f32_gpt2_logits.
  eapply ok_gpt2_logits with (m := m); eassumption.
Qed.

(** * The Qwen3.5 primitives

    Qwen3.5 adds a token mixer the GPT-2 stack has no counterpart for: a gated
    delta recurrence carrying a state matrix through the sequence, in front of
    a depthwise causal convolution, with a logarithm in its decay term. The
    bounds below carry the same propagation relation over those primitives, on
    the same explicit premises. *)

Lemma ok_f32_abs : forall d x r, ok d x r -> ok d (f32_abs x) (Rabs r).
Proof.
  intros d x r [Hf Hd]. split.
  - unfold f32_abs. rewrite is_finite_Babs. exact Hf.
  - unfold f32_abs. rewrite B2R_Babs.
    eapply Rle_trans; [apply Rabs_triang_inv2 | exact Hd].
Qed.

Lemma ok_silu : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx ->
  sig_reg M m x rx ->
  Rabs (B2R x) <= M -> Rabs (Rsigmoid rx) <= M ->
  regz M (B2R x * B2R (f32_sigmoid x)) ->
  ok (errN M L (S (k + 23))) (f32_silu x) (rx * Rsigmoid rx).
Proof.
  intros M m L k x rx HM Hamp Hx Hreg Hbx Hbs Hz.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  unfold f32_silu.
  eapply ok_mult_S with (m := m); try eassumption.
  - eapply ok_errN_mono with (n := k); [exact HM0 | exact HL1 | lia | exact Hx].
  - eapply ok_sigmoid with (m := m); eassumption.
Qed.

(** The comparison-based maximum stays close even when the float and the real
    comparison select different operands. *)
Lemma ok_max2 : forall d x y rx ry,
  ok d x rx -> ok d y ry -> ok d (f32_max2 x y) (Rmax rx ry).
Proof.
  intros d x y rx ry Hx Hy.
  destruct Hx as [Hfx Hdx]. destruct Hy as [Hfy Hdy].
  unfold f32_max2. destruct (f32_lt x y) eqn:E.
  - rewrite (f32_lt_correct x y Hfx Hfy) in E.
    split; [exact Hfy|].
    unfold Rmax. destruct (Rle_dec rx ry) as [Hle|Hlt]; [exact Hdy|].
    apply Rabs_le. apply Rabs_le_inv in Hdx. apply Rabs_le_inv in Hdy. lra.
  - assert (Hge : ~ (B2R x < B2R y)).
    { intro Hc. rewrite <- (f32_lt_correct x y Hfx Hfy) in Hc.
      rewrite Hc in E. discriminate. }
    split; [exact Hfx|].
    unfold Rmax. destruct (Rle_dec rx ry) as [Hle|Hlt]; [|exact Hdx].
    apply Rabs_le. apply Rabs_le_inv in Hdx. apply Rabs_le_inv in Hdy. lra.
Qed.

(** The logarithm on (1, 2]: a division, seven powers, six divisions, a plus
    chain and a final doubling. *)
Record log_reg (M m : R) (mv : binary32) : Prop := {
  lgr_m3 : m <= Rabs (B2R f32_three);   lgr_m5 : m <= Rabs (B2R f32_five);
  lgr_m7 : m <= Rabs (B2R f32_seven);   lgr_m9 : m <= Rabs (B2R f32_nine);
  lgr_m11 : m <= Rabs (B2R f32_eleven); lgr_m13 : m <= Rabs (B2R f32_thirteen);
  lgr_fin : is_finite f32_three = true /\ is_finite f32_five = true
            /\ is_finite f32_seven = true /\ is_finite f32_nine = true
            /\ is_finite f32_eleven = true /\ is_finite f32_thirteen = true;
  lgr_bnd : Rabs (B2R mv) <= M
}.

Lemma fin_odd_consts :
  is_finite f32_three = true /\ is_finite f32_five = true
  /\ is_finite f32_seven = true /\ is_finite f32_nine = true
  /\ is_finite f32_eleven = true /\ is_finite f32_thirteen = true.
Proof. repeat split; vm_compute; reflexivity. Qed.

(** The DeltaNet decay term, given the bound on softplus. *)
Lemma ok_delta_decay_of : forall M m L k a_log dt_bias a ra_log rdt rsp,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) a_log ra_log ->
  ok (errN M L (k + 21)) (f32_exp_approx a_log) rsp ->
  ok (errN M L (k + 21)) (f32_softplus (f32_plus a dt_bias)) rdt ->
  Rabs (B2R (f32_exp_approx a_log)) <= M -> Rabs rdt <= M ->
  regz M (B2R (f32_exp_approx a_log)
          * B2R (f32_softplus (f32_plus a dt_bias))) ->
  ok (errN M L (S (k + 21)))
     (f32_neg (f32_mult (f32_exp_approx a_log) (f32_softplus (f32_plus a dt_bias))))
     (- (rsp * rdt)).
Proof.
  intros M m L k a_log dt_bias a ra_log rdt rsp HM Hamp Hal He Hs Hbe Hbs Hz.
  apply ok_neg. eapply ok_mult_S with (m := m); eassumption.
Qed.

(** Euclidean normalisation: the squared length, a shift, a square root, a
    reciprocal, and one multiply per entry. *)
Lemma ok_l2norm : forall M m L k epsv reps v rv,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L k) v rv ->
  ok (errN M L k) epsv reps ->
  dotreg M v v f32_zero ->
  Forall (fun r => Rabs r <= M) rv ->
  regz M (B2R (f32_dot v v) + B2R epsv) ->
  m <= B2R (f32_plus (f32_dot v v) epsv) ->
  m <= Rdot rv rv + reps ->
  regz M (sqrt (B2R (f32_plus (f32_dot v v) epsv))) ->
  Rabs 1 <= M ->
  m <= Rabs (B2R (f32_sqrt (f32_plus (f32_dot v v) epsv))) ->
  m <= Rabs (sqrt (Rdot rv rv + reps)) ->
  regz M (B2R f32_one / B2R (f32_sqrt (f32_plus (f32_dot v v) epsv))) ->
  Forall2 (fun x rx =>
      Rabs (B2R x) <= M
      /\ Rabs (1 / sqrt (Rdot rv rv + reps)) <= M
      /\ regz M (B2R x * B2R (f32_div f32_one
                   (f32_sqrt (f32_plus (f32_dot v v) epsv))))) v rv ->
  okv (errN M L (S (S (S (S (k + 2 * List.length v))))))
      (f32_l2norm epsv v)
      (List.map (fun x => x * (1 / sqrt (Rdot rv rv + reps))) rv).
Proof.
  intros M m L k epsv reps v rv HM Hamp Hv Heps Hdreg Hbnd Hz1 Hlo1 Hlo1r Hz2
         Hone Hlo2 Hlo2r Hz3 Hentries.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  set (n0 := (k + 2 * List.length v)%nat).
  assert (Hss : ok (errN M L n0) (f32_dot v v) (Rdot rv rv))
    by (unfold n0; eapply ok_dot_n with (m := m); eassumption).
  assert (Hsh : ok (errN M L (S n0)) (f32_plus (f32_dot v v) epsv)
                   (Rdot rv rv + reps)).
  { eapply ok_plus_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := k); [exact HM0 | exact HL1 | unfold n0; lia | exact Heps]. }
  assert (Hsq : ok (errN M L (S (S n0)))
                   (f32_sqrt (f32_plus (f32_dot v v) epsv))
                   (sqrt (Rdot rv rv + reps)))
    by (eapply ok_sqrt_S with (m := m); eassumption).
  assert (Hinv : ok (errN M L (S (S (S n0))))
                    (f32_div f32_one (f32_sqrt (f32_plus (f32_dot v v) epsv)))
                    (1 / sqrt (Rdot rv rv + reps))).
  { eapply ok_div_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := 0%nat); [exact HM0 | exact HL1 | lia |].
    replace 1 with (B2R f32_one) by apply f32_one_correct.
    apply ok_const; [exact HM0 | exact HL1 | exact f32_one_finite]. }
  unfold f32_l2norm, okv in *.
  eapply Forall2_map2; [exact (Forall2_conj _ _ _ _ _ _ Hv Hentries)|].
  intros x rx [Hxr (Hbx & Hbi & Hzx)].
  eapply ok_mult_S with (m := m); try eassumption.
  eapply ok_errN_mono with (n := k); [exact HM0 | exact HL1 | unfold n0; lia | exact Hxr].
Qed.

(** * The RMSNorm variants

    Both share a scale factor: the reciprocal square root of the mean square,
    shifted by eps. *)

Definition Rsq_sum (rx : list R) : R := Rsum (List.map (fun x => x * x) rx).

Definition Rrms_scale (reps : R) (n : nat) (rx : list R) : R :=
  1 / sqrt (Rsq_sum rx / B2R (f32_of_Z (Z.of_nat n)) + reps).

Record rms_reg (M m : R) (epsv : binary32) (x : list binary32) (rx : list R) : Prop := {
  rmr_sum : sumreg M (List.map (fun xi => f32_mult xi xi) x) f32_zero;
  rmr_sq : Forall2 (fun xi rxi => Rabs (B2R xi) <= M /\ Rabs rxi <= M
                                  /\ regz M (B2R xi * B2R xi)) x rx;
  rmr_fin_n : is_finite (f32_of_Z (Z.of_nat (List.length x))) = true;
  rmr_lo_n : m <= Rabs (B2R (f32_of_Z (Z.of_nat (List.length x))));
  rmr_z_ms : regz M (B2R (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                     / B2R (f32_of_Z (Z.of_nat (List.length x))));
  rmr_z_sh : regz M (B2R (f32_div (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                            (f32_of_Z (Z.of_nat (List.length x)))) + B2R epsv);
  rmr_rad : m <= B2R (f32_plus (f32_div (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                                 (f32_of_Z (Z.of_nat (List.length x)))) epsv);
  rmr_z_sq : regz M (sqrt (B2R (f32_plus
                (f32_div (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                         (f32_of_Z (Z.of_nat (List.length x)))) epsv)));
  rmr_den : m <= Rabs (B2R (f32_sqrt (f32_plus
                (f32_div (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                         (f32_of_Z (Z.of_nat (List.length x)))) epsv)));
  rmr_one : Rabs 1 <= M;
  rmr_z_inv : regz M (B2R f32_one / B2R (f32_sqrt (f32_plus
                (f32_div (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                         (f32_of_Z (Z.of_nat (List.length x)))) epsv)))
}.

(** The gate a full-attention layer applies to its output. *)
Lemma ok_gate_sigmoid : forall M m L k gate rgate v rv,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L k) gate rgate -> okv (errN M L k) v rv ->
  Forall2 (fun gi ri => sig_reg M m gi ri) gate rgate ->
  Forall2 (fun vi rvi => Rabs (B2R vi) <= M) v rv ->
  Forall2 (fun gi ri => Rabs (Rsigmoid ri) <= M) gate rgate ->
  Forall2 (fun vi gi => regz M (B2R vi * B2R (f32_sigmoid gi))) v gate ->
  okv (errN M L (S (k + 23))) (f32_gate_sigmoid gate v)
      (List.map (fun '(vi, gi) => vi * Rsigmoid gi) (List.combine rv rgate)).
Proof.
  intros M m L k gate rgate v rv HM Hamp Hg Hv Hsig Hbv Hbs Hz.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  unfold f32_gate_sigmoid, okv in *.
  revert rv rgate Hv Hg Hsig Hbv Hbs.
  induction Hz as [|vi gi vs gs Hzi Hz IH]; intros rv rgate Hv Hg Hsig Hbv Hbs;
    cbn [List.map List.combine].
  - inversion Hv; subst. cbn [List.combine List.map]. constructor.
  - inversion Hv as [|v' rv' vs' rvs Hvr Hvs]; subst.
    inversion Hg as [|g' rg' gs' rgs Hgr Hgs]; subst.
    inversion Hsig as [|g2 rg2 gs2 rgs2 Hsr Hss]; subst.
    inversion Hbv as [|v2 rv2 vs2 rvs2 Hbvr Hbvs]; subst.
    inversion Hbs as [|g3 rg3 gs3 rgs3 Hbsr Hbss]; subst.
    cbn [List.combine List.map]. constructor.
    + eapply ok_mult_S with (m := m); try eassumption.
      * eapply ok_errN_mono with (n := k); [exact HM0 | exact HL1 | lia | exact Hvr].
      * eapply ok_sigmoid with (m := m); eassumption.
    + apply IH; assumption.
Qed.

(** The squared entries, which the mean square sums. *)
Lemma ok_squares : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L k) x rx ->
  Forall2 (fun xi rxi => Rabs (B2R xi) <= M /\ Rabs rxi <= M
                         /\ regz M (B2R xi * B2R xi)) x rx ->
  okv (errN M L (S k)) (List.map (fun xi => f32_mult xi xi) x)
      (List.map (fun r => r * r) rx).
Proof.
  intros M m L k x rx HM Hamp Hx Hreg.
  unfold okv in *.
  eapply Forall2_map2; [exact (Forall2_conj _ _ _ _ _ _ Hx Hreg)|].
  intros xi rxi [Hxr (Hb1 & Hb2 & Hz)].
  eapply ok_mult_S with (m := m); eassumption.
Qed.

(** The scale factor both RMSNorm variants share: the reciprocal square root of
    the mean square, shifted by eps. *)
Lemma ok_rms_scale : forall M m L k epsv reps x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  okv (errN M L k) x rx -> ok (errN M L k) epsv reps ->
  rms_reg M m epsv x rx ->
  Rabs (Rsq_sum rx) <= M ->
  m <= Rsq_sum rx / B2R (f32_of_Z (Z.of_nat (List.length x))) + reps ->
  m <= Rabs (sqrt (Rsq_sum rx / B2R (f32_of_Z (Z.of_nat (List.length x))) + reps)) ->
  ok (errN M L (S (S (S (S (S (k + List.length x)))))))
     (f32_div f32_one
        (f32_sqrt (f32_plus (f32_div (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                                     (f32_of_Z (Z.of_nat (List.length x)))) epsv)))
     (Rrms_scale reps (List.length x) rx).
Proof.
  intros M m L k epsv reps x rx HM Hamp Hx Heps Hreg Hbss Hlo Hlor.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  destruct Hreg as [Hsum Hsq Hfinn Hlon Hzms Hzsh Hrad Hzsq Hden Hone Hzinv].
  assert (Hlen : List.length (List.map (fun xi => f32_mult xi xi) x) = List.length x)
    by apply List.length_map.
  assert (Hsqs : okv (errN M L (S k)) (List.map (fun xi => f32_mult xi xi) x)
                     (List.map (fun r => r * r) rx))
    by (eapply ok_squares with (m := m); eassumption).
  assert (Hss : ok (errN M L (S k + List.length x))
                   (f32_sum (List.map (fun xi => f32_mult xi xi) x)) (Rsq_sum rx)).
  { unfold Rsq_sum.
    pose proof (ok_sum M m L (S k) _ _ HM Hamp Hsum Hsqs) as H.
    rewrite Hlen in H. exact H. }
  assert (Hms : ok (errN M L (S (S k + List.length x)))
                   (f32_div (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                            (f32_of_Z (Z.of_nat (List.length x))))
                   (Rsq_sum rx / B2R (f32_of_Z (Z.of_nat (List.length x))))).
  { eapply ok_div_S with (m := m); try eassumption.
    apply ok_const; [exact HM0 | exact HL1 | exact Hfinn]. }
  assert (Hsh : ok (errN M L (S (S (S k + List.length x))))
                   (f32_plus (f32_div (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                                      (f32_of_Z (Z.of_nat (List.length x)))) epsv)
                   (Rsq_sum rx / B2R (f32_of_Z (Z.of_nat (List.length x))) + reps)).
  { eapply ok_plus_S with (m := m); try eassumption.
    eapply ok_errN_mono with (n := k); [exact HM0 | exact HL1 | lia | exact Heps]. }
  assert (Hsq2 : ok (errN M L (S (S (S (S k + List.length x)))))
                    (f32_sqrt (f32_plus (f32_div
                       (f32_sum (List.map (fun xi => f32_mult xi xi) x))
                       (f32_of_Z (Z.of_nat (List.length x)))) epsv))
                    (sqrt (Rsq_sum rx / B2R (f32_of_Z (Z.of_nat (List.length x))) + reps)))
    by (eapply ok_sqrt_S with (m := m); eassumption).
  unfold Rrms_scale.
  replace (S (S (S (S (S (k + List.length x))))))
    with (S (S (S (S (S k + List.length x))))) by lia.
  eapply ok_div_S with (m := m); try eassumption.
  eapply ok_errN_mono with (n := 0%nat); [exact HM0 | exact HL1 | lia |].
  replace 1 with (B2R f32_one) by apply f32_one_correct.
  apply ok_const; [exact HM0 | exact HL1 | exact f32_one_finite].
Qed.
