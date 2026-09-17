(** * The native build's arithmetic, at the level of floats

    Float_error.v proves that the native build's second rounding is harmless as
    a statement about real numbers: for binary32 operands, rounding the exact
    real result of [+], [-], [*], [/] or [sqrt] to binary64 and then to
    binary32 gives the same real number as rounding it straight to binary32.
    That says nothing about the values the build actually produces when a
    result overflows, when an operand is an infinity or a NaN, or when the
    result is a zero whose sign has to be chosen.

    This file closes that gap. It gives the two conversions the native build
    performs --- [widen], the exact embedding of a binary32 into binary64 that
    happens when an OCaml [float] holding a binary32 value enters an
    arithmetic operation, and [narrow], the round-to-nearest-even conversion
    back that [Int32.float_of_bits (Int32.bits_of_float _)] performs --- and
    proves for each of the five operations that

      narrow (op64 (widen x) (widen y)) = op32 x y

    for every pair of binary32 values, with no side condition. Infinities, NaNs,
    signed zeros, underflow and overflow are all in scope, because the
    statement quantifies over the constructors of [binary_float] rather than
    over the reals.

    The consequence is that the native and inductive extractions of
    Phases1_15_complete.v compute the same binary32 value at every operation,
    for every input, so a result measured on the native build is a result of
    the inductive build. What the native build still trusts is that the host's
    binary64 arithmetic is IEEE-754 round-to-nearest-even and that
    [Int32.bits_of_float] rounds to nearest; those two assumptions, and no
    others, stand between these theorems and the running program. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import Reals.
From Stdlib Require Import Lra.
From Stdlib Require Import Lia.
From Stdlib Require Import Bool.
From Flocq Require Import Core.
From Flocq Require Import Double_rounding.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Float_error.

Open Scope R_scope.

(** * The two formats *)

(** [prec64], [emax64] and [emin64] are Float_error.v's. *)

Lemma prec64_gt_0 : (0 < prec64)%Z.
Proof. unfold prec64. lia. Qed.

Lemma prec64_lt_emax64 : (prec64 < emax64)%Z.
Proof. unfold prec64, emax64. lia. Qed.

Definition binary64 := binary_float prec64 emax64.

(** The operations of the two formats, at round-to-nearest-even. The binary32
    ones are the primitives of the development; the binary64 ones are what the
    host performs. *)

Definition b64_plus (x y : binary64) : binary64 :=
  @Bplus prec64 emax64 prec64_gt_0 prec64_lt_emax64 mode_NE x y.
Definition b64_mult (x y : binary64) : binary64 :=
  @Bmult prec64 emax64 prec64_gt_0 prec64_lt_emax64 mode_NE x y.
Definition b64_div (x y : binary64) : binary64 :=
  @Bdiv prec64 emax64 prec64_gt_0 prec64_lt_emax64 mode_NE x y.
Definition b64_sqrt (x : binary64) : binary64 :=
  @Bsqrt prec64 emax64 prec64_gt_0 prec64_lt_emax64 mode_NE x.

(** * Widening and narrowing *)

(** [widen] embeds a binary32 into binary64. Every binary32 value is a binary64
    value, so the rounding [binary_normalize] performs is the identity; the
    proof of that is [widen_B2R] below. *)

Definition widen (x : binary32) : binary64 :=
  match x with
  | B754_zero s => B754_zero s
  | B754_infinity s => B754_infinity s
  | B754_nan => B754_nan
  | B754_finite s m e _ =>
      @binary_normalize prec64 emax64 prec64_gt_0 prec64_lt_emax64
        mode_NE (cond_Zopp s (Zpos m)) e s
  end.

(** [narrow] is the conversion back, round-to-nearest-even, overflowing to an
    infinity of the operand's sign and preserving the sign of a zero. This is
    what the [Int32] bit round-trip of Extract.v performs. *)

Definition narrow (d : binary64) : binary32 :=
  match d with
  | B754_zero s => B754_zero s
  | B754_infinity s => B754_infinity s
  | B754_nan => B754_nan
  | B754_finite s m e _ =>
      @binary_normalize prec32 emax32 prec32_gt_0 prec32_lt_emax32
        mode_NE (cond_Zopp s (Zpos m)) e s
  end.

(** * Bounds that make the two formats nest *)

Notation fexp32 := (SpecFloat.fexp prec32 emax32).
Notation fexp64 := (SpecFloat.fexp prec64 emax64).

Lemma valid_exp_fexp32 : Valid_exp fexp32.
Proof.
unfold SpecFloat.fexp. apply FLT_exp_valid. unfold Prec_gt_0, prec32. lia.
Qed.

Lemma valid_exp_fexp64 : Valid_exp fexp64.
Proof.
unfold SpecFloat.fexp. apply FLT_exp_valid. unfold Prec_gt_0, prec64. lia.
Qed.

#[local] Existing Instance valid_exp_fexp32.
#[local] Existing Instance valid_exp_fexp64.
#[local] Existing Instance valid_rnd_round_mode.

#[local] Instance prec32_gt_0_i : Prec_gt_0 prec32 := prec32_gt_0.
#[local] Instance prec64_gt_0_i : Prec_gt_0 prec64 := prec64_gt_0.
#[local] Instance prec32_lt_emax32_i : Prec_lt_emax prec32 emax32 := prec32_lt_emax32.
#[local] Instance prec64_lt_emax64_i : Prec_lt_emax prec64 emax64 := prec64_lt_emax64.

Lemma emax32_lt_emax64 : (bpow radix2 emax32 < bpow radix2 emax64)%R.
Proof. apply bpow_lt. unfold emax32, emax64. lia. Qed.

(** A binary32 value lies in the binary64 format. *)
Lemma B2R_f32_in_b64 : forall x : binary32,
  generic_format radix2 fexp64 (B2R x).
Proof.
intros x.
apply (generic_inclusion_mag radix2 fexp32 fexp64).
- intros _. unfold SpecFloat.fexp, SpecFloat.emin, prec32, emax32, prec64, emax64.
  lia.
- apply generic_format_B2R.
Qed.

(** Its magnitude is under the binary32 overflow threshold. *)
Lemma B2R_f32_lt_emax32 : forall x : binary32,
  (Rabs (B2R x) < bpow radix2 emax32)%R.
Proof.
intros x. apply (abs_B2R_lt_emax prec32 emax32).
Qed.

(** Rounding a binary32 value to binary64 is the identity. *)
Lemma round64_f32 : forall x : binary32,
  round radix2 fexp64 (round_mode mode_NE) (B2R x) = B2R x.
Proof.
intros x. apply round_generic.
- apply valid_rnd_round_mode.
- apply B2R_f32_in_b64.
Qed.

(** * Widening is exact *)

(** A finite float is not a NaN. *)
Lemma is_finite_not_nan : forall (p q : Z) (f : binary_float p q),
  is_finite f = true -> is_nan f = false.
Proof. intros p q [s|s| |s m e H]; simpl; auto. Qed.

(** The sign of a finite float is the sign of its value; a [B754_finite] is
    never zero. *)
Lemma B2R_finite_sign_gen :
  forall (p q : Z) (s : bool) m e (H : SpecFloat.bounded p q m e = true),
  Rcompare (B2R (B754_finite s m e H)) 0 = if s then Lt else Gt.
Proof.
intros p q [|] m e H.
- apply Rcompare_Lt. apply F2R_lt_0. reflexivity.
- apply Rcompare_Gt. apply F2R_gt_0. reflexivity.
Qed.

Lemma B2R_finite_sign :
  forall (s : bool) m e (H : SpecFloat.bounded prec32 emax32 m e = true),
  Rcompare (B2R (B754_finite s m e H)) 0 = if s then Lt else Gt.
Proof. intros. apply B2R_finite_sign_gen. Qed.

Lemma Rlt_bool_B2R_finite :
  forall (p q : Z) (s : bool) m e (H : SpecFloat.bounded p q m e = true),
  Rlt_bool (B2R (B754_finite s m e H)) 0 = s.
Proof.
intros p q [|] m e H.
- apply Rlt_bool_true. apply F2R_lt_0. reflexivity.
- apply Rlt_bool_false. apply Rlt_le. apply F2R_gt_0. reflexivity.
Qed.

(** Everything the finite case of [widen] needs, proved once. *)
Lemma widen_finite_spec :
  forall (sx : bool) mx ex (Hx : SpecFloat.bounded prec32 emax32 mx ex = true),
  B2R (widen (B754_finite sx mx ex Hx)) = B2R (B754_finite sx mx ex Hx)
  /\ is_finite (widen (B754_finite sx mx ex Hx)) = true
  /\ Bsign (widen (B754_finite sx mx ex Hx)) = sx.
Proof.
intros sx mx ex Hx.
unfold widen.
generalize (@binary_normalize_correct prec64 emax64 prec64_gt_0 prec64_lt_emax64
              mode_NE (cond_Zopp sx (Zpos mx)) ex sx).
intros H. cbv zeta in H.
change (F2R {| Fnum := SpecFloat.cond_Zopp sx (Z.pos mx); Fexp := ex |})
  with (B2R (B754_finite sx mx ex Hx)) in H.
rewrite round64_f32 in H.
rewrite Rlt_bool_true in H.
2:{ apply Rlt_trans with (bpow radix2 emax32).
    - apply B2R_f32_lt_emax32.
    - apply emax32_lt_emax64. }
destruct H as (H1 & H2 & H3).
repeat split; try assumption.
rewrite H3, B2R_finite_sign. now destruct sx.
Qed.

Lemma widen_B2R : forall x : binary32, B2R (widen x) = B2R x.
Proof.
intros [sx|sx| |sx mx ex Hx]; try reflexivity.
now destruct (widen_finite_spec sx mx ex Hx) as (H & _ & _).
Qed.

Lemma widen_is_finite : forall x : binary32,
  is_finite (widen x) = is_finite x.
Proof.
intros [sx|sx| |sx mx ex Hx]; try reflexivity.
now destruct (widen_finite_spec sx mx ex Hx) as (_ & H & _).
Qed.

Lemma widen_is_nan : forall x : binary32, is_nan (widen x) = is_nan x.
Proof.
intros [sx|sx| |sx mx ex Hx]; try reflexivity.
destruct (widen_finite_spec sx mx ex Hx) as (_ & H & _).
simpl. now apply is_finite_not_nan.
Qed.

Lemma widen_Bsign : forall x : binary32,
  is_nan x = false -> Bsign (widen x) = Bsign x.
Proof.
intros [sx|sx| |sx mx ex Hx] Hn; try reflexivity.
now destruct (widen_finite_spec sx mx ex Hx) as (_ & _ & H).
Qed.

(** * Narrowing is correct rounding, with overflow to an infinity *)

Lemma B2R_finite_neq_0 :
  forall (p q : Z) (s : bool) m e (H : SpecFloat.bounded p q m e = true),
  B2R (B754_finite s m e H) <> 0.
Proof.
intros p q s m e H Hc.
generalize (B2R_finite_sign_gen p q s m e H).
rewrite Hc, Rcompare_Eq by reflexivity. now destruct s.
Qed.

(** What [narrow] does to any finite operand: correct rounding to binary32,
    overflowing to an infinity of the operand's sign, and preserving the sign
    of a zero and of an underflow. *)
Lemma narrow_spec_finite : forall d : binary64,
  is_finite d = true ->
  if Rlt_bool (Rabs (round radix2 fexp32 (round_mode mode_NE) (B2R d)))
              (bpow radix2 emax32)
  then B2R (narrow d) = round radix2 fexp32 (round_mode mode_NE) (B2R d)
       /\ is_finite (narrow d) = true
       /\ Bsign (narrow d) = Bsign d
  else B2SF (narrow d) = binary_overflow prec32 emax32 mode_NE (Bsign d).
Proof.
intros [s|s| |s m e H] Hf; try discriminate Hf.
- (* zero *)
  simpl (B2R _). rewrite round_0 by apply valid_rnd_round_mode.
  rewrite Rabs_R0, Rlt_bool_true by apply bpow_gt_0.
  repeat split.
- (* finite *)
  unfold narrow.
  generalize (@binary_normalize_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32
                mode_NE (cond_Zopp s (Zpos m)) e s).
  intros Hn. cbv zeta in Hn.
  change (F2R {| Fnum := SpecFloat.cond_Zopp s (Z.pos m); Fexp := e |})
    with (B2R (B754_finite s m e H)) in Hn.
  destruct (Rlt_bool (Rabs (round radix2 fexp32 (round_mode mode_NE)
                              (B2R (B754_finite s m e H))))
                     (bpow radix2 emax32)).
  + destruct Hn as (H1 & H2 & H3). repeat split; try assumption.
    rewrite H3, B2R_finite_sign_gen. now destruct s.
  + rewrite Hn. now rewrite Rlt_bool_B2R_finite.
Qed.

(** A finite float with a nonzero value is a [B754_finite]. *)
Lemma finite_nonzero_strict : forall (p q : Z) (f : binary_float p q),
  is_finite f = true -> B2R f <> 0 -> is_finite_strict f = true.
Proof.
intros p q [s|s| |s m e H] Hf Hz; try discriminate Hf; try reflexivity.
now elim Hz.
Qed.

(** Widening a [B754_finite] gives a [B754_finite]. *)
Lemma widen_strict :
  forall (s : bool) m e (H : SpecFloat.bounded prec32 emax32 m e = true),
  is_finite_strict (widen (B754_finite s m e H)) = true.
Proof.
intros s m e H.
destruct (widen_finite_spec s m e H) as (HR & HF & _).
apply finite_nonzero_strict; [assumption|].
rewrite HR. apply B2R_finite_neq_0.
Qed.

(** Sign bits bound the value of a finite float on one side of zero. *)
Lemma B2R_le_0_of_sign : forall (p q : Z) (f : binary_float p q),
  is_finite f = true -> Bsign f = true -> (B2R f <= 0)%R.
Proof.
intros p q [s|s| |s m e H] Hf Hs; try discriminate Hf.
- simpl. apply Rle_refl.
- simpl in Hs. subst s. apply Rlt_le. apply F2R_lt_0. reflexivity.
Qed.

Lemma B2R_ge_0_of_sign : forall (p q : Z) (f : binary_float p q),
  is_finite f = true -> Bsign f = false -> (0 <= B2R f)%R.
Proof.
intros p q [s|s| |s m e H] Hf Hs; try discriminate Hf.
- simpl. apply Rle_refl.
- simpl in Hs. subst s. apply Rlt_le. apply F2R_gt_0. reflexivity.
Qed.

(** Narrowing undoes widening. *)
Theorem narrow_widen : forall x : binary32, narrow (widen x) = x.
Proof.
intros x.
destruct x as [sx|sx| |sx mx ex Hx]; try reflexivity.
destruct (widen_finite_spec sx mx ex Hx) as (HR & HF & HS).
generalize (narrow_spec_finite _ HF).
rewrite HR.
rewrite (round_generic radix2 fexp32 (round_mode mode_NE))
  by (apply valid_rnd_round_mode || apply generic_format_B2R).
rewrite Rlt_bool_true by apply B2R_f32_lt_emax32.
intros (K1 & K2 & K3).
apply B2R_Bsign_inj; try assumption; try reflexivity.
now rewrite K3, HS.
Qed.

(** * The bridge

    A binary64 result that is the correct rounding of [v], narrowed, is the
    binary32 result that is the correct rounding of [v] --- overflow and sign
    included. Each operation supplies [v], the two correctness facts, and the
    double-rounding equation for its own operator. *)

Notation rne := (round_mode mode_NE).

Lemma narrow_eq_of :
  forall (v : R) (d : binary64) (z : binary32),
  round radix2 fexp32 rne (round radix2 fexp64 rne v)
    = round radix2 fexp32 rne v ->
  is_finite d = true ->
  B2R d = round radix2 fexp64 rne v ->
  (if Rlt_bool (Rabs (round radix2 fexp32 rne v)) (bpow radix2 emax32)
   then B2R z = round radix2 fexp32 rne v
        /\ is_finite z = true
        /\ Bsign z = Bsign d
   else B2SF z = binary_overflow prec32 emax32 mode_NE (Bsign d)) ->
  narrow d = z.
Proof.
intros v d z Hdr Hf Hd Hz.
generalize (narrow_spec_finite d Hf).
rewrite Hd, Hdr.
destruct (Rlt_bool (Rabs (round radix2 fexp32 rne v)) (bpow radix2 emax32)).
- intros (K1 & K2 & K3). destruct Hz as (Z1 & Z2 & Z3).
  apply B2R_Bsign_inj; congruence.
- intros K. apply B2SF_inj. congruence.
Qed.

(** Nothing a binary32 pair can produce overflows binary64. *)
Lemma b64_no_overflow : forall v : R,
  (Rabs v <= bpow radix2 1000)%R ->
  Rlt_bool (Rabs (round radix2 fexp64 rne v)) (bpow radix2 emax64) = true.
Proof.
intros v Hv.
apply Rlt_bool_true.
apply Rle_lt_trans with (bpow radix2 1000).
- apply (@abs_round_le_generic radix2 fexp64 valid_exp_fexp64 rne
           (valid_rnd_round_mode mode_NE) v (bpow radix2 1000)).
  + apply generic_format_bpow.
    unfold SpecFloat.fexp, SpecFloat.emin, prec64, emax64. lia.
  + exact Hv.
- apply bpow_lt. unfold emax64. lia.
Qed.

Lemma bpow_double : forall e : Z,
  (bpow radix2 e + bpow radix2 e = bpow radix2 (e + 1))%R.
Proof.
intros e. rewrite bpow_plus.
assert (Hb : bpow radix2 1 = 2%R)
  by (first [ reflexivity | (simpl; reflexivity) | (compute; reflexivity) ]).
rewrite Hb. ring.
Qed.

Lemma bound_sum : forall x y : binary32,
  (Rabs (B2R x + B2R y) <= bpow radix2 1000)%R.
Proof.
intros x y.
generalize (B2R_f32_lt_emax32 x) (B2R_f32_lt_emax32 y). intros Hx Hy.
apply Rle_trans with (bpow radix2 emax32 + bpow radix2 emax32).
- apply Rle_trans with (Rabs (B2R x) + Rabs (B2R y));
    [apply Rabs_triang | lra].
- rewrite bpow_double. apply bpow_le. unfold emax32. lia.
Qed.

(** In an overflowing case the rounded value, hence the exact value, is
    nonzero. *)
Lemma overflow_arg_neq_0 : forall v : R,
  Rlt_bool (Rabs (round radix2 fexp32 rne v)) (bpow radix2 emax32) = false ->
  v <> 0.
Proof.
intros v Hb Hc. revert Hb.
rewrite Hc, round_0 by apply valid_rnd_round_mode.
rewrite Rabs_R0, Rlt_bool_true by apply bpow_gt_0.
discriminate.
Qed.

(** When both operands carry the sign bit [s] and the exact result is nonzero,
    the result carries it too. *)
Lemma sign_of_same_sign_sum : forall x y : binary32,
  is_finite x = true -> is_finite y = true ->
  Bsign x = Bsign y ->
  B2R x + B2R y <> 0 ->
  match Rcompare (B2R x + B2R y) 0 with
  | Eq => andb (Bsign x) (Bsign y)
  | Lt => true
  | Gt => false
  end = Bsign x.
Proof.
intros x y Fx Fy Hs Hnz.
destruct (Bsign x) eqn:Bx.
- assert (Hx : B2R x <= 0) by (apply B2R_le_0_of_sign; assumption).
  assert (Hy : B2R y <= 0)
    by (apply B2R_le_0_of_sign; [assumption | now rewrite <- Hs]).
  rewrite Rcompare_Lt by lra. reflexivity.
- assert (Hx : 0 <= B2R x) by (apply B2R_ge_0_of_sign; assumption).
  assert (Hy : 0 <= B2R y)
    by (apply B2R_ge_0_of_sign; [assumption | now rewrite <- Hs]).
  rewrite Rcompare_Gt by lra. reflexivity.
Qed.

(** * The five operations *)

Lemma dr_plus : forall x y : binary32,
  round radix2 fexp32 rne (round radix2 fexp64 rne (B2R x + B2R y))
  = round radix2 fexp32 rne (B2R x + B2R y).
Proof. intros x y. apply f32_double_round_plus. Qed.

Lemma narrow_widen_plus_finite : forall x y : binary32,
  is_finite x = true -> is_finite y = true ->
  narrow (b64_plus (widen x) (widen y)) = f32_plus x y.
Proof.
intros x y Fx Fy.
unfold f32_plus, rnd_NE, b64_plus.
assert (Nx : is_nan x = false) by (apply is_finite_not_nan; exact Fx).
assert (Ny : is_nan y = false) by (apply is_finite_not_nan; exact Fy).
assert (Fwx : is_finite (widen x) = true) by (rewrite widen_is_finite; exact Fx).
assert (Fwy : is_finite (widen y) = true) by (rewrite widen_is_finite; exact Fy).
generalize (@Bplus_correct prec64 emax64 prec64_gt_0 prec64_lt_emax64 mode_NE
              (widen x) (widen y) Fwx Fwy).
rewrite !widen_B2R, !(widen_Bsign _ Nx), !(widen_Bsign _ Ny).
rewrite (b64_no_overflow _ (bound_sum x y)).
intros (D1 & D2 & D3).
apply (narrow_eq_of (B2R x + B2R y)).
- apply dr_plus.
- exact D2.
- exact D1.
- generalize (@Bplus_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE
                x y Fx Fy).
  destruct (Rlt_bool (Rabs (round radix2 fexp32 rne (B2R x + B2R y)))
                     (bpow radix2 emax32)) eqn:E.
  + intros (Z1 & Z2 & Z3). repeat split; try assumption.
    now rewrite Z3, D3.
  + intros (Z1 & Z2). rewrite Z1. f_equal.
    rewrite D3. symmetry.
    apply sign_of_same_sign_sum; try assumption.
    now apply overflow_arg_neq_0.
Qed.

(** Replace [widen] of a [B754_finite] by the [B754_finite] it is, so that the
    operations reduce on the constructors. *)
Ltac shape_widen :=
  repeat match goal with
  | |- context [widen (B754_finite ?s ?m ?e ?H)] =>
      let Hs := fresh "Hs" in
      let HG := fresh "HG" in
      assert (Hs := widen_strict s m e H);
      destruct (widen_finite_spec s m e H) as (_ & _ & HG);
      revert Hs HG;
      destruct (widen (B754_finite s m e H)) as [?|?| |? ? ? ?];
      intros Hs HG; try discriminate Hs;
      cbn in HG; rewrite ?HG; clear Hs HG
  end.

Theorem narrow_widen_plus : forall x y : binary32,
  narrow (b64_plus (widen x) (widen y)) = f32_plus x y.
Proof.
intros x y.
destruct x as [sx|sx| |sx mx ex Hx]; destruct y as [sy|sy| |sy my ey Hy];
  try (apply narrow_widen_plus_finite; reflexivity);
  unfold b64_plus, f32_plus, rnd_NE; shape_widen;
  cbn - [binary_normalize];
  try reflexivity;
  now destruct (Bool.eqb sx sy).
Qed.

(** ** Subtraction

    [f32_minus] is [f32_plus x (f32_neg y)], and Extract.v gives it no
    directive of its own, so the native build negates and adds. Negation is
    exact in both formats and commutes with widening, so subtraction is a
    corollary of addition rather than a separate argument. *)

Lemma widen_Bopp : forall y : binary32, widen (Bopp y) = Bopp (widen y).
Proof.
intros [sy|sy| |sy my ey Hy]; try reflexivity.
cbn [Bopp].
destruct (widen_finite_spec sy my ey Hy) as (HR & HF & HS).
destruct (widen_finite_spec (negb sy) my ey Hy) as (HR' & HF' & HS').
apply B2R_Bsign_inj.
- exact HF'.
- rewrite is_finite_Bopp. exact HF.
- rewrite HR', B2R_Bopp, HR.
  now rewrite <- (@B2R_Bopp prec32 emax32 (B754_finite sy my ey Hy)).
- rewrite HS'. clear HR HF HR' HF' HS'.
  assert (Hs := widen_strict sy my ey Hy).
  revert HS Hs.
  destruct (widen (B754_finite sy my ey Hy)); cbn; intros HS Hs;
    try discriminate Hs.
  now rewrite HS.
Qed.

Theorem narrow_widen_minus : forall x y : binary32,
  narrow (b64_plus (widen x) (Bopp (widen y))) = f32_minus x y.
Proof.
intros x y. unfold f32_minus, f32_neg.
rewrite <- widen_Bopp. apply narrow_widen_plus.
Qed.

(** ** Multiplication *)

Lemma dr_mult : forall x y : binary32,
  round radix2 fexp32 rne (round radix2 fexp64 rne (B2R x * B2R y))
  = round radix2 fexp32 rne (B2R x * B2R y).
Proof. intros x y. apply f32_double_round_mult. Qed.

Lemma bound_prod : forall x y : binary32,
  (Rabs (B2R x * B2R y) <= bpow radix2 1000)%R.
Proof.
intros x y.
rewrite Rabs_mult.
apply Rle_trans with (bpow radix2 emax32 * bpow radix2 emax32)%R.
- apply Rmult_le_compat; try apply Rabs_pos; apply Rlt_le, B2R_f32_lt_emax32.
- rewrite <- bpow_plus. apply bpow_le. unfold emax32. lia.
Qed.

Lemma narrow_widen_mult_finite : forall x y : binary32,
  is_finite x = true -> is_finite y = true ->
  narrow (b64_mult (widen x) (widen y)) = f32_mult x y.
Proof.
intros x y Fx Fy.
unfold f32_mult, rnd_NE, b64_mult.
assert (Nx : is_nan x = false) by (apply is_finite_not_nan; exact Fx).
assert (Ny : is_nan y = false) by (apply is_finite_not_nan; exact Fy).
assert (Fwx : is_finite (widen x) = true) by (rewrite widen_is_finite; exact Fx).
assert (Fwy : is_finite (widen y) = true) by (rewrite widen_is_finite; exact Fy).
generalize (@Bmult_correct prec64 emax64 prec64_gt_0 prec64_lt_emax64 mode_NE
              (widen x) (widen y)).
rewrite !widen_B2R, !(widen_Bsign _ Nx), !(widen_Bsign _ Ny).
rewrite (b64_no_overflow _ (bound_prod x y)).
intros (D1 & D2 & D3).
rewrite Fwx, Fwy in D2. cbn in D2.
assert (Dn : is_nan (Bmult mode_NE (widen x) (widen y)) = false)
  by (apply is_finite_not_nan; exact D2).
apply (narrow_eq_of (B2R x * B2R y)).
- apply dr_mult.
- exact D2.
- exact D1.
- generalize (@Bmult_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y).
  destruct (Rlt_bool (Rabs (round radix2 fexp32 rne (B2R x * B2R y)))
                     (bpow radix2 emax32)) eqn:E.
  + intros (Z1 & Z2 & Z3). rewrite Fx, Fy in Z2. cbn in Z2.
    assert (Zn : is_nan (Bmult mode_NE x y) = false)
      by (apply is_finite_not_nan; exact Z2).
    repeat split; try assumption.
    now rewrite (Z3 Zn), (D3 Dn).
  + intros Z1. now rewrite Z1, (D3 Dn).
Qed.

Theorem narrow_widen_mult : forall x y : binary32,
  narrow (b64_mult (widen x) (widen y)) = f32_mult x y.
Proof.
intros x y.
destruct x as [sx|sx| |sx mx ex Hx]; destruct y as [sy|sy| |sy my ey Hy];
  try (apply narrow_widen_mult_finite; reflexivity);
  unfold b64_mult, f32_mult, rnd_NE; shape_widen;
  cbn - [binary_normalize]; reflexivity.
Qed.

(** ** Division *)

Lemma dr_div : forall x y : binary32, B2R y <> 0 ->
  round radix2 fexp32 rne (round radix2 fexp64 rne (B2R x / B2R y))
  = round radix2 fexp32 rne (B2R x / B2R y).
Proof. intros x y H. now apply f32_double_round_div. Qed.

Lemma bound_quot : forall x y : binary32,
  is_finite_strict y = true ->
  (Rabs (B2R x / B2R y) <= bpow radix2 1000)%R.
Proof.
intros x y Hy.
assert (Hb : (bpow radix2 (SpecFloat.emin prec32 emax32) <= Rabs (B2R y))%R)
  by (apply abs_B2R_ge_emin; exact Hy).
assert (Hbp : (0 < bpow radix2 (SpecFloat.emin prec32 emax32))%R) by apply bpow_gt_0.
assert (Hy0 : (0 < Rabs (B2R y))%R) by lra.
unfold Rdiv. rewrite Rabs_mult, Rabs_inv.
apply Rle_trans
  with (bpow radix2 emax32 * / bpow radix2 (SpecFloat.emin prec32 emax32))%R.
- apply Rmult_le_compat.
  + apply Rabs_pos.
  + apply Rlt_le, Rinv_0_lt_compat, Hy0.
  + apply Rlt_le, B2R_f32_lt_emax32.
  + apply Rinv_le_contravar; assumption.
- rewrite <- bpow_opp, <- bpow_plus. apply bpow_le.
  unfold emax32, prec32, SpecFloat.emin. lia.
Qed.

Lemma narrow_widen_div_finite : forall x y : binary32,
  is_finite x = true -> is_finite_strict y = true ->
  narrow (b64_div (widen x) (widen y)) = f32_div x y.
Proof.
intros x y Fx Sy.
unfold f32_div, rnd_NE, b64_div.
assert (Fy : is_finite y = true)
  by (destruct y; try discriminate Sy; reflexivity).
assert (Nx : is_nan x = false) by (apply is_finite_not_nan; exact Fx).
assert (Ny : is_nan y = false) by (apply is_finite_not_nan; exact Fy).
assert (Zy : B2R y <> 0)
  by (destruct y; try discriminate Sy; apply B2R_finite_neq_0).
assert (Zwy : B2R (widen y) <> 0) by (rewrite widen_B2R; exact Zy).
generalize (@Bdiv_correct prec64 emax64 prec64_gt_0 prec64_lt_emax64 mode_NE
              (widen x) (widen y) Zwy).
rewrite !widen_B2R, !(widen_Bsign _ Nx), !(widen_Bsign _ Ny), !widen_is_finite.
rewrite (b64_no_overflow _ (bound_quot x y Sy)).
intros (D1 & D2 & D3).
rewrite Fx in D2.
assert (Dn : is_nan (Bdiv mode_NE (widen x) (widen y)) = false)
  by (apply is_finite_not_nan; exact D2).
apply (narrow_eq_of (B2R x / B2R y)).
- now apply dr_div.
- exact D2.
- exact D1.
- generalize (@Bdiv_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE
                x y Zy).
  destruct (Rlt_bool (Rabs (round radix2 fexp32 rne (B2R x / B2R y)))
                     (bpow radix2 emax32)) eqn:E.
  + intros (Z1 & Z2 & Z3). rewrite Fx in Z2.
    assert (Zn : is_nan (Bdiv mode_NE x y) = false)
      by (apply is_finite_not_nan; exact Z2).
    repeat split; try assumption.
    now rewrite (Z3 Zn), (D3 Dn).
  + intros Z1. now rewrite Z1, (D3 Dn).
Qed.

Theorem narrow_widen_div : forall x y : binary32,
  narrow (b64_div (widen x) (widen y)) = f32_div x y.
Proof.
intros x y.
destruct x as [sx|sx| |sx mx ex Hx]; destruct y as [sy|sy| |sy my ey Hy];
  try (apply narrow_widen_div_finite; reflexivity);
  unfold b64_div, f32_div, rnd_NE; shape_widen;
  cbn - [binary_normalize]; reflexivity.
Qed.

(** ** Square root *)

Lemma dr_sqrt : forall x : binary32,
  round radix2 fexp32 rne (round radix2 fexp64 rne (sqrt (B2R x)))
  = round radix2 fexp32 rne (sqrt (B2R x)).
Proof. intros x. apply f32_double_round_sqrt. Qed.

Lemma sqrt_le_bpow64 : forall x : binary32,
  (Rabs (sqrt (B2R x)) <= bpow radix2 64)%R.
Proof.
intros x.
rewrite Rabs_pos_eq by apply sqrt_pos.
apply Rle_trans with (sqrt (bpow radix2 (2 * 64))).
- apply sqrt_le_1_alt.
  apply Rle_trans with (Rabs (B2R x)); [apply Rle_abs|].
  apply Rlt_le.
  generalize (B2R_f32_lt_emax32 x). unfold emax32.
  now replace (2 * 64)%Z with 128%Z by ring.
- rewrite sqrt_bpow. apply Rle_refl.
Qed.

Lemma sqrt_no_overflow32 : forall x : binary32,
  Rlt_bool (Rabs (round radix2 fexp32 rne (sqrt (B2R x)))) (bpow radix2 emax32)
  = true.
Proof.
intros x. apply Rlt_bool_true.
apply Rle_lt_trans with (bpow radix2 64).
- apply (@abs_round_le_generic radix2 fexp32 valid_exp_fexp32 rne
           (valid_rnd_round_mode mode_NE) (sqrt (B2R x)) (bpow radix2 64)).
  + apply generic_format_bpow.
    unfold SpecFloat.fexp, SpecFloat.emin, prec32, emax32. lia.
  + apply sqrt_le_bpow64.
- apply bpow_lt. unfold emax32. lia.
Qed.

Theorem narrow_widen_sqrt : forall x : binary32,
  narrow (b64_sqrt (widen x)) = f32_sqrt x.
Proof.
intros x.
destruct x as [sx|sx| |sx mx ex Hx].
- reflexivity.
- now destruct sx.
- reflexivity.
- unfold b64_sqrt, f32_sqrt, rnd_NE.
  assert (Hs := widen_strict sx mx ex Hx).
  destruct (widen_finite_spec sx mx ex Hx) as (HR & HF & HG).
  revert Hs HR HF HG.
  destruct (widen (B754_finite sx mx ex Hx)) as [?|?| |s' m' e' H'];
    intros Hs HR HF HG; try discriminate Hs.
  cbn in HG. subst s'.
  destruct sx.
  + reflexivity.
  + generalize (@Bsqrt_correct prec64 emax64 prec64_gt_0 prec64_lt_emax64
                  mode_NE (B754_finite false m' e' H')).
    rewrite HR. intros (D1 & D2 & D3). cbn in D2.
    assert (Dn : is_nan (Bsqrt mode_NE (B754_finite false m' e' H')) = false)
      by (apply is_finite_not_nan; exact D2).
    apply (narrow_eq_of (sqrt (B2R (B754_finite false mx ex Hx)))).
    * apply dr_sqrt.
    * exact D2.
    * exact D1.
    * rewrite (sqrt_no_overflow32 (B754_finite false mx ex Hx)).
      generalize (@Bsqrt_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32
                    mode_NE (B754_finite false mx ex Hx)).
      intros (Z1 & Z2 & Z3). cbn in Z2.
      assert (Zn : is_nan (Bsqrt mode_NE (B754_finite false mx ex Hx)) = false)
        by (apply is_finite_not_nan; exact Z2).
      repeat split; try assumption.
      now rewrite (Z3 Zn), (D3 Dn).
Qed.
