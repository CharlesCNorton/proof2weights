(** * The native build's other directives

    Narrow.v covers the five arithmetic operations. Extract.v maps six more
    binary32 functions to host operations on the binary64 value a binary32
    occupies once widened: negation, absolute value, the two comparisons, the
    finiteness test, and the conversion of an integer. Each is the binary64
    operation of the same name on the widened operand, and this file proves it,
    so what the native build trusts of them is that the host performs those
    binary64 operations as IEEE-754 specifies. Integer conversion is exact in
    binary64 only for an integer with at most 53 significant bits;
    [f32_of_Z_constants] checks every literal the development converts. *)

From Stdlib Require Import ZArith Reals Lra Lia List.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Float_error Narrow Bound64.

Import ListNotations.

Open Scope R_scope.

#[local] Existing Instance valid_rnd_round_mode.

(** A [B754_finite] widens to a [B754_finite] of the same sign. *)
Lemma widen_finite_shape :
  forall (s : bool) m e (H : SpecFloat.bounded prec32 emax32 m e = true),
  exists m' e' H', widen (B754_finite s m e H) = B754_finite s m' e' H'.
Proof.
intros s m e H.
assert (Hs := widen_strict s m e H).
destruct (widen_finite_spec s m e H) as (_ & _ & HS).
destruct (widen (B754_finite s m e H)) as [?|?| |s' m' e' H']; try discriminate Hs.
cbn in HS. subst s'. now exists m', e', H'.
Qed.

(** ** Negation and absolute value *)

Theorem widen_neg : forall x : binary32, widen (f32_neg x) = b64_neg (widen x).
Proof. intros x. unfold f32_neg, b64_neg. apply widen_Bopp. Qed.

Theorem widen_abs : forall x : binary32, widen (f32_abs x) = b64_abs (widen x).
Proof.
intros [sx|sx| |sx mx ex Hx]; try reflexivity.
unfold f32_abs, b64_abs. cbn [Babs].
destruct (widen_finite_spec sx mx ex Hx) as (HR & HF & HS).
destruct (widen_finite_spec false mx ex Hx) as (HR' & HF' & HS').
apply B2R_Bsign_inj.
- exact HF'.
- rewrite is_finite_Babs. exact HF.
- rewrite HR', B2R_Babs, HR.
  now rewrite <- (@B2R_Babs prec32 emax32 (B754_finite sx mx ex Hx)).
- rewrite HS'. clear HR HF HR' HF' HS'.
  destruct (widen_finite_shape sx mx ex Hx) as (m' & e' & H' & ->).
  reflexivity.
Qed.

(** ** Comparison *)

Lemma widen_Bcompare : forall x y : binary32,
  Bcompare (widen x) (widen y) = Bcompare x y.
Proof.
intros x y.
destruct (Bool.bool_dec (is_finite x && is_finite y) true) as [Hf|Hf].
- apply andb_prop in Hf. destruct Hf as [Hx Hy].
  rewrite (Bcompare_correct prec32 emax32 x y Hx Hy).
  rewrite (Bcompare_correct prec64 emax64 (widen x) (widen y))
    by now rewrite widen_is_finite.
  now rewrite !widen_B2R.
- destruct x as [sx|sx| |sx mx ex Hx]; destruct y as [sy|sy| |sy my ey Hy];
    cbn in Hf; try (elim Hf; reflexivity);
    try reflexivity;
    try (destruct (widen_finite_shape sx mx ex Hx) as (mx' & ex' & Hx' & ->));
    try (destruct (widen_finite_shape sy my ey Hy) as (my' & ey' & Hy' & ->));
    reflexivity.
Qed.

Theorem widen_lt : forall x y : binary32, f32_lt x y = b64_lt (widen x) (widen y).
Proof. intros x y. unfold f32_lt, f32_compare, b64_lt. now rewrite widen_Bcompare. Qed.

Theorem widen_le : forall x y : binary32, f32_le x y = b64_le (widen x) (widen y).
Proof. intros x y. unfold f32_le, f32_compare, b64_le. now rewrite widen_Bcompare. Qed.

(** ** Finiteness *)

Theorem widen_finite : forall x : binary32, f32_finite x = b64_finite (widen x).
Proof. intros x. unfold f32_finite, b64_finite. now rewrite widen_is_finite. Qed.

(** ** Integer conversion

    The directive converts the host integer to binary64, which is exact for an
    integer with at most 53 significant bits, and narrows the result. *)

Definition b64_of_Z (n : Z) : binary64 :=
  @binary_normalize prec64 emax64 prec64_pos prec64_emax mode_NE n 0 false.

Lemma F2R_Z0 : forall n : Z, F2R (Float radix2 n 0) = IZR n.
Proof. intros n. unfold F2R. cbn [Fnum Fexp bpow]. ring. Qed.

Lemma Rcompare_sign : forall n : Z, n <> 0%Z ->
  match Rcompare (IZR n) 0 with Eq => false | Lt => true | Gt => false end
  = Rlt_bool (IZR n) 0.
Proof.
intros n Hn.
destruct (Rcompare_spec (IZR n) 0) as [H|H|H].
- now rewrite Rlt_bool_true.
- apply eq_IZR in H. contradiction.
- rewrite Rlt_bool_false by lra. reflexivity.
Qed.

Theorem narrow_of_Z : forall n : Z,
  generic_format radix2 fexp64 (IZR n) -> (Z.abs n < 2 ^ 1000)%Z ->
  narrow (b64_of_Z n) = f32_of_Z n.
Proof.
intros n Hg Hn.
destruct (Z.eq_dec n 0) as [->|Hnz]; [reflexivity|].
assert (Hb : Rabs (IZR n) <= bpow radix2 1000).
{ rewrite <- abs_IZR. apply Rlt_le. rewrite <- IZR_Zpower by lia.
  apply IZR_lt. change (Zpower radix2 1000) with (2 ^ 1000)%Z. exact Hn. }
generalize (@binary_normalize_correct prec64 emax64 prec64_pos prec64_emax
              mode_NE n 0 false).
generalize (@binary_normalize_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32
              mode_NE n 0 false).
cbv zeta. rewrite !F2R_Z0.
fold (b64_of_Z n). fold (f32_of_Z n).
rewrite (round_generic radix2 fexp64 _ (IZR n) Hg).
rewrite (Rlt_bool_true (Rabs (IZR n)) (bpow radix2 emax64))
  by (eapply Rle_lt_trans; [exact Hb|]; apply bpow_lt; unfold emax64; lia).
intros Hz (D1 & D2 & D3).
apply (narrow_eq_of (IZR n)).
- now rewrite (round_generic radix2 fexp64 _ (IZR n) Hg).
- exact D2.
- now rewrite (round_generic radix2 fexp64 _ (IZR n) Hg).
- revert Hz. rewrite D3, !Rcompare_sign by exact Hnz.
  destruct (Rlt_bool (Rabs (round radix2 fexp32 rne (IZR n))) (bpow radix2 emax32)).
  + intros (Z1 & Z2 & Z3). repeat split; assumption.
  + intros Hz. exact Hz.
Qed.

(** An integer below [2^53] in magnitude is a binary64 value. *)
Lemma format_small_Z : forall n : Z, (Z.abs n < 2 ^ 53)%Z ->
  generic_format radix2 fexp64 (IZR n).
Proof.
intros n Hn.
apply generic_format_FLT.
apply (FLT_spec _ _ _ _ (Float radix2 n 0)).
- now rewrite F2R_Z0.
- cbn [Fnum radix_val radix2]. unfold prec64. exact Hn.
- cbn [Fexp]. unfold SpecFloat.emin, prec64, emax64. lia.
Qed.

(** An integer [m 2^e] with [|m| < 2^53] and [e >= 0] is a binary64 value. *)
Lemma format_scaled_Z : forall m e : Z, (Z.abs m < 2 ^ 53)%Z -> (0 <= e)%Z ->
  generic_format radix2 fexp64 (IZR (m * 2 ^ e)).
Proof.
intros m e Hm He.
apply generic_format_FLT.
apply (FLT_spec _ _ _ _ (Float radix2 m e)).
- unfold F2R. cbn [Fnum Fexp].
  rewrite mult_IZR. f_equal.
  change (2 ^ e)%Z with (Zpower radix2 e).
  now rewrite IZR_Zpower.
- cbn [Fnum radix_val radix2]. unfold prec64. exact Hm.
- cbn [Fexp]. unfold SpecFloat.emin, prec64, emax64. lia.
Qed.

(** The integers the development converts with [f32_of_Z] as literals. Every
    one is below [2^53] in magnitude except [19!], which is
    [1856156927625 * 2^16]. *)
Definition f32_of_Z_literals : list Z :=
  [-1000000000; -88; 1; 2; 3; 4; 5; 6; 7; 8; 9; 11; 13; 15; 16; 24; 32; 64; 88;
   120; 127; 201; 355; 512; 720; 5040; 40320; 44715; 100000; 362880; 1000000;
   3628800; 4194304; 8388608; 10000000; 12102203; 12582912; 14581891; 19353072;
   31415927; 39916800; 479001600; 6227020800; 7978845608; 10000000000;
   68719476736; 87178291200; 1307674368000; 20922789888000; 355687428096000;
   6402373705728000; 121645100408832000]%Z.

Definition native_int (n : Z) : bool :=
  orb (Z.abs n <? 2 ^ 53)%Z (n =? 121645100408832000)%Z.

Lemma native_int_ok : forall n, native_int n = true ->
  generic_format radix2 fexp64 (IZR n) /\ (Z.abs n < 2 ^ 1000)%Z.
Proof.
intros n H. unfold native_int in H.
apply Bool.orb_true_iff in H. destruct H as [H|H].
- apply Z.ltb_lt in H. split; [now apply format_small_Z | lia].
- apply Z.eqb_eq in H. subst n. split.
  + change 121645100408832000%Z with (1856156927625 * 2 ^ 16)%Z.
    apply format_scaled_Z; [vm_compute; reflexivity | lia].
  + vm_compute. reflexivity.
Qed.

Lemma literals_native : forallb native_int f32_of_Z_literals = true.
Proof. vm_compute. reflexivity. Qed.

Theorem f32_of_Z_constants :
  Forall (fun n => generic_format radix2 fexp64 (IZR n) /\ (Z.abs n < 2 ^ 1000)%Z)
         f32_of_Z_literals.
Proof.
apply Forall_forall. intros n Hin. apply native_int_ok.
exact (proj1 (forallb_forall native_int f32_of_Z_literals) literals_native n Hin).
Qed.
