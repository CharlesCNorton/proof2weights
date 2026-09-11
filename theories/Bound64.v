(** * Directed rounding in binary64

    A running error bound is a binary64 value that must stay on one side of the
    real quantity it bounds. Each operation below rounds to nearest and then
    takes the floating-point successor, or the predecessor, so a finite result
    lies above, or below, the exact real result. Rounding to nearest is either
    the rounding down or the rounding up of its argument, and the successor of
    the rounding down is the rounding up, which is what the two lemmas of the
    first section record.

    The native extraction maps these operations to the host's binary64
    arithmetic and to [Float.succ] and [Float.pred]; the inductive extraction
    computes them in Flocq. *)

From Stdlib Require Import ZArith Reals Lra Lia.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Float_error.

Open Scope R_scope.

(** * Rounding to nearest, then one step outward *)

Section Outward.

Variable fexp : Z -> Z.
Context {Hv : Valid_exp fexp}.

Lemma succ_round_N_ge : forall choice z,
  z <= succ radix2 fexp (round radix2 fexp (Znearest choice) z).
Proof.
  intros choice z.
  destruct (round_DN_or_UP radix2 fexp (Znearest choice) z) as [H|H]; rewrite H.
  - destruct (generic_format_EM radix2 fexp z) as [Fz|Fz].
    + rewrite (round_generic radix2 fexp Zfloor z Fz). apply succ_ge_id.
    + rewrite (succ_DN_eq_UP radix2 fexp z Fz). apply (round_UP_pt radix2 fexp z).
  - eapply Rle_trans; [apply (round_UP_pt radix2 fexp z) | apply succ_ge_id].
Qed.

Lemma pred_round_N_le : forall choice z,
  pred radix2 fexp (round radix2 fexp (Znearest choice) z) <= z.
Proof.
  intros choice z.
  destruct (round_DN_or_UP radix2 fexp (Znearest choice) z) as [H|H]; rewrite H.
  - eapply Rle_trans; [apply pred_le_id | apply (round_DN_pt radix2 fexp z)].
  - destruct (generic_format_EM radix2 fexp z) as [Fz|Fz].
    + rewrite (round_generic radix2 fexp Zceil z Fz). apply pred_le_id.
    + rewrite (pred_UP_eq_DN radix2 fexp z Fz). apply (round_DN_pt radix2 fexp z).
Qed.

End Outward.

(** * The format *)

Lemma prec64_pos : Prec_gt_0 prec64.
Proof. unfold Prec_gt_0, prec64. lia. Qed.

Lemma prec64_emax : Prec_lt_emax prec64 emax64.
Proof. unfold Prec_lt_emax, prec64, emax64. lia. Qed.

Definition binary64 := binary_float prec64 emax64.

Definition b64_plus (x y : binary64) : binary64 :=
  @Bplus prec64 emax64 prec64_pos prec64_emax mode_NE x y.
Definition b64_mult (x y : binary64) : binary64 :=
  @Bmult prec64 emax64 prec64_pos prec64_emax mode_NE x y.
Definition b64_div (x y : binary64) : binary64 :=
  @Bdiv prec64 emax64 prec64_pos prec64_emax mode_NE x y.
Definition b64_sqrt (x : binary64) : binary64 :=
  @Bsqrt prec64 emax64 prec64_pos prec64_emax mode_NE x.
Definition b64_succ (x : binary64) : binary64 :=
  @Bsucc prec64 emax64 prec64_pos prec64_emax x.
Definition b64_pred (x : binary64) : binary64 :=
  @Bpred prec64 emax64 prec64_pos prec64_emax x.
Definition b64_neg (x : binary64) : binary64 := Bopp x.
Definition b64_abs (x : binary64) : binary64 := Babs x.
Definition b64_finite (x : binary64) : bool := is_finite x.
Definition b64_lt (x y : binary64) : bool :=
  match Bcompare x y with Some Lt => true | _ => false end.
Definition b64_le (x y : binary64) : bool :=
  match Bcompare x y with Some Lt | Some Eq => true | _ => false end.
Definition b64_max (x y : binary64) : binary64 := if b64_lt x y then y else x.

Definition b64_zero : binary64 := B754_zero false.
Definition b64_inf : binary64 := B754_infinity false.
Definition b64_ninf : binary64 := B754_infinity true.

(** A binary32 value read as the binary64 value it equals. *)
Definition b64_of_f32 (x : binary32) : binary64 :=
  match x with
  | B754_zero s => B754_zero s
  | B754_infinity s => B754_infinity s
  | B754_nan => B754_nan
  | B754_finite s m e _ =>
      @binary_normalize prec64 emax64 prec64_pos prec64_emax mode_NE
        (SpecFloat.cond_Zopp s (Z.pos m)) e s
  end.

(** The outward operations. A non-finite intermediate yields an infinity on
    the side that carries no information. *)
Definition up_plus (x y : binary64) : binary64 :=
  let w := b64_plus x y in if b64_finite w then b64_succ w else b64_inf.
Definition up_mult (x y : binary64) : binary64 :=
  let w := b64_mult x y in if b64_finite w then b64_succ w else b64_inf.
Definition up_div (x y : binary64) : binary64 :=
  let w := b64_div x y in if b64_finite w then b64_succ w else b64_inf.
Definition dn_plus (x y : binary64) : binary64 :=
  let w := b64_plus x y in if b64_finite w then b64_pred w else b64_ninf.
Definition dn_mult (x y : binary64) : binary64 :=
  let w := b64_mult x y in if b64_finite w then b64_pred w else b64_ninf.
Definition dn_sqrt (x : binary64) : binary64 :=
  let w := b64_sqrt x in if b64_finite w then b64_pred w else b64_ninf.

(** The constants the bounds use: [2^-24], [2^-150], [1 + 2^-23], [1/2], [1]
    and [2]. *)
Definition c64_u : binary64 :=
  @B754_finite prec64 emax64 false 4503599627370496 (-76) eq_refl.
Definition c64_eta : binary64 :=
  @B754_finite prec64 emax64 false 4503599627370496 (-202) eq_refl.
Definition c64_1p2u : binary64 :=
  @B754_finite prec64 emax64 false 4503600164241408 (-52) eq_refl.
Definition c64_half : binary64 :=
  @B754_finite prec64 emax64 false 4503599627370496 (-53) eq_refl.
Definition c64_one : binary64 :=
  @B754_finite prec64 emax64 false 4503599627370496 (-52) eq_refl.
Definition c64_two : binary64 :=
  @B754_finite prec64 emax64 false 4503599627370496 (-51) eq_refl.

(** * What the operations compute *)

Notation fexp64 := (SpecFloat.fexp prec64 emax64).

Definition r64 (z : R) : R := round radix2 fexp64 (round_mode mode_NE) z.

Lemma fexp64_valid : Valid_exp fexp64.
Proof. unfold SpecFloat.fexp. apply FLT_exp_valid. exact prec64_pos. Qed.

Lemma succ_r64_ge : forall z, z <= succ radix2 fexp64 (r64 z).
Proof.
  intros z. unfold r64.
  change (round_mode mode_NE) with (Znearest (fun n => negb (Z.even n))).
  apply (succ_round_N_ge fexp64 (Hv := fexp64_valid)).
Qed.

Lemma pred_r64_le : forall z, pred radix2 fexp64 (r64 z) <= z.
Proof.
  intros z. unfold r64.
  change (round_mode mode_NE) with (Znearest (fun n => negb (Z.even n))).
  apply (pred_round_N_le fexp64 (Hv := fexp64_valid)).
Qed.

Lemma overflow_not_finite : forall (w : binary64) s,
  B2SF w = binary_overflow prec64 emax64 mode_NE s -> is_finite w = false.
Proof.
  intros w s H. rewrite <- is_finite_SF_B2SF, H. reflexivity.
Qed.

Lemma b64_plus_fin : forall x y : binary64,
  is_finite (b64_plus x y) = true -> is_finite x = true /\ is_finite y = true.
Proof.
  intros [sx|sx| |sx mx ex Hx] [sy|sy| |sy my ey Hy]; unfold b64_plus; simpl;
    try (intros; split; reflexivity); try discriminate;
    destruct (Bool.eqb sx sy); discriminate.
Qed.

Lemma b64_plus_exact : forall x y : binary64,
  is_finite (b64_plus x y) = true ->
  is_finite x = true /\ is_finite y = true
  /\ B2R (b64_plus x y) = r64 (B2R x + B2R y).
Proof.
  intros x y Hw.
  destruct (b64_plus_fin x y Hw) as [Hx Hy].
  split; [exact Hx | split; [exact Hy|]].
  pose proof (Bplus_correct prec64 emax64 prec64_pos prec64_emax mode_NE x y Hx Hy) as H.
  unfold b64_plus in *. unfold r64.
  destruct (Rlt_bool _ _).
  - apply H.
  - destruct H as [H _]. rewrite (overflow_not_finite _ _ H) in Hw. discriminate.
Qed.

Lemma b64_mult_exact : forall x y : binary64,
  is_finite (b64_mult x y) = true ->
  is_finite x = true /\ is_finite y = true
  /\ B2R (b64_mult x y) = r64 (B2R x * B2R y).
Proof.
  intros x y Hw.
  pose proof (Bmult_correct prec64 emax64 prec64_pos prec64_emax mode_NE x y) as H.
  unfold b64_mult in *. unfold r64.
  destruct (Rlt_bool _ _).
  - destruct H as [H1 [H2 _]]. rewrite H2 in Hw.
    apply andb_prop in Hw. destruct Hw as [Hx Hy]. auto.
  - rewrite (overflow_not_finite _ _ H) in Hw. discriminate.
Qed.

Lemma b64_div_exact : forall x y : binary64,
  B2R y <> 0 -> is_finite (b64_div x y) = true ->
  is_finite x = true /\ B2R (b64_div x y) = r64 (B2R x / B2R y).
Proof.
  intros x y Hy Hw.
  pose proof (Bdiv_correct prec64 emax64 prec64_pos prec64_emax mode_NE x y Hy) as H.
  unfold b64_div in *. unfold r64.
  destruct (Rlt_bool _ _).
  - destruct H as [H1 [H2 _]]. rewrite H2 in Hw. auto.
  - rewrite (overflow_not_finite _ _ H) in Hw. discriminate.
Qed.

Lemma b64_sqrt_exact : forall x : binary64,
  B2R (b64_sqrt x) = r64 (sqrt (B2R x)).
Proof.
  intros x. unfold b64_sqrt, r64.
  apply (Bsqrt_correct prec64 emax64 prec64_pos prec64_emax mode_NE x).
Qed.

Lemma b64_succ_exact : forall w : binary64,
  is_finite w = true -> is_finite (b64_succ w) = true ->
  B2R (b64_succ w) = succ radix2 fexp64 (B2R w).
Proof.
  intros w Hw Hs.
  pose proof (Bsucc_correct prec64 emax64 prec64_pos prec64_emax w Hw) as H.
  unfold b64_succ in *.
  destruct (Rlt_bool _ _).
  - apply H.
  - rewrite <- is_finite_SF_B2SF, H in Hs. discriminate.
Qed.

Lemma b64_pred_exact : forall w : binary64,
  is_finite w = true -> is_finite (b64_pred w) = true ->
  B2R (b64_pred w) = pred radix2 fexp64 (B2R w).
Proof.
  intros w Hw Hs.
  pose proof (Bpred_correct prec64 emax64 prec64_pos prec64_emax w Hw) as H.
  unfold b64_pred in *.
  destruct (Rlt_bool _ _).
  - apply H.
  - rewrite <- is_finite_SF_B2SF, H in Hs. discriminate.
Qed.

(** * The outward operations bound the exact results *)

Lemma up_plus_spec : forall x y : binary64,
  is_finite (up_plus x y) = true ->
  is_finite x = true /\ is_finite y = true /\ B2R x + B2R y <= B2R (up_plus x y).
Proof.
  intros x y H. unfold up_plus in *.
  destruct (b64_finite (b64_plus x y)) eqn:Hw; [|discriminate].
  destruct (b64_plus_exact x y Hw) as (Hx & Hy & He).
  repeat split; try assumption.
  rewrite (b64_succ_exact _ Hw H), He. apply succ_r64_ge.
Qed.

Lemma up_mult_spec : forall x y : binary64,
  is_finite (up_mult x y) = true ->
  is_finite x = true /\ is_finite y = true /\ B2R x * B2R y <= B2R (up_mult x y).
Proof.
  intros x y H. unfold up_mult in *.
  destruct (b64_finite (b64_mult x y)) eqn:Hw; [|discriminate].
  destruct (b64_mult_exact x y Hw) as (Hx & Hy & He).
  repeat split; try assumption.
  rewrite (b64_succ_exact _ Hw H), He. apply succ_r64_ge.
Qed.

Lemma up_div_spec : forall x y : binary64,
  B2R y <> 0 -> is_finite (up_div x y) = true ->
  is_finite x = true /\ B2R x / B2R y <= B2R (up_div x y).
Proof.
  intros x y Hy H. unfold up_div in *.
  destruct (b64_finite (b64_div x y)) eqn:Hw; [|discriminate].
  destruct (b64_div_exact x y Hy Hw) as (Hx & He).
  split; [exact Hx|].
  rewrite (b64_succ_exact _ Hw H), He. apply succ_r64_ge.
Qed.

Lemma dn_plus_spec : forall x y : binary64,
  is_finite (dn_plus x y) = true ->
  is_finite x = true /\ is_finite y = true /\ B2R (dn_plus x y) <= B2R x + B2R y.
Proof.
  intros x y H. unfold dn_plus in *.
  destruct (b64_finite (b64_plus x y)) eqn:Hw; [|discriminate].
  destruct (b64_plus_exact x y Hw) as (Hx & Hy & He).
  repeat split; try assumption.
  rewrite (b64_pred_exact _ Hw H), He. apply pred_r64_le.
Qed.

Lemma dn_mult_spec : forall x y : binary64,
  is_finite (dn_mult x y) = true ->
  is_finite x = true /\ is_finite y = true /\ B2R (dn_mult x y) <= B2R x * B2R y.
Proof.
  intros x y H. unfold dn_mult in *.
  destruct (b64_finite (b64_mult x y)) eqn:Hw; [|discriminate].
  destruct (b64_mult_exact x y Hw) as (Hx & Hy & He).
  repeat split; try assumption.
  rewrite (b64_pred_exact _ Hw H), He. apply pred_r64_le.
Qed.

Lemma dn_sqrt_spec : forall x : binary64,
  is_finite (dn_sqrt x) = true -> B2R (dn_sqrt x) <= sqrt (B2R x).
Proof.
  intros x H. unfold dn_sqrt in *.
  destruct (b64_finite (b64_sqrt x)) eqn:Hw; [|discriminate].
  rewrite (b64_pred_exact _ Hw H), b64_sqrt_exact. apply pred_r64_le.
Qed.

Lemma b64_lt_spec : forall x y : binary64,
  is_finite x = true -> is_finite y = true -> b64_lt x y = true -> B2R x < B2R y.
Proof.
  intros x y Hx Hy H. unfold b64_lt in H.
  rewrite (Bcompare_correct prec64 emax64 x y Hx Hy) in H.
  destruct (Rcompare_spec (B2R x) (B2R y)); try discriminate. assumption.
Qed.

Lemma b64_le_spec : forall x y : binary64,
  is_finite x = true -> is_finite y = true -> b64_le x y = true -> B2R x <= B2R y.
Proof.
  intros x y Hx Hy H. unfold b64_le in H.
  rewrite (Bcompare_correct prec64 emax64 x y Hx Hy) in H.
  destruct (Rcompare_spec (B2R x) (B2R y)); try discriminate; lra.
Qed.

Lemma b64_max_spec : forall x y : binary64,
  is_finite x = true -> is_finite y = true ->
  is_finite (b64_max x y) = true /\ B2R (b64_max x y) = Rmax (B2R x) (B2R y).
Proof.
  intros x y Hx Hy. unfold b64_max, b64_lt.
  rewrite (Bcompare_correct prec64 emax64 x y Hx Hy).
  destruct (Rcompare_spec (B2R x) (B2R y)) as [H|H|H]; split; try assumption.
  - rewrite Rmax_right; lra.
  - rewrite Rmax_left; lra.
  - rewrite Rmax_left; lra.
Qed.

Lemma b64_neg_spec : forall x : binary64,
  is_finite (b64_neg x) = is_finite x /\ B2R (b64_neg x) = - B2R x.
Proof. intros x. unfold b64_neg. split; [apply is_finite_Bopp | apply B2R_Bopp]. Qed.

Lemma b64_abs_spec : forall x : binary64,
  is_finite (b64_abs x) = is_finite x /\ B2R (b64_abs x) = Rabs (B2R x).
Proof. intros x. unfold b64_abs. split; [apply is_finite_Babs | apply B2R_Babs]. Qed.

(** * Reading a binary32 value in binary64 *)

Lemma b64_of_f32_spec : forall x : binary32,
  is_finite x = true ->
  is_finite (b64_of_f32 x) = true /\ B2R (b64_of_f32 x) = B2R x.
Proof.
  intros x Hx.
  assert (Hfmt : generic_format radix2 fexp64 (B2R x)).
  { destruct (FLT_format_B2R x) as [f Hf Hm He].
    apply generic_format_FLT. apply (FLT_spec _ _ _ _ f Hf).
    - eapply Z.lt_le_trans; [exact Hm|].
      apply Z.pow_le_mono_r; cbn [radix_val radix2]; unfold prec32, prec64; lia.
    - unfold f32_emin, SpecFloat.emin, prec32, emax32, prec64, emax64 in *. lia. }
  assert (Hlt : Rabs (B2R x) < bpow radix2 emax64).
  { eapply Rlt_le_trans; [apply abs_B2R_lt_emax|].
    apply bpow_le. unfold emax32, emax64. lia. }
  destruct x as [s|s| |s m e Hb]; try discriminate.
  - split; reflexivity.
  - cbn [b64_of_f32].
    pose proof (binary_normalize_correct prec64 emax64 prec64_pos prec64_emax mode_NE
                  (SpecFloat.cond_Zopp s (Z.pos m)) e s) as H.
    cbv zeta in H.
    change (F2R (Float radix2 (SpecFloat.cond_Zopp s (Z.pos m)) e))
      with (B2R (B754_finite s m e Hb : binary32)) in H.
    rewrite (round_generic radix2 fexp64 _ _ Hfmt) in H.
    rewrite Rlt_bool_true in H by exact Hlt.
    destruct H as [H1 [H2 _]]. split; assumption.
Qed.

(** * The constants *)

Lemma B2R_c64_u : B2R c64_u = f32_u.
Proof.
  unfold c64_u, f32_u, prec32, B2R, F2R. cbn [Fnum Fexp SpecFloat.cond_Zopp].
  change (IZR 4503599627370496) with (bpow radix2 52).
  rewrite <- bpow_plus.
  change (/ 2) with (bpow radix2 (-1)).
  rewrite <- bpow_plus. f_equal.
Qed.

Lemma B2R_c64_eta : B2R c64_eta = f32_eta.
Proof.
  unfold c64_eta, f32_eta, f32_emin, prec32, emax32, B2R, F2R.
  cbn [Fnum Fexp SpecFloat.cond_Zopp].
  change (IZR 4503599627370496) with (bpow radix2 52).
  rewrite <- bpow_plus.
  change (/ 2) with (bpow radix2 (-1)).
  rewrite <- bpow_plus. f_equal.
Qed.

Lemma B2R_c64_1p2u : B2R c64_1p2u = 1 + 2 * f32_u.
Proof.
  rewrite <- B2R_c64_u.
  unfold c64_1p2u, c64_u, B2R, F2R. cbn [Fnum Fexp SpecFloat.cond_Zopp].
  replace (IZR 4503600164241408) with (IZR 4503599627370496 + IZR 536870912)
    by (rewrite <- plus_IZR; reflexivity).
  change (IZR 4503599627370496) with (bpow radix2 52).
  change (IZR 536870912) with (bpow radix2 29).
  change 2 with (bpow radix2 1).
  rewrite Rmult_plus_distr_r, <- !bpow_plus.
  change 1 with (bpow radix2 0). f_equal.
Qed.

Lemma B2R_c64_half : B2R c64_half = / 2.
Proof.
  unfold c64_half, B2R, F2R. cbn [Fnum Fexp SpecFloat.cond_Zopp].
  change (IZR 4503599627370496) with (bpow radix2 52).
  rewrite <- bpow_plus. reflexivity.
Qed.

Lemma B2R_c64_one : B2R c64_one = 1.
Proof.
  unfold c64_one, B2R, F2R. cbn [Fnum Fexp SpecFloat.cond_Zopp].
  change (IZR 4503599627370496) with (bpow radix2 52).
  rewrite <- bpow_plus. reflexivity.
Qed.

Lemma B2R_c64_two : B2R c64_two = 2.
Proof.
  unfold c64_two, B2R, F2R. cbn [Fnum Fexp SpecFloat.cond_Zopp].
  change (IZR 4503599627370496) with (bpow radix2 52).
  rewrite <- bpow_plus. reflexivity.
Qed.

Lemma c64_finite :
  is_finite c64_u = true /\ is_finite c64_eta = true
  /\ is_finite c64_1p2u = true /\ is_finite c64_half = true.
Proof. repeat split; reflexivity. Qed.
