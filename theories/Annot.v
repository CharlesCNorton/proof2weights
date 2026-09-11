(** * Annotated binary32 arithmetic

    An annotated value pairs a binary32 value with a binary64 bound on its
    distance from a real number. Each operation below computes the binary32
    result exactly as the extracted primitive does, and alongside it a bound
    on the distance from the result of the same operation performed on the
    real numbers the operands stand for. The bound is computed with outward
    rounding (Bound64.v), so the relation [aok] is preserved by every operation
    with no hypothesis at all: wherever a condition the analysis needs fails,
    the operation returns an infinite bound, which asserts nothing.

    The rounding error of one operation is read off its result: rounding [z]
    to [w] moves it by at most [(u |w| + eta) (1 + 2u)], a consequence of
    [f32_round_mixed]. *)

From Stdlib Require Import ZArith Reals Lra Lia Bool.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Qwen Float_error Bound64.

Open Scope bool_scope.
Open Scope R_scope.

(** * Facts about binary32 results *)

Lemma overflow32_not_finite : forall (w : binary32) s,
  B2SF w = binary_overflow prec32 emax32 mode_NE s -> is_finite w = false.
Proof.
  intros w s H. rewrite <- is_finite_SF_B2SF, H. reflexivity.
Qed.

Lemma f32_plus_exact : forall x y : binary32,
  is_finite x = true -> is_finite y = true -> is_finite (f32_plus x y) = true ->
  B2R (f32_plus x y) = f32_round (B2R x + B2R y).
Proof.
  intros x y Hx Hy Hw.
  pose proof (Bplus_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y Hx Hy) as H.
  unfold f32_plus, rnd_NE in *. unfold f32_round, f32_fexp.
  destruct (Rlt_bool _ _).
  - apply H.
  - destruct H as [H _]. rewrite (overflow32_not_finite _ _ H) in Hw. discriminate.
Qed.

Lemma f32_mult_exact : forall x y : binary32,
  is_finite (f32_mult x y) = true ->
  B2R (f32_mult x y) = f32_round (B2R x * B2R y).
Proof.
  intros x y Hw.
  pose proof (Bmult_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y) as H.
  unfold f32_mult, rnd_NE in *. unfold f32_round, f32_fexp.
  destruct (Rlt_bool _ _).
  - apply H.
  - rewrite (overflow32_not_finite _ _ H) in Hw. discriminate.
Qed.

Lemma f32_div_exact : forall x y : binary32,
  B2R y <> 0 -> is_finite (f32_div x y) = true ->
  B2R (f32_div x y) = f32_round (B2R x / B2R y).
Proof.
  intros x y Hy Hw.
  pose proof (Bdiv_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y Hy) as H.
  unfold f32_div, rnd_NE in *. unfold f32_round, f32_fexp.
  destruct (Rlt_bool _ _).
  - apply H.
  - rewrite (overflow32_not_finite _ _ H) in Hw. discriminate.
Qed.

(** The distance rounding moves a value, bounded by the rounded value. *)
Lemma round_err_out : forall z,
  Rabs (f32_round z - z) <= (f32_u * Rabs (f32_round z) + f32_eta) * (1 + 2 * f32_u).
Proof.
  intros z.
  pose proof f32_u_pos as Hu. pose proof u_le_half as Hu2. pose proof f32_eta_pos as He.
  destruct (f32_round_mixed z) as [d [e (Hd & Hee & Hde & Hr)]].
  rewrite Hr.
  apply Rmult_integral in Hde. destruct Hde as [H0|H0]; subst.
  - replace (z * (1 + 0) + e - z) with e by ring.
    pose proof (Rabs_pos (z * (1 + 0) + e)). nra.
  - replace (z * (1 + d) + 0 - z) with (z * d) by ring.
    replace (z * (1 + d) + 0) with (z * (1 + d)) by ring.
    rewrite !Rabs_mult.
    assert (H1 : 1 - f32_u <= Rabs (1 + d)).
    { apply Rabs_le_inv in Hd.
      rewrite Rabs_pos_eq by lra. lra. }
    pose proof (Rabs_pos z) as Hz. pose proof (Rabs_pos d) as Hd0.
    assert (H2 : Rabs z * Rabs d <= f32_u * Rabs z) by nra.
    assert (H3 : f32_u * Rabs z <= f32_u * (Rabs z * Rabs (1 + d)) * (1 + 2 * f32_u)).
    { assert (H4 : Rabs z <= Rabs z * Rabs (1 + d) * (1 + 2 * f32_u)).
      { assert (H5 : 1 <= (1 - f32_u) * (1 + 2 * f32_u)) by nra.
        assert (H6 : (1 - f32_u) * (1 + 2 * f32_u) <= Rabs (1 + d) * (1 + 2 * f32_u)) by nra.
        nra. }
      nra. }
    nra.
Qed.

Lemma f32_le_correct : forall x y : binary32,
  is_finite x = true -> is_finite y = true -> f32_le x y = true -> B2R x <= B2R y.
Proof.
  intros x y Hx Hy H. unfold f32_le, f32_compare in H.
  rewrite (Bcompare_correct prec32 emax32 x y Hx Hy) in H.
  destruct (Rcompare_spec (B2R x) (B2R y)); try discriminate; lra.
Qed.

(** An integer below [2^24] in magnitude converts exactly. *)
Lemma f32_of_Z_exact : forall n : Z,
  (Z.abs n < 16777216)%Z ->
  is_finite (f32_of_Z n) = true /\ B2R (f32_of_Z n) = IZR n.
Proof.
  intros n Hn.
  pose proof (binary_normalize_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32
                mode_NE n 0 false) as H.
  cbv zeta in H.
  assert (HF : F2R (Float radix2 n 0) = IZR n) by (unfold F2R; simpl; ring).
  rewrite HF in H.
  assert (Hfmt : generic_format radix2 f32_fexp (IZR n)).
  { rewrite <- HF. apply generic_format_FLT.
    apply (FLT_spec _ _ _ _ (Float radix2 n 0)); [reflexivity | | ].
    - cbn [Fnum]. unfold prec32. simpl. lia.
    - cbn [Fexp]. unfold SpecFloat.emin, prec32, emax32. lia. }
  unfold f32_fexp in Hfmt.
  rewrite (round_generic radix2 _ _ _ Hfmt) in H.
  assert (Hlt : Rabs (IZR n) < bpow radix2 emax32).
  { rewrite <- abs_IZR. apply Rlt_trans with (IZR 16777216).
    - apply IZR_lt. exact Hn.
    - unfold emax32. simpl. apply IZR_lt. reflexivity. }
  rewrite Rlt_bool_true in H by exact Hlt.
  destruct H as [H1 [H2 _]]. unfold f32_of_Z. split; assumption.
Qed.

(** * Rounding to the nearest integer lands on an integer *)

Lemma f32_magic_exact : is_finite f32_magic = true /\ B2R f32_magic = 12582912.
Proof. apply f32_of_Z_exact. simpl. lia. Qed.

Lemma f32_round_int_integer : forall p : binary32,
  is_finite p = true -> Rabs (B2R p) < 4194304 ->
  is_finite (f32_round_int p) = true ->
  exists n : Z, B2R (f32_round_int p) = IZR n.
Proof.
  intros p Hp Hbound Hk.
  destruct f32_magic_exact as [Hmf Hm].
  apply Rabs_lt_inv in Hbound.
  set (z1 := B2R p + B2R f32_magic).
  assert (Hz1 : 8388608 < z1 < 16777216) by (unfold z1; rewrite Hm; lra).
  assert (F23 : generic_format radix2 f32_fexp 8388608).
  { replace 8388608 with (IZR 8388608) by reflexivity.
    destruct (f32_of_Z_exact 8388608 ltac:(simpl; lia)) as [_ H].
    rewrite <- H. apply generic_format_B2R. }
  assert (F24 : generic_format radix2 f32_fexp 16777216).
  { replace 16777216 with (bpow radix2 24) by reflexivity.
    apply generic_format_bpow. unfold f32_fexp, SpecFloat.fexp, SpecFloat.emin, prec32, emax32.
    lia. }
  (* the first rounding lands on an integer in [2^23, 2^24] *)
  assert (Hr1 : exists N : Z, f32_round z1 = IZR N /\ (8388608 <= N <= 16777216)%Z).
  { assert (Hc : cexp radix2 f32_fexp z1 = 0%Z).
    { unfold cexp. rewrite (mag_unique_pos radix2 z1 24).
      - unfold f32_fexp, SpecFloat.fexp, SpecFloat.emin, prec32, emax32. lia.
      - simpl. split; lra. }
    exists (Znearest (fun n => negb (Z.even n)) (scaled_mantissa radix2 f32_fexp z1)).
    assert (He : f32_round z1
                 = IZR (Znearest (fun n => negb (Z.even n)) (scaled_mantissa radix2 f32_fexp z1))).
    { unfold f32_round, round, F2R. rewrite Hc. cbn [Fnum Fexp].
      change (round_mode mode_NE) with (Znearest (fun n => negb (Z.even n))).
      change (bpow radix2 0) with 1. ring. }
    split; [exact He|].
    assert (L : 8388608 <= f32_round z1).
    { unfold f32_round.
      apply (@round_ge_generic radix2 f32_fexp f32_fexp_valid (round_mode mode_NE)
               (valid_rnd_round_mode mode_NE)); [exact F23 | lra]. }
    assert (U : f32_round z1 <= 16777216).
    { unfold f32_round.
      apply (@round_le_generic radix2 f32_fexp f32_fexp_valid (round_mode mode_NE)
               (valid_rnd_round_mode mode_NE)); [exact F24 | lra]. }
    rewrite He in L, U. split; [apply le_IZR in L | apply le_IZR in U]; lia. }
  destruct Hr1 as [N [HN HNb]].
  assert (Hs : is_finite (f32_plus p f32_magic) = true /\ B2R (f32_plus p f32_magic) = IZR N).
  { pose proof (Bplus_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE p f32_magic Hp Hmf) as H.
    fold z1 in H. change (round radix2 (SpecFloat.fexp prec32 emax32) (round_mode mode_NE) z1)
      with (f32_round z1) in H.
    rewrite HN in H.
    rewrite Rlt_bool_true in H.
    - destruct H as [H1 [H2 _]]. unfold f32_plus. split; assumption.
    - rewrite <- abs_IZR. apply Rle_lt_trans with (IZR 16777216).
      + apply IZR_le. lia.
      + unfold emax32. simpl. apply IZR_lt. reflexivity. }
  destruct Hs as [Hsf Hsv].
  exists (N - 12582912)%Z.
  unfold f32_round_int, f32_minus.
  assert (Hnm : is_finite (f32_neg f32_magic) = true /\ B2R (f32_neg f32_magic) = -12582912).
  { unfold f32_neg. rewrite is_finite_Bopp, B2R_Bopp, Hm. split; [exact Hmf | reflexivity]. }
  destruct Hnm as [Hnf Hnv].
  rewrite (f32_plus_exact _ _ Hsf Hnf Hk). rewrite Hsv, Hnv.
  destruct (f32_of_Z_exact (N - 12582912) ltac:(lia)) as [_ Hq].
  replace (IZR N + -12582912) with (IZR (N - 12582912))
    by (rewrite minus_IZR; reflexivity).
  rewrite <- Hq. unfold f32_round. apply round_generic; [auto with typeclass_instances|].
  apply generic_format_B2R.
Qed.

(** * Annotated values *)

Definition ann := (binary32 * binary64)%type.

(** [aok a r]: when the bound is finite, the value is finite and within the
    bound of [r]. *)
Definition aok (a : ann) (r : R) : Prop :=
  is_finite (snd a) = true ->
  is_finite (fst a) = true /\ Rabs (B2R (fst a) - r) <= B2R (snd a).

(** The rounding error of the operation that produced [w]. *)
Definition rt (w : binary32) : binary64 :=
  up_mult (up_plus (up_mult c64_u (b64_abs (b64_of_f32 w))) c64_eta) c64_1p2u.

Lemma rt_spec : forall w, is_finite w = true -> is_finite (rt w) = true ->
  (f32_u * Rabs (B2R w) + f32_eta) * (1 + 2 * f32_u) <= B2R (rt w).
Proof.
  intros w Hw H. unfold rt in *.
  destruct (up_mult_spec _ _ H) as (H1 & _ & Hm).
  destruct (up_plus_spec _ _ H1) as (H2 & _ & Hp).
  destruct (up_mult_spec _ _ H2) as (_ & _ & Hm2).
  destruct (b64_of_f32_spec w Hw) as [_ Hwv].
  rewrite (proj2 (b64_abs_spec (b64_of_f32 w))), Hwv, B2R_c64_u in Hm2.
  rewrite B2R_c64_eta in Hp. rewrite B2R_c64_1p2u in Hm.
  pose proof f32_u_pos. pose proof f32_eta_pos.
  apply Rle_trans
    with (B2R (up_plus (up_mult c64_u (b64_abs (b64_of_f32 w))) c64_eta) * (1 + 2 * f32_u));
    [|exact Hm].
  apply Rmult_le_compat_r; lra.
Qed.

Lemma rt_round : forall w z,
  is_finite w = true -> is_finite (rt w) = true -> B2R w = f32_round z ->
  Rabs (B2R w - z) <= B2R (rt w).
Proof.
  intros w z Hw Hr He. pose proof (round_err_out z) as H. rewrite <- He in H.
  eapply Rle_trans; [exact H | apply rt_spec; assumption].
Qed.

(** ** Constants *)

Definition ann_const (c : binary32) : ann :=
  (c, if f32_finite c then b64_zero else b64_inf).

Lemma aok_const : forall c, aok (ann_const c) (B2R c).
Proof.
  intros c. unfold aok, ann_const. cbn [fst snd].
  destruct (f32_finite c) eqn:E; intros H; [|discriminate].
  split; [exact E|]. rewrite Rminus_diag_eq by reflexivity. rewrite Rabs_R0.
  apply Rle_refl.
Qed.

(** ** Negation *)

Definition ann_neg (a : ann) : ann := (f32_neg (fst a), snd a).

Lemma aok_neg : forall a r, aok a r -> aok (ann_neg a) (- r).
Proof.
  intros [x d] r H. unfold aok, ann_neg in *. cbn [fst snd] in *. intros Hf.
  destruct (H Hf) as [Hx Hd]. unfold f32_neg. rewrite is_finite_Bopp, B2R_Bopp.
  split; [exact Hx|].
  replace (- B2R x - - r) with (- (B2R x - r)) by ring. rewrite Rabs_Ropp. exact Hd.
Qed.

(** ** Addition *)

Definition ann_plus (a b : ann) : ann :=
  let w := f32_plus (fst a) (fst b) in
  (w, if b64_finite (snd a) && b64_finite (snd b) && f32_finite w
      then up_plus (up_plus (rt w) (snd a)) (snd b) else b64_inf).

Lemma aok_plus : forall a b ra rb, aok a ra -> aok b rb -> aok (ann_plus a b) (ra + rb).
Proof.
  intros [x da] [y db] ra rb Ha Hb. unfold aok, ann_plus in *. cbv zeta in *.
  cbn [fst snd] in *.
  destruct (b64_finite da && b64_finite db && f32_finite (f32_plus x y)) eqn:E;
    [|intros H; discriminate].
  apply andb_prop in E as [E Hw]. apply andb_prop in E as [Hda Hdb].
  intros Hf.
  destruct (Ha Hda) as [Hx Hdx]. destruct (Hb Hdb) as [Hy Hdy].
  split; [exact Hw|].
  destruct (up_plus_spec _ _ Hf) as (H1 & _ & U1).
  destruct (up_plus_spec _ _ H1) as (H2 & _ & U2).
  pose proof (rt_round _ _ Hw H2 (f32_plus_exact x y Hx Hy Hw)) as R.
  replace (B2R (f32_plus x y) - (ra + rb))
    with ((B2R (f32_plus x y) - (B2R x + B2R y)) + ((B2R x - ra) + (B2R y - rb))) by ring.
  eapply Rle_trans; [apply Rabs_triang|].
  eapply Rle_trans; [apply Rplus_le_compat_l, Rabs_triang|].
  lra.
Qed.

(** ** Multiplication *)

Definition ann_mult (a b : ann) : ann :=
  let w := f32_mult (fst a) (fst b) in
  let ax := b64_abs (b64_of_f32 (fst a)) in
  let ay := b64_abs (b64_of_f32 (fst b)) in
  (w, if b64_finite (snd a) && b64_finite (snd b) && f32_finite w
      then up_plus (up_plus (up_plus (rt w) (up_mult ax (snd b))) (up_mult ay (snd a)))
                   (up_mult (snd a) (snd b))
      else b64_inf).

Lemma aok_mult : forall a b ra rb, aok a ra -> aok b rb -> aok (ann_mult a b) (ra * rb).
Proof.
  intros [x da] [y db] ra rb Ha Hb. unfold aok, ann_mult in *. cbv zeta in *.
  cbn [fst snd] in *.
  destruct (b64_finite da && b64_finite db && f32_finite (f32_mult x y)) eqn:E;
    [|intros H; discriminate].
  apply andb_prop in E as [E Hw]. apply andb_prop in E as [Hda Hdb].
  intros Hf.
  destruct (Ha Hda) as [Hx Hdx]. destruct (Hb Hdb) as [Hy Hdy].
  split; [exact Hw|].
  destruct (up_plus_spec _ _ Hf) as (H1 & H1' & U1).
  destruct (up_plus_spec _ _ H1) as (H2 & H2' & U2).
  destruct (up_plus_spec _ _ H2) as (H3 & H3' & U3).
  destruct (up_mult_spec _ _ H1') as (_ & _ & M1).
  destruct (up_mult_spec _ _ H2') as (_ & _ & M2).
  destruct (up_mult_spec _ _ H3') as (_ & _ & M3).
  pose proof (rt_round _ _ Hw H3 (f32_mult_exact x y Hw)) as R.
  destruct (b64_of_f32_spec x Hx) as [_ Hxv]. destruct (b64_of_f32_spec y Hy) as [_ Hyv].
  rewrite (proj2 (b64_abs_spec (b64_of_f32 x))), Hxv in M3.
  rewrite (proj2 (b64_abs_spec (b64_of_f32 y))), Hyv in M2.
  pose proof (Rabs_pos (B2R x - ra)). pose proof (Rabs_pos (B2R y - rb)).
  assert (Hrb : Rabs rb <= Rabs (B2R y) + B2R db).
  { replace rb with (B2R y + - (B2R y - rb)) at 1 by ring.
    eapply Rle_trans; [apply Rabs_triang|]. rewrite Rabs_Ropp. lra. }
  replace (B2R (f32_mult x y) - ra * rb)
    with ((B2R (f32_mult x y) - B2R x * B2R y)
          + (B2R x * (B2R y - rb) + (B2R x - ra) * rb)) by ring.
  eapply Rle_trans; [apply Rabs_triang|].
  eapply Rle_trans; [apply Rplus_le_compat_l, Rabs_triang|].
  rewrite !Rabs_mult.
  assert (T1 : Rabs (B2R x) * Rabs (B2R y - rb) <= Rabs (B2R x) * B2R db)
    by (apply Rmult_le_compat_l; [apply Rabs_pos | exact Hdy]).
  assert (T2 : Rabs (B2R x - ra) * Rabs rb <= B2R da * (Rabs (B2R y) + B2R db))
    by (apply Rmult_le_compat; [apply Rabs_pos | apply Rabs_pos | exact Hdx | exact Hrb]).
  lra.
Qed.

(** ** Division *)

Definition ann_div (a b : ann) : ann :=
  let w := f32_div (fst a) (fst b) in
  let ax := b64_abs (b64_of_f32 (fst a)) in
  let ay := b64_abs (b64_of_f32 (fst b)) in
  let lo := dn_plus ay (b64_neg (snd b)) in
  let den := dn_mult ay lo in
  (w, if b64_finite (snd a) && b64_finite (snd b) && f32_finite w
         && b64_finite lo && b64_lt b64_zero lo && b64_finite den && b64_lt b64_zero den
      then up_plus (rt w) (up_div (up_plus (up_mult ax (snd b)) (up_mult ay (snd a))) den)
      else b64_inf).

Lemma aok_div : forall a b ra rb, aok a ra -> aok b rb -> aok (ann_div a b) (ra / rb).
Proof.
  intros [x da] [y db] ra rb Ha Hb. unfold aok, ann_div in *. cbv zeta in *.
  cbn [fst snd] in *.
  set (lo := dn_plus (b64_abs (b64_of_f32 y)) (b64_neg db)) in *.
  set (den := dn_mult (b64_abs (b64_of_f32 y)) lo) in *.
  destruct (b64_finite da && b64_finite db && f32_finite (f32_div x y) && b64_finite lo
            && b64_lt b64_zero lo && b64_finite den && b64_lt b64_zero den) eqn:E;
    [|intros H; discriminate].
  apply andb_prop in E as [E Hden0]. apply andb_prop in E as [E Hdenf].
  apply andb_prop in E as [E Hlo0]. apply andb_prop in E as [E Hlof].
  apply andb_prop in E as [E Hw]. apply andb_prop in E as [Hda Hdb].
  intros Hf.
  destruct (Ha Hda) as [Hx Hdx]. destruct (Hb Hdb) as [Hy Hdy].
  split; [exact Hw|].
  destruct (b64_of_f32_spec x Hx) as [_ Hxv]. destruct (b64_of_f32_spec y Hy) as [_ Hyv].
  destruct (dn_plus_spec _ _ Hlof) as (_ & _ & L1).
  change (dn_plus (b64_abs (b64_of_f32 y)) (b64_neg db)) with lo in L1.
  rewrite (proj2 (b64_abs_spec (b64_of_f32 y))), Hyv, (proj2 (b64_neg_spec db)) in L1.
  apply (b64_lt_spec b64_zero lo eq_refl Hlof) in Hlo0.
  destruct (dn_mult_spec _ _ Hdenf) as (_ & _ & L2).
  change (dn_mult (b64_abs (b64_of_f32 y)) lo) with den in L2.
  rewrite (proj2 (b64_abs_spec (b64_of_f32 y))), Hyv in L2.
  apply (b64_lt_spec b64_zero den eq_refl Hdenf) in Hden0.
  change (B2R b64_zero) with 0 in Hlo0, Hden0.
  pose proof (Rabs_pos (B2R x - ra)) as Pda. pose proof (Rabs_pos (B2R y - rb)) as Pdb.
  assert (Hy0 : B2R y <> 0).
  { intro Hz. rewrite Hz, Rabs_R0 in L1. lra. }
  assert (Hrb : Rabs (B2R y) - B2R db <= Rabs rb).
  { assert (Ht : Rabs (B2R y) <= Rabs rb + Rabs (B2R y - rb)).
    { replace (B2R y) with (rb + (B2R y - rb)) at 1 by ring. apply Rabs_triang. }
    lra. }
  assert (Hrb0 : rb <> 0).
  { intro Hz. rewrite Hz, Rabs_R0 in Hrb. lra. }
  destruct (up_plus_spec _ _ Hf) as (H1 & H2 & U1).
  pose proof (rt_round _ _ Hw H1 (f32_div_exact x y Hy0 Hw)) as R.
  assert (Hden_ne : B2R den <> 0) by lra.
  destruct (up_div_spec _ _ Hden_ne H2) as (H3 & U2).
  destruct (up_plus_spec _ _ H3) as (H4 & H5 & U3).
  destruct (up_mult_spec _ _ H4) as (_ & _ & M1).
  destruct (up_mult_spec _ _ H5) as (_ & _ & M2).
  rewrite (proj2 (b64_abs_spec (b64_of_f32 x))), Hxv in M1.
  rewrite (proj2 (b64_abs_spec (b64_of_f32 y))), Hyv in M2.
  set (N := up_plus (up_mult (b64_abs (b64_of_f32 x)) db)
                    (up_mult (b64_abs (b64_of_f32 y)) da)) in *.
  assert (Q : Rabs (B2R x / B2R y - ra / rb) <= B2R N / B2R den).
  { replace (B2R x / B2R y - ra / rb)
      with ((B2R x * (rb - B2R y) + (B2R x - ra) * B2R y) / (B2R y * rb))
      by (field; split; assumption).
    apply abs_div_le.
    - exact Hden0.
    - rewrite Rabs_mult.
      apply Rle_trans with (Rabs (B2R y) * B2R lo); [exact L2|].
      apply Rmult_le_compat_l; [apply Rabs_pos | lra].
    - eapply Rle_trans; [apply Rabs_triang|].
      rewrite !Rabs_mult.
      rewrite (Rabs_minus_sym rb (B2R y)).
      assert (T1 : Rabs (B2R x) * Rabs (B2R y - rb) <= Rabs (B2R x) * B2R db)
        by (apply Rmult_le_compat_l; [apply Rabs_pos | exact Hdy]).
      assert (T2 : Rabs (B2R x - ra) * Rabs (B2R y) <= B2R da * Rabs (B2R y))
        by (apply Rmult_le_compat_r; [apply Rabs_pos | exact Hdx]).
      lra. }
  replace (B2R (f32_div x y) - ra / rb)
    with ((B2R (f32_div x y) - B2R x / B2R y) + (B2R x / B2R y - ra / rb)) by ring.
  eapply Rle_trans; [apply Rabs_triang|].
  lra.
Qed.

(** ** Square root *)

Definition ann_sqrt (a : ann) : ann :=
  let w := f32_sqrt (fst a) in
  let x64 := b64_of_f32 (fst a) in
  let lo := dn_plus x64 (b64_neg (snd a)) in
  let den := dn_plus (dn_sqrt x64) (dn_sqrt lo) in
  (w, if b64_finite (snd a) && f32_finite w && b64_finite lo && b64_lt b64_zero lo
         && b64_finite den && b64_lt b64_zero den
      then up_plus (rt w) (up_div (snd a) den) else b64_inf).

Lemma aok_sqrt : forall a r, aok a r -> aok (ann_sqrt a) (sqrt r).
Proof.
  intros [x d] r Ha. unfold aok, ann_sqrt in *. cbv zeta in *. cbn [fst snd] in *.
  set (lo := dn_plus (b64_of_f32 x) (b64_neg d)) in *.
  set (s := dn_sqrt (b64_of_f32 x)) in *.
  set (s2 := dn_sqrt lo) in *.
  set (den := dn_plus s s2) in *.
  destruct (b64_finite d && f32_finite (f32_sqrt x) && b64_finite lo && b64_lt b64_zero lo
            && b64_finite den && b64_lt b64_zero den) eqn:E; [|intros H; discriminate].
  apply andb_prop in E as [E Hden0]. apply andb_prop in E as [E Hdenf].
  apply andb_prop in E as [E Hlo0]. apply andb_prop in E as [E Hlof].
  apply andb_prop in E as [Hd Hw].
  intros Hf.
  destruct (Ha Hd) as [Hx Hdx].
  split; [exact Hw|].
  destruct (b64_of_f32_spec x Hx) as [_ Hxv].
  destruct (dn_plus_spec _ _ Hlof) as (_ & _ & L1).
  change (dn_plus (b64_of_f32 x) (b64_neg d)) with lo in L1.
  rewrite Hxv, (proj2 (b64_neg_spec d)) in L1.
  apply (b64_lt_spec b64_zero lo eq_refl Hlof) in Hlo0.
  destruct (dn_plus_spec _ _ Hdenf) as (Hsf & Hs2f & L0).
  change (dn_plus s s2) with den in L0.
  pose proof (dn_sqrt_spec _ Hsf) as L2.
  change (dn_sqrt (b64_of_f32 x)) with s in L2. rewrite Hxv in L2.
  pose proof (dn_sqrt_spec _ Hs2f) as L3.
  change (dn_sqrt lo) with s2 in L3.
  apply (b64_lt_spec b64_zero den eq_refl Hdenf) in Hden0.
  change (B2R b64_zero) with 0 in Hlo0, Hden0.
  pose proof (Rabs_pos (B2R x - r)) as Pd.
  pose proof Hdx as Hdx'. apply Rabs_le_inv in Hdx'.
  assert (Hxpos : 0 < B2R x) by lra.
  assert (Hrpos : 0 < r) by lra.
  assert (L4 : sqrt (B2R lo) <= sqrt r) by (apply sqrt_le_1_alt; lra).
  destruct (up_plus_spec _ _ Hf) as (H1 & H2 & U1).
  pose proof (rt_round _ (sqrt (B2R x)) Hw H1 (f32_sqrt_correct x)) as R.
  assert (Hden_ne : B2R den <> 0) by lra.
  destruct (up_div_spec _ _ Hden_ne H2) as (_ & U2).
  assert (S : Rabs (sqrt (B2R x) - sqrt r) <= B2R d / B2R den).
  { assert (Hsx : 0 < sqrt (B2R x)) by (apply sqrt_lt_R0; exact Hxpos).
    assert (Hsr : 0 <= sqrt r) by apply sqrt_pos.
    assert (Hprod : (sqrt (B2R x) - sqrt r) * (sqrt (B2R x) + sqrt r) = B2R x - r).
    { replace ((sqrt (B2R x) - sqrt r) * (sqrt (B2R x) + sqrt r))
        with (sqrt (B2R x) * sqrt (B2R x) - sqrt r * sqrt r) by ring.
      rewrite !sqrt_sqrt by lra. reflexivity. }
    assert (Hk : Rabs (sqrt (B2R x) - sqrt r) * (sqrt (B2R x) + sqrt r) <= B2R d).
    { rewrite <- (Rabs_pos_eq (sqrt (B2R x) + sqrt r)) by lra.
      rewrite <- Rabs_mult, Hprod. exact Hdx. }
    apply Rmult_le_reg_r with (B2R den); [exact Hden0|].
    unfold Rdiv. rewrite Rmult_assoc, Rinv_l by lra. rewrite Rmult_1_r.
    apply Rle_trans with (Rabs (sqrt (B2R x) - sqrt r) * (sqrt (B2R x) + sqrt r));
      [|exact Hk].
    apply Rmult_le_compat_l; [apply Rabs_pos | lra]. }
  replace (B2R (f32_sqrt x) - sqrt r)
    with ((B2R (f32_sqrt x) - sqrt (B2R x)) + (sqrt (B2R x) - sqrt r)) by ring.
  eapply Rle_trans; [apply Rabs_triang|].
  lra.
Qed.

(** ** The comparison-based maximum *)

Definition ann_max (a b : ann) : ann :=
  (f32_max2 (fst a) (fst b),
   if b64_finite (snd a) && b64_finite (snd b) then b64_max (snd a) (snd b) else b64_inf).

Lemma aok_max : forall a b ra rb, aok a ra -> aok b rb -> aok (ann_max a b) (Rmax ra rb).
Proof.
  intros [x da] [y db] ra rb Ha Hb. unfold aok, ann_max in *. cbn [fst snd] in *.
  destruct (b64_finite da && b64_finite db) eqn:E; [|intros H; discriminate].
  apply andb_prop in E as [Hda Hdb]. intros _.
  destruct (Ha Hda) as [Hx Hdx]. destruct (Hb Hdb) as [Hy Hdy].
  destruct (b64_max_spec da db Hda Hdb) as [_ Hm]. rewrite Hm.
  apply ok_max2.
  - split; [exact Hx|]. eapply Rle_trans; [exact Hdx | apply Rmax_l].
  - split; [exact Hy|]. eapply Rle_trans; [exact Hdy | apply Rmax_r].
Qed.

(** * Powers of two

    Exact when the argument is an integer of magnitude at most 127, which is
    what the reduction of the exponential produces. *)

Lemma f32_pow2_int : forall (k : binary32) (n : Z),
  is_finite k = true -> B2R k = IZR n -> (-127 <= n <= 127)%Z ->
  is_finite (f32_pow2 k) = true /\ B2R (f32_pow2 k) = bpow radix2 n.
Proof.
  intros k n Hk Hkn Hnr.
  destruct (exp_k_facts n Hnr) as (Hnf & Hnv & Hpf & Hpv).
  assert (Heq : f32_pow2 k = f32_pow2 (f32_of_Z n)).
  { destruct (Z.eq_dec n 0) as [H0|H0].
    - subst n. clear -Hk Hkn. destruct k as [s|s| |s m e Hb].
      + apply B2SF_inj. destruct s; vm_compute; reflexivity.
      + discriminate Hk.
      + discriminate Hk.
      + exfalso. revert Hkn. cbn [B2R]. apply F2R_neq_0. cbn [Fnum].
        destruct s; discriminate.
    - f_equal. apply B2R_inj.
      + apply is_finite_strict_B2R. rewrite Hkn. apply eq_IZR_contrapositive. exact H0.
      + apply is_finite_strict_B2R. rewrite Hnv. apply eq_IZR_contrapositive. exact H0.
      + rewrite Hkn, Hnv. reflexivity. }
  rewrite Heq. split; assumption.
Qed.
