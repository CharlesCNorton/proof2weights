(** * The exponential, annotated against [exp]

    [ann_exp] computes [f32_exp_approx] and, alongside it, a bound on the
    distance from the exponential of the saturated real argument. The float
    reduction selects some integer [n], and whichever integer it is,
    [exp xc = 2^n exp (xc - n log 2)]. The bound therefore needs no agreement
    between the float reduction and a real one, only that [n] is an integer of
    magnitude at most 127 and that the float reduced argument, widened by its
    bound, lies where the series is accurate. Both are checked. The distance
    from [exp xc] to [exp (Rsat r)] is charged relative to [exp xc], through an
    upper bound on [exp d - 1] at the argument's own bound [d]. *)

From Stdlib Require Import ZArith Reals Lra Lia Bool.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Qwen Float_error Series Truth Bound64 Annot.

Open Scope bool_scope.
Open Scope R_scope.

(** * Constants *)

Definition f32_c2p22 : binary32 := f32_of_Z 4194304.
Definition f32_c127 : binary32 := f32_of_Z 127.

(** [1 + 2^-20], [2^-25] and [179/512]. *)
Definition c64_1p20 : binary64 :=
  @B754_finite prec64 emax64 false 4503603922337792 (-52) eq_refl.
Definition c64_tr : binary64 :=
  @B754_finite prec64 emax64 false 4503599627370496 (-77) eq_refl.
Definition c64_r : binary64 :=
  @B754_finite prec64 emax64 false 6298002603900928 (-54) eq_refl.

Lemma B2R_c64_1p20 : B2R c64_1p20 = 1 + 1 / 1048576.
Proof.
  unfold c64_1p20, B2R, F2R. cbn [Fnum Fexp SpecFloat.cond_Zopp].
  replace (IZR 4503603922337792) with (IZR 4503599627370496 + IZR 4294967296)
    by (rewrite <- plus_IZR; reflexivity).
  change (IZR 4503599627370496) with (bpow radix2 52).
  change (IZR 4294967296) with (bpow radix2 32).
  rewrite Rmult_plus_distr_r, <- !bpow_plus.
  change (bpow radix2 (52 + -52)) with 1.
  change (bpow radix2 (32 + -52)) with (/ IZR 1048576).
  lra.
Qed.

Lemma B2R_c64_tr : B2R c64_tr = 1 / 33554432.
Proof.
  unfold c64_tr, B2R, F2R. cbn [Fnum Fexp SpecFloat.cond_Zopp].
  change (IZR 4503599627370496) with (bpow radix2 52).
  rewrite <- bpow_plus.
  change (bpow radix2 (52 + -77)) with (/ IZR 33554432).
  lra.
Qed.

Lemma B2R_c64_r : B2R c64_r = 179 / 512.
Proof.
  unfold c64_r, B2R, F2R. cbn [Fnum Fexp SpecFloat.cond_Zopp].
  replace (IZR 6298002603900928) with (IZR 179 * IZR 35184372088832)
    by (rewrite <- mult_IZR; reflexivity).
  change (IZR 35184372088832) with (bpow radix2 45).
  rewrite Rmult_assoc, <- bpow_plus.
  change (bpow radix2 (45 + -54)) with (/ IZR 512).
  lra.
Qed.

Lemma c64_exp_finite :
  is_finite c64_1p20 = true /\ is_finite c64_tr = true /\ is_finite c64_r = true.
Proof. repeat split; reflexivity. Qed.

(** * The exponential near one *)

Lemma exp_le_mono : forall x y, x <= y -> exp x <= exp y.
Proof.
  intros x y [H|H]; [left; apply exp_increasing, H | rewrite H; lra].
Qed.

Lemma exp_sub_one_quad : forall t, 0 <= t <= / 2 -> exp t - 1 <= t * (1 + 2 * t).
Proof.
  intros t [H0 H1].
  pose proof (exp_ineq1_le (- t)) as H.
  rewrite exp_Ropp in H.
  pose proof (exp_pos t) as Hp.
  assert (H2 : exp t * (1 - t) <= 1).
  { apply (Rmult_le_compat_l (exp t)) in H; [|lra].
    rewrite Rinv_r in H by (apply Rgt_not_eq; exact Hp). lra. }
  assert (H3 : 0 <= t * t * (1 - 2 * t))
    by (apply Rmult_le_pos; [apply Rmult_le_pos|]; lra).
  assert (H4 : exp t <= 1 + t * (1 + 2 * t)).
  { apply Rmult_le_reg_r with (1 - t); [lra|].
    apply Rle_trans with 1; [exact H2 | nra]. }
  lra.
Qed.

Lemma exp_abs_sub_one : forall w, Rabs (exp w - 1) <= exp (Rabs w) - 1.
Proof.
  intros w. pose proof (exp_ineq1_le w) as H1.
  destruct (Rle_or_lt 0 w) as [H|H].
  - rewrite (Rabs_pos_eq w H), Rabs_pos_eq by lra. lra.
  - pose proof (exp_ineq1_le (- w)) as H2.
    assert (H3 : exp w < 1) by (rewrite <- exp_0; apply exp_increasing, H).
    rewrite (Rabs_left w H), Rabs_left by lra. lra.
Qed.

(** Moving the argument by at most [d] moves the exponential by at most
    [exp d - 1] relative to itself. *)
Lemma exp_shift_bound : forall X Y d em,
  Rabs (X - Y) <= d -> exp d - 1 <= em -> Rabs (exp X - exp Y) <= exp X * em.
Proof.
  intros X Y d em Hd Hem.
  assert (HY : exp Y = exp X * exp (Y - X)) by (rewrite <- exp_plus; f_equal; ring).
  rewrite HY.
  replace (exp X - exp X * exp (Y - X)) with (exp X * (- (exp (Y - X) - 1))) by ring.
  rewrite Rabs_mult, Rabs_Ropp, (Rabs_pos_eq (exp X)) by (left; apply exp_pos).
  apply Rmult_le_compat_l; [left; apply exp_pos|].
  eapply Rle_trans; [apply exp_abs_sub_one|].
  assert (H1 : Rabs (Y - X) <= d) by (rewrite Rabs_minus_sym; exact Hd).
  pose proof (exp_le_mono _ _ H1). lra.
Qed.

(** * An upper bound on [exp d - 1]

    For [d] at most one half the bound is [d (1 + 2d)]. Above that, [d] is
    halved to [h] and the bound squared through
    [exp (2h) - 1 = (exp h - 1) (exp h + 1)]. *)

Fixpoint em1_up_aux (n : nat) (d : binary64) : binary64 :=
  match n with
  | O => b64_inf
  | S n' =>
      if b64_le d c64_half then up_mult d (up_plus c64_one (up_mult c64_two d))
      else let e := em1_up_aux n' (up_mult d c64_half) in up_mult e (up_plus e c64_two)
  end.

Definition em1_up (d : binary64) : binary64 := em1_up_aux 12 d.

Lemma em1_up_aux_spec : forall n d,
  is_finite (em1_up_aux n d) = true ->
  is_finite d = true /\ (0 <= B2R d -> exp (B2R d) - 1 <= B2R (em1_up_aux n d)).
Proof.
  induction n as [|n IH]; intros d H; [discriminate H|].
  cbn [em1_up_aux] in H |- *.
  destruct (b64_le d c64_half) eqn:E.
  - destruct (up_mult_spec _ _ H) as (Hd & H1 & M1).
    split; [exact Hd|]. intros H0.
    destruct (up_plus_spec _ _ H1) as (_ & H2 & P1).
    destruct (up_mult_spec _ _ H2) as (_ & _ & M2).
    rewrite B2R_c64_two in M2. rewrite B2R_c64_one in P1.
    destruct c64_finite as (_ & _ & _ & Hh).
    pose proof (b64_le_spec _ _ Hd Hh E) as Hle. rewrite B2R_c64_half in Hle.
    pose proof (exp_sub_one_quad (B2R d) (conj H0 Hle)) as Hq.
    eapply Rle_trans; [exact Hq|]. eapply Rle_trans; [|exact M1].
    apply Rmult_le_compat_l; lra.
  - destruct (up_mult_spec _ _ H) as (He & H1 & M1).
    destruct (IH _ He) as [Hh IHh].
    destruct (up_mult_spec _ _ Hh) as (Hd & _ & Mh).
    rewrite B2R_c64_half in Mh.
    split; [exact Hd|]. intros H0.
    destruct (up_plus_spec _ _ H1) as (_ & _ & P1). rewrite B2R_c64_two in P1.
    specialize (IHh ltac:(lra)).
    set (h := up_mult d c64_half) in *.
    set (e := em1_up_aux n h) in *.
    assert (Hh1 : 1 <= exp (B2R h)) by (rewrite <- exp_0; apply exp_le_mono; lra).
    assert (Hdh : exp (B2R d) <= exp (B2R h) * exp (B2R h))
      by (rewrite <- exp_plus; apply exp_le_mono; lra).
    assert (T1 : (exp (B2R h) - 1) * (exp (B2R h) + 1)
                 <= B2R e * B2R (up_plus e c64_two))
      by (apply Rmult_le_compat; lra).
    nra.
Qed.

Lemma em1_up_spec : forall d,
  is_finite (em1_up d) = true -> 0 <= B2R d -> exp (B2R d) - 1 <= B2R (em1_up d).
Proof. intros d H H0. apply (em1_up_aux_spec 12 d H), H0. Qed.

(** * The reduced argument and the polynomial *)

Definition ae_r (xc k : binary32) : ann :=
  ann_plus (ann_plus (ann_const xc) (ann_neg (ann_mult (ann_const k) (ann_const f32_ln2_hi))))
           (ann_mult (ann_const k) (ann_const f32_ln2_lo)).

Definition rn_of (xc k : binary32) : R :=
  B2R xc + - (B2R k * B2R f32_ln2_hi) + B2R k * B2R f32_ln2_lo.

Lemma ae_r_aok : forall xc k, aok (ae_r xc k) (rn_of xc k).
Proof.
  intros xc k. unfold ae_r, rn_of.
  apply aok_plus; [apply aok_plus; [apply aok_const|] | apply aok_mult; apply aok_const].
  apply aok_neg. apply aok_mult; apply aok_const.
Qed.

Definition ae_poly (r : ann) : ann :=
  let r2 := ann_mult r r in
  let r3 := ann_mult r2 r in
  let r4 := ann_mult r3 r in
  let r5 := ann_mult r4 r in
  let r6 := ann_mult r5 r in
  let r7 := ann_mult r6 r in
  ann_plus (ann_const f32_one)
    (ann_plus r
      (ann_plus (ann_div r2 (ann_const f32_two))
        (ann_plus (ann_div r3 (ann_const f32_six))
          (ann_plus (ann_div r4 (ann_const f32_twenty_four))
            (ann_plus (ann_div r5 (ann_const f32_one_twenty))
              (ann_plus (ann_div r6 (ann_const f32_seven_twenty))
                        (ann_div r7 (ann_const f32_five_thousand_forty)))))))).

Definition rpoly (r : R) : R :=
  let r2 := r * r in
  let r3 := r2 * r in
  let r4 := r3 * r in
  let r5 := r4 * r in
  let r6 := r5 * r in
  let r7 := r6 * r in
  B2R f32_one
    + (r + (r2 / B2R f32_two + (r3 / B2R f32_six + (r4 / B2R f32_twenty_four
      + (r5 / B2R f32_one_twenty + (r6 / B2R f32_seven_twenty
        + r7 / B2R f32_five_thousand_forty)))))).

Lemma ae_poly_aok : forall a ra, aok a ra -> aok (ae_poly a) (rpoly ra).
Proof.
  intros a ra H. unfold ae_poly, rpoly. cbv zeta.
  repeat first [ apply aok_plus | apply aok_mult | apply aok_div | apply aok_const
               | exact H ].
Qed.

Lemma rpoly_expand : forall r : R,
  rpoly r = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 + r^7/5040.
Proof.
  intros r. unfold rpoly. cbv zeta.
  rewrite f32_one_correct, B2R_two, B2R_six, B2R_24, B2R_120, B2R_720, B2R_5040.
  field.
Qed.

(** For any integer [n] of magnitude at most 127, the polynomial at the real
    reduced argument, scaled by [2^n], is within a relative [1.6e-8] of
    [exp xc]. *)
Lemma ae_core : forall (xc k : binary32) (n : Z),
  B2R k = IZR n -> (-127 <= n <= 127)%Z -> Rabs (rn_of xc k) <= 179 / 512 ->
  Rabs (rpoly (rn_of xc k) * bpow radix2 n - exp (B2R xc))
  <= exp (B2R xc) * (16 / 1000000000).
Proof.
  intros xc k n Hk [Hn1 Hn2] Hr.
  set (rn := rn_of xc k) in *.
  assert (Hrn : rn = B2R xc - IZR n * (355 / 512 - 14581891 / 68719476736)).
  { unfold rn, rn_of. rewrite Hk, B2R_ln2_hi, B2R_ln2_lo. ring. }
  assert (HK : Rabs (IZR n) <= 127).
  { apply Rabs_le. split.
    - assert (Hc : (Z.opp 127 <= n)%Z) by lia.
      apply IZR_le in Hc. rewrite opp_IZR in Hc. exact Hc.
    - apply IZR_le in Hn2. exact Hn2. }
  pose proof ln2_split_bound as He.
  assert (HKe : Rabs (IZR n * (355 / 512 - 14581891 / 68719476736 - ln 2))
                <= 3 / 10000000000).
  { rewrite Rabs_mult.
    apply Rle_trans with (127 * (2 / 1000000000000)).
    - apply Rmult_le_compat; try apply Rabs_pos; assumption.
    - lra. }
  apply Rabs_le_inv in HKe. apply Rabs_le_inv in Hr.
  apply (exp_split_bound (rpoly rn) rn
           (rn + IZR n * (355 / 512 - 14581891 / 68719476736 - ln 2))
           (bpow radix2 n) (exp (B2R xc))).
  - apply bpow_gt_0.
  - rewrite rpoly_expand. apply exp_series_bound. split; lra.
  - apply exp_upper_bound. split; lra.
  - replace (rn + IZR n * (355 / 512 - 14581891 / 68719476736 - ln 2) - rn)
      with (IZR n * (355 / 512 - 14581891 / 68719476736 - ln 2)) by ring.
    apply exp_near_zero. split; lra.
  - apply exp_lower_bound. split; lra.
  - rewrite bpow_exp, <- exp_plus. f_equal. rewrite Hrn. ring.
Qed.

(** The arithmetic of the final bound, on plain variables: the result [v] is
    within [dres] of [rp], which is within a relative [1.6e-8] of [Ex]; [U]
    bounds [Ex] from above; and [em] bounds the relative move from [Ex] to the
    target [eS]. *)
Lemma exp_assembly : forall v rp Ex eS em dres S U T2 T3 T4 F : R,
  Rabs (v - rp) <= dres -> Rabs (rp - Ex) <= Ex * (16 / 1000000000) -> 0 < Ex ->
  Rabs (Ex - eS) <= Ex * em -> 0 <= em -> Rabs v + dres <= S ->
  S * (1 + 1 / 1048576) <= U -> 1 / 33554432 * U <= T3 -> dres + T3 <= T2 ->
  U * em <= T4 -> T2 + T4 <= F ->
  Rabs (v - eS) <= F.
Proof.
  intros v rp Ex eS em dres S U T2 T3 T4 F Hv Hc HE Hs Hem HS HU H3 H2 H4 H1.
  assert (Hvx : Rabs (v - Ex) <= dres + Ex * (16 / 1000000000)).
  { replace (v - Ex) with ((v - rp) + (rp - Ex)) by ring.
    eapply Rle_trans; [apply Rabs_triang|]. lra. }
  assert (HEU : Ex <= U).
  { pose proof (Rle_abs v) as A1.
    pose proof (Rle_abs (Ex - v)) as A2.
    rewrite Rabs_minus_sym in A2. lra. }
  assert (T1 : Ex * (16 / 1000000000) <= 1 / 33554432 * U) by lra.
  assert (T2' : Ex * em <= U * em) by (apply Rmult_le_compat_r; assumption).
  replace (v - eS) with ((v - Ex) + (Ex - eS)) by ring.
  eapply Rle_trans; [apply Rabs_triang|]. lra.
Qed.

(** * The annotated exponential *)

Definition ann_exp (a : ann) : ann :=
  let xc := e_sat (fst a) in
  let p := f32_mult xc f32_inv_ln2 in
  let k := f32_round_int p in
  let r := ae_r xc k in
  let res := ann_mult (ae_poly r) (ann_const (f32_pow2 k)) in
  let rb := up_plus (b64_abs (b64_of_f32 (fst r))) (snd r) in
  let U := up_mult (up_plus (b64_abs (b64_of_f32 (fst res))) (snd res)) c64_1p20 in
  (fst res,
   if b64_finite (snd a) && f32_finite p && f32_lt (f32_abs p) f32_c2p22 && f32_finite k
      && f32_le (f32_abs k) f32_c127 && b64_finite rb && b64_le rb c64_r
   then up_plus (up_plus (snd res) (up_mult c64_tr U)) (up_mult U (em1_up (snd a)))
   else b64_inf).

Lemma ann_exp_fst : forall a, fst (ann_exp a) = f32_exp_approx (fst a).
Proof.
  intros a. unfold ann_exp, ae_poly, ae_r. cbv zeta.
  lazy [fst ann_plus ann_mult ann_div ann_neg ann_const].
  unfold f32_exp_approx, f32_exp_k, f32_minus, e_sat. cbv zeta.
  reflexivity.
Qed.

Lemma fst_pair : forall (A B : Type) (u : A) (w : B), fst (u, w) = u.
Proof. reflexivity. Qed.

Lemma snd_pair : forall (A B : Type) (u : A) (w : B), snd (u, w) = w.
Proof. reflexivity. Qed.

Theorem aok_exp : forall a ra, aok a ra -> aok (ann_exp a) (exp (Rsat ra)).
Proof.
  intros [x dx] ra Ha. unfold aok in *. unfold ann_exp. cbv zeta.
  rewrite ?fst_pair, ?snd_pair in *.
  set (xc := e_sat x).
  set (p := f32_mult xc f32_inv_ln2).
  set (k := f32_round_int p).
  set (r := ae_r xc k).
  set (res := ann_mult (ae_poly r) (ann_const (f32_pow2 k))).
  set (rb := up_plus (b64_abs (b64_of_f32 (fst r))) (snd r)).
  set (U := up_mult (up_plus (b64_abs (b64_of_f32 (fst res))) (snd res)) c64_1p20).
  destruct (b64_finite dx && f32_finite p && f32_lt (f32_abs p) f32_c2p22 && f32_finite k
            && f32_le (f32_abs k) f32_c127 && b64_finite rb && b64_le rb c64_r) eqn:E;
    [|intros H; discriminate].
  apply andb_prop in E as [E Hrb]. apply andb_prop in E as [E Hrbf].
  apply andb_prop in E as [E Hk127]. apply andb_prop in E as [E Hkf].
  apply andb_prop in E as [E Hp22]. apply andb_prop in E as [Hdx Hpf].
  intros Hf.
  (* the input, saturated *)
  destruct (Ha Hdx) as [Hx Hxv].
  destruct (ok_sat (B2R dx) x ra (conj Hx Hxv)) as [Hxc Hxcv].
  change (is_finite xc = true) in Hxc.
  change (Rabs (B2R xc - Rsat ra) <= B2R dx) in Hxcv.
  (* the integer the reduction selects *)
  change (is_finite p = true) in Hpf. change (is_finite k = true) in Hkf.
  destruct (f32_of_Z_exact 4194304 ltac:(simpl; lia)) as [Hc22f Hc22v].
  assert (Hpa : is_finite (f32_abs p) = true)
    by (unfold f32_abs; rewrite is_finite_Babs; exact Hpf).
  unfold f32_c2p22 in Hp22.
  apply (f32_lt_correct _ _ Hpa Hc22f) in Hp22.
  unfold f32_abs in Hp22. rewrite B2R_Babs, Hc22v in Hp22.
  destruct (f32_round_int_integer p Hpf Hp22 Hkf) as [n Hn].
  change (B2R k = IZR n) in Hn.
  destruct (f32_of_Z_exact 127 ltac:(simpl; lia)) as [Hc127f Hc127v].
  assert (Hka : is_finite (f32_abs k) = true)
    by (unfold f32_abs; rewrite is_finite_Babs; exact Hkf).
  unfold f32_c127 in Hk127.
  apply (f32_le_correct _ _ Hka Hc127f) in Hk127.
  unfold f32_abs in Hk127. rewrite B2R_Babs, Hc127v, Hn in Hk127.
  assert (Hnr : (-127 <= n <= 127)%Z).
  { apply Rabs_le_inv in Hk127. destruct Hk127 as [L1 L2].
    split; apply le_IZR; lra. }
  destruct (f32_pow2_int k n Hkf Hn Hnr) as [Hpwf Hpwv].
  (* the reduced argument lies where the series is accurate *)
  pose proof (ae_r_aok xc k) as Hr. change (aok r (rn_of xc k)) in Hr.
  destruct c64_exp_finite as (H1p20 & Htrf & Hcrf).
  change (is_finite rb = true) in Hrbf.
  apply (b64_le_spec _ _ Hrbf Hcrf) in Hrb. rewrite B2R_c64_r in Hrb.
  destruct (up_plus_spec _ _ Hrbf) as (Hra & Hdr & Urb).
  destruct (Hr Hdr) as [Hr1 Hr1v].
  destruct (b64_of_f32_spec (fst r) Hr1) as [_ Hr64].
  rewrite (proj2 (b64_abs_spec (b64_of_f32 (fst r)))), Hr64 in Urb.
  change (Rabs (B2R (fst r)) + B2R (snd r) <= B2R rb) in Urb.
  assert (Hrn : Rabs (rn_of xc k) <= 179 / 512).
  { assert (T : Rabs (rn_of xc k) <= Rabs (B2R (fst r)) + B2R (snd r)).
    { replace (rn_of xc k) with (B2R (fst r) - (B2R (fst r) - rn_of xc k)) by ring.
      eapply Rle_trans; [apply Rabs_triang|]. rewrite Rabs_Ropp. lra. }
    lra. }
  pose proof (ae_core xc k n Hn Hnr Hrn) as Hcore.
  (* the result, before the input's own error *)
  assert (Hres : aok res (rpoly (rn_of xc k) * B2R (f32_pow2 k))).
  { unfold res. apply aok_mult; [apply ae_poly_aok, Hr | apply aok_const]. }
  rewrite Hpwv in Hres.
  (* the pieces of the bound *)
  destruct (up_plus_spec _ _ Hf) as (Hf1 & Hf2 & U1).
  destruct (up_plus_spec _ _ Hf1) as (Hdres & Hf3 & U2).
  destruct (up_mult_spec _ _ Hf3) as (_ & HU & U3).
  destruct (up_mult_spec _ _ Hf2) as (_ & Hem & U4).
  rewrite B2R_c64_tr in U3.
  assert (Hdx0 : 0 <= B2R dx) by (eapply Rle_trans; [apply Rabs_pos | exact Hxv]).
  pose proof (em1_up_spec dx Hem Hdx0) as Hemv.
  destruct (up_mult_spec _ _ HU) as (HS & _ & U5).
  rewrite B2R_c64_1p20 in U5.
  change (B2R (up_plus (b64_abs (b64_of_f32 (fst res))) (snd res)) * (1 + 1 / 1048576)
          <= B2R U) in U5.
  destruct (up_plus_spec _ _ HS) as (Hva & _ & U6).
  destruct (Hres Hdres) as [Hv Hvv].
  destruct (b64_of_f32_spec (fst res) Hv) as [_ Hv64].
  rewrite (proj2 (b64_abs_spec (b64_of_f32 (fst res)))), Hv64 in U6.
  split; [exact Hv|].
  (* assembly *)
  assert (Hem0 : 0 <= B2R (em1_up dx)).
  { assert (1 <= exp (B2R dx)) by (rewrite <- exp_0; apply exp_le_mono, Hdx0). lra. }
  exact (exp_assembly _ _ _ _ _ _ _ _ _ _ _ _ Hvv Hcore (exp_pos (B2R xc))
           (exp_shift_bound _ _ _ _ Hxcv Hemv) Hem0 U6 U5 U3 U2 U4 U1).
Qed.
