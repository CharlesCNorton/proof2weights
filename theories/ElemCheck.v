(** * An interval checker for binary32 programs

    A straight-line binary32 program is evaluated over an interval of inputs.
    Each value it computes is abstracted by an interval enclosing the binary32
    value, an interval enclosing its shadow, the value the same program takes
    in exact real arithmetic, and an interval containing a bound on the
    distance between the two, together with the binary32 value itself when it
    is the same for every input. Addition, multiplication and division charge
    the rounding error of the operation at half a unit in the last place of
    the largest magnitude the exact result can take, and at least [2^-150];
    a sum is charged at most the magnitude of either addend, each being a
    binary32 value the rounding could have returned. Nothing is charged where
    the result is known exactly: Sterbenz's lemma for a subtraction of
    operands within a factor of two, a product by a power of two whose result
    stays normal, and an operation on known values, which is evaluated.
    Enclosures are CoqInterval intervals over its [Z]-based floats.
    [aeval_sound] says that the abstraction is sound for every input in the
    interval. *)

From Stdlib Require Import ZArith Reals Lra Lia List Bool.
From Flocq Require Import Core IEEE754.BinarySingleNaN Sterbenz Mult_error.
From Interval Require Import Specific_stdz Specific_ops Float_full Interval Xreal Basic.
Require Import Phases1_15_complete Float_error Annot Enclose Llama Qwen.

Import ListNotations.
Open Scope R_scope.

#[local] Existing Instance f32_fexp_valid.

(** * Intervals *)

Definition P : EF.precision := 80%Z.

Definition ctn (i : EI.type) (x : R) : Prop := contains (EI.convert i) (Xreal x).

Definition dy (m e : Z) : EF.type := Specific_ops.Float m e.
Definition ivd (m1 e1 m2 e2 : Z) : EI.type := Float.Ibnd (dy m1 e1) (dy m2 e2).
Definition ptd (m e : Z) : EI.type := ivd m e m e.

Definition dR (m e : Z) : R := IZR m * bpow radix2 e.

Lemma ctn_ivd : forall m1 e1 m2 e2 x, dR m1 e1 <= x <= dR m2 e2 -> ctn (ivd m1 e1 m2 e2) x.
Proof.
  intros m1 e1 m2 e2 x H. unfold ctn, ivd, dy. cbn [EI.convert].
  rewrite !toX_Float. cbn. unfold dR in H. lra.
Qed.

Lemma ctn_ptd : forall m e, ctn (ptd m e) (dR m e).
Proof. intros. apply ctn_ivd. lra. Qed.

Lemma ctn_add : forall a b x y, ctn a x -> ctn b y -> ctn (EI.add P a b) (x + y).
Proof. intros a b x y Ha Hb. exact (EI.add_correct P a b (Xreal x) (Xreal y) Ha Hb). Qed.

Lemma ctn_sub : forall a b x y, ctn a x -> ctn b y -> ctn (EI.sub P a b) (x - y).
Proof. intros a b x y Ha Hb. exact (EI.sub_correct P a b (Xreal x) (Xreal y) Ha Hb). Qed.

Lemma ctn_mul : forall a b x y, ctn a x -> ctn b y -> ctn (EI.mul P a b) (x * y).
Proof. intros a b x y Ha Hb. exact (EI.mul_correct P a b (Xreal x) (Xreal y) Ha Hb). Qed.

Lemma ctn_div : forall a b x y, y <> 0 -> ctn a x -> ctn b y -> ctn (EI.div P a b) (x / y).
Proof.
  intros a b x y Hy Ha Hb.
  pose proof (EI.div_correct P a b (Xreal x) (Xreal y) Ha Hb) as H.
  cbn in H. unfold Xdiv' in H. rewrite is_zero_false in H by exact Hy. exact H.
Qed.

Lemma ctn_neg : forall a x, ctn a x -> ctn (EI.neg a) (- x).
Proof. intros a x Ha. exact (EI.neg_correct a (Xreal x) Ha). Qed.

Lemma ctn_abs : forall a x, ctn a x -> ctn (EI.abs a) (Rabs x).
Proof. intros a x Ha. exact (EI.abs_correct a (Xreal x) Ha). Qed.

Lemma ctn_exp : forall a x, ctn a x -> ctn (EI.exp P a) (exp x).
Proof. intros a x Ha. exact (EI.exp_correct P a (Xreal x) Ha). Qed.

Lemma ctn_join_l : forall a b x, ctn a x -> ctn (EI.join a b) x.
Proof. intros a b x H. apply EI.join_correct. left. exact H. Qed.

Lemma ctn_join_r : forall a b x, ctn b x -> ctn (EI.join a b) x.
Proof. intros a b x H. apply EI.join_correct. right. exact H. Qed.

(** Endpoints read as dyadic numbers. *)
Definition ub (i : EI.type) : option (Z * Z) :=
  match EI.upper i with Specific_ops.Float m e => Some (m, e) | _ => None end.
Definition lb (i : EI.type) : option (Z * Z) :=
  match EI.lower i with Specific_ops.Float m e => Some (m, e) | _ => None end.

Lemma ub_spec : forall i x m e, ctn i x -> ub i = Some (m, e) -> x <= dR m e.
Proof.
  intros [|l u] x m e H E; unfold ub in E; cbn in E; [discriminate|].
  destruct u as [|mu eu]; [discriminate|]. injection E as <- <-.
  unfold ctn in H. cbn [EI.convert] in H. unfold dR.
  destruct (EI.F.valid_lb l && EI.F.valid_ub (Specific_ops.Float mu eu))%bool.
  - rewrite toX_Float in H. cbn in H. destruct (EI.F.toX l); cbn in H; tauto.
  - cbn in H. lra.
Qed.

Lemma lb_spec : forall i x m e, ctn i x -> lb i = Some (m, e) -> dR m e <= x.
Proof.
  intros [|l u] x m e H E; unfold lb in E; cbn in E; [discriminate|].
  destruct l as [|ml el]; [discriminate|]. injection E as <- <-.
  unfold ctn in H. cbn [EI.convert] in H. unfold dR.
  destruct (EI.F.valid_lb (Specific_ops.Float ml el) && EI.F.valid_ub u)%bool.
  - rewrite toX_Float in H. cbn in H. destruct (EI.F.toX u); cbn in H; tauto.
  - cbn in H. lra.
Qed.

(** Exact comparison of dyadic numbers. *)
Definition dle (m1 e1 m2 e2 : Z) : bool :=
  let e := Z.min e1 e2 in (Z.shiftl m1 (e1 - e) <=? Z.shiftl m2 (e2 - e))%Z.
Definition dlt (m1 e1 m2 e2 : Z) : bool :=
  let e := Z.min e1 e2 in (Z.shiftl m1 (e1 - e) <? Z.shiftl m2 (e2 - e))%Z.

Lemma dR_shift : forall m e e', (e' <= e)%Z -> dR m e = IZR (Z.shiftl m (e - e')) * bpow radix2 e'.
Proof.
  intros m e e' H. unfold dR. rewrite Z.shiftl_mul_pow2 by lia.
  rewrite mult_IZR, IZR_pow2 by lia. rewrite Rmult_assoc, <- bpow_plus.
  f_equal. f_equal. lia.
Qed.

Lemma dle_spec : forall m1 e1 m2 e2, dle m1 e1 m2 e2 = true -> dR m1 e1 <= dR m2 e2.
Proof.
  intros m1 e1 m2 e2 H. unfold dle in H. apply Z.leb_le in H.
  rewrite (dR_shift m1 e1 (Z.min e1 e2)), (dR_shift m2 e2 (Z.min e1 e2)) by lia.
  apply Rmult_le_compat_r; [apply bpow_ge_0 | apply IZR_le; exact H].
Qed.

Lemma dlt_spec : forall m1 e1 m2 e2, dlt m1 e1 m2 e2 = true -> dR m1 e1 < dR m2 e2.
Proof.
  intros m1 e1 m2 e2 H. unfold dlt in H. apply Z.ltb_lt in H.
  rewrite (dR_shift m1 e1 (Z.min e1 e2)), (dR_shift m2 e2 (Z.min e1 e2)) by lia.
  apply Rmult_lt_compat_r; [apply bpow_gt_0 | apply IZR_lt; exact H].
Qed.

(** An interval lies below, or above, a dyadic bound. *)
Definition ub_le (i : EI.type) (m e : Z) : bool :=
  match ub i with Some (m1, e1) => dle m1 e1 m e | None => false end.
Definition ub_lt (i : EI.type) (m e : Z) : bool :=
  match ub i with Some (m1, e1) => dlt m1 e1 m e | None => false end.
Definition lb_ge (i : EI.type) (m e : Z) : bool :=
  match lb i with Some (m1, e1) => dle m e m1 e1 | None => false end.
Definition lb_gt (i : EI.type) (m e : Z) : bool :=
  match lb i with Some (m1, e1) => dlt m e m1 e1 | None => false end.

Lemma ub_le_spec : forall i m e x, ub_le i m e = true -> ctn i x -> x <= dR m e.
Proof.
  intros i m e x H Hx. unfold ub_le in H. destruct (ub i) as [[m1 e1]|] eqn:E; [|discriminate].
  eapply Rle_trans; [apply (ub_spec i x m1 e1 Hx E) | apply dle_spec; exact H].
Qed.

Lemma ub_lt_spec : forall i m e x, ub_lt i m e = true -> ctn i x -> x < dR m e.
Proof.
  intros i m e x H Hx. unfold ub_lt in H. destruct (ub i) as [[m1 e1]|] eqn:E; [|discriminate].
  eapply Rle_lt_trans; [apply (ub_spec i x m1 e1 Hx E) | apply dlt_spec; exact H].
Qed.

Lemma lb_ge_spec : forall i m e x, lb_ge i m e = true -> ctn i x -> dR m e <= x.
Proof.
  intros i m e x H Hx. unfold lb_ge in H. destruct (lb i) as [[m1 e1]|] eqn:E; [|discriminate].
  eapply Rle_trans; [apply dle_spec; exact H | apply (lb_spec i x m1 e1 Hx E)].
Qed.

Lemma lb_gt_spec : forall i m e x, lb_gt i m e = true -> ctn i x -> dR m e < x.
Proof.
  intros i m e x H Hx. unfold lb_gt in H. destruct (lb i) as [[m1 e1]|] eqn:E; [|discriminate].
  eapply Rlt_le_trans; [apply dlt_spec; exact H | apply (lb_spec i x m1 e1 Hx E)].
Qed.

(** The interval [[-E, E]] for the upper bound [E] of an interval. *)
Definition sym (r : EI.type) : EI.type :=
  match ub r with Some (m, e) => ivd (- m) e m e | None => Float.Inan end.

Lemma ctn_sym : forall r e d, ctn r e -> Rabs d <= e -> ctn (sym r) d.
Proof.
  intros r e d Hr Hd. unfold sym. destruct (ub r) as [[m k]|] eqn:E.
  - pose proof (ub_spec r e m k Hr E) as H. apply ctn_ivd.
    unfold dR in *. rewrite opp_IZR. apply Rabs_le_inv in Hd. split; lra.
  - unfold ctn. cbn. exact I.
Qed.

(** * Binary32 constants *)

Definition iv32 (c : binary32) : EI.type := let (m, e) := b32_decomp c in ptd m e.

Lemma ctn_iv32 : forall c, ctn (iv32 c) (B2R c).
Proof.
  intros c. unfold iv32. pose proof (b32_decomp_spec c) as H.
  destruct (b32_decomp c) as [m e]. rewrite H. apply ctn_ptd.
Qed.

Definition zero_iv : EI.type := ptd 0 0.

Lemma ctn_zero : ctn zero_iv 0.
Proof. replace 0 with (dR 0 0) by (unfold dR; ring). apply ctn_ptd. Qed.

(** * Rounding *)

Lemma f32_u_eq : f32_u = dR 1 (-24).
Proof. unfold f32_u, dR, prec32. cbn. lra. Qed.

Lemma f32_eta_eq : f32_eta = dR 1 (-150).
Proof. unfold f32_eta, dR, f32_emin, emax32, prec32. cbn. lra. Qed.

Lemma round_abs_err : forall z, Rabs (f32_round z - z) <= f32_u * Rabs z + f32_eta.
Proof.
  intros z. destruct (f32_round_mixed z) as [d [e (Hd & He & _ & Hr)]].
  rewrite Hr. replace (z * (1 + d) + e - z) with (z * d + e) by ring.
  eapply Rle_trans; [apply Rabs_triang|]. rewrite Rabs_mult.
  pose proof (Rabs_pos z). pose proof f32_u_pos. nra.
Qed.

(** Rounding moves [z] by at most [max (u |z|, eta)]: [eta] below the normal
    range, [u |z|] within it. *)
Lemma round_err_max : forall z, Rabs (f32_round z - z) <= Rmax (f32_u * Rabs z) f32_eta.
Proof.
  intros z. destruct (f32_round_mixed z) as [d [e (Hd & He & Hde & Hr)]].
  rewrite Hr. apply Rmult_integral in Hde. pose proof f32_u_pos as Pu. pose proof (Rabs_pos z) as Pz.
  destruct Hde as [Z0|Z0]; subst.
  - replace (z * (1 + 0) + e - z) with e by ring. eapply Rle_trans; [exact He | apply Rmax_r].
  - replace (z * (1 + d) + 0 - z) with (z * d) by ring. rewrite Rabs_mult.
    eapply Rle_trans; [|apply Rmax_l]. rewrite Rmult_comm. apply Rmult_le_compat_r; assumption.
Qed.

(** Rounding moves [z] by at most half a unit in the last place: at most
    [2^(E-25)] when [|z| < 2^E], and at most [eta] below the normal range. *)
Lemma half_ulp_bound : forall z E, Rabs z < bpow radix2 E ->
  Rabs (f32_round z - z) <= bpow radix2 (Z.max (E - 25) (-150)).
Proof.
  intros z E H. unfold f32_round.
  change (round_mode mode_NE) with (Znearest (fun n => negb (Z.even n))).
  destruct (Req_dec z 0) as [Z0|Z0].
  - subst z. rewrite round_0 by apply valid_rnd_N.
    rewrite Rminus_0_r, Rabs_R0. apply bpow_ge_0.
  - eapply Rle_trans; [apply Ulp.error_le_half_ulp; apply f32_fexp_valid|].
    rewrite Ulp.ulp_neq_0 by exact Z0. unfold cexp.
    change f32_fexp with (FLT_exp f32_emin prec32). unfold FLT_exp.
    replace (/ 2) with (bpow radix2 (-1)) by reflexivity.
    rewrite <- bpow_plus. apply bpow_le.
    pose proof (mag_le_bpow radix2 z E Z0 H) as Hm. unfold f32_emin, emax32, prec32 in *. lia.
Qed.

Definition rnd_iv (z : EI.type) : EI.type :=
  match ub (EI.abs z) with
  | Some (m, e) => ptd 1 (Z.max (e + Z.log2 (Z.abs m) + 1 - 25) (-150))
  | None => Float.Inan
  end.

Lemma rnd_iv_spec : forall Z z, ctn Z z -> exists r, ctn (rnd_iv Z) r /\ Rabs (f32_round z - z) <= r.
Proof.
  intros Z z H. unfold rnd_iv. destruct (ub (EI.abs Z)) as [[m e]|] eqn:U.
  - set (E := (e + Z.log2 (Z.abs m) + 1)%Z).
    exists (bpow radix2 (Z.max (E - 25) (-150))). split.
    + replace (bpow radix2 (Z.max (E - 25) (-150))) with (dR 1 (Z.max (E - 25) (-150)))
        by (unfold dR; ring).
      apply ctn_ptd.
    + apply half_ulp_bound. pose proof (ub_spec _ _ _ _ (ctn_abs _ _ H) U) as L.
      eapply Rle_lt_trans; [exact L|]. unfold dR, E.
      destruct (Z.le_gt_cases m 0) as [Hm|Hm].
      * assert (IZR m * bpow radix2 e <= 0).
        { assert (IZR m <= 0) by (apply IZR_le; exact Hm). pose proof (bpow_ge_0 radix2 e). nra. }
        pose proof (bpow_gt_0 radix2 (e + Z.log2 (Z.abs m) + 1)). lra.
      * rewrite Z.abs_eq by lia.
        destruct (Z.log2_spec m Hm) as [_ L2].
        replace (e + Z.log2 m + 1)%Z with (Z.succ (Z.log2 m) + e)%Z by lia.
        rewrite bpow_plus. apply Rmult_lt_compat_r; [apply bpow_gt_0|].
        rewrite <- IZR_pow2 by (pose proof (Z.log2_nonneg m); lia). apply IZR_lt. exact L2.
  - exists (Rabs (f32_round z - z)). split; [unfold ctn; cbn; exact I | lra].
Qed.

Lemma f32_format : forall f : binary32, generic_format radix2 f32_fexp (B2R f).
Proof. intros f. apply generic_format_B2R. Qed.

(** Finite results of the three operations, from a magnitude below [2^128]. *)
Lemma f32_plus_fin : forall x y, is_finite x = true -> is_finite y = true ->
  Rabs (f32_round (B2R x + B2R y)) < bpow radix2 emax32 ->
  is_finite (f32_plus x y) = true /\ B2R (f32_plus x y) = f32_round (B2R x + B2R y).
Proof.
  intros x y Hx Hy Hb.
  pose proof (Bplus_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y Hx Hy) as H.
  unfold f32_plus, rnd_NE, f32_round, f32_fexp in *. rewrite Rlt_bool_true in H by exact Hb.
  destruct H as [H1 [H2 _]]. split; [exact H2 | exact H1].
Qed.

Lemma f32_mult_fin : forall x y, is_finite x = true -> is_finite y = true ->
  Rabs (f32_round (B2R x * B2R y)) < bpow radix2 emax32 ->
  is_finite (f32_mult x y) = true /\ B2R (f32_mult x y) = f32_round (B2R x * B2R y).
Proof.
  intros x y Hx Hy Hb.
  pose proof (Bmult_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y) as H.
  unfold f32_mult, rnd_NE, f32_round, f32_fexp in *. rewrite Rlt_bool_true in H by exact Hb.
  destruct H as [H1 [H2 _]]. rewrite H2, Hx, Hy. split; [reflexivity | exact H1].
Qed.

Lemma f32_div_fin : forall x y, is_finite x = true -> is_finite y = true -> B2R y <> 0 ->
  Rabs (f32_round (B2R x / B2R y)) < bpow radix2 emax32 ->
  is_finite (f32_div x y) = true /\ B2R (f32_div x y) = f32_round (B2R x / B2R y).
Proof.
  intros x y Hx Hy Hy0 Hb.
  pose proof (Bdiv_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y Hy0) as H.
  unfold f32_div, rnd_NE, f32_round, f32_fexp in *. rewrite Rlt_bool_true in H by exact Hb.
  destruct H as [H1 [H2 _]]. rewrite H2, Hx. split; [reflexivity | exact H1].
Qed.

(** A finite result means the rounded magnitude stayed below [2^128]. *)
Lemma f32_plus_small : forall x y, is_finite x = true -> is_finite y = true ->
  is_finite (f32_plus x y) = true -> Rabs (f32_round (B2R x + B2R y)) < bpow radix2 emax32.
Proof.
  intros x y Hx Hy Hf.
  pose proof (Bplus_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y Hx Hy) as H.
  revert H. case Rlt_bool_spec; intros L H; [exact L|].
  destruct H as [H _]. unfold f32_plus, rnd_NE in Hf.
  rewrite (overflow32_not_finite _ _ H) in Hf. discriminate.
Qed.

Lemma f32_mult_small : forall x y,
  is_finite (f32_mult x y) = true -> Rabs (f32_round (B2R x * B2R y)) < bpow radix2 emax32.
Proof.
  intros x y Hf.
  pose proof (Bmult_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y) as H.
  revert H. case Rlt_bool_spec; intros L H; [exact L|].
  unfold f32_mult, rnd_NE in Hf. rewrite (overflow32_not_finite _ _ H) in Hf. discriminate.
Qed.

Lemma f32_div_small : forall x y, B2R y <> 0 ->
  is_finite (f32_div x y) = true -> Rabs (f32_round (B2R x / B2R y)) < bpow radix2 emax32.
Proof.
  intros x y Hy Hf.
  pose proof (Bdiv_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE x y Hy) as H.
  revert H. case Rlt_bool_spec; intros L H; [exact L|].
  unfold f32_div, rnd_NE in Hf. rewrite (overflow32_not_finite _ _ H) in Hf. discriminate.
Qed.

(** * Abstract values *)

Record aval := AV { av : EI.type; ash : EI.type; ad : EI.type; ak : option binary32 }.

Definition asound (a : aval) (f : binary32) (s : R) : Prop :=
  is_finite f = true /\ ctn (av a) (B2R f) /\ ctn (ash a) s /\
  (exists e, ctn (ad a) e /\ Rabs (B2R f - s) <= e) /\
  (forall c, ak a = Some c -> is_finite c = true /\ B2R c = B2R f).

Definition a_const (c : binary32) : aval := AV (iv32 c) (iv32 c) zero_iv (Some c).

Lemma a_const_sound : forall c, is_finite c = true -> asound (a_const c) c (B2R c).
Proof.
  intros c Hc. unfold asound, a_const. cbn [av ash ad ak].
  split; [exact Hc|]. split; [apply ctn_iv32|]. split; [apply ctn_iv32|].
  split.
  - exists 0. split; [exact ctn_zero|]. unfold Rminus. rewrite Rplus_opp_r, Rabs_R0. lra.
  - intros c' E. injection E as <-. split; [exact Hc | reflexivity].
Qed.

(** [2^127] bounds a magnitude safely below overflow. *)
Definition no_ovf (i : EI.type) : bool := ub_lt (EI.abs i) 1 127.

Lemma no_ovf_spec : forall i x, no_ovf i = true -> ctn i x -> Rabs x < bpow radix2 emax32.
Proof.
  intros i x H Hx. pose proof (ub_lt_spec _ _ _ _ H (ctn_abs _ _ Hx)) as L.
  eapply Rlt_trans; [exact L|]. unfold dR, emax32. cbn. lra.
Qed.

(** Sterbenz's condition on enclosures of the two operands of an addition of
    opposite signs. *)
Definition dle_opt (a b : option (Z * Z)) (f : Z * Z -> Z * Z -> bool) : bool :=
  match a, b with Some x, Some y => f x y | _, _ => false end.

Definition sterbenz_ok (A B : EI.type) : bool :=
  let pos A := match lb A with Some (m, e) => (0 <? m)%Z | None => false end in
  let neg B := match ub B with Some (m, e) => (m <? 0)%Z | None => false end in
  let case1 A B :=
    (pos A && neg B &&
     dle_opt (lb B) (lb A) (fun '(mb, eb) '(ma, ea) => dle (- mb) (eb - 1) ma ea) &&
     dle_opt (ub A) (ub B) (fun '(ma, ea) '(mb, eb) => dle ma ea (- mb) (eb + 1)))%bool in
  (case1 A B || case1 B A)%bool.

Lemma pos_spec : forall A x, match lb A with Some (m, e) => (0 <? m)%Z | None => false end = true ->
  ctn A x -> 0 < x.
Proof.
  intros A x H Hx. destruct (lb A) as [[m e]|] eqn:E; [|discriminate].
  apply Z.ltb_lt in H. pose proof (lb_spec A x m e Hx E) as L. unfold dR in L.
  assert (0 < IZR m * bpow radix2 e) by (apply Rmult_lt_0_compat; [apply IZR_lt; lia | apply bpow_gt_0]).
  lra.
Qed.

Lemma neg_spec : forall B y, match ub B with Some (m, e) => (m <? 0)%Z | None => false end = true ->
  ctn B y -> y < 0.
Proof.
  intros B y H Hy. destruct (ub B) as [[m e]|] eqn:E; [|discriminate].
  apply Z.ltb_lt in H. pose proof (ub_spec B y m e Hy E) as L. unfold dR in L.
  assert (IZR m * bpow radix2 e < 0).
  { assert (IZR m < 0) by (apply IZR_lt; lia). pose proof (bpow_gt_0 radix2 e). nra. }
  lra.
Qed.

Lemma dR_half : forall m e, dR (- m) (e - 1) = - dR m e / 2.
Proof.
  intros. unfold dR. rewrite opp_IZR. replace (e - 1)%Z with (e + -1)%Z by lia.
  rewrite bpow_plus. change (bpow radix2 (-1)) with (/ 2). field.
Qed.

Lemma dR_double : forall m e, dR (- m) (e + 1) = - dR m e * 2.
Proof.
  intros. unfold dR. rewrite opp_IZR, bpow_plus. change (bpow radix2 1) with 2. ring.
Qed.

Lemma sterbenz_case : forall A B x y, ctn A x -> ctn B y ->
  (match lb A with Some (m, e) => (0 <? m)%Z | None => false end &&
   match ub B with Some (m, e) => (m <? 0)%Z | None => false end &&
   dle_opt (lb B) (lb A) (fun '(mb, eb) '(ma, ea) => dle (- mb) (eb - 1) ma ea) &&
   dle_opt (ub A) (ub B) (fun '(ma, ea) '(mb, eb) => dle ma ea (- mb) (eb + 1)))%bool = true ->
  - y / 2 <= x <= 2 * - y.
Proof.
  intros A B x y Hx Hy H. apply andb_prop in H as [H H4]. apply andb_prop in H as [H H3].
  apply andb_prop in H as [H1 H2].
  unfold dle_opt in H3, H4.
  destruct (lb B) as [[mb eb]|] eqn:EB; [|discriminate].
  destruct (lb A) as [[ma ea]|] eqn:EA; [|discriminate].
  destruct (ub A) as [[ma' ea']|] eqn:EA'; [|discriminate].
  destruct (ub B) as [[mb' eb']|] eqn:EB'; [|discriminate].
  apply dle_spec in H3. apply dle_spec in H4.
  rewrite dR_half in H3. rewrite dR_double in H4.
  pose proof (lb_spec B y mb eb Hy EB). pose proof (lb_spec A x ma ea Hx EA).
  pose proof (ub_spec A x ma' ea' Hx EA'). pose proof (ub_spec B y mb' eb' Hy EB').
  split; lra.
Qed.

Lemma sterbenz_exact : forall A B (fa fb : binary32),
  sterbenz_ok A B = true -> ctn A (B2R fa) -> ctn B (B2R fb) ->
  f32_round (B2R fa + B2R fb) = B2R fa + B2R fb.
Proof.
  intros A B fa fb H Ha Hb. unfold f32_round. apply round_generic.
  { apply valid_rnd_round_mode. }
  assert (Mon : Monotone_exp f32_fexp).
  { change f32_fexp with (FLT_exp f32_emin prec32). apply FLT_exp_monotone. }
  unfold sterbenz_ok in H. apply orb_prop in H as [H|H].
  - pose proof (sterbenz_case A B _ _ Ha Hb H) as S.
    replace (B2R fa + B2R fb) with (B2R fa - - B2R fb) by ring.
    apply (sterbenz radix2 f32_fexp); [apply f32_format | apply generic_format_opp, f32_format | lra].
  - pose proof (sterbenz_case B A _ _ Hb Ha H) as S.
    replace (B2R fa + B2R fb) with (B2R fb - - B2R fa) by ring.
    apply (sterbenz radix2 f32_fexp); [apply f32_format | apply generic_format_opp, f32_format | lra].
Qed.

(** ** Addition *)

(** Rounding a sum moves it by no more than either addend: the other addend
    is a binary32 value, and rounding to nearest picks the closest one. *)
Lemma round_err_addend : forall fa fb : binary32,
  Rabs (f32_round (B2R fa + B2R fb) - (B2R fa + B2R fb)) <= Rabs (B2R fb).
Proof.
  intros fa fb. unfold f32_round.
  change (round_mode mode_NE) with (Znearest (fun n => negb (Z.even n))).
  destruct (round_N_pt radix2 f32_fexp (fun n => negb (Z.even n)) (B2R fa + B2R fb)) as [_ H].
  specialize (H (B2R fa) (f32_format fa)).
  replace (B2R fa - (B2R fa + B2R fb)) with (- B2R fb) in H by ring.
  rewrite Rabs_Ropp in H. exact H.
Qed.

(** The smaller of two error intervals, by upper bound. *)
Definition pick (X Y : EI.type) : EI.type :=
  match ub X, ub Y with
  | Some (m1, e1), Some (m2, e2) => if dle m1 e1 m2 e2 then X else Y
  | Some _, None => X
  | _, _ => Y
  end.

Lemma pick_spec : forall X Y (P : R -> Prop),
  (exists r, ctn X r /\ P r) -> (exists r, ctn Y r /\ P r) -> exists r, ctn (pick X Y) r /\ P r.
Proof.
  intros X Y P HX HY. unfold pick.
  destruct (ub X) as [[m1 e1]|]; destruct (ub Y) as [[m2 e2]|];
    try destruct (dle m1 e1 m2 e2); assumption.
Qed.

(** Both operands known. *)
Definition opt2 {A B : Type} (x : option A) (y : option B) : option (A * B) :=
  match x, y with Some a, Some b => Some (a, b) | _, _ => None end.

Lemma opt2_some : forall {A B : Type} (x : option A) (y : option B) a b,
  opt2 x y = Some (a, b) -> x = Some a /\ y = Some b.
Proof. intros A B [x|] [y|] a b E; cbn in E; try discriminate. injection E as <- <-. split; reflexivity. Qed.

(** A known addend [c] and a nonnegative other one: the rounded sum is at
    least [c], since [c] is a binary32 value and rounding is monotone. *)
Definition known_lb (a b : aval) : option binary32 :=
  match ak a with
  | Some c => if lb_ge (av b) 0 0 then Some c else None
  | None => match ak b with Some c => if lb_ge (av a) 0 0 then Some c else None | None => None end
  end.

Definition refine_lb (V : EI.type) (c : option binary32) : EI.type :=
  match c with
  | Some c => let (m, e) := b32_decomp c in EI.meet V (ivd m e 1 200)
  | None => V
  end.

Definition a_plus (a b : aval) : option aval :=
  let S := EI.add P (ash a) (ash b) in
  let D0 := EI.add P (ad a) (ad b) in
  match opt2 (ak a) (ak b) with
  | Some (ca, cb) =>
      let cf := f32_plus ca cb in
      if is_finite cf then
        Some (AV (iv32 cf) S
                 (EI.add P D0 (EI.abs (EI.sub P (iv32 cf) (EI.add P (iv32 ca) (iv32 cb)))))
                 (Some cf))
      else None
  | None =>
      let Z := EI.add P (av a) (av b) in
      let R := if sterbenz_ok (av a) (av b) then zero_iv
               else pick (pick (rnd_iv Z) (EI.abs (av a))) (EI.abs (av b)) in
      let V := refine_lb (EI.add P Z (sym R)) (known_lb a b) in
      if no_ovf V then Some (AV V S (EI.add P D0 R) None) else None
  end.

Lemma round_near_abs : forall z, Rabs (f32_round z) <= 2 * Rabs z + 1.
Proof.
  intros z. pose proof (round_abs_err z) as E. pose proof f32_u_pos. pose proof u_le_half.
  rewrite f32_eta_eq in E. unfold dR in E. cbn in E.
  assert (Rabs (f32_round z) <= Rabs z + Rabs (f32_round z - z)).
  { replace (f32_round z) with (z + (f32_round z - z)) at 1 by ring. apply Rabs_triang. }
  pose proof (Rabs_pos z). nra.
Qed.

Lemma refine_lb_sound : forall a b (fa fb : binary32) sa sb V,
  asound a fa sa -> asound b fb sb -> ctn V (f32_round (B2R fa + B2R fb)) ->
  ctn (refine_lb V (known_lb a b)) (f32_round (B2R fa + B2R fb)).
Proof.
  intros a b fa fb sa sb V (Fa & Va & _ & _ & Ka) (Fb & Vb & _ & _ & Kb) H. unfold known_lb.
  set (z := B2R fa + B2R fb).
  assert (Up : f32_round z <= bpow radix2 200).
  { pose proof (round_near_abs z) as N. apply Rabs_le_inv in N.
    apply Rle_trans with (2 * Rabs z + 1); [lra|].
    pose proof (abs_B2R_lt_emax _ _ fa) as A1. pose proof (abs_B2R_lt_emax _ _ fb) as A2.
    assert (Rabs z <= Rabs (B2R fa) + Rabs (B2R fb)) by apply Rabs_triang.
    assert (bpow radix2 emax32 * 4 + 1 <= bpow radix2 200) by (unfold emax32; cbn; lra).
    lra. }
  assert (G : forall c : binary32, is_finite c = true -> B2R c <= z ->
            ctn (refine_lb V (Some c)) (f32_round z)).
  { intros c Fc Lc. unfold refine_lb. pose proof (b32_decomp_spec c) as Hs.
    destruct (b32_decomp c) as [m e]. cbn [fst snd] in Hs.
    apply EI.meet_correct; [exact H|]. apply ctn_ivd. unfold dR. rewrite <- Hs. split.
    - unfold f32_round. apply round_ge_generic; [apply f32_fexp_valid | apply valid_rnd_round_mode | apply f32_format | exact Lc].
    - rewrite Rmult_1_l. exact Up. }
  destruct (ak a) as [c|] eqn:KA.
  - destruct (lb_ge (av b) 0 0) eqn:L; [|exact H].
    destruct (Ka c eq_refl) as [Fc Vc]. apply G; [exact Fc|].
    pose proof (lb_ge_spec _ _ _ _ L Vb) as Lb. unfold dR in Lb. cbn in Lb. unfold z. lra.
  - destruct (ak b) as [c|] eqn:KB; [|exact H].
    destruct (lb_ge (av a) 0 0) eqn:L; [|exact H].
    destruct (Kb c eq_refl) as [Fc Vc]. apply G; [exact Fc|].
    pose proof (lb_ge_spec _ _ _ _ L Va) as La. unfold dR in La. cbn in La. unfold z. lra.
Qed.

Lemma err_plus : forall (fa fb : binary32) sa sb ea eb r,
  Rabs (B2R fa - sa) <= ea -> Rabs (B2R fb - sb) <= eb ->
  forall w, Rabs (w - (B2R fa + B2R fb)) <= r ->
  Rabs (w - (sa + sb)) <= ea + eb + r.
Proof.
  intros fa fb sa sb ea eb r Ha Hb w Hw.
  replace (w - (sa + sb)) with ((w - (B2R fa + B2R fb)) + (B2R fa - sa) + (B2R fb - sb)) by ring.
  eapply Rle_trans; [apply Rabs_triang|]. eapply Rle_trans; [apply Rplus_le_compat_r, Rabs_triang|].
  lra.
Qed.

Lemma a_plus_sound : forall a b fa fb sa sb r,
  asound a fa sa -> asound b fb sb -> a_plus a b = Some r ->
  asound r (f32_plus fa fb) (sa + sb).
Proof.
  intros a b fa fb sa sb r HA HB E.
  pose proof HA as (Fa & Va & Sa & (ea & Ea & Ha) & Ka).
  pose proof HB as (Fb & Vb & Sb & (eb & Eb & Hb) & Kb).
  unfold a_plus in E.
  destruct (opt2 (ak a) (ak b)) as [[ca cb]|] eqn:O.
  - destruct (opt2_some _ _ _ _ O) as [KA KB].
    destruct (Ka ca KA) as [Fca Vca]. destruct (Kb cb KB) as [Fcb Vcb].
    destruct (is_finite (f32_plus ca cb)) eqn:Fc; [|discriminate].
    injection E as <-.
    pose proof (f32_plus_small ca cb Fca Fcb Fc) as Sm. rewrite Vca, Vcb in Sm.
    destruct (f32_plus_fin fa fb Fa Fb Sm) as [Ff Vf].
    destruct (f32_plus_fin ca cb Fca Fcb ltac:(rewrite Vca, Vcb; exact Sm)) as [_ Vc].
    rewrite Vca, Vcb in Vc.
    assert (Heq : B2R (f32_plus fa fb) = B2R (f32_plus ca cb)) by (rewrite Vf, Vc; reflexivity).
    unfold asound. cbn [av ash ad ak]. split; [exact Ff|].
    split; [rewrite Heq; apply ctn_iv32|].
    split; [apply ctn_add; assumption|].
    split.
    + exists (ea + eb + Rabs (B2R (f32_plus ca cb) - (B2R ca + B2R cb))). split.
      * apply ctn_add; [apply ctn_add; assumption|].
        apply ctn_abs, ctn_sub; [apply ctn_iv32 | apply ctn_add; apply ctn_iv32].
      * rewrite Heq. apply err_plus with fa fb; try assumption.
        rewrite Vca, Vcb. lra.
    + intros c E. injection E as <-. split; [exact Fc | exact (eq_sym Heq)].
  - set (Z := EI.add P (av a) (av b)) in E.
    set (R := if sterbenz_ok (av a) (av b) then zero_iv
              else pick (pick (rnd_iv Z) (EI.abs (av a))) (EI.abs (av b))) in E.
    set (V := refine_lb (EI.add P Z (sym R)) (known_lb a b)) in E.
    destruct (no_ovf V) eqn:NO; [|discriminate]. injection E as <-.
    set (z := B2R fa + B2R fb).
    assert (Hz : ctn Z z) by (apply ctn_add; assumption).
    assert (HR : exists r, ctn R r /\ Rabs (f32_round z - z) <= r).
    { unfold R. destruct (sterbenz_ok (av a) (av b)) eqn:St.
      - exists 0. split; [exact ctn_zero|]. unfold z. rewrite (sterbenz_exact _ _ fa fb St Va Vb).
        unfold Rminus. rewrite Rplus_opp_r, Rabs_R0. lra.
      - apply pick_spec; [apply pick_spec|].
        + exact (rnd_iv_spec Z z Hz).
        + exists (Rabs (B2R fa)). split; [apply ctn_abs; exact Va|].
          unfold z. rewrite Rplus_comm. apply round_err_addend.
        + exists (Rabs (B2R fb)). split; [apply ctn_abs; exact Vb | apply round_err_addend]. }
    destruct HR as [rr [HRc HRb]].
    assert (Hv0 : ctn (EI.add P Z (sym R)) (f32_round z)).
    { replace (f32_round z) with (z + (f32_round z - z)) by ring.
      apply ctn_add; [exact Hz | apply (ctn_sym R rr); assumption]. }
    assert (Hv : ctn V (f32_round z)) by exact (refine_lb_sound a b fa fb sa sb _ HA HB Hv0).
    pose proof (no_ovf_spec _ _ NO Hv) as Sm.
    destruct (f32_plus_fin fa fb Fa Fb Sm) as [Ff Vf].
    unfold asound. cbn [av ash ad ak]. split; [exact Ff|]. rewrite Vf.
    split; [exact Hv|]. split; [apply ctn_add; assumption|].
    split; [|intros c E; discriminate].
    exists (ea + eb + rr). split.
    + apply ctn_add; [apply ctn_add; assumption | exact HRc].
    + apply err_plus with fa fb; assumption.
Qed.


(** ** Negation *)

Definition a_neg (a : aval) : aval :=
  AV (EI.neg (av a)) (EI.neg (ash a)) (ad a) (option_map f32_neg (ak a)).

Lemma a_neg_sound : forall a f s, asound a f s -> asound (a_neg a) (f32_neg f) (- s).
Proof.
  intros a f s (F & V & S & (e & E & H) & K). unfold asound, a_neg, f32_neg. cbn [av ash ad ak].
  rewrite is_finite_Bopp, B2R_Bopp. split; [exact F|].
  split; [apply ctn_neg; exact V|]. split; [apply ctn_neg; exact S|].
  split.
  - exists e. split; [exact E|].
    replace (- B2R f - - s) with (- (B2R f - s)) by ring. rewrite Rabs_Ropp. exact H.
  - intros c Ec. destruct (ak a) as [c0|]; cbn in Ec; [|discriminate]. injection Ec as <-.
    destruct (K c0 eq_refl) as [Fc Vc]. rewrite is_finite_Bopp, !B2R_Bopp, Vc.
    split; [exact Fc | reflexivity].
Qed.

Definition a_minus (a b : aval) : option aval := a_plus a (a_neg b).

Lemma a_minus_sound : forall a b fa fb sa sb r,
  asound a fa sa -> asound b fb sb -> a_minus a b = Some r ->
  asound r (f32_minus fa fb) (sa - sb).
Proof.
  intros a b fa fb sa sb r Ha Hb E. unfold f32_minus, Rminus.
  exact (a_plus_sound a (a_neg b) fa (f32_neg fb) sa (- sb) r Ha (a_neg_sound b fb sb Hb) E).
Qed.

(** ** Multiplication *)

(** A known binary32 power of two. *)
Definition pow2_of (c : binary32) : option Z :=
  let (m, e) := b32_decomp c in
  if ((0 <? m)%Z && (Z.shiftl 1 (Z.log2 m) =? m)%Z)%bool then Some (Z.log2 m + e)%Z else None.

Lemma pow2_of_spec : forall c n, pow2_of c = Some n -> B2R c = bpow radix2 n.
Proof.
  intros c n E. unfold pow2_of in E. pose proof (b32_decomp_spec c) as Hs.
  destruct (b32_decomp c) as [m e]. cbn [fst snd] in Hs.
  destruct ((0 <? m)%Z && (Z.shiftl 1 (Z.log2 m) =? m)%Z)%bool eqn:C; [|discriminate].
  injection E as <-. apply andb_prop in C as [C1 C2].
  apply Z.ltb_lt in C1. apply Z.eqb_eq in C2.
  assert (Hm : IZR m = bpow radix2 (Z.log2 m)).
  { rewrite <- C2 at 1. rewrite Z.shiftl_mul_pow2 by apply Z.log2_nonneg.
    rewrite Z.mul_1_l. apply IZR_pow2, Z.log2_nonneg. }
  rewrite Hs, Hm, bpow_plus. reflexivity.
Qed.

Definition exact_mul (b : aval) (Z : EI.type) : bool :=
  match ak b with
  | Some c => match pow2_of c with Some _ => lb_ge (EI.abs Z) 1 (-126) | None => false end
  | None => false
  end.

Lemma exact_mul_spec : forall b Z (fa fb : binary32) sb,
  asound b fb sb -> exact_mul b Z = true -> ctn Z (B2R fa * B2R fb) ->
  f32_round (B2R fa * B2R fb) = B2R fa * B2R fb.
Proof.
  intros b Z fa fb sb (_ & _ & _ & _ & K) H Hz. unfold exact_mul in H.
  destruct (ak b) as [c|] eqn:Kb; [|discriminate].
  destruct (pow2_of c) as [n|] eqn:Pc; [|discriminate].
  destruct (K c eq_refl) as [_ Vc]. rewrite <- Vc, (pow2_of_spec c n Pc) in *.
  pose proof (lb_ge_spec _ _ _ _ H (ctn_abs _ _ Hz)) as L.
  unfold f32_round. apply round_generic; [apply valid_rnd_round_mode|].
  change f32_fexp with (FLT_exp f32_emin prec32).
  apply mult_bpow_exact_FLT; [apply f32_format |].
  destruct (Req_dec (B2R fa) 0) as [Z0|Z0].
  - exfalso. rewrite Z0, Rmult_0_l, Rabs_R0 in L. unfold dR in L.
    pose proof (bpow_gt_0 radix2 (-126)). cbn in L. lra.
  - assert (Hm : (- 125 - n <= mag radix2 (B2R fa))%Z).
    { apply mag_ge_bpow. rewrite Rabs_mult, (Rabs_pos_eq (bpow radix2 n)) in L by apply bpow_ge_0.
      unfold dR in L. rewrite Rmult_1_l in L.
      apply Rmult_le_reg_r with (bpow radix2 n); [apply bpow_gt_0|].
      rewrite <- bpow_plus. replace (- 125 - n - 1 + n)%Z with (-126)%Z by lia. exact L. }
    unfold f32_emin, emax32, prec32 in *. lia.
Qed.

Definition a_mult (a b : aval) : option aval :=
  let S := EI.mul P (ash a) (ash b) in
  let D0 := EI.add P (EI.mul P (EI.abs (av a)) (ad b)) (EI.mul P (EI.abs (ash b)) (ad a)) in
  match opt2 (ak a) (ak b) with
  | Some (ca, cb) =>
      let cf := f32_mult ca cb in
      if is_finite cf then
        Some (AV (iv32 cf) S
                 (EI.add P D0 (EI.abs (EI.sub P (iv32 cf) (EI.mul P (iv32 ca) (iv32 cb)))))
                 (Some cf))
      else None
  | None =>
      let Z := EI.mul P (av a) (av b) in
      let R := if exact_mul b Z then zero_iv else rnd_iv Z in
      let V := EI.add P Z (sym R) in
      if no_ovf V then Some (AV V S (EI.add P D0 R) None) else None
  end.

Lemma err_mult : forall (fa fb : binary32) sa sb ea eb r,
  Rabs (B2R fa - sa) <= ea -> Rabs (B2R fb - sb) <= eb ->
  forall w, Rabs (w - B2R fa * B2R fb) <= r ->
  Rabs (w - sa * sb) <= Rabs (B2R fa) * eb + Rabs sb * ea + r.
Proof.
  intros fa fb sa sb ea eb r Ha Hb w Hw.
  replace (w - sa * sb)
    with ((w - B2R fa * B2R fb) + B2R fa * (B2R fb - sb) + sb * (B2R fa - sa)) by ring.
  eapply Rle_trans; [apply Rabs_triang|]. eapply Rle_trans; [apply Rplus_le_compat_r, Rabs_triang|].
  rewrite !Rabs_mult.
  assert (Rabs (B2R fa) * Rabs (B2R fb - sb) <= Rabs (B2R fa) * eb)
    by (apply Rmult_le_compat_l; [apply Rabs_pos | assumption]).
  assert (Rabs sb * Rabs (B2R fa - sa) <= Rabs sb * ea)
    by (apply Rmult_le_compat_l; [apply Rabs_pos | assumption]).
  lra.
Qed.

Lemma a_mult_sound : forall a b fa fb sa sb r,
  asound a fa sa -> asound b fb sb -> a_mult a b = Some r ->
  asound r (f32_mult fa fb) (sa * sb).
Proof.
  intros a b fa fb sa sb r Ha Hb E.
  pose proof Hb as Hb'.
  destruct Ha as (Fa & Va & Sa & (ea & Ea & Ha) & Ka).
  destruct Hb as (Fb & Vb & Sb & (eb & Eb & Hb) & Kb).
  assert (HD0 : ctn (EI.add P (EI.mul P (EI.abs (av a)) (ad b)) (EI.mul P (EI.abs (ash b)) (ad a)))
                    (Rabs (B2R fa) * eb + Rabs sb * ea)).
  { apply ctn_add; apply ctn_mul; try assumption; apply ctn_abs; assumption. }
  unfold a_mult in E.
  destruct (opt2 (ak a) (ak b)) as [[ca cb]|] eqn:O.
  - destruct (opt2_some _ _ _ _ O) as [KA KB].
    destruct (Ka ca KA) as [Fca Vca]. destruct (Kb cb KB) as [Fcb Vcb].
    destruct (is_finite (f32_mult ca cb)) eqn:Fc; [|discriminate].
    injection E as <-.
    pose proof (f32_mult_small ca cb Fc) as Sm. rewrite Vca, Vcb in Sm.
    destruct (f32_mult_fin fa fb Fa Fb Sm) as [Ff Vf].
    destruct (f32_mult_fin ca cb Fca Fcb ltac:(rewrite Vca, Vcb; exact Sm)) as [_ Vc].
    rewrite Vca, Vcb in Vc.
    assert (Heq : B2R (f32_mult fa fb) = B2R (f32_mult ca cb)) by (rewrite Vf, Vc; reflexivity).
    unfold asound. cbn [av ash ad ak]. split; [exact Ff|].
    split; [rewrite Heq; apply ctn_iv32|].
    split; [apply ctn_mul; assumption|].
    split.
    + exists (Rabs (B2R fa) * eb + Rabs sb * ea + Rabs (B2R (f32_mult ca cb) - B2R ca * B2R cb)).
      split.
      * apply ctn_add; [exact HD0|].
        apply ctn_abs, ctn_sub; [apply ctn_iv32 | apply ctn_mul; apply ctn_iv32].
      * rewrite Heq. apply (err_mult fa fb); try assumption. rewrite Vca, Vcb. lra.
    + intros c Ec. injection Ec as <-. split; [exact Fc | exact (eq_sym Heq)].
  - set (Z := EI.mul P (av a) (av b)) in E.
    set (R := if exact_mul b Z then zero_iv else rnd_iv Z) in E.
    destruct (no_ovf (EI.add P Z (sym R))) eqn:NO; [|discriminate]. injection E as <-.
    set (z := B2R fa * B2R fb).
    assert (Hz : ctn Z z) by (apply ctn_mul; assumption).
    assert (HR : exists r, ctn R r /\ Rabs (f32_round z - z) <= r).
    { unfold R. destruct (exact_mul b Z) eqn:Ex.
      - exists 0. split; [exact ctn_zero|]. unfold z.
        rewrite (exact_mul_spec b Z fa fb sb Hb' Ex Hz).
        unfold Rminus. rewrite Rplus_opp_r, Rabs_R0. lra.
      - exact (rnd_iv_spec Z z Hz). }
    destruct HR as [rr [HRc HRb]].
    assert (Hv : ctn (EI.add P Z (sym R)) (f32_round z)).
    { replace (f32_round z) with (z + (f32_round z - z)) by ring.
      apply ctn_add; [exact Hz | apply (ctn_sym R rr); assumption]. }
    pose proof (no_ovf_spec _ _ NO Hv) as Sm.
    destruct (f32_mult_fin fa fb Fa Fb Sm) as [Ff Vf].
    unfold asound. cbn [av ash ad ak]. split; [exact Ff|]. rewrite Vf.
    split; [exact Hv|]. split; [apply ctn_mul; assumption|].
    split; [|intros c Ec; discriminate].
    exists (Rabs (B2R fa) * eb + Rabs sb * ea + rr). split.
    + apply ctn_add; [exact HD0 | exact HRc].
    + apply (err_mult fa fb); assumption.
Qed.

(** ** Division *)

Definition nz (i : EI.type) : bool := (lb_gt i 0 0 || ub_lt i 0 0)%bool.

Lemma nz_spec : forall i x, nz i = true -> ctn i x -> x <> 0.
Proof.
  intros i x H Hx. unfold nz in H. apply orb_prop in H as [H|H].
  - pose proof (lb_gt_spec _ _ _ _ H Hx). unfold dR in *. cbn in *. lra.
  - pose proof (ub_lt_spec _ _ _ _ H Hx). unfold dR in *. cbn in *. lra.
Qed.

Definition a_div (a b : aval) : option aval :=
  if negb (nz (av b) && nz (ash b))%bool then None else
  let S := EI.div P (ash a) (ash b) in
  let D0 := EI.add P (EI.div P (ad a) (EI.abs (av b)))
              (EI.div P (EI.mul P (EI.abs (ash a)) (ad b))
                        (EI.mul P (EI.abs (av b)) (EI.abs (ash b)))) in
  match opt2 (ak a) (ak b) with
  | Some (ca, cb) =>
      let cf := f32_div ca cb in
      if is_finite cf then
        Some (AV (iv32 cf) S
                 (EI.add P D0 (EI.abs (EI.sub P (iv32 cf) (EI.div P (iv32 ca) (iv32 cb)))))
                 (Some cf))
      else None
  | None =>
      let Z := EI.div P (av a) (av b) in
      let R := rnd_iv Z in
      let V := EI.add P Z (sym R) in
      if no_ovf V then Some (AV V S (EI.add P D0 R) None) else None
  end.

Lemma err_div : forall (fa fb : binary32) sa sb ea eb r,
  B2R fb <> 0 -> sb <> 0 ->
  Rabs (B2R fa - sa) <= ea -> Rabs (B2R fb - sb) <= eb ->
  forall w, Rabs (w - B2R fa / B2R fb) <= r ->
  Rabs (w - sa / sb) <= ea / Rabs (B2R fb) + Rabs sa * eb / (Rabs (B2R fb) * Rabs sb) + r.
Proof.
  intros fa fb sa sb ea eb r Hf Hs Ha Hb w Hw.
  replace (w - sa / sb)
    with ((w - B2R fa / B2R fb) + (B2R fa - sa) / B2R fb + sa * (sb - B2R fb) / (B2R fb * sb))
    by (field; split; assumption).
  eapply Rle_trans; [apply Rabs_triang|]. eapply Rle_trans; [apply Rplus_le_compat_r, Rabs_triang|].
  assert (P1 : 0 < Rabs (B2R fb)) by (apply Rabs_pos_lt; exact Hf).
  assert (P2 : 0 < Rabs sb) by (apply Rabs_pos_lt; exact Hs).
  unfold Rdiv. rewrite !Rabs_mult, !Rabs_inv, Rabs_mult.
  rewrite <- Rabs_Ropp with (x := sb - B2R fb), Ropp_minus_distr.
  assert (Rabs (B2R fa - sa) * / Rabs (B2R fb) <= ea * / Rabs (B2R fb))
    by (apply Rmult_le_compat_r; [left; apply Rinv_0_lt_compat; exact P1 | exact Ha]).
  assert (Rabs sa * Rabs (B2R fb - sb) * / (Rabs (B2R fb) * Rabs sb)
          <= Rabs sa * eb * / (Rabs (B2R fb) * Rabs sb)).
  { apply Rmult_le_compat_r; [left; apply Rinv_0_lt_compat; nra|].
    apply Rmult_le_compat_l; [apply Rabs_pos | exact Hb]. }
  lra.
Qed.

Lemma a_div_sound : forall a b fa fb sa sb r,
  asound a fa sa -> asound b fb sb -> a_div a b = Some r ->
  asound r (f32_div fa fb) (sa / sb).
Proof.
  intros a b fa fb sa sb r Ha Hb E.
  destruct Ha as (Fa & Va & Sa & (ea & Ea & Ha) & Ka).
  destruct Hb as (Fb & Vb & Sb & (eb & Eb & Hb) & Kb).
  unfold a_div in E.
  destruct (nz (av b) && nz (ash b))%bool eqn:NZ; cbn [negb] in E; [|discriminate].
  apply andb_prop in NZ as [NZ1 NZ2].
  pose proof (nz_spec _ _ NZ1 Vb) as Hf0. pose proof (nz_spec _ _ NZ2 Sb) as Hs0.
  assert (HD0 : ctn (EI.add P (EI.div P (ad a) (EI.abs (av b)))
                  (EI.div P (EI.mul P (EI.abs (ash a)) (ad b))
                            (EI.mul P (EI.abs (av b)) (EI.abs (ash b)))))
                    (ea / Rabs (B2R fb) + Rabs sa * eb / (Rabs (B2R fb) * Rabs sb))).
  { apply ctn_add.
    - apply ctn_div; [apply Rabs_no_R0; exact Hf0 | exact Ea | apply ctn_abs; exact Vb].
    - apply ctn_div.
      + apply Rmult_integral_contrapositive; split; apply Rabs_no_R0; assumption.
      + apply ctn_mul; [apply ctn_abs; exact Sa | exact Eb].
      + apply ctn_mul; apply ctn_abs; assumption. }
  destruct (opt2 (ak a) (ak b)) as [[ca cb]|] eqn:O.
  - destruct (opt2_some _ _ _ _ O) as [KA KB].
    destruct (Ka ca KA) as [Fca Vca]. destruct (Kb cb KB) as [Fcb Vcb].
    destruct (is_finite (f32_div ca cb)) eqn:Fc; [|discriminate].
    injection E as <-.
    assert (Hc0 : B2R cb <> 0) by (rewrite Vcb; exact Hf0).
    pose proof (f32_div_small ca cb Hc0 Fc) as Sm. rewrite Vca, Vcb in Sm.
    destruct (f32_div_fin fa fb Fa Fb Hf0 Sm) as [Ff Vf].
    destruct (f32_div_fin ca cb Fca Fcb Hc0 ltac:(rewrite Vca, Vcb; exact Sm)) as [_ Vc].
    rewrite Vca, Vcb in Vc.
    assert (Heq : B2R (f32_div fa fb) = B2R (f32_div ca cb)) by (rewrite Vf, Vc; reflexivity).
    unfold asound. cbn [av ash ad ak]. split; [exact Ff|].
    split; [rewrite Heq; apply ctn_iv32|].
    split; [apply ctn_div; assumption|].
    split.
    + exists (ea / Rabs (B2R fb) + Rabs sa * eb / (Rabs (B2R fb) * Rabs sb)
              + Rabs (B2R (f32_div ca cb) - B2R ca / B2R cb)).
      split.
      * apply ctn_add; [exact HD0|].
        apply ctn_abs, ctn_sub; [apply ctn_iv32 | apply ctn_div; [exact Hc0 | apply ctn_iv32 | apply ctn_iv32]].
      * rewrite Heq. apply (err_div fa fb); try assumption. rewrite Vca, Vcb. lra.
    + intros c Ec. injection Ec as <-. split; [exact Fc | exact (eq_sym Heq)].
  - set (Z := EI.div P (av a) (av b)) in E.
    destruct (no_ovf (EI.add P Z (sym (rnd_iv Z)))) eqn:NO; [|discriminate]. injection E as <-.
    set (z := B2R fa / B2R fb).
    assert (Hz : ctn Z z) by (apply ctn_div; assumption).
    destruct (rnd_iv_spec Z z Hz) as [rr [HRc HRb]].
    assert (Hv : ctn (EI.add P Z (sym (rnd_iv Z))) (f32_round z)).
    { replace (f32_round z) with (z + (f32_round z - z)) by ring.
      apply ctn_add; [exact Hz | apply (ctn_sym _ _ _ HRc HRb)]. }
    pose proof (no_ovf_spec _ _ NO Hv) as Sm.
    destruct (f32_div_fin fa fb Fa Fb Hf0 Sm) as [Ff Vf].
    unfold asound. cbn [av ash ad ak]. split; [exact Ff|]. rewrite Vf.
    split; [exact Hv|]. split; [apply ctn_div; assumption|].
    split; [|intros c Ec; discriminate].
    exists (ea / Rabs (B2R fb) + Rabs sa * eb / (Rabs (B2R fb) * Rabs sb) + rr).
    split.
    + apply ctn_add; [exact HD0 | exact HRc].
    + apply (err_div fa fb); assumption.
Qed.

(** ** Absolute value and the positive part *)

Definition a_abs (a : aval) : aval :=
  AV (EI.abs (av a)) (EI.abs (ash a)) (ad a) (option_map f32_abs (ak a)).

Lemma a_abs_sound : forall a f s, asound a f s -> asound (a_abs a) (f32_abs f) (Rabs s).
Proof.
  intros a f s (F & V & S & (e & E & H) & K). unfold asound, a_abs, f32_abs. cbn [av ash ad ak].
  rewrite is_finite_Babs, B2R_Babs. split; [exact F|].
  split; [apply ctn_abs; exact V|]. split; [apply ctn_abs; exact S|].
  split.
  - exists e. split; [exact E|]. eapply Rle_trans; [apply Rabs_triang_inv2 | exact H].
  - intros c Ec. destruct (ak a) as [c0|]; cbn in Ec; [|discriminate]. injection Ec as <-.
    destruct (K c0 eq_refl) as [Fc Vc]. rewrite is_finite_Babs, !B2R_Babs, Vc.
    split; [exact Fc | reflexivity].
Qed.

Lemma f32_max2_zero : forall f : binary32, is_finite f = true ->
  is_finite (f32_max2 f f32_zero) = true /\ B2R (f32_max2 f f32_zero) = Rmax (B2R f) 0.
Proof.
  intros f F. unfold f32_max2, f32_lt, f32_compare.
  rewrite (Bcompare_correct prec32 emax32 f f32_zero F eq_refl).
  change (B2R f32_zero) with 0.
  destruct (Rcompare_spec (B2R f) 0) as [L|L|L].
  - split; [reflexivity|]. rewrite Rmax_right by lra. reflexivity.
  - split; [exact F|]. rewrite Rmax_left by lra. reflexivity.
  - split; [exact F|]. rewrite Rmax_left by lra. reflexivity.
Qed.

Lemma Rmax0_lip : forall a b, Rabs (Rmax a 0 - Rmax b 0) <= Rabs (a - b).
Proof.
  intros a b. unfold Rmax. destruct (Rle_dec a 0), (Rle_dec b 0);
    unfold Rabs; repeat destruct Rcase_abs; lra.
Qed.

(** The positive part: [0] when the enclosure is nonpositive, the enclosure
    itself when it is nonnegative, and the hull of the two otherwise. *)
Definition max0_iv (A : EI.type) : EI.type :=
  if ub_le A 0 0 then zero_iv else if lb_ge A 0 0 then A else EI.join A zero_iv.

Lemma ctn_max0 : forall A x, ctn A x -> ctn (max0_iv A) (Rmax x 0).
Proof.
  intros A x H. unfold max0_iv.
  destruct (ub_le A 0 0) eqn:U.
  - pose proof (ub_le_spec _ _ _ _ U H) as L. unfold dR in L. cbn in L.
    rewrite Rmax_right by lra. exact ctn_zero.
  - destruct (lb_ge A 0 0) eqn:Lo.
    + pose proof (lb_ge_spec _ _ _ _ Lo H) as L. unfold dR in L. cbn in L.
      rewrite Rmax_left by lra. exact H.
    + unfold Rmax. destruct (Rle_dec x 0).
      * apply ctn_join_r. exact ctn_zero.
      * apply ctn_join_l. exact H.
Qed.

Definition a_max0 (a : aval) : aval :=
  AV (max0_iv (av a)) (max0_iv (ash a)) (ad a)
     (option_map (fun c => f32_max2 c f32_zero) (ak a)).

Lemma a_max0_sound : forall a f s, asound a f s ->
  asound (a_max0 a) (f32_max2 f f32_zero) (Rmax s 0).
Proof.
  intros a f s (F & V & S & (e & E & H) & K). unfold asound, a_max0. cbn [av ash ad ak].
  destruct (f32_max2_zero f F) as [Ff Vf]. rewrite Vf. split; [exact Ff|].
  split; [apply ctn_max0; exact V|]. split; [apply ctn_max0; exact S|].
  split.
  - exists e. split; [exact E|]. eapply Rle_trans; [apply Rmax0_lip | exact H].
  - intros c Ec. destruct (ak a) as [c0|]; cbn in Ec; [|discriminate]. injection Ec as <-.
    destruct (K c0 eq_refl) as [Fc Vc]. destruct (f32_max2_zero c0 Fc) as [Fc' Vc'].
    rewrite Vc', Vc. split; [exact Fc' | reflexivity].
Qed.

(** ** Powers of two *)

Definition int_of (c : binary32) : option Z :=
  let (m, e) := b32_decomp c in
  if (0 <=? e)%Z then Some (Z.shiftl m e)
  else let n := Z.shiftr m (- e) in if (Z.shiftl n (- e) =? m)%Z then Some n else None.

Lemma int_of_spec : forall c n, int_of c = Some n -> B2R c = IZR n.
Proof.
  intros c n E. unfold int_of in E. pose proof (b32_decomp_spec c) as Hs.
  destruct (b32_decomp c) as [m e]. cbn [fst snd] in Hs. rewrite Hs.
  destruct (Z.leb_spec 0 e) as [He|He].
  - injection E as <-. rewrite Z.shiftl_mul_pow2 by exact He.
    rewrite mult_IZR, IZR_pow2 by exact He. reflexivity.
  - destruct (Z.shiftl (Z.shiftr m (- e)) (- e) =? m)%Z eqn:C; [|discriminate].
    injection E as <-. apply Z.eqb_eq in C.
    rewrite <- C at 1. rewrite Z.shiftl_mul_pow2 by lia.
    rewrite mult_IZR, IZR_pow2 by lia. rewrite Rmult_assoc, <- bpow_plus.
    replace (- e + e)%Z with 0%Z by lia. cbn. ring.
Qed.

Lemma bpow_exp' : forall z : Z, bpow radix2 z = exp (IZR z * ln 2).
Proof.
  intros z. rewrite bpow_powerRZ. rewrite powerRZ_Rpower by (simpl; lra).
  unfold Rpower. reflexivity.
Qed.

Definition a_pow2 (a : aval) : option aval :=
  match ak a with
  | Some c =>
      match int_of c with
      | Some n =>
          if ((-127 <=? n)%Z && (n <=? 127)%Z && ub_le (ad a) 0 0)%bool
          then let p := f32_pow2 c in Some (AV (iv32 p) (iv32 p) zero_iv (Some p))
          else None
      | None => None
      end
  | None => None
  end.

Lemma a_pow2_sound : forall a f s r, asound a f s -> a_pow2 a = Some r ->
  asound r (f32_pow2 f) (exp (s * ln 2)).
Proof.
  intros a f s r (F & V & S & (e & E & H) & K) Ep. unfold a_pow2 in Ep.
  destruct (ak a) as [c|] eqn:KA; [|discriminate].
  destruct (int_of c) as [n|] eqn:In; [|discriminate].
  destruct ((-127 <=? n)%Z && (n <=? 127)%Z && ub_le (ad a) 0 0)%bool eqn:C; [|discriminate].
  apply andb_prop in C as [C Cz]. apply andb_prop in C as [C1 C2].
  apply Z.leb_le in C1. apply Z.leb_le in C2.
  injection Ep as <-.
  destruct (K c eq_refl) as [Fc Vc].
  pose proof (int_of_spec c n In) as Hn.
  assert (Hs : s = IZR n).
  { pose proof (ub_le_spec _ _ _ _ Cz E) as Z0. unfold dR in Z0. rewrite Rmult_0_l in Z0.
    assert (Hd : Rabs (B2R f - s) <= 0) by lra. apply Rabs_le_inv in Hd.
    rewrite <- Vc, Hn in Hd. lra. }
  destruct (f32_pow2_int f n F ltac:(rewrite <- Vc; exact Hn) ltac:(lia)) as [Ff Vf].
  destruct (f32_pow2_int c n Fc Hn ltac:(lia)) as [Fp Vp].
  unfold asound. cbn [av ash ad ak].
  assert (Hsh : exp (s * ln 2) = bpow radix2 n) by (rewrite Hs, bpow_exp'; reflexivity).
  split; [exact Ff|]. split; [rewrite Vf, <- Vp; apply ctn_iv32|].
  split; [rewrite Hsh, <- Vp; apply ctn_iv32|].
  split.
  - exists 0. split; [exact ctn_zero|]. rewrite Vf, Hsh. unfold Rminus. rewrite Rplus_opp_r, Rabs_R0. lra.
  - intros c' Ec. injection Ec as <-. split; [exact Fp | rewrite Vp, Vf; reflexivity].
Qed.

(** ** The exponential and the logarithm, from bounds proved on them *)

Definition exp_hyp (ce : R) : Prop :=
  forall v : binary32, is_finite v = true -> -88 <= B2R v <= 88 ->
  is_finite (f32_exp_approx v) = true /\
  Rabs (B2R (f32_exp_approx v) - exp (B2R v)) <= ce * exp (B2R v).

Definition log_hyp (cl : R) : Prop :=
  forall m : binary32, is_finite m = true -> 1 <= B2R m <= 16385 / 8192 ->
  is_finite (f32_log_unit m) = true /\ Rabs (B2R (f32_log_unit m) - ln (B2R m)) <= cl.

Lemma exp_lip : forall v s, Rabs (exp v - exp s) <= exp (Rmax v s) * Rabs (v - s).
Proof.
  assert (A : forall v s, s <= v -> exp v - exp s <= exp v * (v - s)).
  { intros v s Hvs. destruct (Req_dec v s) as [Eq|Ne]; [subst; lra|].
    pose proof (exp_ineq1 (s - v) ltac:(lra)) as I.
    replace (exp s) with (exp v * exp (s - v)) by (rewrite <- exp_plus; f_equal; ring).
    pose proof (exp_pos v). nra. }
  intros v s. unfold Rmax. destruct (Rle_dec v s) as [L|L].
  - rewrite Rabs_minus_sym, (Rabs_minus_sym v s).
    pose proof (A s v L). rewrite Rabs_pos_eq by (destruct (Req_dec v s) as [Ev|Nv];
      [subst; lra | pose proof (exp_increasing v s ltac:(lra)); lra]).
    rewrite Rabs_pos_eq by lra. exact H.
  - pose proof (A v s ltac:(lra)). rewrite Rabs_pos_eq.
    + rewrite Rabs_pos_eq by lra. exact H.
    + pose proof (exp_increasing s v ltac:(lra)). lra.
Qed.

Definition a_exp (ce : Z * Z) (a : aval) : option aval :=
  if (lb_ge (av a) (-88) 0 && ub_le (av a) 88 0)%bool then
    let C := ptd (fst ce) (snd ce) in
    let Ev := EI.exp P (av a) in
    Some (AV (EI.mul P Ev (EI.add P (ptd 1 0) (sym C)))
             (EI.exp P (ash a))
             (EI.add P (EI.mul P C Ev) (EI.mul P (EI.exp P (EI.join (av a) (ash a))) (ad a)))
             None)
  else None.

Lemma a_exp_sound : forall ce a f s r, exp_hyp (dR (fst ce) (snd ce)) ->
  asound a f s -> a_exp ce a = Some r -> asound r (f32_exp_approx f) (exp s).
Proof.
  intros [mc ec] a f s r Hexp (F & V & S & (e & E & H) & K) Ea. unfold a_exp in Ea.
  cbn [fst snd] in *.
  destruct (lb_ge (av a) (-88) 0 && ub_le (av a) 88 0)%bool eqn:C; [|discriminate].
  apply andb_prop in C as [C1 C2]. injection Ea as <-.
  pose proof (lb_ge_spec _ _ _ _ C1 V) as L1. pose proof (ub_le_spec _ _ _ _ C2 V) as L2.
  unfold dR in L1, L2. cbn in L1, L2. rewrite Rmult_1_r in L1, L2.
  destruct (Hexp f F ltac:(lra)) as [Ff Hf].
  set (ce := dR mc ec) in *.
  pose proof (exp_pos (B2R f)) as Pe.
  unfold asound. cbn [av ash ad ak]. split; [exact Ff|].
  split.
  - assert (Ht : ctn (EI.add P (ptd 1 0) (sym (ptd mc ec)))
                     (dR 1 0 + (B2R (f32_exp_approx f) - exp (B2R f)) / exp (B2R f))).
    { apply ctn_add; [apply ctn_ptd|]. apply (ctn_sym _ ce); [apply ctn_ptd|].
      unfold Rdiv. rewrite Rabs_mult, Rabs_inv, (Rabs_pos_eq (exp (B2R f))) by lra.
      apply Rmult_le_reg_r with (exp (B2R f)); [exact Pe|].
      rewrite Rmult_assoc, Rinv_l by lra. lra. }
    assert (E1 : dR 1 0 = 1) by (unfold dR; cbn; ring). rewrite E1 in Ht.
    pose proof (ctn_mul _ _ _ _ (ctn_exp _ _ V) Ht) as M.
    replace (exp (B2R f) * (1 + (B2R (f32_exp_approx f) - exp (B2R f)) / exp (B2R f)))
      with (B2R (f32_exp_approx f)) in M by (field; lra).
    exact M.
  - split; [apply ctn_exp; exact S|]. split; [|intros c Ec; discriminate].
    exists (ce * exp (B2R f) + exp (Rmax (B2R f) s) * e). split.
    + pose proof (ctn_mul _ _ _ _ (ctn_ptd mc ec) (ctn_exp _ _ V)) as M1.
      assert (M2 : ctn (EI.exp P (EI.join (av a) (ash a))) (exp (Rmax (B2R f) s))).
      { apply ctn_exp. unfold Rmax. destruct (Rle_dec (B2R f) s);
          [apply ctn_join_r; exact S | apply ctn_join_l; exact V]. }
      exact (ctn_add _ _ _ _ M1 (ctn_mul _ _ _ _ M2 E)).
    + replace (B2R (f32_exp_approx f) - exp s)
        with ((B2R (f32_exp_approx f) - exp (B2R f)) + (exp (B2R f) - exp s)) by ring.
      eapply Rle_trans; [apply Rabs_triang|]. apply Rplus_le_compat; [exact Hf|].
      eapply Rle_trans; [apply exp_lip|].
      apply Rmult_le_compat_l; [left; apply exp_pos | exact H].
Qed.

Lemma ctn_ln : forall a x, 0 < x -> ctn a x -> ctn (EI.ln P a) (ln x).
Proof.
  intros a x Hx Ha. pose proof (EI.ln_correct P a (Xreal x) Ha) as H.
  cbn in H. unfold Xln' in H. rewrite is_positive_true in H by exact Hx. exact H.
Qed.

Lemma ln_lip1 : forall a b, 1 <= a -> 1 <= b -> Rabs (ln a - ln b) <= Rabs (a - b).
Proof.
  assert (A : forall a b, 1 <= b <= a -> ln a - ln b <= a - b).
  { intros a b [Hb Hab]. destruct (Req_dec a b) as [Eq|Ne]; [subst; lra|].
    assert (Hr : 1 < a / b) by (apply Rmult_lt_reg_r with b; [lra|]; unfold Rdiv;
      rewrite Rmult_assoc, Rinv_l, Rmult_1_r by lra; lra).
    assert (Hl : ln a - ln b = ln (a / b)).
    { replace a with ((a / b) * b) at 1 by (field; lra). rewrite ln_mult by lra. ring. }
    rewrite Hl.
    pose proof (exp_ineq1 (a / b - 1) ltac:(lra)) as I.
    assert (ln (a / b) < a / b - 1).
    { rewrite <- (ln_exp (a / b - 1)). apply ln_increasing; [lra|]. lra. }
    assert (a / b - 1 <= a - b).
    { unfold Rdiv. apply Rmult_le_reg_r with b; [lra|].
      replace ((a * / b - 1) * b) with (a - b) by (field; lra). nra. }
    lra. }
  intros a b Ha Hb. destruct (Rle_dec b a) as [L|L].
  - assert (0 <= ln a - ln b).
    { destruct (Req_dec a b) as [Eab|Nab]; [subst; lra|].
      pose proof (ln_increasing b a ltac:(lra) ltac:(lra)). lra. }
    rewrite !Rabs_pos_eq by lra. apply A. lra.
  - assert (0 <= ln b - ln a) by (pose proof (ln_increasing a b ltac:(lra) ltac:(lra)); lra).
    rewrite (Rabs_minus_sym (ln a)), (Rabs_minus_sym a).
    rewrite !Rabs_pos_eq by lra. apply A. lra.
Qed.

Lemma ln_sub_le : forall a b, 0 < b <= a -> ln a - ln b <= (a - b) / b.
Proof.
  intros a b [Hb Hab]. destruct (Req_dec a b) as [Eq|Ne].
  - subst. replace (b - b) with 0 by ring. unfold Rdiv. lra.
  - assert (Hq : 1 < a / b).
    { apply Rmult_lt_reg_r with b; [lra|]. unfold Rdiv.
      rewrite Rmult_assoc, Rinv_l, Rmult_1_r by lra. lra. }
    assert (Hl : ln a - ln b = ln (a / b)).
    { replace a with ((a / b) * b) at 1 by (field; lra). rewrite ln_mult by lra. ring. }
    rewrite Hl.
    pose proof (exp_ineq1 (a / b - 1) ltac:(lra)) as I.
    assert (ln (a / b) < a / b - 1).
    { rewrite <- (ln_exp (a / b - 1)). apply ln_increasing; lra. }
    replace ((a - b) / b) with (a / b - 1) by (field; lra). lra.
Qed.

Lemma ln_lip_m : forall a b m, 0 < m -> m <= a -> m <= b -> Rabs (ln a - ln b) <= Rabs (a - b) / m.
Proof.
  assert (A : forall a b m, 0 < m -> m <= b <= a -> Rabs (ln a - ln b) <= Rabs (a - b) / m).
  { intros a b m Hm [Hb Hab]. pose proof (ln_sub_le a b ltac:(lra)) as H.
    assert (0 <= ln a - ln b).
    { destruct (Req_dec a b) as [E|E]; [subst; lra|].
      pose proof (ln_increasing b a ltac:(lra) ltac:(lra)). lra. }
    rewrite !Rabs_pos_eq by lra. eapply Rle_trans; [exact H|].
    unfold Rdiv. apply Rmult_le_compat_l; [lra|]. apply Rinv_le_contravar; lra. }
  intros a b m Hm Ha Hb. destruct (Rle_dec b a) as [L|L].
  - apply A; lra.
  - rewrite (Rabs_minus_sym (ln a)), (Rabs_minus_sym a). apply A; lra.
Qed.

(** The input error enters through the Lipschitz constant [1/m] of [ln] on
    [[m, oo)], with [m] the least value either enclosure admits. *)
Definition a_log (cl : Z * Z) (a : aval) : option aval :=
  if (lb_ge (av a) 1 0 && ub_le (av a) 16385 (-13) && lb_ge (ash a) 1 0)%bool then
    match lb (EI.join (av a) (ash a)) with
    | Some (ml, el) =>
        if dle 1 0 ml el then
          let C := ptd (fst cl) (snd cl) in
          Some (AV (EI.add P (EI.ln P (av a)) (sym C)) (EI.ln P (ash a))
                   (EI.add P C (EI.div P (ad a) (ptd ml el))) None)
        else None
    | None => None
    end
  else None.

Lemma a_log_sound : forall cl a f s r, log_hyp (dR (fst cl) (snd cl)) ->
  asound a f s -> a_log cl a = Some r -> asound r (f32_log_unit f) (ln s).
Proof.
  intros [mc ec] a f s r Hlog (F & V & S & (e & E & H) & K) Ea. unfold a_log in Ea.
  cbn [fst snd] in *.
  destruct (lb_ge (av a) 1 0 && ub_le (av a) 16385 (-13) && lb_ge (ash a) 1 0)%bool eqn:C;
    [|discriminate].
  apply andb_prop in C as [C C3]. apply andb_prop in C as [C1 C2].
  destruct (lb (EI.join (av a) (ash a))) as [[ml el]|] eqn:Lj; [|discriminate].
  destruct (dle 1 0 ml el) eqn:Dm; [|discriminate]. injection Ea as <-.
  pose proof (lb_ge_spec _ _ _ _ C1 V) as L1. pose proof (ub_le_spec _ _ _ _ C2 V) as L2.
  pose proof (lb_ge_spec _ _ _ _ C3 S) as L3.
  unfold dR in L1, L2, L3. cbn in L1, L2, L3. rewrite Rmult_1_r in L1, L3.
  assert (M1 : 1 <= dR ml el).
  { pose proof (dle_spec _ _ _ _ Dm) as D. unfold dR in D |- *. cbn in D. lra. }
  pose proof (lb_spec _ _ _ _ (ctn_join_l _ _ _ V) Lj) as Mf.
  pose proof (lb_spec _ _ _ _ (ctn_join_r _ _ _ S) Lj) as Ms.
  set (m := dR ml el) in *.
  destruct (Hlog f F ltac:(lra)) as [Ff Hf].
  set (cl := dR mc ec) in *.
  unfold asound. cbn [av ash ad ak]. split; [exact Ff|].
  split.
  - replace (B2R (f32_log_unit f)) with (ln (B2R f) + (B2R (f32_log_unit f) - ln (B2R f))) by ring.
    apply ctn_add; [apply ctn_ln; [lra | exact V]|].
    apply (ctn_sym _ cl); [apply ctn_ptd | exact Hf].
  - split; [apply ctn_ln; [lra | exact S]|]. split; [|intros c Ec; discriminate].
    exists (cl + e / m).
    split; [exact (ctn_add _ _ _ _ (ctn_ptd mc ec)
                     (ctn_div _ _ _ _ (Rgt_not_eq _ _ (Rlt_le_trans 0 1 m Rlt_0_1 M1)) E (ctn_ptd ml el)))|].
    replace (B2R (f32_log_unit f) - ln s)
      with ((B2R (f32_log_unit f) - ln (B2R f)) + (ln (B2R f) - ln s)) by ring.
    eapply Rle_trans; [apply Rabs_triang|]. apply Rplus_le_compat; [exact Hf|].
    eapply Rle_trans; [apply (ln_lip_m _ _ m); lra|].
    unfold Rdiv. apply Rmult_le_compat_r; [left; apply Rinv_0_lt_compat; lra | exact H].
Qed.

(** * Programs

    A program is a list of instructions, each naming earlier values by their
    position; the environment starts with the inputs. *)

Inductive instr :=
| IC (c : binary32)
| IAdd (i j : nat) | ISub (i j : nat) | IMul (i j : nat) | IDiv (i j : nat)
| INeg (i : nat) | IAbs (i : nat) | IMax0 (i : nat)
| IPow2 (i : nat) | IExp (i : nat) | ILog (i : nat).

Definition fnth (env : list binary32) (i : nat) : binary32 := nth i env f32_zero.
Definition snth (env : list R) (i : nat) : R := nth i env 0.
Definition anth (env : list aval) (i : nat) : aval := nth i env (a_const f32_zero).

Definition fstep (env : list binary32) (ins : instr) : binary32 :=
  match ins with
  | IC c => c
  | IAdd i j => f32_plus (fnth env i) (fnth env j)
  | ISub i j => f32_minus (fnth env i) (fnth env j)
  | IMul i j => f32_mult (fnth env i) (fnth env j)
  | IDiv i j => f32_div (fnth env i) (fnth env j)
  | INeg i => f32_neg (fnth env i)
  | IAbs i => f32_abs (fnth env i)
  | IMax0 i => f32_max2 (fnth env i) f32_zero
  | IPow2 i => f32_pow2 (fnth env i)
  | IExp i => f32_exp_approx (fnth env i)
  | ILog i => f32_log_unit (fnth env i)
  end.

Definition sstep (env : list R) (ins : instr) : R :=
  match ins with
  | IC c => B2R c
  | IAdd i j => snth env i + snth env j
  | ISub i j => snth env i - snth env j
  | IMul i j => snth env i * snth env j
  | IDiv i j => snth env i / snth env j
  | INeg i => - snth env i
  | IAbs i => Rabs (snth env i)
  | IMax0 i => Rmax (snth env i) 0
  | IPow2 i => exp (snth env i * ln 2)
  | IExp i => exp (snth env i)
  | ILog i => ln (snth env i)
  end.

Definition astep (ce cl : Z * Z) (env : list aval) (ins : instr) : option aval :=
  match ins with
  | IC c => if is_finite c then Some (a_const c) else None
  | IAdd i j => a_plus (anth env i) (anth env j)
  | ISub i j => a_minus (anth env i) (anth env j)
  | IMul i j => a_mult (anth env i) (anth env j)
  | IDiv i j => a_div (anth env i) (anth env j)
  | INeg i => Some (a_neg (anth env i))
  | IAbs i => Some (a_abs (anth env i))
  | IMax0 i => Some (a_max0 (anth env i))
  | IPow2 i => a_pow2 (anth env i)
  | IExp i => a_exp ce (anth env i)
  | ILog i => a_log cl (anth env i)
  end.

Fixpoint feval (p : list instr) (env : list binary32) : list binary32 :=
  match p with [] => env | ins :: p' => feval p' (env ++ [fstep env ins]) end.

Fixpoint seval (p : list instr) (env : list R) : list R :=
  match p with [] => env | ins :: p' => seval p' (env ++ [sstep env ins]) end.

Fixpoint aeval (ce cl : Z * Z) (p : list instr) (env : list aval) : option (list aval) :=
  match p with
  | [] => Some env
  | ins :: p' => match astep ce cl env ins with
                 | Some a => aeval ce cl p' (env ++ [a])
                 | None => None
                 end
  end.

Inductive F3 : list aval -> list binary32 -> list R -> Prop :=
| F3nil : F3 [] [] []
| F3cons : forall a f s A Fs Ss, asound a f s -> F3 A Fs Ss -> F3 (a :: A) (f :: Fs) (s :: Ss).

Lemma F3_nth : forall A Fs Ss i, F3 A Fs Ss -> asound (anth A i) (fnth Fs i) (snth Ss i).
Proof.
  intros A Fs Ss i H. revert i. induction H as [|a f s A Fs Ss Ha H IH]; intros i.
  - unfold anth, fnth, snth. destruct i; apply a_const_sound; reflexivity.
  - destruct i; [exact Ha | apply IH].
Qed.

Lemma F3_app : forall A Fs Ss a f s, F3 A Fs Ss -> asound a f s ->
  F3 (A ++ [a]) (Fs ++ [f]) (Ss ++ [s]).
Proof.
  intros A Fs Ss a f s H Ha. induction H as [|a0 f0 s0 A Fs Ss H0 H IH]; cbn.
  - constructor; [exact Ha | constructor].
  - constructor; [exact H0 | exact IH].
Qed.

Section Sound.

Variables ce cl : Z * Z.
Hypothesis Hexp : exp_hyp (dR (fst ce) (snd ce)).
Hypothesis Hlog : log_hyp (dR (fst cl) (snd cl)).

Lemma astep_sound : forall A Fs Ss ins a, F3 A Fs Ss -> astep ce cl A ins = Some a ->
  asound a (fstep Fs ins) (sstep Ss ins).
Proof.
  intros A Fs Ss ins a H E.
  destruct ins as [c|i j|i j|i j|i j|i|i|i|i|i|i]; cbn [astep fstep sstep] in *.
  - destruct (is_finite c) eqn:Fc; [|discriminate]. injection E as <-. apply a_const_sound; exact Fc.
  - exact (a_plus_sound _ _ _ _ _ _ _ (F3_nth _ _ _ i H) (F3_nth _ _ _ j H) E).
  - exact (a_minus_sound _ _ _ _ _ _ _ (F3_nth _ _ _ i H) (F3_nth _ _ _ j H) E).
  - exact (a_mult_sound _ _ _ _ _ _ _ (F3_nth _ _ _ i H) (F3_nth _ _ _ j H) E).
  - exact (a_div_sound _ _ _ _ _ _ _ (F3_nth _ _ _ i H) (F3_nth _ _ _ j H) E).
  - injection E as <-. exact (a_neg_sound _ _ _ (F3_nth _ _ _ i H)).
  - injection E as <-. exact (a_abs_sound _ _ _ (F3_nth _ _ _ i H)).
  - injection E as <-. exact (a_max0_sound _ _ _ (F3_nth _ _ _ i H)).
  - exact (a_pow2_sound _ _ _ _ (F3_nth _ _ _ i H) E).
  - exact (a_exp_sound ce _ _ _ _ Hexp (F3_nth _ _ _ i H) E).
  - exact (a_log_sound cl _ _ _ _ Hlog (F3_nth _ _ _ i H) E).
Qed.

Theorem aeval_sound : forall p A Fs Ss A', F3 A Fs Ss -> aeval ce cl p A = Some A' ->
  F3 A' (feval p Fs) (seval p Ss).
Proof.
  induction p as [|ins p IH]; intros A Fs Ss A' H E; cbn in E |- *.
  - injection E as <-. exact H.
  - destruct (astep ce cl A ins) as [a|] eqn:Ea; [|discriminate].
    apply (IH (A ++ [a])); [|exact E].
    apply F3_app; [exact H | exact (astep_sound A Fs Ss ins a H Ea)].
Qed.

End Sound.

(** Programs without the exponential and the logarithm need neither bound. *)
Definition plain (ins : instr) : bool :=
  match ins with IExp _ | ILog _ => false | _ => true end.

Lemma astep_sound_plain : forall ce cl A Fs Ss ins a, plain ins = true -> F3 A Fs Ss ->
  astep ce cl A ins = Some a -> asound a (fstep Fs ins) (sstep Ss ins).
Proof.
  intros ce cl A Fs Ss ins a Hp H E.
  destruct ins as [c|i j|i j|i j|i j|i|i|i|i|i|i]; cbn [astep fstep sstep plain] in *;
    try discriminate.
  - destruct (is_finite c) eqn:Fc; [|discriminate]. injection E as <-. apply a_const_sound; exact Fc.
  - exact (a_plus_sound _ _ _ _ _ _ _ (F3_nth _ _ _ i H) (F3_nth _ _ _ j H) E).
  - exact (a_minus_sound _ _ _ _ _ _ _ (F3_nth _ _ _ i H) (F3_nth _ _ _ j H) E).
  - exact (a_mult_sound _ _ _ _ _ _ _ (F3_nth _ _ _ i H) (F3_nth _ _ _ j H) E).
  - exact (a_div_sound _ _ _ _ _ _ _ (F3_nth _ _ _ i H) (F3_nth _ _ _ j H) E).
  - injection E as <-. exact (a_neg_sound _ _ _ (F3_nth _ _ _ i H)).
  - injection E as <-. exact (a_abs_sound _ _ _ (F3_nth _ _ _ i H)).
  - injection E as <-. exact (a_max0_sound _ _ _ (F3_nth _ _ _ i H)).
  - exact (a_pow2_sound _ _ _ _ (F3_nth _ _ _ i H) E).
Qed.

Theorem aeval_sound_plain : forall ce cl p A Fs Ss A', forallb plain p = true ->
  F3 A Fs Ss -> aeval ce cl p A = Some A' -> F3 A' (feval p Fs) (seval p Ss).
Proof.
  intros ce cl. induction p as [|ins p IH]; intros A Fs Ss A' Hp H E; cbn in E |- *.
  - injection E as <-. exact H.
  - cbn [forallb] in Hp. apply andb_prop in Hp as [Hi Hp].
    destruct (astep ce cl A ins) as [a|] eqn:Ea; [|discriminate].
    apply (IH (A ++ [a])); [exact Hp | | exact E].
    apply F3_app; [exact H | exact (astep_sound_plain ce cl A Fs Ss ins a Hi H Ea)].
Qed.

(** * Inputs, and reading off a bound *)

(** The interval between two binary32 values. *)
Definition ivab (a b : binary32) : EI.type :=
  let (ma, ea) := b32_decomp a in let (mb, eb) := b32_decomp b in ivd ma ea mb eb.

Lemma ctn_ivab : forall a b x, B2R a <= x <= B2R b -> ctn (ivab a b) x.
Proof.
  intros a b x H. unfold ivab. pose proof (b32_decomp_spec a) as Ha. pose proof (b32_decomp_spec b) as Hb.
  destruct (b32_decomp a) as [ma ea]. destruct (b32_decomp b) as [mb eb]. cbn [fst snd] in *.
  apply ctn_ivd. unfold dR. rewrite <- Ha, <- Hb. exact H.
Qed.

(** An input ranging over [[a, b]], held exactly, and known when [a = b]. *)
Definition ax (a b : binary32) : aval :=
  AV (ivab a b) (ivab a b) zero_iv
     (if (is_finite a && is_finite b && Beqb a b)%bool then Some a else None).

Lemma ax_sound : forall a b x, is_finite x = true -> B2R a <= B2R x <= B2R b ->
  asound (ax a b) x (B2R x).
Proof.
  intros a b x F H. unfold asound, ax. cbn [av ash ad ak].
  split; [exact F|]. split; [apply ctn_ivab; exact H|]. split; [apply ctn_ivab; exact H|].
  split.
  - exists 0. split; [exact ctn_zero|]. unfold Rminus. rewrite Rplus_opp_r, Rabs_R0. lra.
  - intros c E. destruct (is_finite a && is_finite b && Beqb a b)%bool eqn:C; [|discriminate].
    injection E as <-. apply andb_prop in C as [C Ce]. apply andb_prop in C as [Fa Fb].
    rewrite (Beqb_correct prec32 emax32 a b Fa Fb) in Ce.
    revert Ce. case Req_bool_spec; [intros Eab _ | intros _ Ce; discriminate].
    split; [exact Fa | lra].
Qed.

(** A known input, for any binary32 value with the same real value. *)
Lemma a_const_same : forall c f, is_finite c = true -> is_finite f = true -> B2R f = B2R c ->
  asound (a_const c) f (B2R c).
Proof.
  intros c f Fc Ff E. pose proof (a_const_sound c Fc) as (_ & V & S & (e & Ee & He) & K).
  unfold asound. split; [exact Ff|]. rewrite E. split; [exact V|]. split; [exact S|].
  split; [exists e; split; assumption|].
  intros c' Ec. destruct (K c' Ec) as [F' V']. split; [exact F' | rewrite V'; reflexivity].
Qed.

(** One interval below another. *)
Definition iv_le (X Y : EI.type) : bool :=
  match ub X, lb Y with Some (m1, e1), Some (m2, e2) => dle m1 e1 m2 e2 | _, _ => false end.

Lemma iv_le_spec : forall X Y x y, iv_le X Y = true -> ctn X x -> ctn Y y -> x <= y.
Proof.
  intros X Y x y H Hx Hy. unfold iv_le in H.
  destruct (ub X) as [[m1 e1]|] eqn:E1; [|discriminate].
  destruct (lb Y) as [[m2 e2]|] eqn:E2; [|discriminate].
  eapply Rle_trans; [apply (ub_spec X x m1 e1 Hx E1)|].
  eapply Rle_trans; [apply dle_spec; exact H | apply (lb_spec Y y m2 e2 Hy E2)].
Qed.

(** The bound an abstract value carries. *)
Lemma asound_err : forall a f s X e0, asound a f s -> iv_le (ad a) X = true -> ctn X e0 ->
  Rabs (B2R f - s) <= e0.
Proof.
  intros a f s X e0 (_ & _ & _ & (e & E & H) & _) L Hx.
  eapply Rle_trans; [exact H | exact (iv_le_spec _ _ _ _ L E Hx)].
Qed.
