(** * Bounds against the mathematical functions

    Float_error.v bounds the float evaluation of each elementary function
    against the real evaluation of the same expression, and Series.v bounds a
    real polynomial against the function it approximates. This file composes
    the two, so each theorem below is the distance from an extracted binary32
    function to the mathematical function.

    The real evaluations divide by the constants the code holds, read off the
    floats by [Qb] and made faithful by [Qb_correct]. For the trigonometric
    series those are not all factorials: [14!] through [19!] need more than
    twenty-four significant bits, and the stored value of [15!] is
    1307674411008. Series.v states its bounds on the stored constants, so the
    composition is the triangle inequality.

    Sine and cosine are bounded twice: on the reduced argument, as [ok_sin] and
    [ok_cos] are, and on the argument itself, by carrying the propagation
    relation through the reduction and bounding what the stored split of
    [2 pi] costs. *)

From Stdlib Require Import ZArith QArith Qreals Reals Lra Lia.
From Stdlib Require List.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Llama Qwen Float_error Series.

Open Scope R_scope.

(** * The constants the code holds *)

Ltac b2r_const v :=
  rewrite Qb_correct;
  rewrite (Qeq_eqR _ (inject_Z v)) by (vm_compute; reflexivity);
  apply Q2R_inject.

Lemma B2R_fac3  : B2R f32_fac3  = 6.                   Proof. b2r_const 6%Z. Qed.
Lemma B2R_fac5  : B2R f32_fac5  = 120.                 Proof. b2r_const 120%Z. Qed.
Lemma B2R_fac7  : B2R f32_fac7  = 5040.                Proof. b2r_const 5040%Z. Qed.
Lemma B2R_fac9  : B2R f32_fac9  = 362880.              Proof. b2r_const 362880%Z. Qed.
Lemma B2R_fac11 : B2R f32_fac11 = 39916800.            Proof. b2r_const 39916800%Z. Qed.
Lemma B2R_fac13 : B2R f32_fac13 = 6227020800.          Proof. b2r_const 6227020800%Z. Qed.
Lemma B2R_fac15 : B2R f32_fac15 = 1307674411008.       Proof. b2r_const 1307674411008%Z. Qed.
Lemma B2R_fac17 : B2R f32_fac17 = 355687414628352.     Proof. b2r_const 355687414628352%Z. Qed.
Lemma B2R_fac19 : B2R f32_fac19 = 121645104594157568.  Proof. b2r_const 121645104594157568%Z. Qed.

Lemma B2R_fac2  : B2R f32_fac2  = 2.                   Proof. b2r_const 2%Z. Qed.
Lemma B2R_fac4  : B2R f32_fac4  = 24.                  Proof. b2r_const 24%Z. Qed.
Lemma B2R_fac6  : B2R f32_fac6  = 720.                 Proof. b2r_const 720%Z. Qed.
Lemma B2R_fac8  : B2R f32_fac8  = 40320.               Proof. b2r_const 40320%Z. Qed.
Lemma B2R_fac10 : B2R f32_fac10 = 3628800.             Proof. b2r_const 3628800%Z. Qed.
Lemma B2R_fac12 : B2R f32_fac12 = 479001600.           Proof. b2r_const 479001600%Z. Qed.
Lemma B2R_fac14 : B2R f32_fac14 = 87178289152.         Proof. b2r_const 87178289152%Z. Qed.
Lemma B2R_fac16 : B2R f32_fac16 = 20922790576128.      Proof. b2r_const 20922790576128%Z. Qed.
Lemma B2R_fac18 : B2R f32_fac18 = 6402373530419200.    Proof. b2r_const 6402373530419200%Z. Qed.

(** The six stored constants that differ from the factorial they approximate. *)
Lemma fac14_not_exact : B2R f32_fac14 <> 87178291200.
Proof. rewrite B2R_fac14. apply eq_IZR_contrapositive. lia. Qed.
Lemma fac15_not_exact : B2R f32_fac15 <> 1307674368000.
Proof. rewrite B2R_fac15. apply eq_IZR_contrapositive. lia. Qed.
Lemma fac16_not_exact : B2R f32_fac16 <> 20922789888000.
Proof. rewrite B2R_fac16. apply eq_IZR_contrapositive. lia. Qed.
Lemma fac17_not_exact : B2R f32_fac17 <> 355687428096000.
Proof. rewrite B2R_fac17. apply eq_IZR_contrapositive. lia. Qed.
Lemma fac18_not_exact : B2R f32_fac18 <> 6402373705728000.
Proof. rewrite B2R_fac18. apply eq_IZR_contrapositive. lia. Qed.
Lemma fac19_not_exact : B2R f32_fac19 <> 121645100408832000.
Proof. rewrite B2R_fac19. apply eq_IZR_contrapositive. lia. Qed.

(** * The real evaluations, as plain polynomials *)

Lemma Rs_poly_expand : forall r : R,
  Rs_poly r
  = r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880 - r^11/39916800
      + r^13/6227020800 - r^15/1307674411008 + r^17/355687414628352
      - r^19/121645104594157568.
Proof.
  intros r. unfold Rs_poly, Rs_a7, Rs_a6, Rs_a5, Rs_a4, Rs_a3, Rs_a2, Rs_a1,
    Rs_a0, Rs_d19, Rs_d17, Rs_d15, Rs_d13, Rs_d11, Rs_d9, Rs_d7, Rs_d5, Rs_d3,
    Rs_r19, Rs_r17, Rs_r15, Rs_r13, Rs_r11, Rs_r9, Rs_r7, Rs_r5, Rs_r3, Rs_r2.
  rewrite B2R_fac3, B2R_fac5, B2R_fac7, B2R_fac9, B2R_fac11, B2R_fac13,
          B2R_fac15, B2R_fac17, B2R_fac19.
  field.
Qed.

Lemma Rc_poly_expand : forall r : R,
  Rc_poly r
  = 1 - r^2/2 + r^4/24 - r^6/720 + r^8/40320 - r^10/3628800
      + r^12/479001600 - r^14/87178289152 + r^16/20922790576128
      - r^18/6402373530419200.
Proof.
  intros r. unfold Rc_poly, Rc_a7, Rc_a6, Rc_a5, Rc_a4, Rc_a3, Rc_a2, Rc_a1,
    Rc_a0, Rc_d18, Rc_d16, Rc_d14, Rc_d12, Rc_d10, Rc_d8, Rc_d6, Rc_d4, Rc_d2,
    Rc_r18, Rc_r16, Rc_r14, Rc_r12, Rc_r10, Rc_r8, Rc_r6, Rc_r4, Rc_r2.
  rewrite B2R_fac2, B2R_fac4, B2R_fac6, B2R_fac8, B2R_fac10, B2R_fac12,
          B2R_fac14, B2R_fac16, B2R_fac18.
  field.
Qed.

(** * Sine and cosine *)

Theorem sin_poly_actual : forall r : R,
  -3.15 <= r <= 3.15 -> Rabs (Rs_poly r - sin r) <= 2/1000000000.
Proof.
  intros r Hr. rewrite Rs_poly_expand. apply sin_series_bound, Hr.
Qed.

Theorem cos_poly_actual : forall r : R,
  -3.15 <= r <= 3.15 -> Rabs (Rc_poly r - cos r) <= 2/100000000.
Proof.
  intros r Hr. rewrite Rc_poly_expand. apply cos_series_bound, Hr.
Qed.

(** Splitting a distance through an intermediate value, on plain variables so
    the normalisation never touches the polynomials. *)
Lemma abs_split : forall a b c : R, Rabs (a - c) <= Rabs (a - b) + Rabs (b - c).
Proof.
  intros a b c.
  replace (a - c) with ((a - b) + (b - c)) by ring.
  apply Rabs_triang.
Qed.

(** The extracted sine and cosine, against the mathematical functions of the
    reduced argument: the rounding chain of Float_error.v plus the truncation
    of the series. *)

Theorem ok_sin_true : forall M m L k x rr,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) (f32_reduce_2pi x) rr ->
  sin_reg M m (f32_reduce_2pi x) rr ->
  -3.15 <= rr <= 3.15 ->
  Rabs (B2R (f32_sin x) - sin rr) <= errN M L (k + 20) + 2/1000000000.
Proof.
  intros M m L k x rr HM Hamp Hr Hreg Hint.
  destruct (ok_sin M m L k x rr HM Hamp Hr Hreg) as [_ Hd].
  pose proof (sin_poly_actual rr Hint) as Ht.
  eapply Rle_trans; [apply (abs_split _ (Rs_poly rr)) | lra].
Qed.

Theorem ok_cos_true : forall M m L k x rr,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) (f32_reduce_2pi x) rr ->
  cos_reg M m (f32_reduce_2pi x) rr ->
  -3.15 <= rr <= 3.15 ->
  Rabs (B2R (f32_cos x) - cos rr) <= errN M L (k + 19) + 2/100000000.
Proof.
  intros M m L k x rr HM Hamp Hr Hreg Hint.
  destruct (ok_cos M m L k x rr HM Hamp Hr Hreg) as [_ Hd].
  pose proof (cos_poly_actual rr Hint) as Ht.
  eapply Rle_trans; [apply (abs_split _ (Rc_poly rr)) | lra].
Qed.

(** * The argument reduction

    [f32_reduce_2pi] rounds [x / (2 pi)] to an integer [k] and subtracts [k]
    times the stored split of [2 pi]. The real evaluation selects [k] by
    rounding the exact product and performs the same two subtractions. The one
    branch that is a hypothesis is [k] itself, as for the exponential: the
    float reduction must land on the integer the real evaluation selects. For
    [|x| <= 4000] the reduced argument stays within [3.15] of zero and differs
    from [x - 2 pi k] by at most [1.3e-8], so the series bounds carry over to
    [sin x] and [cos x]. *)

Definition trig_k_range : list Z :=
  List.map (fun n => (Z.of_nat n - 1024)%Z) (List.seq 0 2049).

Definition trig_k_ok (z : Z) : bool :=
  is_finite (f32_of_Z z) && Qeq_bool (Qb (f32_of_Z z)) (inject_Z z).

Lemma trig_k_ok_all : List.forallb trig_k_ok trig_k_range = true.
Proof. vm_compute. reflexivity. Qed.

Lemma in_trig_k_range : forall z, (-1024 <= z <= 1024)%Z -> List.In z trig_k_range.
Proof.
  intros z Hz. unfold trig_k_range. apply List.in_map_iff.
  exists (Z.to_nat (z + 1024)). split.
  - rewrite Z2Nat.id by lia. lia.
  - apply List.in_seq. lia.
Qed.

Lemma of_Z_exact : forall z, (-1024 <= z <= 1024)%Z ->
  is_finite (f32_of_Z z) = true /\ B2R (f32_of_Z z) = IZR z.
Proof.
  intros z Hz.
  pose proof (proj1 (List.forallb_forall trig_k_ok trig_k_range) trig_k_ok_all z
                (in_trig_k_range z Hz)) as H.
  unfold trig_k_ok in H. apply andb_prop in H as [H1 H2].
  apply Qeq_bool_eq in H2.
  split; [exact H1|].
  rewrite Qb_correct, (Qeq_eqR _ _ H2). apply Q2R_inject.
Qed.

Lemma B2R_2pi_hi : B2R f32_2pi_hi = 201 / 32.
Proof.
  rewrite Qb_correct.
  rewrite (Qeq_eqR _ (Qmake 201 32)) by (vm_compute; reflexivity).
  unfold Q2R; cbn [Qnum Qden]; unfold Rdiv; reflexivity.
Qed.

Lemma B2R_2pi_lo : B2R f32_2pi_lo = 8312081 / 4294967296.
Proof.
  rewrite Qb_correct.
  rewrite (Qeq_eqR _ (Qmake 8312081 4294967296)) by (vm_compute; reflexivity).
  unfold Q2R; cbn [Qnum Qden]; unfold Rdiv; reflexivity.
Qed.

Lemma B2R_inv2pi : B2R f32_inv2pi = 10680707 / 67108864.
Proof.
  rewrite Qb_correct.
  rewrite (Qeq_eqR _ (Qmake 10680707 67108864)) by (vm_compute; reflexivity).
  unfold Q2R; cbn [Qnum Qden]; unfold Rdiv; reflexivity.
Qed.

Definition Rs_k (rx : R) : Z :=
  Znearest (fun n => negb (Z.even n)) (rx * B2R f32_inv2pi).

Definition Rreduce (rx : R) : R :=
  rx - IZR (Rs_k rx) * B2R f32_2pi_hi - IZR (Rs_k rx) * B2R f32_2pi_lo.

(** The side conditions of the reduction. *)
Record red_reg (M m : R) (x : binary32) (rx : R) : Prop := {
  rdr_kz : B2SF (f32_round_int (f32_mult x f32_inv2pi)) = B2SF (f32_of_Z (Rs_k rx));
  rdr_bk : Rabs (B2R (f32_of_Z (Rs_k rx))) <= M;
  rdr_bhi : Rabs (B2R f32_2pi_hi) <= M;
  rdr_blo : Rabs (B2R f32_2pi_lo) <= M;
  rdr_zkhi : regz M (B2R (f32_of_Z (Rs_k rx)) * B2R f32_2pi_hi);
  rdr_zklo : regz M (B2R (f32_of_Z (Rs_k rx)) * B2R f32_2pi_lo);
  rdr_z1 : regz M (B2R x + B2R (f32_neg (f32_mult (f32_of_Z (Rs_k rx)) f32_2pi_hi)));
  rdr_z2 : regz M (B2R (f32_minus x (f32_mult (f32_of_Z (Rs_k rx)) f32_2pi_hi))
                   + B2R (f32_neg (f32_mult (f32_of_Z (Rs_k rx)) f32_2pi_lo)))
}.

Lemma ok_reduce_2pi : forall M m L n x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L n) x rx -> red_reg M m x rx -> (-1024 <= Rs_k rx <= 1024)%Z ->
  ok (errN M L (n + 3)) (f32_reduce_2pi x) (Rreduce rx).
Proof.
  intros M m L n x rx HM Hamp Hx Hreg Hk.
  assert (HM0 : 0 <= M) by (destruct Hamp as (_ & H & _); lra).
  assert (HL1 : 1 <= L) by (eapply amp_L_pos; eassumption).
  destruct Hreg as [Hkz Hbk Hbhi Hblo Hzkhi Hzklo Hz1 Hz2].
  assert (Hkeq : f32_round_int (f32_mult x f32_inv2pi) = f32_of_Z (Rs_k rx))
    by (apply B2SF_inj; exact Hkz).
  destruct (of_Z_exact (Rs_k rx) Hk) as [Hkfin HkB].
  unfold f32_reduce_2pi. cbv zeta. rewrite Hkeq.
  set (kf := f32_of_Z (Rs_k rx)) in *.
  assert (Hk0 : ok (errN M L n) kf (IZR (Rs_k rx)))
    by (rewrite <- HkB; apply ok_const; assumption).
  assert (Hhi : ok (errN M L n) f32_2pi_hi (B2R f32_2pi_hi))
    by (apply ok_const; [exact HM0 | exact HL1 | vm_compute; reflexivity]).
  assert (Hlo : ok (errN M L n) f32_2pi_lo (B2R f32_2pi_lo))
    by (apply ok_const; [exact HM0 | exact HL1 | vm_compute; reflexivity]).
  assert (Hkhi : ok (errN M L (S n)) (f32_mult kf f32_2pi_hi)
                    (IZR (Rs_k rx) * B2R f32_2pi_hi))
    by (eapply ok_mult_S with (m := m); eassumption).
  assert (Hklo : ok (errN M L (S n)) (f32_mult kf f32_2pi_lo)
                    (IZR (Rs_k rx) * B2R f32_2pi_lo))
    by (eapply ok_mult_S with (m := m); eassumption).
  assert (Hx1 : ok (errN M L (S n)) x rx)
    by (eapply ok_errN_mono with (n := n); [exact HM0 | exact HL1 | lia | exact Hx]).
  assert (Hr1 : ok (errN M L (S (S n))) (f32_minus x (f32_mult kf f32_2pi_hi))
                   (rx - IZR (Rs_k rx) * B2R f32_2pi_hi))
    by (eapply ok_minus_S with (m := m); eassumption).
  assert (Hklo2 : ok (errN M L (S (S n))) (f32_mult kf f32_2pi_lo)
                     (IZR (Rs_k rx) * B2R f32_2pi_lo))
    by (eapply ok_errN_mono with (n := S n); [exact HM0 | exact HL1 | lia | exact Hklo]).
  unfold Rreduce. replace (n + 3)%nat with (S (S (S n))) by lia.
  eapply ok_minus_S with (m := m); eassumption.
Qed.

Lemma Rs_k_half : forall rx : R,
  Rabs (rx * (10680707 / 67108864) - IZR (Rs_k rx)) <= / 2.
Proof. intros rx. unfold Rs_k. rewrite B2R_inv2pi. apply Znearest_half. Qed.

Lemma Rs_k_bounds : forall rx : R, -4000 <= rx <= 4000 ->
  (-637 <= Rs_k rx <= 637)%Z /\ Rabs (rx - 2 * PI * IZR (Rs_k rx)) <= 31419 / 10000.
Proof.
  intros rx Hr.
  pose proof (Rs_k_half rx) as H. apply Rabs_le_inv in H.
  pose proof pi_bounds as Hpi.
  split; [split|].
  - destruct (Z_lt_le_dec (Rs_k rx) (-637)) as [Hc|Hc]; [|exact Hc].
    exfalso. assert (Hc' : (Rs_k rx <= Z.opp 638)%Z) by lia.
    apply IZR_le in Hc'. rewrite opp_IZR in Hc'. lra.
  - destruct (Z_lt_le_dec 637 (Rs_k rx)) as [Hc|Hc]; [|exact Hc].
    exfalso. assert (Hc' : (638 <= Rs_k rx)%Z) by lia.
    apply IZR_le in Hc'. lra.
  - pose proof inv_two_pi_bound as Hi.
    assert (Heq : rx - 2 * PI * IZR (Rs_k rx)
                  = 2 * PI * ((rx * (10680707 / 67108864) - IZR (Rs_k rx))
                              - rx * (10680707 / 67108864 - / (2 * PI)))).
    { field. lra. }
    rewrite Heq, Rabs_mult, (Rabs_pos_eq (2 * PI)) by lra.
    assert (Hq : Rabs ((rx * (10680707 / 67108864) - IZR (Rs_k rx))
                       - rx * (10680707 / 67108864 - / (2 * PI)))
                 <= / 2 + 4000 * (1 / 100000000)).
    { unfold Rminus at 1. eapply Rle_trans; [apply Rabs_triang|].
      rewrite Rabs_Ropp, Rabs_mult.
      assert (Hxr : Rabs rx <= 4000) by (apply Rabs_le; lra).
      assert (Hp : Rabs rx * Rabs (10680707 / 67108864 - / (2 * PI))
                   <= 4000 * (1 / 100000000))
        by (apply Rmult_le_compat; try apply Rabs_pos; assumption).
      apply Rabs_le in H. lra. }
    apply Rle_trans with (2 * PI * (/ 2 + 4000 * (1 / 100000000))).
    + apply Rmult_le_compat_l; [lra | exact Hq].
    + lra.
Qed.

Lemma sin_period_Z : forall (x : R) (k : Z), sin (x + 2 * IZR k * PI) = sin x.
Proof.
  intros x k. destruct k as [|p|p].
  - replace (x + 2 * IZR 0 * PI) with x by (simpl; ring). reflexivity.
  - rewrite <- (positive_nat_Z p), <- INR_IZR_INZ. apply sin_period.
  - replace (Z.neg p) with (- Z.pos p)%Z by reflexivity.
    rewrite opp_IZR, <- (positive_nat_Z p), <- INR_IZR_INZ.
    rewrite <- (sin_period (x + 2 * - INR (Pos.to_nat p) * PI) (Pos.to_nat p)).
    f_equal. ring.
Qed.

Lemma cos_period_Z : forall (x : R) (k : Z), cos (x + 2 * IZR k * PI) = cos x.
Proof.
  intros x k. destruct k as [|p|p].
  - replace (x + 2 * IZR 0 * PI) with x by (simpl; ring). reflexivity.
  - rewrite <- (positive_nat_Z p), <- INR_IZR_INZ. apply cos_period.
  - replace (Z.neg p) with (- Z.pos p)%Z by reflexivity.
    rewrite opp_IZR, <- (positive_nat_Z p), <- INR_IZR_INZ.
    rewrite <- (cos_period (x + 2 * - INR (Pos.to_nat p) * PI) (Pos.to_nat p)).
    f_equal. ring.
Qed.

Lemma abs_sin_le : forall t : R, Rabs (sin t) <= Rabs t.
Proof.
  intros t.
  assert (Hpos : forall s, 0 <= s -> Rabs (sin s) <= s).
  { intros s Hs. destruct (Req_dec s 0) as [H0|H0].
    - subst s. rewrite sin_0, Rabs_R0. lra.
    - pose proof (sin_lt_x s ltac:(lra)) as Hl.
      pose proof (SIN_bound s) as Hb.
      apply Rabs_le. split; [|lra].
      destruct (Rle_or_lt s PI) as [Hp|Hp].
      + pose proof (sin_ge_0 s Hs Hp). lra.
      + pose proof pi_bounds. lra. }
  destruct (Rle_or_lt 0 t) as [Ht|Ht].
  - rewrite (Rabs_pos_eq t) by exact Ht. apply Hpos, Ht.
  - rewrite (Rabs_left t) by exact Ht. rewrite <- Rabs_Ropp, <- sin_neg.
    apply Hpos. lra.
Qed.

Lemma sin_lipschitz : forall p q : R, Rabs (sin p - sin q) <= Rabs (p - q).
Proof.
  intros p q. rewrite form4.
  rewrite !Rabs_mult, (Rabs_pos_eq 2) by lra.
  pose proof (COS_bound ((p + q) / 2)) as Hc.
  assert (Hc' : Rabs (cos ((p + q) / 2)) <= 1) by (apply Rabs_le; lra).
  pose proof (abs_sin_le ((p - q) / 2)) as Hs.
  assert (Hd : Rabs ((p - q) / 2) = Rabs (p - q) / 2).
  { unfold Rdiv. rewrite Rabs_mult, (Rabs_pos_eq (/ 2)) by lra. reflexivity. }
  rewrite Hd in Hs.
  pose proof (Rabs_pos (cos ((p + q) / 2))). pose proof (Rabs_pos (sin ((p - q) / 2))).
  nra.
Qed.

Lemma cos_lipschitz : forall p q : R, Rabs (cos p - cos q) <= Rabs (p - q).
Proof.
  intros p q. rewrite <- (sin_shift p), <- (sin_shift q).
  eapply Rle_trans; [apply sin_lipschitz|].
  replace (PI / 2 - p - (PI / 2 - q)) with (- (p - q)) by ring.
  rewrite Rabs_Ropp. lra.
Qed.

Theorem reduce_vs_true : forall rx : R, -4000 <= rx <= 4000 ->
  -3.15 <= Rreduce rx <= 3.15
  /\ Rabs (sin (Rreduce rx) - sin rx) <= 13 / 1000000000
  /\ Rabs (cos (Rreduce rx) - cos rx) <= 13 / 1000000000.
Proof.
  intros rx Hr.
  destruct (Rs_k_bounds rx Hr) as [[Hk1 Hk2] H2pi].
  pose proof two_pi_split_bound as Hs.
  set (k := Rs_k rx) in *.
  assert (HK : Rabs (IZR k) <= 637).
  { apply Rabs_le. split.
    - assert (Hc : (Z.opp 637 <= k)%Z) by lia.
      apply IZR_le in Hc. rewrite opp_IZR in Hc. exact Hc.
    - apply IZR_le in Hk2. exact Hk2. }
  assert (Heq : Rreduce rx = (rx + 2 * IZR (- k) * PI)
                             - IZR k * (201 / 32 + 8312081 / 4294967296 - 2 * PI)).
  { unfold Rreduce. fold k. rewrite B2R_2pi_hi, B2R_2pi_lo, opp_IZR. ring. }
  assert (Hke : Rabs (IZR k * (201 / 32 + 8312081 / 4294967296 - 2 * PI))
                <= 13 / 1000000000).
  { rewrite Rabs_mult.
    apply Rle_trans with (637 * (2 / 100000000000)); [|lra].
    apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  assert (Hshift : rx + 2 * IZR (- k) * PI = rx - 2 * PI * IZR k)
    by (rewrite opp_IZR; ring).
  split; [|split].
  - rewrite Heq, Hshift.
    apply Rabs_le_inv in H2pi. apply Rabs_le_inv in Hke. lra.
  - rewrite Heq, <- (sin_period_Z rx (- k)).
    eapply Rle_trans; [apply sin_lipschitz|].
    replace (rx + 2 * IZR (- k) * PI
             - IZR k * (201 / 32 + 8312081 / 4294967296 - 2 * PI)
             - (rx + 2 * IZR (- k) * PI))
      with (- (IZR k * (201 / 32 + 8312081 / 4294967296 - 2 * PI))) by ring.
    rewrite Rabs_Ropp. exact Hke.
  - rewrite Heq, <- (cos_period_Z rx (- k)).
    eapply Rle_trans; [apply cos_lipschitz|].
    replace (rx + 2 * IZR (- k) * PI
             - IZR k * (201 / 32 + 8312081 / 4294967296 - 2 * PI)
             - (rx + 2 * IZR (- k) * PI))
      with (- (IZR k * (201 / 32 + 8312081 / 4294967296 - 2 * PI))) by ring.
    rewrite Rabs_Ropp. exact Hke.
Qed.

(** The extracted sine and cosine against [sin x] and [cos x]. *)
Theorem ok_sin_true_full : forall M m L n x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L n) x rx -> red_reg M m x rx ->
  sin_reg M m (f32_reduce_2pi x) (Rreduce rx) -> -4000 <= rx <= 4000 ->
  Rabs (B2R (f32_sin x) - sin rx) <= errN M L (n + 23) + 15 / 1000000000.
Proof.
  intros M m L n x rx HM Hamp Hx Hred Hsin Hr.
  destruct (Rs_k_bounds rx Hr) as [[Hk1 Hk2] _].
  assert (Hrd : ok (errN M L (n + 3)) (f32_reduce_2pi x) (Rreduce rx))
    by (apply ok_reduce_2pi with (m := m); try assumption; lia).
  destruct (reduce_vs_true rx Hr) as (Hrr & Hs & _).
  pose proof (ok_sin_true M m L (n + 3) x (Rreduce rx) HM Hamp Hrd Hsin Hrr) as H.
  replace (n + 3 + 20)%nat with (n + 23)%nat in H by lia.
  eapply Rle_trans; [apply (abs_split _ (sin (Rreduce rx))) | lra].
Qed.

Theorem ok_cos_true_full : forall M m L n x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L n) x rx -> red_reg M m x rx ->
  cos_reg M m (f32_reduce_2pi x) (Rreduce rx) -> -4000 <= rx <= 4000 ->
  Rabs (B2R (f32_cos x) - cos rx) <= errN M L (n + 22) + 33 / 1000000000.
Proof.
  intros M m L n x rx HM Hamp Hx Hred Hcos Hr.
  destruct (Rs_k_bounds rx Hr) as [[Hk1 Hk2] _].
  assert (Hrd : ok (errN M L (n + 3)) (f32_reduce_2pi x) (Rreduce rx))
    by (apply ok_reduce_2pi with (m := m); try assumption; lia).
  destruct (reduce_vs_true rx Hr) as (Hrr & _ & Hc).
  pose proof (ok_cos_true M m L (n + 3) x (Rreduce rx) HM Hamp Hrd Hcos Hrr) as H.
  replace (n + 3 + 19)%nat with (n + 22)%nat in H by lia.
  eapply Rle_trans; [apply (abs_split _ (cos (Rreduce rx))) | lra].
Qed.

(** * The exponential

    After saturation, [f32_exp_approx] writes its argument as [k log 2 + r],
    evaluates the degree-seven Taylor polynomial at [r] and multiplies by
    [2^k]. The divisors
    are exact in binary32, and the reduction subtracts [k] times
    [355/512 - 14581891/2^36] rather than [k log 2]. The distance from the
    exponential therefore has two sources, the truncation of the series at
    [|r| <= 0.35] and the split of [log 2], and together they come to a
    relative [1.6e-8]. *)

Lemma B2R_two    : B2R f32_two = 2.            Proof. b2r_const 2%Z. Qed.
Lemma B2R_six    : B2R f32_six = 6.            Proof. b2r_const 6%Z. Qed.
Lemma B2R_24     : B2R f32_twenty_four = 24.   Proof. b2r_const 24%Z. Qed.
Lemma B2R_120    : B2R f32_one_twenty = 120.   Proof. b2r_const 120%Z. Qed.
Lemma B2R_720    : B2R f32_seven_twenty = 720. Proof. b2r_const 720%Z. Qed.
Lemma B2R_5040   : B2R f32_five_thousand_forty = 5040. Proof. b2r_const 5040%Z. Qed.

Lemma bpow_exp : forall z : Z, bpow radix2 z = exp (IZR z * ln 2).
Proof.
  intros z. rewrite bpow_powerRZ. rewrite powerRZ_Rpower by (simpl; lra).
  unfold Rpower. reflexivity.
Qed.

Lemma Re_poly_expand : forall rx : R,
  Re_poly rx
  = 1 + Re_r rx + (Re_r rx)^2/2 + (Re_r rx)^3/6 + (Re_r rx)^4/24
      + (Re_r rx)^5/120 + (Re_r rx)^6/720 + (Re_r rx)^7/5040.
Proof.
  intros rx.
  unfold Re_poly, Re_p6, Re_p5, Re_p4, Re_p3, Re_p2, Re_p1,
         Re_d7, Re_d6, Re_d5, Re_d4, Re_d3, Re_d2,
         Re_r7, Re_r6, Re_r5, Re_r4, Re_r3, Re_r2.
  rewrite B2R_two, B2R_six, B2R_24, B2R_120, B2R_720, B2R_5040.
  field.
Qed.

(** The arithmetic of the bound, on plain variables: a series within [1e-8] of
    [exp r], an exponent [t] within [3e-10] of [r], and [E = B exp t]. *)
Lemma exp_split_bound : forall P r t B E : R,
  0 < B ->
  Rabs (P - exp r) <= 1 / 100000000 ->
  exp r <= 144 / 100 ->
  Rabs (exp (t - r) - 1) <= 4 / 10000000000 ->
  69 / 100 <= exp t ->
  E = B * exp t ->
  Rabs (P * B - E) <= E * (16 / 1000000000).
Proof.
  intros P r t B E HB Hser Hup Hnz Hlo HE.
  assert (Ht : exp t = exp r * exp (t - r)).
  { rewrite <- exp_plus. f_equal. ring. }
  assert (Hin : Rabs (P - exp t) <= 1 / 100000000 + 144 / 100 * (4 / 10000000000)).
  { rewrite Ht.
    replace (P - exp r * exp (t - r))
      with ((P - exp r) + exp r * (1 - exp (t - r))) by ring.
    eapply Rle_trans; [apply Rabs_triang|].
    apply Rplus_le_compat; [exact Hser|].
    rewrite Rabs_mult, (Rabs_pos_eq (exp r)) by (left; apply exp_pos).
    apply Rmult_le_compat; [left; apply exp_pos | apply Rabs_pos | exact Hup |].
    rewrite <- Rabs_Ropp.
    replace (- (1 - exp (t - r))) with (exp (t - r) - 1) by ring.
    exact Hnz. }
  rewrite HE.
  replace (P * B - B * exp t) with (B * (P - exp t)) by ring.
  rewrite Rabs_mult, (Rabs_pos_eq B) by lra.
  apply Rle_trans with (B * (1 / 100000000 + 144 / 100 * (4 / 10000000000))).
  - apply Rmult_le_compat_l; [lra | exact Hin].
  - apply Rle_trans with (B * (69 / 100 * (16 / 1000000000))).
    + apply Rmult_le_compat_l; lra.
    + replace (B * exp t * (16 / 1000000000))
        with (B * (exp t * (16 / 1000000000))) by ring.
      apply Rmult_le_compat_l; [lra|].
      apply Rmult_le_compat_r; lra.
Qed.

(** The computation after saturation, on the range saturation leaves. *)
Theorem exp_core_vs_true : forall rx : R,
  -88 <= rx <= 88 ->
  Rabs (Re_exp_core rx - exp rx) <= exp rx * (16 / 1000000000).
Proof.
  intros rx Hrx.
  pose proof (Re_r_range rx Hrx) as Hr. apply Rabs_le_inv in Hr.
  pose proof (Re_k_range rx Hrx) as [Hk1 Hk2].
  assert (HK : Rabs (IZR (Re_k rx)) <= 127).
  { apply Rabs_le. split.
    - assert (Hc : (Z.opp 127 <= Re_k rx)%Z) by lia.
      apply IZR_le in Hc. rewrite opp_IZR in Hc. exact Hc.
    - apply IZR_le in Hk2. exact Hk2. }
  pose proof ln2_split_bound as He.
  assert (HKe : Rabs (IZR (Re_k rx) * (355 / 512 - 14581891 / 68719476736 - ln 2))
                <= 3 / 10000000000).
  { rewrite Rabs_mult.
    apply Rle_trans with (127 * (2 / 1000000000000)).
    - apply Rmult_le_compat; try apply Rabs_pos; assumption.
    - lra. }
  apply Rabs_le_inv in HKe.
  unfold Re_exp_core.
  apply (exp_split_bound (Re_poly rx) (Re_r rx)
           (Re_r rx + IZR (Re_k rx) * (355 / 512 - 14581891 / 68719476736 - ln 2))
           (bpow radix2 (Re_k rx)) (exp rx)).
  - apply bpow_gt_0.
  - rewrite Re_poly_expand. apply exp_series_bound. split; lra.
  - apply exp_upper_bound. split; lra.
  - replace (Re_r rx + IZR (Re_k rx) * (355 / 512 - 14581891 / 68719476736 - ln 2)
             - Re_r rx)
      with (IZR (Re_k rx) * (355 / 512 - 14581891 / 68719476736 - ln 2)) by ring.
    apply exp_near_zero. split; lra.
  - apply exp_lower_bound. split; lra.
  - rewrite bpow_exp, <- exp_plus. f_equal. rewrite Re_r_eq. ring.
Qed.

(** The real evaluation of the extracted exponential tracks the exponential of
    the saturated argument, for every real argument. *)
Theorem exp_approx_vs_true : forall rx : R,
  Rabs (Re_exp_approx rx - exp (Rsat rx)) <= exp (Rsat rx) * (16 / 1000000000).
Proof.
  intros rx. unfold Re_exp_approx. apply exp_core_vs_true, Rsat_range.
Qed.

(** * The logarithm

    [f32_log_unit] is only ever applied on [(1, 2]], where the arctanh series
    converges without range reduction. All six divisors are exact. *)

Lemma B2R_three    : B2R f32_three = 3.    Proof. b2r_const 3%Z. Qed.
Lemma B2R_five     : B2R f32_five = 5.     Proof. b2r_const 5%Z. Qed.
Lemma B2R_seven    : B2R f32_seven = 7.    Proof. b2r_const 7%Z. Qed.
Lemma B2R_nine     : B2R f32_nine = 9.     Proof. b2r_const 9%Z. Qed.
Lemma B2R_eleven   : B2R f32_eleven = 11.  Proof. b2r_const 11%Z. Qed.
Lemma B2R_thirteen : B2R f32_thirteen = 13. Proof. b2r_const 13%Z. Qed.

Lemma Rlog_unit_expand : forall rm : R,
  rm + 1 <> 0 ->
  Rlog_unit rm
  = 2 * ((rm-1)/(rm+1) + ((rm-1)/(rm+1))^3/3 + ((rm-1)/(rm+1))^5/5
         + ((rm-1)/(rm+1))^7/7 + ((rm-1)/(rm+1))^9/9
         + ((rm-1)/(rm+1))^11/11 + ((rm-1)/(rm+1))^13/13).
Proof.
  intros rm Hne.
  unfold Rlog_unit, Rl_s, Rl_p5, Rl_p4, Rl_p3, Rl_p2, Rl_p1,
         Rl_d13, Rl_d11, Rl_d9, Rl_d7, Rl_d5, Rl_d3,
         Rl_u13, Rl_u11, Rl_u9, Rl_u7, Rl_u5, Rl_u3, Rl_u2, Rl_u.
  rewrite B2R_two, B2R_three, B2R_five, B2R_seven, B2R_nine,
          B2R_eleven, B2R_thirteen.
  field. exact Hne.
Qed.

(** The substitution the series performs is exact: with [u = (m-1)/(m+1)],
    [(1+u)/(1-u)] is [m]. *)
Lemma log_arg_id : forall m : R,
  0 < m -> (1 + (m-1)/(m+1)) / (1 - (m-1)/(m+1)) = m.
Proof. intros m Hm. field. lra. Qed.

Theorem log_unit_vs_true : forall rm : R,
  1 <= rm <= 2.0001 -> Rabs (Rlog_unit rm - ln rm) <= 2/100000000.
Proof.
  intros rm H.
  assert (Hp : 0 < rm + 1) by lra.
  assert (Hu : 0 <= (rm-1)/(rm+1) <= 0.3334).
  { split.
    - unfold Rdiv. apply Rmult_le_pos;
        [lra | left; apply Rinv_0_lt_compat; lra].
    - unfold Rdiv. apply Rmult_le_reg_r with (rm+1); [lra|].
      rewrite Rmult_assoc, Rinv_l by lra. lra. }
  pose proof (log_series_bound _ Hu) as Hb.
  rewrite log_arg_id in Hb by lra.
  rewrite Rlog_unit_expand by lra.
  exact Hb.
Qed.

(** * The float functions against the mathematical ones

    Composing the rounding chain of Float_error.v with the truncation bounds
    above. Each theorem is the distance from the extracted binary32 function to
    the mathematical function of the value the real evaluation was given. *)

Theorem ok_exp_true : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx ->
  exp_reg M m x rx ->
  Rabs (B2R (f32_exp_approx x) - exp (Rsat rx))
  <= errN M L (k + 21) + exp (Rsat rx) * (16 / 1000000000).
Proof.
  intros M m L k x rx HM Hamp Hx Hreg.
  destruct (ok_exp_approx M m L k x rx HM Hamp Hx Hreg) as [_ Hd].
  pose proof (exp_approx_vs_true rx) as Ht.
  eapply Rle_trans; [apply (abs_split _ (Re_exp_approx rx)) | lra].
Qed.

Corollary ok_exp_true_in_range : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx ->
  exp_reg M m x rx ->
  -88 <= rx <= 88 ->
  Rabs (B2R (f32_exp_approx x) - exp rx)
  <= errN M L (k + 21) + exp rx * (16 / 1000000000).
Proof.
  intros M m L k x rx HM Hamp Hx Hreg Hrng.
  pose proof (ok_exp_true M m L k x rx HM Hamp Hx Hreg) as H.
  rewrite Rsat_id in H by exact Hrng. exact H.
Qed.

Theorem ok_log_true : forall M m L k mv rm,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) mv rm ->
  log_reg M m mv rm ->
  1 <= rm <= 2 ->
  Rabs (B2R (f32_log_unit mv) - ln rm)
  <= errN M L (k + 17) + 2/100000000.
Proof.
  intros M m L k mv rm HM Hamp Hx Hreg Hrng.
  destruct (ok_log_unit M m L k mv rm HM Hamp Hx Hreg) as [_ Hd].
  pose proof (log_unit_vs_true rm ltac:(lra)) as Ht.
  eapply Rle_trans; [apply (abs_split _ (Rlog_unit rm)) | lra].
Qed.

(** Square root needs no series: Flocq's [Bsqrt] is correctly rounded, so the
    extracted function is within half an ulp of the mathematical square root of
    the value it is given. *)
Theorem ok_sqrt_true : forall x : binary32,
  Rabs (B2R (f32_sqrt x) - sqrt (B2R x)) <= / 2 * f32_ulp (sqrt (B2R x)).
Proof. exact f32_sqrt_error. Qed.

(** * Sigmoid, tanh, SiLU, softplus and GELU

    These compose the exponential, and softplus the logarithm as well, so their
    distance to the mathematical functions follows from the bounds above. GELU
    is compared with the form GPT-2 is trained with; its two constants are
    stored as [6693141/2^23] and [12003091/2^28]. *)

Definition sigmoid (x : R) : R := 1 / (1 + exp (- x)).

Definition gelu (x : R) : R :=
  x / 2 * (1 + tanh (sqrt (2 / PI) * (x + 44715 / 1000000 * x ^ 3))).

Lemma recip_one_plus : forall a b d : R,
  0 < b -> 0 <= d <= 1 -> Rabs (a - b) <= b * d ->
  Rabs (1 / (1 + a) - 1 / (1 + b)) <= d.
Proof.
  intros a b d Hb Hd Hab.
  apply Rabs_le_inv in Hab.
  assert (Ha : 0 <= a) by nra.
  assert (Hp : 0 < (1 + a) * (1 + b)) by nra.
  assert (Hpb : b <= (1 + a) * (1 + b)) by nra.
  assert (Hdd : b * d <= d * ((1 + a) * (1 + b))).
  { rewrite (Rmult_comm b d). apply Rmult_le_compat_l; lra. }
  replace (1 / (1 + a) - 1 / (1 + b)) with ((b - a) * / ((1 + a) * (1 + b)))
    by (field; repeat split; apply Rgt_not_eq; lra).
  apply Rabs_le. split;
    apply (Rmult_le_reg_r ((1 + a) * (1 + b))); try exact Hp;
    rewrite Rmult_assoc, Rinv_l, Rmult_1_r by (apply Rgt_not_eq; exact Hp); lra.
Qed.

Theorem sigmoid_vs_true : forall rx : R,
  -88 <= rx <= 88 -> Rabs (Rsigmoid rx - sigmoid rx) <= 16 / 1000000000.
Proof.
  intros rx Hr. unfold Rsigmoid, sigmoid.
  pose proof (exp_approx_vs_true (- rx)) as H.
  rewrite Rsat_id in H by lra.
  apply recip_one_plus; [apply exp_pos | lra | exact H].
Qed.

Lemma tanh_as_sigmoid : forall y : R, tanh y = 2 * sigmoid (2 * y) - 1.
Proof.
  intros y. unfold tanh, sinh, cosh, sigmoid.
  rewrite !exp_Ropp.
  replace (exp (2 * y)) with (exp y * exp y) by (rewrite <- exp_plus; f_equal; ring).
  pose proof (exp_pos y) as Hp.
  field; repeat split; apply Rgt_not_eq; nra.
Qed.

Theorem tanh_vs_true : forall ry : R,
  -44 <= ry <= 44 -> Rabs (Rtanh ry - tanh ry) <= 32 / 1000000000.
Proof.
  intros ry Hr. unfold Rtanh. rewrite B2R_two, tanh_as_sigmoid.
  pose proof (sigmoid_vs_true (2 * ry) ltac:(lra)) as H.
  replace (2 * Rsigmoid (2 * ry) - 1 - (2 * sigmoid (2 * ry) - 1))
    with (2 * (Rsigmoid (2 * ry) - sigmoid (2 * ry))) by ring.
  rewrite Rabs_mult, (Rabs_pos_eq 2) by lra. lra.
Qed.

Theorem silu_vs_true : forall rx : R,
  -88 <= rx <= 88 ->
  Rabs (Rsilu rx - rx * sigmoid rx) <= Rabs rx * (16 / 1000000000).
Proof.
  intros rx Hr. unfold Rsilu.
  replace (rx * Rsigmoid rx - rx * sigmoid rx) with (rx * (Rsigmoid rx - sigmoid rx))
    by ring.
  rewrite Rabs_mult.
  apply Rmult_le_compat_l; [apply Rabs_pos | apply sigmoid_vs_true, Hr].
Qed.

(** The logarithm moves no more than its argument above one. *)
Lemma ln_le_sub_one : forall t : R, 0 < t -> ln t <= t - 1.
Proof.
  intros t Ht. pose proof (exp_ineq1_le (ln t)) as H.
  rewrite exp_ln in H by exact Ht. lra.
Qed.

Lemma ln_diff_le : forall x y : R, 1 <= x -> 1 <= y -> ln x - ln y <= Rabs (x - y).
Proof.
  intros x y Hx Hy.
  assert (Hiy : 0 < / y) by (apply Rinv_0_lt_compat; lra).
  assert (Hl : ln x - ln y = ln (x * / y)).
  { rewrite ln_mult by lra. rewrite ln_Rinv by lra. ring. }
  rewrite Hl.
  eapply Rle_trans; [apply ln_le_sub_one; apply Rmult_lt_0_compat; lra|].
  replace (x * / y - 1) with ((x - y) * / y) by (field; apply Rgt_not_eq; lra).
  destruct (Rle_or_lt y x) as [Hle|Hlt].
  - rewrite Rabs_pos_eq by lra.
    rewrite <- (Rmult_1_r (x - y)) at 2.
    apply Rmult_le_compat_l; [lra|].
    rewrite <- Rinv_1. apply Rinv_le_contravar; lra.
  - apply Rle_trans with 0; [nra | apply Rabs_pos].
Qed.

Lemma ln_lipschitz : forall x y : R,
  1 <= x -> 1 <= y -> Rabs (ln x - ln y) <= Rabs (x - y).
Proof.
  intros x y Hx Hy. apply Rabs_le. split.
  - pose proof (ln_diff_le y x Hy Hx) as H. rewrite Rabs_minus_sym in H. lra.
  - apply ln_diff_le; assumption.
Qed.

Lemma softplus_stable : forall x : R,
  Rmax x 0 + ln (1 + exp (- Rabs x)) = ln (1 + exp x).
Proof.
  intros x. destruct (Rle_or_lt 0 x) as [H|H].
  - rewrite Rmax_left by lra. rewrite Rabs_pos_eq by lra.
    pose proof (exp_pos x) as Hx. pose proof (exp_pos (- x)) as Hnx.
    rewrite <- (ln_exp x) at 1.
    rewrite <- ln_mult by lra.
    f_equal. rewrite Rmult_plus_distr_l, Rmult_1_r, <- exp_plus.
    replace (x + - x) with 0 by ring. rewrite exp_0. ring.
  - rewrite Rmax_right by lra. rewrite Rabs_left by lra.
    rewrite Ropp_involutive. ring.
Qed.

Lemma exp_le_one : forall t : R, 0 <= t -> exp (- t) <= 1.
Proof.
  intros t Ht. pose proof (exp_ineq1_le t) as H.
  rewrite exp_Ropp, <- Rinv_1. apply Rinv_le_contravar; lra.
Qed.

Theorem softplus_vs_true : forall rx : R,
  -88 <= rx <= 88 -> Rabs (Rsoftplus rx - ln (1 + exp rx)) <= 4 / 100000000.
Proof.
  intros rx Hr. unfold Rsoftplus. rewrite <- softplus_stable.
  assert (Hax : Rabs rx <= 88) by (apply Rabs_le; lra).
  pose proof (Rabs_pos rx) as Hap.
  pose proof (exp_approx_vs_true (- Rabs rx)) as H.
  rewrite Rsat_id in H by lra.
  pose proof (exp_pos (- Rabs rx)) as Hb0.
  pose proof (exp_le_one (Rabs rx) Hap) as Hb1.
  set (a := Re_exp_approx (- Rabs rx)) in *.
  set (b := exp (- Rabs rx)) in *.
  assert (Hab : Rabs (a - b) <= 16 / 1000000000).
  { eapply Rle_trans; [exact H|]. nra. }
  apply Rabs_le_inv in H.
  assert (Hlog : Rabs (Rlog_unit (1 + a) - ln (1 + a)) <= 2 / 100000000)
    by (apply log_unit_vs_true; nra).
  assert (Hln : Rabs (ln (1 + a) - ln (1 + b)) <= 16 / 1000000000).
  { eapply Rle_trans; [apply ln_lipschitz; nra|].
    replace (1 + a - (1 + b)) with (a - b) by ring. exact Hab. }
  replace (Rmax rx 0 + Rlog_unit (1 + a) - (Rmax rx 0 + ln (1 + b)))
    with (Rlog_unit (1 + a) - ln (1 + b)) by ring.
  eapply Rle_trans; [apply (abs_split _ (ln (1 + a))) | lra].
Qed.

(** An exponential near zero, from [1 + w <= exp w] alone. *)
Lemma exp_sub_one_le : forall w : R,
  Rabs w <= / 4 -> Rabs (exp w - 1) <= 2 * Rabs w.
Proof.
  intros w Hw.
  pose proof (exp_ineq1_le w) as H1.
  pose proof (exp_ineq1_le (- w)) as H2.
  pose proof (exp_pos w) as Hp.
  rewrite exp_Ropp in H2.
  assert (H3 : exp w * (1 + - w) <= 1).
  { apply (Rmult_le_compat_l (exp w)) in H2; [|lra].
    rewrite Rinv_r in H2 by (apply Rgt_not_eq; exact Hp). exact H2. }
  apply Rabs_le_inv in Hw.
  destruct (Rle_or_lt 0 w) as [Hw0|Hw0].
  - rewrite (Rabs_pos_eq w) by exact Hw0.
    assert (He : exp w <= 2) by nra.
    apply Rabs_le. split; nra.
  - rewrite (Rabs_left w) by exact Hw0.
    assert (He : exp w <= 1) by nra.
    apply Rabs_le. split; nra.
Qed.

Lemma sigmoid_close : forall u v : R,
  Rabs (u - v) <= / 8 -> Rabs (sigmoid u - sigmoid v) <= 2 * Rabs (u - v).
Proof.
  intros u v Huv. unfold sigmoid.
  pose proof (Rabs_pos (u - v)) as H0.
  apply recip_one_plus; [apply exp_pos | lra |].
  assert (He : exp (- u) = exp (- v) * exp (v - u))
    by (rewrite <- exp_plus; f_equal; ring).
  rewrite He.
  replace (exp (- v) * exp (v - u) - exp (- v)) with (exp (- v) * (exp (v - u) - 1))
    by ring.
  rewrite Rabs_mult, (Rabs_pos_eq (exp (- v))) by (left; apply exp_pos).
  apply Rmult_le_compat_l; [left; apply exp_pos|].
  rewrite (Rabs_minus_sym u v). apply exp_sub_one_le.
  rewrite <- (Rabs_minus_sym u v). lra.
Qed.

Lemma tanh_close : forall s t : R,
  Rabs (s - t) <= / 16 -> Rabs (tanh s - tanh t) <= 8 * Rabs (s - t).
Proof.
  intros s t H. rewrite !tanh_as_sigmoid.
  replace (2 * sigmoid (2 * s) - 1 - (2 * sigmoid (2 * t) - 1))
    with (2 * (sigmoid (2 * s) - sigmoid (2 * t))) by ring.
  assert (H2 : Rabs (2 * s - 2 * t) = 2 * Rabs (s - t)).
  { replace (2 * s - 2 * t) with (2 * (s - t)) by ring.
    rewrite Rabs_mult, (Rabs_pos_eq 2) by lra. reflexivity. }
  pose proof (sigmoid_close (2 * s) (2 * t) ltac:(rewrite H2; lra)) as Hs.
  rewrite H2 in Hs. rewrite Rabs_mult, (Rabs_pos_eq 2) by lra. lra.
Qed.

Lemma B2R_gelu_c1 : B2R f32_gelu_c1 = 6693141 / 8388608.
Proof.
  rewrite Qb_correct.
  rewrite (Qeq_eqR _ (Qmake 6693141 8388608)) by (vm_compute; reflexivity).
  unfold Q2R; cbn [Qnum Qden]; unfold Rdiv; reflexivity.
Qed.

Lemma B2R_gelu_c2 : B2R f32_gelu_c2 = 12003091 / 268435456.
Proof.
  rewrite Qb_correct.
  rewrite (Qeq_eqR _ (Qmake 12003091 268435456)) by (vm_compute; reflexivity).
  unfold Q2R; cbn [Qnum Qden]; unfold Rdiv; reflexivity.
Qed.

Lemma B2R_half : B2R f32_half = 1 / 2.
Proof.
  rewrite Qb_correct.
  rewrite (Qeq_eqR _ (Qmake 1 2)) by (vm_compute; reflexivity).
  unfold Q2R; cbn [Qnum Qden]; unfold Rdiv; reflexivity.
Qed.

Lemma gelu_cubic_bound : forall rx : R,
  Rabs rx <= 8 ->
  Rabs (rx * (rx * rx)) <= 512
  /\ Rabs (rx + 12003091 / 268435456 * (rx * (rx * rx))) <= 31.
Proof.
  intros rx Hr.
  assert (H3 : Rabs (rx * (rx * rx)) <= 512).
  { rewrite !Rabs_mult. pose proof (Rabs_pos rx) as H0.
    replace 512 with (8 * (8 * 8)) by lra.
    apply Rmult_le_compat; [exact H0 | apply Rmult_le_pos; exact H0 | exact Hr |].
    apply Rmult_le_compat; [exact H0 | exact H0 | exact Hr | exact Hr]. }
  split; [exact H3|].
  eapply Rle_trans; [apply Rabs_triang|].
  rewrite (Rabs_mult (12003091 / 268435456)), (Rabs_pos_eq (12003091 / 268435456)) by lra.
  lra.
Qed.

Lemma gelu_inner_close : forall rx : R,
  Rabs rx <= 8 ->
  Rabs (Rgelu_inner rx - sqrt (2 / PI) * (rx + 44715 / 1000000 * rx ^ 3))
  <= 14 / 10000000.
Proof.
  intros rx Hr. unfold Rgelu_inner. rewrite B2R_gelu_c1, B2R_gelu_c2.
  pose proof gelu_c1_bound as Hc1.
  pose proof sqrt_2_pi_le as Hc.
  pose proof (sqrt_pos (2 / PI)) as Hc0.
  destruct (gelu_cubic_bound rx Hr) as [H3 Hs].
  set (C1 := sqrt (2 / PI)) in *.
  set (S := rx + 12003091 / 268435456 * (rx * (rx * rx))) in *.
  set (R3 := rx * (rx * rx)) in *.
  replace (6693141 / 8388608 * S - C1 * (rx + 44715 / 1000000 * rx ^ 3))
    with ((6693141 / 8388608 - C1) * S
          + C1 * ((12003091 / 268435456 - 44715 / 1000000) * R3))
    by (unfold S, R3; ring).
  assert (Hd : Rabs (12003091 / 268435456 - 44715 / 1000000) <= 16 / 10000000000)
    by (apply Rabs_le; lra).
  assert (T1 : Rabs ((6693141 / 8388608 - C1) * S) <= 23 / 1000000000 * 31).
  { rewrite Rabs_mult. apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  assert (T2 : Rabs ((12003091 / 268435456 - 44715 / 1000000) * R3)
               <= 16 / 10000000000 * 512).
  { rewrite Rabs_mult. apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  assert (T3 : Rabs (C1 * ((12003091 / 268435456 - 44715 / 1000000) * R3))
               <= 4 / 5 * (16 / 10000000000 * 512)).
  { rewrite Rabs_mult, (Rabs_pos_eq C1) by exact Hc0.
    apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  eapply Rle_trans; [apply Rabs_triang | lra].
Qed.

Theorem gelu_vs_true : forall rx : R,
  -8 <= rx <= 8 -> Rabs (Rgelu rx - gelu rx) <= 5 / 100000.
Proof.
  intros rx Hr.
  assert (Hra : Rabs rx <= 8) by (apply Rabs_le; lra).
  pose proof (gelu_inner_close rx Hra) as Hin.
  destruct (gelu_cubic_bound rx Hra) as [_ Hs].
  set (t := sqrt (2 / PI) * (rx + 44715 / 1000000 * rx ^ 3)) in *.
  assert (Hsr : -44 <= Rgelu_inner rx <= 44).
  { apply Rabs_le_inv. unfold Rgelu_inner. rewrite B2R_gelu_c1, B2R_gelu_c2.
    rewrite Rabs_mult, (Rabs_pos_eq (6693141 / 8388608)) by lra. lra. }
  pose proof (tanh_vs_true (Rgelu_inner rx) Hsr) as Ht.
  pose proof (tanh_close (Rgelu_inner rx) t ltac:(lra)) as Hc.
  unfold Rgelu, gelu. rewrite B2R_half. fold t.
  replace (1 / 2 * rx * (1 + Rtanh (Rgelu_inner rx)) - rx / 2 * (1 + tanh t))
    with (rx / 2 * ((Rtanh (Rgelu_inner rx) - tanh (Rgelu_inner rx))
                    + (tanh (Rgelu_inner rx) - tanh t))) by field.
  rewrite Rabs_mult.
  assert (Hh : Rabs (rx / 2) <= 4).
  { unfold Rdiv. rewrite Rabs_mult, (Rabs_pos_eq (/ 2)) by lra. lra. }
  assert (Hsum : Rabs ((Rtanh (Rgelu_inner rx) - tanh (Rgelu_inner rx))
                       + (tanh (Rgelu_inner rx) - tanh t))
                 <= 32 / 1000000000 + 8 * (14 / 10000000)).
  { eapply Rle_trans; [apply Rabs_triang | lra]. }
  apply Rle_trans with (4 * (32 / 1000000000 + 8 * (14 / 10000000))); [|lra].
  apply Rmult_le_compat; try apply Rabs_pos; assumption.
Qed.

(** The float functions against the mathematical ones. *)

Theorem ok_sigmoid_true : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx -> sig_reg M m x rx -> -88 <= rx <= 88 ->
  Rabs (B2R (f32_sigmoid x) - sigmoid rx) <= errN M L (k + 23) + 16 / 1000000000.
Proof.
  intros M m L k x rx HM Hamp Hx Hreg Hr.
  destruct (ok_sigmoid M m L k x rx HM Hamp Hx Hreg) as [_ Hd].
  pose proof (sigmoid_vs_true rx Hr) as Ht.
  eapply Rle_trans; [apply (abs_split _ (Rsigmoid rx)) | lra].
Qed.

Theorem ok_tanh_true : forall M m L k y ry,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) y ry -> tanh_reg M m y ry -> -44 <= ry <= 44 ->
  Rabs (B2R (f32_tanh y) - tanh ry) <= errN M L (k + 26) + 32 / 1000000000.
Proof.
  intros M m L k y ry HM Hamp Hy Hreg Hr.
  destruct (ok_tanh M m L k y ry HM Hamp Hy Hreg) as [_ Hd].
  pose proof (tanh_vs_true ry Hr) as Ht.
  eapply Rle_trans; [apply (abs_split _ (Rtanh ry)) | lra].
Qed.

Theorem ok_silu_true : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx -> sig_reg M m x rx ->
  Rabs (B2R x) <= M -> Rabs (Rsigmoid rx) <= M ->
  regz M (B2R x * B2R (f32_sigmoid x)) -> -88 <= rx <= 88 ->
  Rabs (B2R (f32_silu x) - rx * sigmoid rx)
  <= errN M L (S (k + 23)) + Rabs rx * (16 / 1000000000).
Proof.
  intros M m L k x rx HM Hamp Hx Hreg Hbx Hbs Hz Hr.
  destruct (ok_silu M m L k x rx HM Hamp Hx Hreg Hbx Hbs Hz) as [_ Hd].
  pose proof (silu_vs_true rx Hr) as Ht. unfold Rsilu in Ht.
  eapply Rle_trans; [apply (abs_split _ (rx * Rsigmoid rx)) | lra].
Qed.

Theorem ok_softplus_true : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx -> sp_reg M m x rx -> -88 <= rx <= 88 ->
  Rabs (B2R (f32_softplus x) - ln (1 + exp rx))
  <= errN M L (k + 40) + 4 / 100000000.
Proof.
  intros M m L k x rx HM Hamp Hx Hreg Hr.
  destruct (ok_softplus M m L k x rx HM Hamp Hx Hreg) as [_ Hd].
  pose proof (softplus_vs_true rx Hr) as Ht.
  eapply Rle_trans; [apply (abs_split _ (Rsoftplus rx)) | lra].
Qed.

Theorem ok_gelu_true : forall M m L k x rx,
  M < bpow radix2 emax32 -> amp_ok M m L ->
  ok (errN M L k) x rx -> gelu_reg M m x rx -> -8 <= rx <= 8 ->
  Rabs (B2R (f32_gelu x) - gelu rx) <= errN M L (k + 33) + 5 / 100000.
Proof.
  intros M m L k x rx HM Hamp Hx Hreg Hr.
  destruct (ok_gelu M m L k x rx HM Hamp Hx Hreg) as [_ Hd].
  pose proof (gelu_vs_true rx Hr) as Ht.
  eapply Rle_trans; [apply (abs_split _ (Rgelu rx)) | lra].
Qed.
