(** * Enclosures of the reference at high precision

    A value of this arithmetic is a point [m 2^e] held exactly, an interval
    [[lo 2^-FB, hi 2^-FB]] with integer endpoints, or the whole line. A
    constant is a binary32 value and is held as its point. Addition, negation
    and the maximum are exact on the endpoints; products, quotients and square
    roots are computed exactly in integers and rounded outward once; the
    exponential goes through CoqInterval's verified interval exponential. A dot
    product against a row of points scales the row by its least exponent [E],
    so each term is an integer [m 2^(e - E)] times an endpoint, and the sum is
    exact until it is rounded outward once.

    [enc_ops_rel] says that every operation keeps the reference value of
    [real_ops] inside the interval, so by the relational theorem of RunErr.v
    the GPT-2 pass in this arithmetic encloses every logit of the reference. *)

From Stdlib Require Import ZArith Reals Lra Lia List Bool.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
From Interval Require Import Specific_stdz Specific_ops Float_full Interval Xreal Basic.
Require Import Phases1_15_complete Float_error RunErr.

Import ListNotations.
Open Scope R_scope.

Module EF := SpecificFloat StdZRadix2.
Module EI := FloatIntervalFull EF.

(** * Integer scaling, rounded down and up *)

Definition flr_scale (x e : Z) : Z :=
  if (0 <=? e)%Z then Z.shiftl x e else Z.shiftr x (- e).

Definition cel_scale (x e : Z) : Z :=
  if (0 <=? e)%Z then Z.shiftl x e else (- Z.shiftr (- x) (- e))%Z.

Lemma IZR_pow2 : forall k, (0 <= k)%Z -> IZR (2 ^ k) = bpow radix2 k.
Proof. intros [|p|p] Hk; [reflexivity | reflexivity | lia]. Qed.

Lemma shiftr_floor : forall x k, (0 <= k)%Z ->
  IZR (Z.shiftr x k) * bpow radix2 k <= IZR x.
Proof.
  intros x k Hk. rewrite Z.shiftr_div_pow2 by lia.
  rewrite <- IZR_pow2 by lia. rewrite <- mult_IZR. apply IZR_le.
  rewrite Z.mul_comm. apply Z.mul_div_le. apply Z.pow_pos_nonneg; lia.
Qed.

Lemma flr_scale_le : forall x e,
  IZR (flr_scale x e) <= IZR x * bpow radix2 e.
Proof.
  intros x e. unfold flr_scale. destruct (Z.leb_spec 0 e) as [He|He].
  - rewrite Z.shiftl_mul_pow2 by lia. rewrite mult_IZR, IZR_pow2 by lia. lra.
  - pose proof (shiftr_floor x (- e) ltac:(lia)) as H.
    assert (P : 0 < bpow radix2 (- e)) by apply bpow_gt_0.
    replace e with (- - e)%Z at 2 by lia. rewrite bpow_opp.
    apply Rmult_le_reg_r with (bpow radix2 (- e)); [exact P|].
    rewrite Rmult_assoc, Rinv_l by lra. lra.
Qed.

Lemma cel_scale_ge : forall x e,
  IZR x * bpow radix2 e <= IZR (cel_scale x e).
Proof.
  intros x e. unfold cel_scale. destruct (Z.leb_spec 0 e) as [He|He].
  - rewrite Z.shiftl_mul_pow2 by lia. rewrite mult_IZR, IZR_pow2 by lia. lra.
  - pose proof (shiftr_floor (- x) (- e) ltac:(lia)) as H.
    assert (P : 0 < bpow radix2 (- e)) by apply bpow_gt_0.
    rewrite opp_IZR. rewrite opp_IZR in H.
    replace e with (- - e)%Z at 1 by lia. rewrite bpow_opp.
    apply Rmult_le_reg_r with (bpow radix2 (- e)); [exact P|].
    rewrite Rmult_assoc, Rinv_l by lra. lra.
Qed.

(** * Real bounds on products and quotients of intervals *)

Definition Rmin4 (a b c d : R) : R := Rmin (Rmin a b) (Rmin c d).
Definition Rmax4 (a b c d : R) : R := Rmax (Rmax a b) (Rmax c d).

Lemma mul_between : forall a b s r, a <= r <= b ->
  Rmin (a * s) (b * s) <= r * s <= Rmax (a * s) (b * s).
Proof.
  intros a b s r [H1 H2].
  destruct (Rle_dec 0 s) as [Hs|Hs].
  - split; [apply Rle_trans with (a * s); [apply Rmin_l | nra]
           | apply Rle_trans with (b * s); [nra | apply Rmax_r]].
  - split; [apply Rle_trans with (b * s); [apply Rmin_r | nra]
           | apply Rle_trans with (a * s); [nra | apply Rmax_l]].
Qed.

Lemma mul_bounds : forall a b c d r s, a <= r <= b -> c <= s <= d ->
  Rmin4 (a * c) (a * d) (b * c) (b * d) <= r * s
  <= Rmax4 (a * c) (a * d) (b * c) (b * d).
Proof.
  intros a b c d r s Hr Hs.
  destruct (mul_between a b s r Hr) as [L U].
  assert (Ha : Rmin (a * c) (a * d) <= a * s <= Rmax (a * c) (a * d)).
  { rewrite !(Rmult_comm a). apply mul_between. exact Hs. }
  assert (Hb : Rmin (b * c) (b * d) <= b * s <= Rmax (b * c) (b * d)).
  { rewrite !(Rmult_comm b). apply mul_between. exact Hs. }
  unfold Rmin4, Rmax4. split.
  - apply Rle_trans with (Rmin (a * s) (b * s)); [|exact L].
    apply Rmin_glb.
    + apply Rle_trans with (Rmin (a * c) (a * d)); [apply Rmin_l | apply Ha].
    + apply Rle_trans with (Rmin (b * c) (b * d)); [apply Rmin_r | apply Hb].
  - apply Rle_trans with (Rmax (a * s) (b * s)); [exact U|].
    apply Rmax_lub.
    + apply Rle_trans with (Rmax (a * c) (a * d)); [apply Ha | apply Rmax_l].
    + apply Rle_trans with (Rmax (b * c) (b * d)); [apply Hb | apply Rmax_r].
Qed.

Lemma inv_bounds : forall c d s, c <= s <= d -> (0 < c \/ d < 0) ->
  / d <= / s <= / c.
Proof.
  intros c d s [H1 H2] [Hc|Hd].
  - split; apply Rinv_le_contravar; lra.
  - split.
    + apply Ropp_le_cancel. rewrite <- !Rinv_opp. apply Rinv_le_contravar; lra.
    + apply Ropp_le_cancel. rewrite <- !Rinv_opp. apply Rinv_le_contravar; lra.
Qed.


(** * The arithmetic *)

(** The mantissa and exponent of a binary32 value, zero for the others. *)
Definition b32_decomp (x : binary32) : Z * Z :=
  match x with
  | B754_finite s m e _ => (cond_Zopp s (Zpos m), e)
  | _ => (0%Z, 0%Z)
  end.

Lemma b32_decomp_spec : forall x,
  B2R x = IZR (fst (b32_decomp x)) * bpow radix2 (snd (b32_decomp x)).
Proof.
  intros [s|s| |s m e Hb]; cbn [b32_decomp fst snd B2R]; try lra.
  unfold F2R. cbn [Fnum Fexp]. reflexivity.
Qed.

Lemma b32_decomp_exp : forall x, (-149 <= snd (b32_decomp x))%Z.
Proof.
  intros [s|s| |s m e Hb]; cbn [b32_decomp snd]; try lia.
  unfold SpecFloat.bounded in Hb. apply andb_prop in Hb as [Hc _].
  unfold SpecFloat.canonical_mantissa in Hc. apply Z.eqb_eq in Hc.
  unfold SpecFloat.fexp in Hc. rewrite <- Hc. apply Z.le_max_r.
Qed.

(** A value is a point [m 2^e] held exactly, an interval
    [[lo 2^-FB, hi 2^-FB]], or the whole line. *)
Inductive enc := EFail | EPt (m e : Z) | EIv (lo hi : Z).

Section Arith.

Variable FB : Z.
Hypothesis FB_ge : (149 <= FB)%Z.
Variable prec : Z.

Definition sc (z : Z) : R := IZR z * bpow radix2 (- FB).

Definition enc_in (a : enc) (r : R) : Prop :=
  match a with
  | EFail => True
  | EPt m e => r = IZR m * bpow radix2 e
  | EIv lo hi => sc lo <= r <= sc hi
  end.

Definition iv (a : enc) : option (Z * Z) :=
  match a with
  | EFail => None
  | EPt m e => Some (flr_scale m (e + FB), cel_scale m (e + FB))
  | EIv lo hi => Some (lo, hi)
  end.

Definition enc_const (c : binary32) : enc := let (m, e) := b32_decomp c in EPt m e.

Definition enc_plus (a b : enc) : enc :=
  match iv a, iv b with
  | Some (l1, h1), Some (l2, h2) => EIv (l1 + l2) (h1 + h2)
  | _, _ => EFail
  end.

Definition enc_neg (a : enc) : enc :=
  match a with
  | EFail => EFail
  | EPt m e => EPt (- m) e
  | EIv lo hi => EIv (- hi) (- lo)
  end.

Definition enc_max (a b : enc) : enc :=
  match iv a, iv b with
  | Some (l1, h1), Some (l2, h2) => EIv (Z.max l1 l2) (Z.max h1 h2)
  | _, _ => EFail
  end.

(** Products of the endpoints at scale [2^-2FB], rounded out to [2^-FB]. *)
Definition zmin4 (a b c d : Z) : Z := Z.min (Z.min a b) (Z.min c d).
Definition zmax4 (a b c d : Z) : Z := Z.max (Z.max a b) (Z.max c d).

Definition enc_mult (a b : enc) : enc :=
  match iv a, iv b with
  | Some (l1, h1), Some (l2, h2) =>
      let p1 := (l1 * l2)%Z in let p2 := (l1 * h2)%Z in
      let p3 := (h1 * l2)%Z in let p4 := (h1 * h2)%Z in
      EIv (flr_scale (zmin4 p1 p2 p3 p4) (- FB)) (cel_scale (zmax4 p1 p2 p3 p4) (- FB))
  | _, _ => EFail
  end.

(** Quotients [x 2^FB / y] of the endpoints, rounded out, when the divisor
    keeps its sign. *)
Definition zdiv_dn (x y : Z) : Z := Z.div (x * 2 ^ FB) y.
Definition zdiv_up (x y : Z) : Z := (- Z.div (- (x * 2 ^ FB)) y)%Z.

Definition enc_div (a b : enc) : enc :=
  match iv a, iv b with
  | Some (x1, x2), Some (y1, y2) =>
      if (0 <? y1)%Z || (y2 <? 0)%Z then
        EIv (zmin4 (zdiv_dn x1 y1) (zdiv_dn x1 y2) (zdiv_dn x2 y1) (zdiv_dn x2 y2))
            (zmax4 (zdiv_up x1 y1) (zdiv_up x1 y2) (zdiv_up x2 y1) (zdiv_up x2 y2))
      else EFail
  | _, _ => EFail
  end.

(** The square root; the reference takes [sqrt] of a negative number to 0. *)
Definition zsqrt_up (n : Z) : Z := let s := Z.sqrt n in if (s * s =? n)%Z then s else (s + 1)%Z.

Definition enc_sqrt (a : enc) : enc :=
  match iv a with
  | Some (lo, hi) =>
      if (hi <? 0)%Z then EIv 0 0
      else EIv (if (lo <=? 0)%Z then 0%Z else Z.sqrt (lo * 2 ^ FB)) (zsqrt_up (hi * 2 ^ FB))
  | None => EFail
  end.

(** The exponential of the argument saturated to [[-88, 88]], through
    CoqInterval. *)
Definition sat88 (z : Z) : Z := Z.max (- 88 * 2 ^ FB) (Z.min (88 * 2 ^ FB) z).

Definition of_interval (J : EI.type) : enc :=
  match J with
  | Float.Ibnd (Specific_ops.Float ml el) (Specific_ops.Float mu eu) =>
      EIv (flr_scale ml (el + FB)) (cel_scale mu (eu + FB))
  | _ => EFail
  end.

Definition to_interval (lo hi : Z) : EI.type :=
  Float.Ibnd (Specific_ops.Float lo (- FB)%Z) (Specific_ops.Float hi (- FB)%Z).

Definition enc_exp (a : enc) : enc :=
  match iv a with
  | Some (lo, hi) => of_interval (EI.exp prec (to_interval (sat88 lo) (sat88 hi)))
  | None => EFail
  end.

(** A dot product against a row of points [m_i 2^e_i]. With [E] at most every
    [e_i], each term is the integer [m_i 2^(e_i - E)] times an endpoint at
    scale [2^(E - FB)], so the sum is exact and is rounded out once. *)
Fixpoint row_emin (xs : list enc) (E : Z) : Z :=
  match xs with
  | EPt _ e :: xs' => row_emin xs' (Z.min E e)
  | _ :: xs' => row_emin xs' E
  | [] => E
  end.

Fixpoint enc_dot_pts (E : Z) (xs ys : list enc) (lo hi : Z) : option (Z * Z) :=
  match xs, ys with
  | x :: xs', y :: ys' =>
      match x, iv y with
      | EPt m e, Some (l, h) =>
          if (E <=? e)%Z then
            let w := Z.shiftl m (e - E) in
            if (0 <=? w)%Z then enc_dot_pts E xs' ys' (lo + w * l) (hi + w * h)
            else enc_dot_pts E xs' ys' (lo + w * h) (hi + w * l)
          else None
      | _, _ => None
      end
  | _, _ => Some (lo, hi)
  end.

(** Otherwise the terms are accumulated in the order [f32_dot] takes. *)
Fixpoint enc_dot_aux (xs ys : list enc) (acc : enc) : enc :=
  match xs, ys with
  | x :: xs', y :: ys' => enc_dot_aux xs' ys' (enc_plus acc (enc_mult x y))
  | _, _ => acc
  end.

Definition enc_dot_row (xs ys : list enc) : option enc :=
  let E := row_emin xs 0 in
  match enc_dot_pts E xs ys 0 0 with
  | Some (lo, hi) => Some (EIv (flr_scale lo E) (cel_scale hi E))
  | None => None
  end.

(** The row of points may be either operand. *)
Definition enc_dot (xs ys : list enc) : enc :=
  match enc_dot_row xs ys with
  | Some r => r
  | None =>
      match enc_dot_row ys xs with
      | Some r => r
      | None => enc_dot_aux xs ys (enc_const f32_zero)
      end
  end.

Definition enc_ops : ops enc :=
  mk_ops enc enc_const enc_plus enc_neg enc_mult enc_div enc_sqrt enc_max enc_exp enc_dot.

(** * Every operation keeps the reference inside *)

Lemma bpow_mFB_pos : 0 < bpow radix2 (- FB).
Proof. apply bpow_gt_0. Qed.

Definition sc2 (z : Z) : R := IZR z * bpow radix2 (- FB) * bpow radix2 (- FB).

Lemma sc_plus : forall a b, sc (a + b) = sc a + sc b.
Proof. intros. unfold sc. rewrite plus_IZR. ring. Qed.

Lemma sc_opp : forall a, sc (- a) = - sc a.
Proof. intros. unfold sc. rewrite opp_IZR. ring. Qed.

Lemma sc_mult : forall a b, sc a * sc b = sc2 (a * b).
Proof. intros. unfold sc, sc2. rewrite mult_IZR. ring. Qed.

Lemma sc_flr : forall z, sc (flr_scale z (- FB)) <= sc2 z.
Proof.
  intros z. unfold sc, sc2. pose proof (flr_scale_le z (- FB)) as H.
  apply Rmult_le_compat_r; [apply bpow_ge_0 | exact H].
Qed.

Lemma sc_cel : forall z, sc2 z <= sc (cel_scale z (- FB)).
Proof.
  intros z. unfold sc, sc2. pose proof (cel_scale_ge z (- FB)) as H.
  apply Rmult_le_compat_r; [apply bpow_ge_0 | exact H].
Qed.

Lemma min_scale : forall a b K, 0 < K -> Rmin a b * K = Rmin (a * K) (b * K).
Proof. intros a b K HK. unfold Rmin. destruct (Rle_dec a b), (Rle_dec (a * K) (b * K)); nra. Qed.

Lemma max_scale : forall a b K, 0 < K -> Rmax a b * K = Rmax (a * K) (b * K).
Proof. intros a b K HK. unfold Rmax. destruct (Rle_dec a b), (Rle_dec (a * K) (b * K)); nra. Qed.

Lemma IZR_zmin : forall a b, IZR (Z.min a b) = Rmin (IZR a) (IZR b).
Proof.
  intros a b. destruct (Z.min_spec a b) as [[H E]|[H E]]; rewrite E.
  - rewrite Rmin_left; [reflexivity | apply IZR_le; lia].
  - rewrite Rmin_right; [reflexivity | apply IZR_le; lia].
Qed.

Lemma IZR_zmax : forall a b, IZR (Z.max a b) = Rmax (IZR a) (IZR b).
Proof.
  intros a b. destruct (Z.max_spec a b) as [[H E]|[H E]]; rewrite E.
  - rewrite Rmax_right; [reflexivity | apply IZR_le; lia].
  - rewrite Rmax_left; [reflexivity | apply IZR_le; lia].
Qed.

Lemma K2_pos : 0 < bpow radix2 (- FB) * bpow radix2 (- FB).
Proof. pose proof bpow_mFB_pos. nra. Qed.

Lemma sc2_min4 : forall a b c d,
  sc2 (zmin4 a b c d) = Rmin4 (sc2 a) (sc2 b) (sc2 c) (sc2 d).
Proof.
  intros. unfold sc2, zmin4, Rmin4. rewrite !IZR_zmin, !Rmult_assoc.
  pose proof K2_pos as K. rewrite !min_scale by exact K. reflexivity.
Qed.

Lemma sc2_max4 : forall a b c d,
  sc2 (zmax4 a b c d) = Rmax4 (sc2 a) (sc2 b) (sc2 c) (sc2 d).
Proof.
  intros. unfold sc2, zmax4, Rmax4. rewrite !IZR_zmax, !Rmult_assoc.
  pose proof K2_pos as K. rewrite !max_scale by exact K. reflexivity.
Qed.

Lemma sc_zmax : forall a b, sc (Z.max a b) = Rmax (sc a) (sc b).
Proof. intros. unfold sc. rewrite IZR_zmax. apply max_scale, bpow_mFB_pos. Qed.

Lemma sc_zmin : forall a b, sc (Z.min a b) = Rmin (sc a) (sc b).
Proof. intros. unfold sc. rewrite IZR_zmin. apply min_scale, bpow_mFB_pos. Qed.

Lemma sc_scale : forall m e, sc (flr_scale m (e + FB)) <= IZR m * bpow radix2 e
                             <= sc (cel_scale m (e + FB)).
Proof.
  intros m e. unfold sc. pose proof bpow_mFB_pos as P.
  replace (IZR m * bpow radix2 e) with (IZR m * bpow radix2 (e + FB) * bpow radix2 (- FB))
    by (rewrite Rmult_assoc, <- bpow_plus; f_equal; f_equal; lia).
  split; apply Rmult_le_compat_r; try lra; [apply flr_scale_le | apply cel_scale_ge].
Qed.

Lemma iv_in : forall a r lo hi, enc_in a r -> iv a = Some (lo, hi) -> sc lo <= r <= sc hi.
Proof.
  intros [|m e|l h] r lo hi Ha E; cbn [iv] in E; try discriminate; injection E as <- <-.
  - cbn [enc_in] in Ha. rewrite Ha. apply sc_scale.
  - exact Ha.
Qed.

Lemma iv_le : forall a r lo hi, enc_in a r -> iv a = Some (lo, hi) -> (lo <= hi)%Z.
Proof.
  intros a r lo hi Ha E. destruct (iv_in a r lo hi Ha E) as [H1 H2].
  apply le_IZR. apply Rmult_le_reg_r with (bpow radix2 (- FB)); [apply bpow_mFB_pos|].
  unfold sc in H1, H2. lra.
Qed.

Lemma enc_const_in : forall c, enc_in (enc_const c) (B2R c).
Proof.
  intros c. unfold enc_const. pose proof (b32_decomp_spec c) as Hs.
  destruct (b32_decomp c) as [m e]. exact Hs.
Qed.

Ltac iv_cases a b :=
  destruct (iv a) as [[l1 h1]|] eqn:Ea; [|exact I];
  destruct (iv b) as [[l2 h2]|] eqn:Eb; [|exact I].

Lemma enc_plus_in : forall a b r s, enc_in a r -> enc_in b s -> enc_in (enc_plus a b) (r + s).
Proof.
  intros a b r s Ha Hb. unfold enc_plus. iv_cases a b.
  destruct (iv_in a r l1 h1 Ha Ea), (iv_in b s l2 h2 Hb Eb).
  cbn [enc_in]. rewrite !sc_plus. lra.
Qed.

Lemma enc_neg_in : forall a r, enc_in a r -> enc_in (enc_neg a) (- r).
Proof.
  intros [|m e|l h] r Ha; cbn [enc_neg enc_in] in *; [exact I | |].
  - rewrite Ha, opp_IZR. ring.
  - rewrite !sc_opp. lra.
Qed.

Lemma enc_max_in : forall a b r s, enc_in a r -> enc_in b s -> enc_in (enc_max a b) (Rmax r s).
Proof.
  intros a b r s Ha Hb. unfold enc_max. iv_cases a b.
  destruct (iv_in a r l1 h1 Ha Ea) as [A1 A2], (iv_in b s l2 h2 Hb Eb) as [B1 B2].
  cbn [enc_in]. rewrite !sc_zmax.
  unfold Rmax. destruct (Rle_dec (sc l1) (sc l2)), (Rle_dec r s), (Rle_dec (sc h1) (sc h2)); lra.
Qed.

Lemma enc_mult_in : forall a b r s, enc_in a r -> enc_in b s -> enc_in (enc_mult a b) (r * s).
Proof.
  intros a b r s Ha Hb. unfold enc_mult. iv_cases a b.
  pose proof (iv_in a r l1 h1 Ha Ea) as A. pose proof (iv_in b s l2 h2 Hb Eb) as B.
  cbn [enc_in].
  pose proof (mul_bounds _ _ _ _ _ _ A B) as [L U].
  rewrite !sc_mult in L, U. rewrite <- sc2_min4 in L. rewrite <- sc2_max4 in U.
  split.
  - eapply Rle_trans; [apply sc_flr | exact L].
  - eapply Rle_trans; [exact U | apply sc_cel].
Qed.

(** Integer division rounds down; the negation of the quotient of the
    negation rounds up. *)
Lemma zdiv_le : forall n y, y <> 0%Z -> IZR (n / y) <= IZR n / IZR y.
Proof.
  intros n y Hy. pose proof (Z.div_mod n y Hy) as D.
  assert (Ry : IZR y <> 0) by (apply not_0_IZR; exact Hy).
  destruct (Z.lt_ge_cases 0 y) as [Pos|Neg].
  - pose proof (Z.mod_pos_bound n y Pos) as [M1 M2].
    apply Rmult_le_reg_r with (IZR y); [apply IZR_lt; exact Pos|].
    unfold Rdiv. rewrite Rmult_assoc, Rinv_l by exact Ry. rewrite Rmult_1_r.
    rewrite <- mult_IZR. apply IZR_le. lia.
  - assert (Neg' : (y < 0)%Z) by lia.
    pose proof (Z.mod_neg_bound n y Neg') as [M1 M2].
    apply Ropp_le_cancel.
    apply Rmult_le_reg_r with (- IZR y); [apply Ropp_0_gt_lt_contravar; apply IZR_lt; exact Neg'|].
    unfold Rdiv. replace (- (IZR n * / IZR y) * - IZR y) with (IZR n)
      by (field; exact Ry).
    replace (- IZR (n / y) * - IZR y) with (IZR (y * (n / y))) by (rewrite mult_IZR; ring).
    apply IZR_le. lia.
Qed.

Lemma zdiv_up_ge : forall n y, y <> 0%Z -> IZR n / IZR y <= IZR (- (- n / y)).
Proof.
  intros n y Hy. pose proof (zdiv_le (- n) y Hy) as H.
  rewrite opp_IZR in H. rewrite opp_IZR.
  unfold Rdiv in *. lra.
Qed.

Lemma sc_div : forall x y, y <> 0%Z -> sc x / sc y = IZR (x * 2 ^ FB) / IZR y * bpow radix2 (- FB).
Proof.
  intros x y Hy. unfold sc. assert (Ry : IZR y <> 0) by (apply not_0_IZR; exact Hy).
  pose proof bpow_mFB_pos as P.
  rewrite mult_IZR, IZR_pow2 by lia.
  replace (bpow radix2 FB) with (/ bpow radix2 (- FB)) by (rewrite bpow_opp, Rinv_inv; reflexivity).
  field. split; lra.
Qed.

Lemma enc_div_in : forall a b r s, enc_in a r -> enc_in b s -> enc_in (enc_div a b) (r / s).
Proof.
  intros a b r s Ha Hb. unfold enc_div. iv_cases a b.
  destruct ((0 <? l2)%Z || (h2 <? 0)%Z) eqn:Sg; [|exact I].
  pose proof (iv_in a r l1 h1 Ha Ea) as A. pose proof (iv_in b s l2 h2 Hb Eb) as B.
  pose proof (iv_le b s l2 h2 Hb Eb) as LE.
  cbn [enc_in].
  apply orb_prop in Sg.
  assert (Nz : (l2 <> 0 /\ h2 <> 0)%Z)
    by (destruct Sg as [S|S]; [apply Z.ltb_lt in S | apply Z.ltb_lt in S]; split; lia).
  assert (SgR : 0 < sc l2 \/ sc h2 < 0).
  { unfold sc. pose proof bpow_mFB_pos as P.
    destruct Sg as [S|S]; apply Z.ltb_lt in S; [left | right].
    - apply Rmult_lt_0_compat; [apply IZR_lt; exact S | exact P].
    - assert (IZR h2 < 0) by (apply IZR_lt; exact S). nra. }
  pose proof (inv_bounds _ _ _ B SgR) as Iv.
  pose proof (mul_bounds _ _ _ _ _ _ A Iv) as [L U].
  unfold Rdiv. unfold Rdiv in L, U.
  assert (Q : forall x y, y <> 0%Z ->
            sc (zdiv_dn x y) <= sc x * / sc y /\ sc x * / sc y <= sc (zdiv_up x y)).
  { intros x y Hy. rewrite <- (Rdiv_def (sc x) (sc y)), (sc_div x y Hy).
    unfold zdiv_dn, zdiv_up, sc. pose proof bpow_mFB_pos as P. split.
    - apply Rmult_le_compat_r; [lra | apply zdiv_le; exact Hy].
    - apply Rmult_le_compat_r; [lra | apply zdiv_up_ge; exact Hy]. }
  destruct Nz as [N1 N2].
  destruct (Q l1 l2 N1) as [Q1 Q1'], (Q l1 h2 N2) as [Q2 Q2'],
           (Q h1 l2 N1) as [Q3 Q3'], (Q h1 h2 N2) as [Q4 Q4'].
  split.
  - unfold zmin4. rewrite sc_zmin, sc_zmin, sc_zmin.
    eapply Rle_trans; [|exact L]. unfold Rmin4.
    apply Rmin_glb; apply Rmin_glb.
    + eapply Rle_trans; [apply Rmin_l|]. eapply Rle_trans; [apply Rmin_r|]. exact Q2.
    + eapply Rle_trans; [apply Rmin_l|]. eapply Rle_trans; [apply Rmin_l|]. exact Q1.
    + eapply Rle_trans; [apply Rmin_r|]. eapply Rle_trans; [apply Rmin_r|]. exact Q4.
    + eapply Rle_trans; [apply Rmin_r|]. eapply Rle_trans; [apply Rmin_l|]. exact Q3.
  - unfold zmax4. rewrite sc_zmax, sc_zmax, sc_zmax.
    eapply Rle_trans; [exact U|]. unfold Rmax4.
    apply Rmax_lub; apply Rmax_lub.
    + eapply Rle_trans; [|apply Rmax_l]. eapply Rle_trans; [|apply Rmax_r]. exact Q2'.
    + eapply Rle_trans; [|apply Rmax_l]. eapply Rle_trans; [|apply Rmax_l]. exact Q1'.
    + eapply Rle_trans; [|apply Rmax_r]. eapply Rle_trans; [|apply Rmax_r]. exact Q4'.
    + eapply Rle_trans; [|apply Rmax_r]. eapply Rle_trans; [|apply Rmax_l]. exact Q3'.
Qed.

(** The square root. *)
Lemma sqrt_sc : forall z, (0 <= z)%Z ->
  sqrt (sc z) = sqrt (IZR (z * 2 ^ FB)) * bpow radix2 (- FB).
Proof.
  intros z Hz. pose proof bpow_mFB_pos as P.
  assert (E : sc z = IZR (z * 2 ^ FB) * (bpow radix2 (- FB) * bpow radix2 (- FB))).
  { unfold sc. rewrite mult_IZR, IZR_pow2 by lia.
    replace (bpow radix2 FB) with (/ bpow radix2 (- FB)) by (rewrite bpow_opp, Rinv_inv; reflexivity).
    field. lra. }
  rewrite E, sqrt_mult_alt.
  - rewrite sqrt_square by lra. reflexivity.
  - apply IZR_le. apply Z.mul_nonneg_nonneg; [exact Hz | apply Z.pow_nonneg; lia].
Qed.

Lemma zsqrt_le : forall n, (0 <= n)%Z -> IZR (Z.sqrt n) <= sqrt (IZR n).
Proof.
  intros n Hn. pose proof (Z.sqrt_spec n Hn) as [H1 _].
  pose proof (Z.sqrt_nonneg n) as H0.
  rewrite <- (sqrt_square (IZR (Z.sqrt n))) by (apply IZR_le; exact H0).
  apply sqrt_le_1_alt. rewrite <- mult_IZR. apply IZR_le. exact H1.
Qed.

Lemma zsqrt_up_ge : forall n, (0 <= n)%Z -> sqrt (IZR n) <= IZR (zsqrt_up n).
Proof.
  intros n Hn. unfold zsqrt_up. pose proof (Z.sqrt_spec n Hn) as [H1 H2].
  pose proof (Z.sqrt_nonneg n) as H0.
  destruct (Z.eqb_spec (Z.sqrt n * Z.sqrt n) n) as [E|E].
  - replace (IZR n) with (IZR (Z.sqrt n) * IZR (Z.sqrt n)) by (rewrite <- mult_IZR, E; reflexivity).
    rewrite sqrt_square by (apply IZR_le; exact H0). lra.
  - rewrite <- (sqrt_square (IZR (Z.sqrt n + 1))) by (apply IZR_le; lia).
    apply sqrt_le_1_alt. rewrite <- mult_IZR. apply IZR_le. lia.
Qed.

Lemma enc_sqrt_in : forall a r, enc_in a r -> enc_in (enc_sqrt a) (sqrt r).
Proof.
  intros a r Ha. unfold enc_sqrt.
  destruct (iv a) as [[lo hi]|] eqn:Ea; [|exact I].
  destruct (iv_in a r lo hi Ha Ea) as [A1 A2]. pose proof bpow_mFB_pos as P.
  destruct (Z.ltb_spec hi 0) as [Hn|Hn]; cbn [enc_in].
  - assert (Hr : r < 0).
    { apply Rle_lt_trans with (sc hi); [exact A2|].
      unfold sc. assert (IZR hi < 0) by (apply IZR_lt; exact Hn). nra. }
    rewrite sqrt_neg_0 by lra. unfold sc. rewrite Rmult_0_l. lra.
  - split.
    + destruct (Z.leb_spec lo 0) as [Hl|Hl].
      * unfold sc. rewrite Rmult_0_l. apply sqrt_pos.
      * eapply Rle_trans; [|apply sqrt_le_1_alt; exact A1].
        rewrite sqrt_sc by lia. unfold sc.
        apply Rmult_le_compat_r; [lra|]. apply zsqrt_le.
        apply Z.mul_nonneg_nonneg; [lia | apply Z.pow_nonneg; lia].
    + eapply Rle_trans; [apply sqrt_le_1_alt; exact A2|].
      rewrite sqrt_sc by lia. unfold sc.
      apply Rmult_le_compat_r; [lra|]. apply zsqrt_up_ge.
      apply Z.mul_nonneg_nonneg; [lia | apply Z.pow_nonneg; lia].
Qed.

(** The saturated exponential. *)
Lemma toX_Float : forall m e,
  EF.toX (Specific_ops.Float m e) = Xreal (IZR m * bpow radix2 e).
Proof.
  intros m e. unfold EF.toX, EF.toF. cbn.
  destruct m as [|p|p]; cbn.
  - f_equal. ring.
  - f_equal. rewrite FtoR_split. reflexivity.
  - f_equal. rewrite FtoR_split. reflexivity.
Qed.

Lemma sc_sat : forall z, sc (sat88 z) = Rsat (sc z).
Proof.
  intros z. unfold sat88, Rsat. rewrite sc_zmax, sc_zmin.
  assert (E : sc (88 * 2 ^ FB) = 88).
  { unfold sc. rewrite mult_IZR, IZR_pow2 by lia. rewrite Rmult_assoc, <- bpow_plus.
    replace (FB + - FB)%Z with 0%Z by lia. cbn. ring. }
  assert (E' : sc (- 88 * 2 ^ FB) = -88).
  { replace (- 88 * 2 ^ FB)%Z with (- (88 * 2 ^ FB))%Z by ring. rewrite sc_opp, E. reflexivity. }
  rewrite E, E'. reflexivity.
Qed.

Lemma Rsat_mono : forall x y, x <= y -> Rsat x <= Rsat y.
Proof.
  intros x y H. unfold Rsat.
  assert (M : Rmin 88 x <= Rmin 88 y)
    by (unfold Rmin; destruct (Rle_dec 88 x), (Rle_dec 88 y); lra).
  unfold Rmax. destruct (Rle_dec (-88) (Rmin 88 x)), (Rle_dec (-88) (Rmin 88 y)); lra.
Qed.

Lemma of_interval_in : forall J y, contains (EI.convert J) (Xreal y) -> enc_in (of_interval J) y.
Proof.
  intros J y HJ. unfold of_interval.
  destruct J as [|l u]; [exact I|].
  destruct l as [|ml el]; [exact I|]. destruct u as [|mu eu]; [exact I|].
  cbn [EI.convert] in HJ. rewrite !toX_Float in HJ. cbn in HJ.
  destruct HJ as [H1 H2]. cbn [enc_in].
  pose proof (sc_scale ml el) as [S1 _]. pose proof (sc_scale mu eu) as [_ S2].
  split; lra.
Qed.

Lemma enc_exp_in : forall a r, enc_in a r -> enc_in (enc_exp a) (exp (Rsat r)).
Proof.
  intros a r Ha. unfold enc_exp.
  destruct (iv a) as [[lo hi]|] eqn:Ea; [|exact I].
  destruct (iv_in a r lo hi Ha Ea) as [A1 A2].
  apply of_interval_in.
  apply (EI.exp_correct prec _ (Xreal (Rsat r))).
  unfold to_interval. cbn [EI.convert].
  rewrite !toX_Float. cbn. fold (sc (sat88 lo)). fold (sc (sat88 hi)).
  rewrite !sc_sat. split; apply Rsat_mono; assumption.
Qed.

(** The dot products. *)
Lemma sc2_zero : sc2 0 = 0.
Proof. unfold sc2. ring. Qed.

Lemma sc2_plus : forall a b, sc2 (a + b) = sc2 a + sc2 b.
Proof. intros. unfold sc2. rewrite plus_IZR. ring. Qed.

Definition scE (E z : Z) : R := IZR z * bpow radix2 (E - FB).

Lemma scE_plus : forall E a b, scE E (a + b) = scE E a + scE E b.
Proof. intros. unfold scE. rewrite plus_IZR. ring. Qed.

Lemma scE_wl : forall E w l, scE E (w * l) = IZR w * bpow radix2 E * sc l.
Proof.
  intros. unfold scE, sc. rewrite mult_IZR.
  replace (E - FB)%Z with (E + - FB)%Z by lia. rewrite bpow_plus. ring.
Qed.

Lemma shiftl_val : forall m e E, (E <= e)%Z ->
  IZR (Z.shiftl m (e - E)) * bpow radix2 E = IZR m * bpow radix2 e.
Proof.
  intros m e E H. rewrite Z.shiftl_mul_pow2 by lia.
  rewrite mult_IZR, IZR_pow2 by lia. rewrite Rmult_assoc, <- bpow_plus.
  f_equal. f_equal. lia.
Qed.

Lemma dot_pts_in : forall E xs rs, Forall2 enc_in xs rs ->
  forall ys ss lo0 hi0 lo hi acc, Forall2 enc_in ys ss ->
  enc_dot_pts E xs ys lo0 hi0 = Some (lo, hi) ->
  scE E lo0 <= acc <= scE E hi0 ->
  scE E lo <= Rdotr rs ss acc <= scE E hi.
Proof.
  intros E xs rs Hx. induction Hx as [|x r xs rs Hxr Hx IH];
    intros ys ss lo0 hi0 lo hi acc Hy Ed Hacc.
  - cbn in Ed. injection Ed as <- <-. destruct ss; exact Hacc.
  - destruct Hy as [|y s ys ss Hys Hy].
    + cbn in Ed. injection Ed as <- <-. exact Hacc.
    + cbn [enc_dot_pts] in Ed. cbn [Rdotr].
      destruct x as [|m e|l h]; try discriminate.
      destruct (iv y) as [[l h]|] eqn:Ey; [|discriminate].
      destruct (Z.leb_spec E e) as [Hs|Hs]; [|discriminate].
      cbn [enc_in] in Hxr. destruct (iv_in y s l h Hys Ey) as [S1 S2].
      rewrite Hxr, <- (shiftl_val m e E Hs).
      set (w := Z.shiftl m (e - E)) in *.
      assert (Pw : 0 < bpow radix2 E) by apply bpow_gt_0.
      destruct (Z.leb_spec 0 w) as [Hm|Hm].
      * eapply IH; [exact Hy | exact Ed|].
        rewrite !scE_plus, !scE_wl.
        assert (0 <= IZR w * bpow radix2 E)
          by (apply Rmult_le_pos; [apply IZR_le; exact Hm | lra]).
        split; nra.
      * eapply IH; [exact Hy | exact Ed|].
        rewrite !scE_plus, !scE_wl.
        assert (IZR w * bpow radix2 E <= 0).
        { assert (IZR w < 0) by (apply IZR_lt; lia). nra. }
        split; nra.
Qed.

Lemma dot_aux_in : forall xs rs, Forall2 enc_in xs rs ->
  forall ys ss a acc, Forall2 enc_in ys ss -> enc_in a acc ->
  enc_in (enc_dot_aux xs ys a) (Rdotr rs ss acc).
Proof.
  intros xs rs Hx. induction Hx as [|x r xs rs Hxr Hx IH]; intros ys ss a acc Hy Ha.
  - destruct Hy; exact Ha.
  - destruct Hy as [|y s ys ss Hys Hy]; [exact Ha|].
    cbn [enc_dot_aux Rdotr]. apply IH; [exact Hy|].
    apply enc_plus_in; [exact Ha | apply enc_mult_in; assumption].
Qed.

Lemma dot_row_in : forall xs rs ys ss r, Forall2 enc_in xs rs -> Forall2 enc_in ys ss ->
  enc_dot_row xs ys = Some r -> enc_in r (Rdot rs ss).
Proof.
  intros xs rs ys ss r Hx Hy Er. unfold enc_dot_row, Rdot in *.
  set (E := row_emin xs 0) in Er.
  destruct (enc_dot_pts E xs ys 0 0) as [[lo hi]|] eqn:Ed; [|discriminate].
  injection Er as <-. cbn [enc_in].
  pose proof (dot_pts_in E xs rs Hx ys ss 0 0 lo hi 0 Hy Ed) as D.
  assert (Z0 : scE E 0 = 0) by (unfold scE; ring).
  rewrite Z0 in D. destruct (D (conj (Rle_refl 0) (Rle_refl 0))) as [D1 D2].
  assert (Hb : forall z, scE E z = IZR z * bpow radix2 E * bpow radix2 (- FB)).
  { intros z. unfold scE. replace (E - FB)%Z with (E + - FB)%Z by lia.
    rewrite bpow_plus. ring. }
  rewrite !Hb in D1, D2. unfold sc.
  split.
  - eapply Rle_trans; [|exact D1]. apply Rmult_le_compat_r; [apply bpow_ge_0|].
    apply flr_scale_le.
  - eapply Rle_trans; [exact D2|]. apply Rmult_le_compat_r; [apply bpow_ge_0|].
    apply cel_scale_ge.
Qed.

Lemma Rdotr_comm : forall rs ss acc, Rdotr rs ss acc = Rdotr ss rs acc.
Proof.
  induction rs as [|r rs IH]; intros [|s ss] acc; try reflexivity.
  cbn [Rdotr]. rewrite (Rmult_comm r s). apply IH.
Qed.

Lemma enc_dot_in : forall xs rs ys ss, Forall2 enc_in xs rs -> Forall2 enc_in ys ss ->
  enc_in (enc_dot xs ys) (Rdot rs ss).
Proof.
  intros xs rs ys ss Hx Hy. unfold enc_dot.
  destruct (enc_dot_row xs ys) as [r|] eqn:E1; [exact (dot_row_in xs rs ys ss r Hx Hy E1)|].
  destruct (enc_dot_row ys xs) as [r|] eqn:E2.
  - unfold Rdot. rewrite Rdotr_comm. exact (dot_row_in ys ss xs rs r Hy Hx E2).
  - unfold Rdot. apply dot_aux_in; [exact Hx | exact Hy |].
    replace 0 with (B2R f32_zero) by reflexivity. apply enc_const_in.
Qed.

Theorem enc_ops_rel : ops_rel enc_ops real_ops enc_in.
Proof.
  constructor; cbn [op_const op_plus op_neg op_mult op_div op_sqrt op_max op_exp op_dot
                    enc_ops real_ops].
  - apply enc_const_in.
  - intros. apply enc_plus_in; assumption.
  - intros. apply enc_neg_in; assumption.
  - intros. apply enc_mult_in; assumption.
  - intros. apply enc_div_in; assumption.
  - intros. apply enc_sqrt_in; assumption.
  - intros. apply enc_max_in; assumption.
  - intros. apply enc_exp_in; assumption.
  - intros. apply enc_dot_in; assumption.
Qed.

End Arith.
