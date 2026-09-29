(** * The elementary functions against mathematics, over every binary32 input

    Each theorem bounds the distance from an extracted binary32 function to the
    mathematical function it computes, for every binary32 input of a domain,
    the rounding of the float evaluation included. The proof runs the interval
    checker of ElemCheck.v by [vm_compute] over a cover of the domain by
    subintervals, split where the check needs it. Within a subinterval the
    integer an argument reduction selects is constant: it is monotone in the
    argument and is checked equal at the two ends. The checker bounds the
    float evaluation against its shadow, the same program in exact real
    arithmetic with that integer, and the series bounds of Series.v take the
    shadow to the function. *)

From Stdlib Require Import ZArith QArith Qreals Reals Lra Lia List Bool.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
From Interval Require Import Specific_stdz Specific_ops Float_full Interval Xreal Basic.
Require Import Phases1_15_complete Float_error Annot Enclose Llama Qwen Series Truth ElemCheck.

Import ListNotations.
Open Scope R_scope.

#[local] Existing Instance f32_fexp_valid.

(** * Covering an interval of binary32 values *)

Definition f32_succ (x : binary32) : binary32 :=
  @Bsucc prec32 emax32 prec32_gt_0 prec32_lt_emax32 x.

(** [chk a b] or else both halves, split after [split a b]. *)
Fixpoint erun (fuel : nat) (chk : binary32 -> binary32 -> bool)
    (split : binary32 -> binary32 -> binary32) (a b : binary32) : bool :=
  match fuel with
  | O => false
  | S n =>
      if chk a b then true else
      let m := split a b in
      if is_finite m then
        if is_finite (f32_succ m) then
          if erun n chk split a m then erun n chk split (f32_succ m) b else false
        else false
      else false
  end.

Lemma erun_sound : forall (Pr : binary32 -> Prop) chk split,
  (forall a b, chk a b = true -> forall x, is_finite x = true -> B2R a <= B2R x <= B2R b -> Pr x) ->
  forall fuel a b, erun fuel chk split a b = true ->
  forall x, is_finite x = true -> B2R a <= B2R x <= B2R b -> Pr x.
Proof.
  intros Pr chk split Hchk. induction fuel as [|n IH]; intros a b E x Fx Hx; [discriminate|].
  cbn [erun] in E.
  destruct (chk a b) eqn:C; [exact (Hchk a b C x Fx Hx)|].
  set (m := split a b) in E.
  destruct (is_finite m) eqn:Fm; [|discriminate].
  destruct (is_finite (f32_succ m)) eqn:Fs; [|discriminate].
  destruct (erun n chk split a m) eqn:E1; [|discriminate].
  destruct (Rle_dec (B2R x) (B2R m)) as [L|L].
  - exact (IH a m E1 x Fx (conj (proj1 Hx) L)).
  - apply (IH (f32_succ m) b E x Fx). split; [|exact (proj2 Hx)].
    pose proof (Bsucc_correct prec32 emax32 prec32_gt_0 prec32_lt_emax32 m Fm) as S.
    unfold f32_succ in *. revert S. case Rlt_bool_spec; intros _ S.
    + destruct S as [S _]. rewrite S.
      apply succ_le_lt; [apply f32_fexp_valid | apply generic_format_B2R | apply generic_format_B2R | lra].
    + rewrite <- is_finite_SF_B2SF, S in Fs. discriminate.
Qed.

Lemma Rle_bool_le : forall x y, Rle_bool x y = true -> x <= y.
Proof. intros x y. case Rle_bool_spec; [auto | intros _ H; discriminate]. Qed.

Lemma Req_bool_eq : forall x y, Req_bool x y = true -> x = y.
Proof. intros x y. case Req_bool_spec; [auto | intros _ H; discriminate]. Qed.

(** The midpoint, rounded, or the left end when that reaches the right one. *)
Definition mid (a b : binary32) : binary32 :=
  let m := f32_mult (f32_plus a b) f32_half in if Bleb b m then a else m.

(** * Rounding, monotone and bounded *)

Lemma f32_round_le : forall a b, a <= b -> f32_round a <= f32_round b.
Proof. intros a b H. apply round_le; [apply f32_fexp_valid | apply valid_rnd_round_mode | exact H]. Qed.

Lemma round_near : forall z, Rabs (f32_round z) <= 2 * Rabs z + 1.
Proof.
  intros z. pose proof (round_abs_err z) as E. pose proof f32_u_pos. pose proof u_le_half.
  rewrite f32_eta_eq in E. unfold dR in E. cbn in E.
  assert (Rabs (f32_round z) <= Rabs z + Rabs (f32_round z - z)).
  { replace (f32_round z) with (z + (f32_round z - z)) at 1 by ring. apply Rabs_triang. }
  pose proof (Rabs_pos z). nra.
Qed.

Lemma round_small : forall z, Rabs z <= 1073741824 -> Rabs (f32_round z) < bpow radix2 emax32.
Proof.
  intros z H. pose proof (round_near z). unfold emax32. cbn. lra.
Qed.

(** * The exponential

    The program of [f32_exp_approx] on an argument in [[-88, 88]], where the
    saturation leaves it unchanged, with the reduction integer [k] as its
    second input. *)

Definition exp_prog : list instr :=
  [IC f32_ln2_hi; IMul 1 2; ISub 0 3; IC f32_ln2_lo; IMul 1 5; IAdd 4 6;
   IMul 7 7; IMul 8 7; IMul 9 7; IMul 10 7; IMul 11 7; IMul 12 7;
   IC f32_five_thousand_forty; IDiv 13 14; IC f32_seven_twenty; IDiv 12 16; IAdd 17 15;
   IC f32_one_twenty; IDiv 11 19; IAdd 20 18; IC f32_twenty_four; IDiv 10 22; IAdd 23 21;
   IC f32_six; IDiv 9 25; IAdd 26 24; IC f32_two; IDiv 8 28; IAdd 29 27; IAdd 7 30;
   IC f32_one; IAdd 32 31; IPow2 1; IMul 33 34].

Lemma B2R_88 : is_finite f32_exp_hi = true /\ B2R f32_exp_hi = 88.
Proof. apply f32_of_Z_exact. cbn. lia. Qed.

Lemma B2R_m88 : is_finite f32_exp_lo = true /\ B2R f32_exp_lo = -88.
Proof.
  destruct (f32_of_Z_exact (-88) ltac:(cbn; lia)) as [F V].
  split; [exact F|]. unfold f32_exp_lo. rewrite V. reflexivity.
Qed.

Lemma exp_in_range : forall x, is_finite x = true -> -88 <= B2R x <= 88 ->
  f32_lt f32_exp_hi x = false /\ f32_lt x f32_exp_lo = false.
Proof.
  intros x F H. destruct B2R_88 as [Fh Vh]. destruct B2R_m88 as [Fl Vl].
  unfold f32_lt, f32_compare.
  rewrite (Bcompare_correct prec32 emax32 f32_exp_hi x Fh F).
  rewrite (Bcompare_correct prec32 emax32 x f32_exp_lo F Fl).
  rewrite Vh, Vl.
  destruct (Rcompare_spec 88 (B2R x)); destruct (Rcompare_spec (B2R x) (-88));
    split; try reflexivity; lra.
Qed.

Lemma exp_prog_eq : forall x : binary32, is_finite x = true -> -88 <= B2R x <= 88 ->
  f32_exp_approx x = fnth (feval exp_prog [x; f32_exp_k x]) 35.
Proof.
  intros x F H. destruct (exp_in_range x F H) as [H1 H2].
  unfold f32_exp_approx. rewrite H1, H2.
  cbv [feval exp_prog fstep fnth nth app]. reflexivity.
Qed.

Definition exp_poly (r : R) : R :=
  1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 + r^7/5040.

Lemma exp_prog_shadow : forall xr kr,
  snth (seval exp_prog [xr; kr]) 7 = xr - kr * (355 / 512) + kr * (14581891 / 68719476736) /\
  snth (seval exp_prog [xr; kr]) 35
    = exp_poly (xr - kr * (355 / 512) + kr * (14581891 / 68719476736)) * exp (kr * ln 2).
Proof.
  intros xr kr. cbv [seval exp_prog sstep snth nth app].
  rewrite B2R_ln2_hi, B2R_ln2_lo, B2R_two, B2R_six, B2R_24, B2R_120, B2R_720, B2R_5040,
    f32_one_correct.
  split; [ring|]. unfold exp_poly. field.
Qed.

(** The reduction integer, and its monotonicity. *)
Lemma exp_k_val : forall x, is_finite x = true -> -88 <= B2R x <= 88 ->
  is_finite (f32_exp_k x) = true /\
  B2R (f32_exp_k x)
  = f32_round (f32_round (f32_round (B2R x * (12102203 / 8388608)) + 12582912) + - 12582912).
Proof.
  intros x F H. destruct f32_magic_exact as [Fm Vm].
  assert (Fi : is_finite f32_inv_ln2 = true) by (vm_compute; reflexivity).
  unfold f32_exp_k, f32_round_int, f32_minus.
  assert (Z1 : Rabs (B2R x * B2R f32_inv_ln2) <= 128).
  { rewrite B2R_inv_ln2, Rabs_mult, (Rabs_pos_eq (12102203 / 8388608)) by lra.
    apply Rabs_le in H. nra. }
  assert (Z1' : Rabs (B2R x * B2R f32_inv_ln2) <= 1073741824) by lra.
  destruct (f32_mult_fin x f32_inv_ln2 F Fi (round_small _ Z1')) as [Ft Vt].
  set (t := f32_mult x f32_inv_ln2) in *.
  pose proof (round_near (B2R x * B2R f32_inv_ln2)) as T. rewrite <- Vt in T.
  assert (Z2 : Rabs (B2R t + B2R f32_magic) <= 12583169).
  { rewrite Vm. eapply Rle_trans; [apply Rabs_triang|]. rewrite (Rabs_pos_eq 12582912) by lra. lra. }
  destruct (f32_plus_fin t f32_magic Ft Fm ltac:(apply round_small; lra)) as [Fq Vq].
  set (q := f32_plus t f32_magic) in *.
  pose proof (round_near (B2R t + B2R f32_magic)) as Q. rewrite <- Vq in Q.
  assert (Fn : is_finite (f32_neg f32_magic) = true /\ B2R (f32_neg f32_magic) = -12582912).
  { unfold f32_neg. rewrite is_finite_Bopp, B2R_Bopp, Vm. split; [exact Fm | reflexivity]. }
  destruct Fn as [Fn Vn].
  assert (Z3 : Rabs (B2R q + B2R (f32_neg f32_magic)) <= 1073741824).
  { rewrite Vn. eapply Rle_trans; [apply Rabs_triang|]. rewrite (Rabs_left (-12582912)) by lra. lra. }
  destruct (f32_plus_fin q (f32_neg f32_magic) Fq Fn ltac:(apply round_small; exact Z3)) as [Fr Vr].
  split; [exact Fr|]. rewrite Vr, Vn, Vq, Vm, Vt. rewrite B2R_inv_ln2. reflexivity.
Qed.

Lemma exp_k_mono : forall x y, is_finite x = true -> is_finite y = true ->
  -88 <= B2R x -> B2R x <= B2R y -> B2R y <= 88 ->
  B2R (f32_exp_k x) <= B2R (f32_exp_k y).
Proof.
  intros x y Fx Fy H1 H2 H3.
  destruct (exp_k_val x Fx ltac:(lra)) as [_ Vx]. destruct (exp_k_val y Fy ltac:(lra)) as [_ Vy].
  rewrite Vx, Vy. apply f32_round_le. apply Rplus_le_compat_r.
  apply f32_round_le. apply Rplus_le_compat_r. apply f32_round_le.
  apply Rmult_le_compat_r; lra.
Qed.

(** The check on one subinterval, against a bound [T e^a] on the distance
    from the shadow. *)
Definition exp_chk (T : Z * Z) (a b : binary32) : bool :=
  if negb (is_finite a && is_finite b)%bool then false else
  if negb (Bleb f32_exp_lo a && Bleb b f32_exp_hi)%bool then false else
  let K := f32_exp_k a in
  if negb (is_finite K && Beqb (f32_exp_k b) K)%bool then false else
  match int_of K with
  | None => false
  | Some n =>
      if negb ((-127 <=? n)%Z && (n <=? 127)%Z && (0 <=? fst T)%Z)%bool then false else
      match aeval (0%Z, 0%Z) (0%Z, 0%Z) exp_prog [ax a b; a_const K] with
      | None => false
      | Some env =>
          (ub_le (EI.abs (ash (anth env 7))) 1433 (-12) &&
           iv_le (ad (anth env 35)) (EI.mul P (ptd (fst T) (snd T)) (EI.exp P (iv32 a))))%bool
      end
  end.

Definition exp_rel (T : Z * Z) (x : binary32) : Prop :=
  is_finite (f32_exp_approx x) = true /\
  Rabs (B2R (f32_exp_approx x) - exp (B2R x)) <= (dR (fst T) (snd T) + 16 / 1000000000) * exp (B2R x).

Lemma exp_chk_sound : forall T a b, exp_chk T a b = true ->
  forall x : binary32, is_finite x = true -> B2R a <= B2R x <= B2R b -> exp_rel T x.
Proof.
  intros T a b E x Fx Hx. unfold exp_chk in E.
  destruct (is_finite a && is_finite b)%bool eqn:C1; cbn [negb] in E; [|discriminate].
  apply andb_prop in C1 as [Fa Fb].
  destruct B2R_88 as [Fh Vh]. destruct B2R_m88 as [Fl Vl].
  destruct (Bleb f32_exp_lo a && Bleb b f32_exp_hi)%bool eqn:C2; cbn [negb] in E; [|discriminate].
  apply andb_prop in C2 as [C2a C2b].
  rewrite (Bleb_correct prec32 emax32 _ _ Fl Fa), Vl in C2a.
  rewrite (Bleb_correct prec32 emax32 _ _ Fb Fh), Vh in C2b.
  apply Rle_bool_le in C2a. apply Rle_bool_le in C2b.
  set (K := f32_exp_k a) in E.
  destruct (is_finite K && Beqb (f32_exp_k b) K)%bool eqn:C3; cbn [negb] in E; [|discriminate].
  apply andb_prop in C3 as [FK C3].
  destruct (exp_k_val b Fb ltac:(lra)) as [Fkb _].
  rewrite (Beqb_correct prec32 emax32 _ _ Fkb FK) in C3. apply Req_bool_eq in C3.
  destruct (int_of K) as [n|] eqn:In; [|discriminate].
  destruct ((-127 <=? n)%Z && (n <=? 127)%Z && (0 <=? fst T)%Z)%bool eqn:C4;
    cbn [negb] in E; [|discriminate].
  apply andb_prop in C4 as [C4 C4t]. apply andb_prop in C4 as [C4a C4b].
  apply Z.leb_le in C4a. apply Z.leb_le in C4b. apply Z.leb_le in C4t.
  destruct (aeval (0%Z, 0%Z) (0%Z, 0%Z) exp_prog [ax a b; a_const K]) as [env|] eqn:Ev;
    [|discriminate].
  apply andb_prop in E as [E7 E35].
  assert (Hxr : -88 <= B2R x <= 88) by lra.
  (* the reduction integer is [K] throughout *)
  destruct (exp_k_val x Fx Hxr) as [Fkx _].
  assert (Hk : B2R (f32_exp_k x) = B2R K).
  { pose proof (exp_k_mono a x Fa Fx ltac:(lra) ltac:(lra) ltac:(lra)).
    pose proof (exp_k_mono x b Fx Fb ltac:(lra) ltac:(lra) ltac:(lra)).
    fold K in H. lra. }
  pose proof (int_of_spec K n In) as Kn.
  (* the checker's abstraction *)
  assert (H3 : F3 [ax a b; a_const K] [x; f32_exp_k x] [B2R x; B2R K]).
  { constructor; [apply ax_sound; [exact Fx | exact Hx]|].
    constructor; [apply a_const_same; [exact FK | exact Fkx | exact Hk] | constructor]. }
  pose proof (aeval_sound_plain _ _ exp_prog _ _ _ _ eq_refl H3 Ev) as Hs.
  pose proof (F3_nth _ _ _ 7 Hs) as A7. pose proof (F3_nth _ _ _ 35 Hs) as A35.
  rewrite <- (exp_prog_eq x Fx Hxr) in A35.
  destruct (exp_prog_shadow (B2R x) (B2R K)) as [S7 S35].
  rewrite S7 in A7. rewrite S35 in A35.
  set (r := B2R x - B2R K * (355 / 512) + B2R K * (14581891 / 68719476736)) in *.
  (* the reduced argument stays where the series is accurate *)
  destruct A7 as (_ & _ & A7s & _).
  pose proof (ub_le_spec _ _ _ _ E7 (ctn_abs _ _ A7s)) as Hr. unfold dR in Hr. cbn in Hr.
  apply Rabs_le_inv in Hr.
  (* the float against its shadow *)
  pose proof (asound_err _ _ _ _ _ A35 E35
                (ctn_mul _ _ _ _ (ctn_ptd (fst T) (snd T)) (ctn_exp _ _ (ctn_iv32 a)))) as Hd.
  destruct A35 as (Ff & _).
  split; [exact Ff|].
  (* the shadow against the exponential *)
  rewrite Kn in *.
  assert (HB : exp (IZR n * ln 2) = bpow radix2 n) by (rewrite bpow_exp; reflexivity).
  rewrite HB in Hd.
  assert (HK : Rabs (IZR n * (355 / 512 - 14581891 / 68719476736 - ln 2)) <= 3 / 10000000000).
  { rewrite Rabs_mult. pose proof ln2_split_bound as Hl.
    assert (Rabs (IZR n) <= 127).
    { apply Rabs_le. change (-(127)) with (IZR (-127)). split; apply IZR_le; lia. }
    apply Rle_trans with (127 * (2 / 1000000000000)); [|lra].
    apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  apply Rabs_le_inv in HK.
  pose proof (exp_split_bound (exp_poly r) r (B2R x - IZR n * ln 2) (bpow radix2 n) (exp (B2R x))
    (bpow_gt_0 radix2 n)
    ltac:(unfold exp_poly; apply exp_series_bound; lra)
    ltac:(apply exp_upper_bound; lra)
    ltac:(replace (B2R x - IZR n * ln 2 - r) with (IZR n * (355 / 512 - 14581891 / 68719476736 - ln 2))
            by (unfold r; try rewrite Kn; ring); apply exp_near_zero; lra)
    ltac:(apply exp_lower_bound; unfold r in *; lra)
    ltac:(rewrite bpow_exp, <- exp_plus; f_equal; ring)) as Hs2.
  pose proof (exp_pos (B2R x)) as Px.
  assert (Ea : exp (B2R a) <= exp (B2R x)).
  { destruct (Req_dec (B2R a) (B2R x)) as [Eq|Ne]; [rewrite Eq; lra|].
    left. apply exp_increasing. lra. }
  assert (Tn : 0 <= dR (fst T) (snd T)).
  { unfold dR. apply Rmult_le_pos; [apply IZR_le; exact C4t | apply bpow_ge_0]. }
  replace (B2R (f32_exp_approx x) - exp (B2R x))
    with ((B2R (f32_exp_approx x) - exp_poly r * bpow radix2 n) + (exp_poly r * bpow radix2 n - exp (B2R x)))
    by ring.
  eapply Rle_trans; [apply Rabs_triang|].
  assert (dR (fst T) (snd T) * exp (B2R a) <= dR (fst T) (snd T) * exp (B2R x))
    by (apply Rmult_le_compat_l; assumption).
  lra.
Qed.

(** * The logarithm on [[1, 2]] *)

Definition log_prog : list instr :=
  [IC f32_one; ISub 0 1; IAdd 0 1; IDiv 2 3; IMul 4 4; IMul 5 4; IMul 6 5; IMul 7 5;
   IMul 8 5; IMul 9 5; IMul 10 5; IC f32_thirteen; IDiv 11 12; IC f32_eleven; IDiv 10 14;
   IAdd 15 13; IC f32_nine; IDiv 9 17; IAdd 18 16; IC f32_seven; IDiv 8 20; IAdd 21 19;
   IC f32_five; IDiv 7 23; IAdd 24 22; IC f32_three; IDiv 6 26; IAdd 27 25; IAdd 4 28;
   IC f32_two; IMul 30 29].

Lemma log_prog_eq : forall m, f32_log_unit m = fnth (feval log_prog [m]) 31.
Proof. intros m. unfold f32_log_unit. cbv [feval log_prog fstep fnth nth app]. reflexivity. Qed.

Definition log_poly (u : R) : R :=
  2 * (u + u^3/3 + u^5/5 + u^7/7 + u^9/9 + u^11/11 + u^13/13).

Lemma log_prog_shadow : forall mr,
  snth (seval log_prog [mr]) 4 = (mr - 1) / (mr + 1) /\
  snth (seval log_prog [mr]) 31 = log_poly ((mr - 1) / (mr + 1)).
Proof.
  intros mr. cbv [seval log_prog sstep snth nth app].
  rewrite f32_one_correct, B2R_two, B2R_three, B2R_five, B2R_seven, B2R_nine, B2R_eleven,
    B2R_thirteen.
  set (u := (mr - 1) / (mr + 1)).
  split; [reflexivity|]. unfold log_poly. field.
Qed.

(** The domain reaches [16385/8192], past [2] by the error the exponential
    can bring to [1 + e] in softplus; [u = 5462/16384] is below [0.3334]. *)
Definition c_log_hi : binary32 := f32_div (f32_of_Z 16385) (f32_of_Z 8192).

Lemma B2R_c_log_hi : is_finite c_log_hi = true /\ B2R c_log_hi = 16385 / 8192.
Proof.
  split; [vm_compute; reflexivity|].
  rewrite Qb_correct.
  rewrite (Qeq_eqR _ (Qmake 16385 8192)) by (vm_compute; reflexivity).
  unfold Q2R; cbn [Qnum Qden]; unfold Rdiv; reflexivity.
Qed.

Definition log_chk (T : Z * Z) (a b : binary32) : bool :=
  if negb (is_finite a && is_finite b)%bool then false else
  if negb (Bleb f32_one a && Bleb b c_log_hi)%bool then false else
  match aeval (0%Z, 0%Z) (0%Z, 0%Z) log_prog [ax a b] with
  | None => false
  | Some env =>
      (lb_ge (ash (anth env 4)) 0 0 && ub_le (ash (anth env 4)) 5462 (-14) &&
       iv_le (ad (anth env 31)) (ptd (fst T) (snd T)))%bool
  end.

Definition log_rel (T : Z * Z) (m : binary32) : Prop :=
  is_finite (f32_log_unit m) = true /\
  Rabs (B2R (f32_log_unit m) - ln (B2R m)) <= dR (fst T) (snd T) + 2 / 100000000.

Lemma log_chk_sound : forall T a b, log_chk T a b = true ->
  forall x : binary32, is_finite x = true -> B2R a <= B2R x <= B2R b -> log_rel T x.
Proof.
  intros T a b E x Fx Hx. unfold log_chk in E.
  destruct (is_finite a && is_finite b)%bool eqn:C1; cbn [negb] in E; [|discriminate].
  apply andb_prop in C1 as [Fa Fb].
  assert (F1 : is_finite f32_one = true) by reflexivity.
  destruct B2R_c_log_hi as [F2 V2].
  destruct (Bleb f32_one a && Bleb b c_log_hi)%bool eqn:C2; cbn [negb] in E; [|discriminate].
  apply andb_prop in C2 as [C2a C2b].
  rewrite (Bleb_correct prec32 emax32 _ _ F1 Fa), f32_one_correct in C2a.
  rewrite (Bleb_correct prec32 emax32 _ _ Fb F2), V2 in C2b.
  apply Rle_bool_le in C2a. apply Rle_bool_le in C2b.
  destruct (aeval (0%Z, 0%Z) (0%Z, 0%Z) log_prog [ax a b]) as [env|] eqn:Ev; [|discriminate].
  apply andb_prop in E as [E E31]. apply andb_prop in E as [E4a E4b].
  assert (H3 : F3 [ax a b] [x] [B2R x]).
  { constructor; [apply ax_sound; [exact Fx | exact Hx] | constructor]. }
  pose proof (aeval_sound_plain _ _ log_prog _ _ _ _ eq_refl H3 Ev) as Hs.
  pose proof (F3_nth _ _ _ 4 Hs) as A4. pose proof (F3_nth _ _ _ 31 Hs) as A31.
  destruct (log_prog_shadow (B2R x)) as [S4 S31].
  rewrite S4 in A4. rewrite S31 in A31.
  destruct A4 as (_ & _ & A4s & _).
  pose proof (lb_ge_spec _ _ _ _ E4a A4s) as U1. pose proof (ub_le_spec _ _ _ _ E4b A4s) as U2.
  unfold dR in U1, U2. cbn in U1, U2.
  pose proof (asound_err _ _ _ _ _ A31 E31 (ctn_ptd (fst T) (snd T))) as Hd0.
  destruct A31 as (Ff & _).
  rewrite <- (log_prog_eq x) in Ff, Hd0.
  unfold log_rel. split; [exact Ff|].
  set (F := f32_log_unit x) in *. clearbody F. rename Hd0 into Hd.
  pose proof (log_series_bound ((B2R x - 1) / (B2R x + 1)) ltac:(lra)) as Hl.
  rewrite log_arg_id in Hl by lra. fold (log_poly ((B2R x - 1) / (B2R x + 1))) in Hl.
  replace (B2R F - ln (B2R x))
    with ((B2R F - log_poly ((B2R x - 1) / (B2R x + 1)))
          + (log_poly ((B2R x - 1) / (B2R x + 1)) - ln (B2R x))) by ring.
  eapply Rle_trans; [apply Rabs_triang|]. lra.
Qed.

(** * Sine and cosine

    [f32_sin] and [f32_cos] evaluate a polynomial on the reduced argument
    [f32_reduce_2pi x]. The reduction is checked once for every integer it
    selects on [[-262144, 262144]], and each polynomial once over the range of
    the reduced argument. *)

Definition f32_trig_k (x : binary32) : binary32 := f32_round_int (f32_mult x f32_inv2pi).

Definition red_prog : list instr :=
  [IC f32_2pi_hi; IMul 1 2; ISub 0 3; IC f32_2pi_lo; IMul 1 5; ISub 4 6].

Lemma red_prog_eq : forall x : binary32,
  f32_reduce_2pi x = fnth (feval red_prog [x; f32_trig_k x]) 7.
Proof.
  intros x. unfold f32_reduce_2pi, f32_trig_k. cbv [feval red_prog fstep fnth nth app].
  reflexivity.
Qed.

Lemma red_prog_shadow : forall xr kr,
  snth (seval red_prog [xr; kr]) 7 = xr - kr * (201 / 32) - kr * (8312081 / 4294967296).
Proof.
  intros xr kr. cbv [seval red_prog sstep snth nth app]. rewrite B2R_2pi_hi, B2R_2pi_lo. ring.
Qed.

Lemma trig_k_val : forall x : binary32, is_finite x = true -> -262144 <= B2R x <= 262144 ->
  is_finite (f32_trig_k x) = true /\
  B2R (f32_trig_k x)
  = f32_round (f32_round (f32_round (B2R x * (10680707 / 67108864)) + 12582912) + - 12582912).
Proof.
  intros x F H. destruct f32_magic_exact as [Fm Vm].
  assert (Fi : is_finite f32_inv2pi = true) by (vm_compute; reflexivity).
  unfold f32_trig_k, f32_round_int, f32_minus.
  assert (Z1 : Rabs (B2R x * B2R f32_inv2pi) <= 41723).
  { rewrite B2R_inv2pi, Rabs_mult, (Rabs_pos_eq (10680707 / 67108864)) by lra.
    apply Rabs_le in H. nra. }
  assert (Z1' : Rabs (B2R x * B2R f32_inv2pi) <= 1073741824) by lra.
  destruct (f32_mult_fin x f32_inv2pi F Fi (round_small _ Z1')) as [Ft Vt].
  set (t := f32_mult x f32_inv2pi) in *.
  pose proof (round_near (B2R x * B2R f32_inv2pi)) as T. rewrite <- Vt in T.
  assert (Z2 : Rabs (B2R t + B2R f32_magic) <= 12666360).
  { rewrite Vm. eapply Rle_trans; [apply Rabs_triang|]. rewrite (Rabs_pos_eq 12582912) by lra. lra. }
  destruct (f32_plus_fin t f32_magic Ft Fm ltac:(apply round_small; lra)) as [Fq Vq].
  set (q := f32_plus t f32_magic) in *.
  pose proof (round_near (B2R t + B2R f32_magic)) as Q. rewrite <- Vq in Q.
  assert (Fn : is_finite (f32_neg f32_magic) = true /\ B2R (f32_neg f32_magic) = -12582912).
  { unfold f32_neg. rewrite is_finite_Bopp, B2R_Bopp, Vm. split; [exact Fm | reflexivity]. }
  destruct Fn as [Fn Vn].
  assert (Z3 : Rabs (B2R q + B2R (f32_neg f32_magic)) <= 1073741824).
  { rewrite Vn. eapply Rle_trans; [apply Rabs_triang|]. rewrite (Rabs_left (-12582912)) by lra. lra. }
  destruct (f32_plus_fin q (f32_neg f32_magic) Fq Fn ltac:(apply round_small; exact Z3)) as [Fr Vr].
  split; [exact Fr|]. rewrite Vr, Vn, Vq, Vm, Vt. rewrite B2R_inv2pi. reflexivity.
Qed.

Lemma trig_k_mono : forall x y : binary32, is_finite x = true -> is_finite y = true ->
  -262144 <= B2R x -> B2R x <= B2R y -> B2R y <= 262144 ->
  B2R (f32_trig_k x) <= B2R (f32_trig_k y).
Proof.
  intros x y Fx Fy H1 H2 H3.
  destruct (trig_k_val x Fx ltac:(lra)) as [_ Vx]. destruct (trig_k_val y Fy ltac:(lra)) as [_ Vy].
  rewrite Vx, Vy. apply f32_round_le. apply Rplus_le_compat_r.
  apply f32_round_le. apply Rplus_le_compat_r. apply f32_round_le.
  apply Rmult_le_compat_r; lra.
Qed.

Definition c262144 : binary32 := f32_of_Z 262144.
Definition c262144n : binary32 := f32_of_Z (-262144).

Lemma B2R_c262144 : is_finite c262144 = true /\ B2R c262144 = 262144.
Proof. apply f32_of_Z_exact. cbn. lia. Qed.

Lemma B2R_c262144n : is_finite c262144n = true /\ B2R c262144n = -262144.
Proof.
  destruct (f32_of_Z_exact (-262144) ltac:(cbn; lia)) as [F V].
  split; [exact F|]. unfold c262144n. rewrite V. reflexivity.
Qed.

(** The bound on the reduced argument, [6492/2048 < 3.17]. *)
Definition red_chk (Tr : Z * Z) (a b : binary32) : bool :=
  if negb (is_finite a && is_finite b)%bool then false else
  if negb (Bleb c262144n a && Bleb b c262144)%bool then false else
  let K := f32_trig_k a in
  if negb (is_finite K && Beqb (f32_trig_k b) K)%bool then false else
  match int_of K with
  | None => false
  | Some n =>
      if negb ((-41722 <=? n)%Z && (n <=? 41722)%Z)%bool then false else
      match aeval (0%Z, 0%Z) (0%Z, 0%Z) red_prog [ax a b; a_const K] with
      | None => false
      | Some env =>
          (ub_le (EI.abs (av (anth env 7))) 6492 (-11) &&
           iv_le (ad (anth env 7)) (ptd (fst Tr) (snd Tr)))%bool
      end
  end.

Definition red_rel (Tr : Z * Z) (x : binary32) : Prop :=
  is_finite (f32_reduce_2pi x) = true /\ Rabs (B2R (f32_reduce_2pi x)) <= 6492 / 2048 /\
  exists n : Z, (-41722 <= n <= 41722)%Z /\
    Rabs (B2R (f32_reduce_2pi x) - (B2R x - IZR n * (201 / 32) - IZR n * (8312081 / 4294967296)))
    <= dR (fst Tr) (snd Tr).

Lemma red_chk_sound : forall Tr a b, red_chk Tr a b = true ->
  forall x : binary32, is_finite x = true -> B2R a <= B2R x <= B2R b -> red_rel Tr x.
Proof.
  intros Tr a b E x Fx Hx. unfold red_chk in E.
  destruct (is_finite a && is_finite b)%bool eqn:C1; cbn [negb] in E; [|discriminate].
  apply andb_prop in C1 as [Fa Fb].
  destruct B2R_c262144 as [Fh Vh]. destruct B2R_c262144n as [Fl Vl].
  destruct (Bleb c262144n a && Bleb b c262144)%bool eqn:C2; cbn [negb] in E; [|discriminate].
  apply andb_prop in C2 as [C2a C2b].
  rewrite (Bleb_correct prec32 emax32 _ _ Fl Fa), Vl in C2a.
  rewrite (Bleb_correct prec32 emax32 _ _ Fb Fh), Vh in C2b.
  apply Rle_bool_le in C2a. apply Rle_bool_le in C2b.
  set (K := f32_trig_k a) in E.
  destruct (is_finite K && Beqb (f32_trig_k b) K)%bool eqn:C3; cbn [negb] in E; [|discriminate].
  apply andb_prop in C3 as [FK C3].
  destruct (trig_k_val b Fb ltac:(lra)) as [Fkb _].
  rewrite (Beqb_correct prec32 emax32 _ _ Fkb FK) in C3. apply Req_bool_eq in C3.
  destruct (int_of K) as [n|] eqn:In; [|discriminate].
  destruct ((-41722 <=? n)%Z && (n <=? 41722)%Z)%bool eqn:C4; cbn [negb] in E; [|discriminate].
  apply andb_prop in C4 as [C4a C4b]. apply Z.leb_le in C4a. apply Z.leb_le in C4b.
  destruct (aeval (0%Z, 0%Z) (0%Z, 0%Z) red_prog [ax a b; a_const K]) as [env|] eqn:Ev;
    [|discriminate].
  apply andb_prop in E as [E7 E7d].
  assert (Hxr : -262144 <= B2R x <= 262144) by lra.
  destruct (trig_k_val x Fx Hxr) as [Fkx _].
  assert (Hk : B2R (f32_trig_k x) = B2R K).
  { pose proof (trig_k_mono a x Fa Fx ltac:(lra) ltac:(lra) ltac:(lra)).
    pose proof (trig_k_mono x b Fx Fb ltac:(lra) ltac:(lra) ltac:(lra)).
    fold K in H. lra. }
  pose proof (int_of_spec K n In) as Kn.
  assert (H3 : F3 [ax a b; a_const K] [x; f32_trig_k x] [B2R x; B2R K]).
  { constructor; [apply ax_sound; [exact Fx | exact Hx]|].
    constructor; [apply a_const_same; [exact FK | exact Fkx | exact Hk] | constructor]. }
  pose proof (aeval_sound_plain _ _ red_prog _ _ _ _ eq_refl H3 Ev) as Hs.
  pose proof (F3_nth _ _ _ 7 Hs) as A7.
  rewrite <- (red_prog_eq x) in A7. rewrite red_prog_shadow in A7.
  pose proof (asound_err _ _ _ _ _ A7 E7d (ctn_ptd (fst Tr) (snd Tr))) as Hd.
  destruct A7 as (Ff & Av & _).
  pose proof (ub_le_spec _ _ _ _ E7 (ctn_abs _ _ Av)) as Hb. unfold dR in Hb. cbn in Hb.
  split; [exact Ff|]. split; [lra|].
  exists n. split; [lia|]. rewrite Kn in Hd. exact Hd.
Qed.

(** A split at the boundary between the middle integers of a range. *)
Definition f32_pred (x : binary32) : binary32 :=
  @Bpred prec32 emax32 prec32_gt_0 prec32_lt_emax32 x.

Fixpoint adjust (fuel : nat) (km m : binary32) : binary32 :=
  match fuel with
  | O => m
  | S n =>
      if Bltb km (f32_trig_k m) then adjust n km (f32_pred m)
      else if Bleb (f32_trig_k (f32_succ m)) km then adjust n km (f32_succ m)
      else m
  end.

Definition rsplit (a b : binary32) : binary32 :=
  match int_of (f32_trig_k a), int_of (f32_trig_k b) with
  | Some na, Some nb =>
      let km := f32_of_Z (Z.div (na + nb) 2) in
      let m := adjust 64 km (f32_div (f32_plus km f32_half) f32_inv2pi) in
      if (Bleb a m && Bltb m b)%bool then m else mid a b
  | _, _ => mid a b
  end.

(** ** The polynomials *)

Definition sin_prog : list instr :=
  [IMul 0 0; IMul 1 0; IMul 2 1; IMul 3 1; IMul 4 1; IMul 5 1; IMul 6 1; IMul 7 1; IMul 8 1;
   IMul 9 1; IC f32_fac17; IDiv 9 11; IC f32_fac19; IDiv 10 13; ISub 12 14;
   IC f32_fac15; IDiv 8 16; ISub 15 17; IC f32_fac13; IDiv 7 19; IAdd 18 20;
   IC f32_fac11; IDiv 6 22; ISub 21 23; IC f32_fac9; IDiv 5 25; IAdd 24 26;
   IC f32_fac7; IDiv 4 28; ISub 27 29; IC f32_fac5; IDiv 3 31; IAdd 30 32;
   IC f32_fac3; IDiv 2 34; ISub 33 35; IAdd 0 36].

Definition cos_prog : list instr :=
  [IMul 0 0; IMul 1 1; IMul 2 1; IMul 3 1; IMul 4 1; IMul 5 1; IMul 6 1; IMul 7 1; IMul 8 1;
   IC f32_fac16; IDiv 8 10; IC f32_fac18; IDiv 9 12; ISub 11 13;
   IC f32_fac14; IDiv 7 15; ISub 14 16; IC f32_fac12; IDiv 6 18; IAdd 17 19;
   IC f32_fac10; IDiv 5 21; ISub 20 22; IC f32_fac8; IDiv 4 24; IAdd 23 25;
   IC f32_fac6; IDiv 3 27; ISub 26 28; IC f32_fac4; IDiv 2 30; IAdd 29 31;
   IC f32_fac2; IDiv 1 33; ISub 32 34; IC f32_one; IAdd 36 35].

Lemma sin_prog_eq : forall x : binary32,
  f32_sin x = fnth (feval sin_prog [f32_reduce_2pi x]) 37.
Proof. intros x. unfold f32_sin. cbv [feval sin_prog fstep fnth nth app]. reflexivity. Qed.

Lemma cos_prog_eq : forall x : binary32,
  f32_cos x = fnth (feval cos_prog [f32_reduce_2pi x]) 37.
Proof. intros x. unfold f32_cos. cbv [feval cos_prog fstep fnth nth app]. reflexivity. Qed.

Definition sin_poly (r : R) : R :=
  r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880 - r^11/39916800
  + r^13/6227020800 - r^15/1307674411008 + r^17/355687414628352
  - r^19/121645104594157568.

Definition cos_poly (r : R) : R :=
  1 - r^2/2 + r^4/24 - r^6/720 + r^8/40320 - r^10/3628800
  + r^12/479001600 - r^14/87178289152 + r^16/20922790576128
  - r^18/6402373530419200.

Lemma sin_prog_shadow : forall rr, snth (seval sin_prog [rr]) 37 = sin_poly rr.
Proof.
  intros rr. cbv [seval sin_prog sstep snth nth app].
  rewrite B2R_fac3, B2R_fac5, B2R_fac7, B2R_fac9, B2R_fac11, B2R_fac13, B2R_fac15,
    B2R_fac17, B2R_fac19.
  unfold sin_poly. field.
Qed.

Lemma cos_prog_shadow : forall rr, snth (seval cos_prog [rr]) 37 = cos_poly rr.
Proof.
  intros rr. cbv [seval cos_prog sstep snth nth app].
  rewrite f32_one_correct, B2R_fac2, B2R_fac4, B2R_fac6, B2R_fac8, B2R_fac10, B2R_fac12,
    B2R_fac14, B2R_fac16, B2R_fac18.
  unfold cos_poly. field.
Qed.

(** The float polynomial within [T] of its shadow on [[a, b]]. *)
Definition poly_chk (p : list instr) (T : Z * Z) (a b : binary32) : bool :=
  if negb (is_finite a && is_finite b)%bool then false else
  match aeval (0%Z, 0%Z) (0%Z, 0%Z) p [ax a b] with
  | None => false
  | Some env => iv_le (ad (anth env 37)) (ptd (fst T) (snd T))
  end.

Lemma poly_chk_sound : forall p T a b, forallb plain p = true -> poly_chk p T a b = true ->
  forall r : binary32, is_finite r = true -> B2R a <= B2R r <= B2R b ->
  is_finite (fnth (feval p [r]) 37) = true /\
  Rabs (B2R (fnth (feval p [r]) 37) - snth (seval p [B2R r]) 37) <= dR (fst T) (snd T).
Proof.
  intros p T a b Hp E r Fr Hr. unfold poly_chk in E.
  destruct (is_finite a && is_finite b)%bool eqn:C1; cbn [negb] in E; [|discriminate].
  destruct (aeval (0%Z, 0%Z) (0%Z, 0%Z) p [ax a b]) as [env|] eqn:Ev; [|discriminate].
  assert (H3 : F3 [ax a b] [r] [B2R r]).
  { constructor; [apply ax_sound; [exact Fr | exact Hr] | constructor]. }
  pose proof (aeval_sound_plain _ _ p _ _ _ _ Hp H3 Ev) as Hs.
  pose proof (F3_nth _ _ _ 37 Hs) as A.
  pose proof (asound_err _ _ _ _ _ A E (ctn_ptd (fst T) (snd T))) as Hd.
  destruct A as (Ff & _). split; [exact Ff | exact Hd].
Qed.

(** * Sigmoid, tanh, softplus and GELU

    These programs call the exponential and, for softplus, the logarithm, and
    are checked with the bounds proved on those two as the abstraction of the
    call. The shadows of sigmoid, tanh and softplus are the functions
    themselves. *)

Definition sig_prog : list instr :=
  [INeg 0; IExp 1; IC f32_one; IAdd 3 2; IDiv 3 4].

Definition tanh_prog : list instr :=
  [IC f32_two; IMul 1 0; INeg 2; IExp 3; IC f32_one; IAdd 5 4; IDiv 5 6; IMul 1 7; ISub 8 5].

(** Softplus with the logarithm's operations in place, so that each is charged
    at the magnitude it has here. *)
Definition softplus_prog : list instr :=
  [IAbs 0; INeg 1; IExp 2; IMax0 0; IC f32_one; IAdd 5 3;
   IC f32_one; ISub 6 7; IAdd 6 7; IDiv 8 9; IMul 10 10; IMul 11 10; IMul 12 11; IMul 13 11;
   IMul 14 11; IMul 15 11; IMul 16 11; IC f32_thirteen; IDiv 17 18; IC f32_eleven; IDiv 16 20;
   IAdd 21 19; IC f32_nine; IDiv 15 23; IAdd 24 22; IC f32_seven; IDiv 14 26; IAdd 27 25;
   IC f32_five; IDiv 13 29; IAdd 30 28; IC f32_three; IDiv 12 32; IAdd 33 31; IAdd 10 34;
   IC f32_two; IMul 36 35; IAdd 4 37].

Definition gelu_prog : list instr :=
  [IMul 0 0; IMul 0 1; IC f32_gelu_c2; IMul 3 2; IAdd 0 4; IC f32_gelu_c1; IMul 6 5;
   IC f32_two; IMul 8 7; INeg 9; IExp 10; IC f32_one; IAdd 12 11; IDiv 12 13; IMul 8 14;
   ISub 15 12; IC f32_half; IMul 17 0; IAdd 12 16; IMul 18 19].

Lemma sig_prog_eq : forall x : binary32, f32_sigmoid x = fnth (feval sig_prog [x]) 5.
Proof. intros x. unfold f32_sigmoid. cbv [feval sig_prog fstep fnth nth app]. reflexivity. Qed.

Lemma tanh_prog_eq : forall x : binary32, f32_tanh x = fnth (feval tanh_prog [x]) 9.
Proof.
  intros x. unfold f32_tanh, f32_sigmoid. cbv [feval tanh_prog fstep fnth nth app]. reflexivity.
Qed.

Lemma softplus_prog_eq : forall x : binary32, f32_softplus x = fnth (feval softplus_prog [x]) 38.
Proof.
  intros x. unfold f32_softplus, f32_log_unit. cbv [feval softplus_prog fstep fnth nth app].
  reflexivity.
Qed.

Lemma gelu_prog_eq : forall x : binary32, f32_gelu x = fnth (feval gelu_prog [x]) 20.
Proof.
  intros x. unfold f32_gelu, f32_tanh, f32_sigmoid. cbv [feval gelu_prog fstep fnth nth app].
  reflexivity.
Qed.

Lemma sig_prog_shadow : forall xr, snth (seval sig_prog [xr]) 5 = sigmoid xr.
Proof.
  intros xr. cbv [seval sig_prog sstep snth nth app]. rewrite f32_one_correct.
  unfold sigmoid. reflexivity.
Qed.

Lemma tanh_prog_shadow : forall xr, snth (seval tanh_prog [xr]) 9 = tanh xr.
Proof.
  intros xr. cbv [seval tanh_prog sstep snth nth app]. rewrite f32_one_correct, B2R_two.
  rewrite tanh_as_sigmoid. unfold sigmoid. reflexivity.
Qed.

Lemma softplus_prog_shadow : forall xr,
  snth (seval softplus_prog [xr]) 38
  = Rmax xr 0 + log_poly ((1 + exp (- Rabs xr) - 1) / (1 + exp (- Rabs xr) + 1)).
Proof.
  intros xr. cbv [seval softplus_prog sstep snth nth app].
  rewrite f32_one_correct, B2R_two, B2R_three, B2R_five, B2R_seven, B2R_nine, B2R_eleven,
    B2R_thirteen.
  set (u := (1 + exp (- Rabs xr) - 1) / (1 + exp (- Rabs xr) + 1)).
  unfold log_poly. field.
Qed.

(** The shadow against softplus, through the series bound on [log]. *)
Lemma softplus_shadow_true : forall xr,
  Rabs (Rmax xr 0 + log_poly ((1 + exp (- Rabs xr) - 1) / (1 + exp (- Rabs xr) + 1))
        - ln (1 + exp xr)) <= 2 / 100000000.
Proof.
  intros xr. rewrite <- softplus_stable.
  set (m := 1 + exp (- Rabs xr)).
  assert (Hm : 1 < m <= 2).
  { unfold m. pose proof (exp_pos (- Rabs xr)). pose proof (exp_le_one (Rabs xr) (Rabs_pos xr)).
    split; lra. }
  assert (Hu : 0 <= (m - 1) / (m + 1) <= 0.3334).
  { split.
    - unfold Rdiv. apply Rmult_le_pos; [lra | left; apply Rinv_0_lt_compat; lra].
    - unfold Rdiv. apply Rmult_le_reg_r with (m + 1); [lra|].
      rewrite Rmult_assoc, Rinv_l by lra. lra. }
  pose proof (log_series_bound _ Hu) as Hb. rewrite log_arg_id in Hb by lra.
  fold (log_poly ((m - 1) / (m + 1))) in Hb.
  replace (Rmax xr 0 + log_poly ((m - 1) / (m + 1)) - (Rmax xr 0 + ln m))
    with (log_poly ((m - 1) / (m + 1)) - ln m) by ring.
  exact Hb.
Qed.

Definition gelu_shadow (x : R) : R :=
  1 / 2 * x * (1 + tanh (6693141 / 8388608 * (x + 12003091 / 268435456 * (x * (x * x))))).

Lemma gelu_prog_shadow : forall xr, snth (seval gelu_prog [xr]) 20 = gelu_shadow xr.
Proof.
  intros xr. cbv [seval gelu_prog sstep snth nth app].
  rewrite f32_one_correct, B2R_two, B2R_half, B2R_gelu_c1, B2R_gelu_c2.
  unfold gelu_shadow. rewrite (tanh_as_sigmoid (6693141 / 8388608 * _)). unfold sigmoid.
  reflexivity.
Qed.

(** One program on one input interval [[a, b]] within [[lo, hi]], within [T] of
    its shadow at position [res]. *)
Definition comp_chk (p : list instr) (res : nat) (ce cl T : Z * Z) (lo hi a b : binary32) : bool :=
  if negb (is_finite a && is_finite b && is_finite lo && is_finite hi && Bleb lo a && Bleb b hi)%bool
  then false else
  match aeval ce cl p [ax a b] with
  | None => false
  | Some env => iv_le (ad (anth env res)) (ptd (fst T) (snd T))
  end.

Lemma comp_chk_sound : forall p res ce cl T lo hi,
  exp_hyp (dR (fst ce) (snd ce)) -> log_hyp (dR (fst cl) (snd cl)) ->
  forall a b, comp_chk p res ce cl T lo hi a b = true ->
  forall x : binary32, is_finite x = true -> B2R a <= B2R x <= B2R b ->
  B2R lo <= B2R x <= B2R hi /\
  is_finite (fnth (feval p [x]) res) = true /\
  Rabs (B2R (fnth (feval p [x]) res) - snth (seval p [B2R x]) res) <= dR (fst T) (snd T).
Proof.
  intros p res ce cl T lo hi He Hl a b E x Fx Hx. unfold comp_chk in E.
  destruct (is_finite a && is_finite b && is_finite lo && is_finite hi && Bleb lo a && Bleb b hi)%bool
    eqn:C; cbn [negb] in E; [|discriminate].
  apply andb_prop in C as [C Cb]. apply andb_prop in C as [C Ca].
  apply andb_prop in C as [C Fh]. apply andb_prop in C as [C Fl]. apply andb_prop in C as [Fa Fb].
  rewrite (Bleb_correct prec32 emax32 _ _ Fl Fa) in Ca. rewrite (Bleb_correct prec32 emax32 _ _ Fb Fh) in Cb.
  apply Rle_bool_le in Ca. apply Rle_bool_le in Cb.
  destruct (aeval ce cl p [ax a b]) as [env|] eqn:Ev; [|discriminate].
  assert (H3 : F3 [ax a b] [x] [B2R x]).
  { constructor; [apply ax_sound; [exact Fx | exact Hx] | constructor]. }
  pose proof (aeval_sound ce cl He Hl p _ _ _ _ H3 Ev) as Hs.
  pose proof (F3_nth _ _ _ res Hs) as A.
  pose proof (asound_err _ _ _ _ _ A E (ctn_ptd (fst T) (snd T))) as Hd.
  destruct A as (Ff & _). split; [lra|]. split; [exact Ff | exact Hd].
Qed.
