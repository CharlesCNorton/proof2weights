(** * The bounds of the elementary-function table

    Each checker of ElemBounds.v run over its whole domain, by [vm_compute]
    when the proof is checked, and the resulting bound on the distance from
    each extracted binary32 function to the mathematical function, for every
    binary32 input of the domain. *)

From Stdlib Require Import ZArith QArith Qreals Reals Lra Lia List Bool.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
From Interval Require Import Specific_stdz Specific_ops Float_full Interval Xreal Basic.
Require Import Phases1_15_complete Float_error Annot Enclose Llama Qwen Series Truth.
Require Import ElemCheck ElemBounds.

Import ListNotations.
Open Scope R_scope.

(** * The targets *)

Definition exp_T : Z * Z := (3%Z, (-24)%Z).
Definition exp_c : Z * Z := (105%Z, (-29)%Z).
Definition log_T : Z * Z := (13%Z, (-26)%Z).
Definition log_c : Z * Z := (230%Z, (-30)%Z).
Definition red_T : Z * Z := (17%Z, (-22)%Z).
Definition sinp_T : Z * Z := (3%Z, (-20)%Z).
Definition cosp_T : Z * Z := (3%Z, (-20)%Z).
Definition sig_T : Z * Z := (9%Z, (-26)%Z).
Definition tanh_T : Z * Z := (1%Z, (-21)%Z).
Definition softplus_T : Z * Z := (5%Z, (-23)%Z).
Definition gelu_T : Z * Z := (1%Z, (-18)%Z).

Definition c_red : binary32 := f32_div (f32_of_Z 6492) (f32_of_Z 2048).
Definition c44 : binary32 := f32_of_Z 44.
Definition c44n : binary32 := f32_of_Z (-44).
Definition c8 : binary32 := f32_of_Z 8.
Definition c8n : binary32 := f32_of_Z (-8).

(** * The checker runs *)

Lemma exp_run : erun 80 (exp_chk exp_T) mid f32_exp_lo f32_exp_hi = true.
Proof. vm_cast_no_check (eq_refl true). Qed.

Lemma log_run : erun 60 (log_chk log_T) mid f32_one c_log_hi = true.
Proof. vm_cast_no_check (eq_refl true). Qed.

Lemma red_run : erun 60 (red_chk red_T) rsplit c262144n c262144 = true.
Proof. vm_cast_no_check (eq_refl true). Qed.

Lemma sinp_run : erun 60 (poly_chk sin_prog sinp_T) mid (f32_neg c_red) c_red = true.
Proof. vm_cast_no_check (eq_refl true). Qed.

Lemma cosp_run : erun 60 (poly_chk cos_prog cosp_T) mid (f32_neg c_red) c_red = true.
Proof. vm_cast_no_check (eq_refl true). Qed.

Lemma sig_run :
  erun 60 (comp_chk sig_prog 5 exp_c log_c sig_T f32_exp_lo f32_exp_hi) mid f32_exp_lo f32_exp_hi = true.
Proof. vm_cast_no_check (eq_refl true). Qed.

Lemma tanh_run : erun 60 (comp_chk tanh_prog 9 exp_c log_c tanh_T c44n c44) mid c44n c44 = true.
Proof. vm_cast_no_check (eq_refl true). Qed.

Lemma softplus_run :
  erun 60 (comp_chk softplus_prog 38 exp_c log_c softplus_T f32_exp_lo f32_exp_hi) mid
    f32_exp_lo f32_exp_hi = true.
Proof. vm_cast_no_check (eq_refl true). Qed.

Lemma gelu_run : erun 60 (comp_chk gelu_prog 20 exp_c log_c gelu_T c8n c8) mid c8n c8 = true.
Proof. vm_cast_no_check (eq_refl true). Qed.

(** * The exponential, relative to [exp x] on [[-88, 88]] *)

Theorem exp_bound : exp_hyp (dR (fst exp_c) (snd exp_c)).
Proof.
  intros v Fv Hv.
  destruct B2R_88 as [Fh Vh]. destruct B2R_m88 as [Fl Vl].
  destruct (erun_sound (exp_rel exp_T) (exp_chk exp_T) mid (exp_chk_sound exp_T) 80
              f32_exp_lo f32_exp_hi exp_run v Fv ltac:(rewrite Vl, Vh; exact Hv)) as [Ff Hf].
  split; [exact Ff|]. eapply Rle_trans; [exact Hf|].
  apply Rmult_le_compat_r; [left; apply exp_pos|]. unfold dR; cbn; lra.
Qed.

(** * The logarithm on [[1, 16385/8192]] *)

Theorem log_bound : log_hyp (dR (fst log_c) (snd log_c)).
Proof.
  intros m Fm Hm. destruct B2R_c_log_hi as [Fh Vh].
  destruct (erun_sound (log_rel log_T) (log_chk log_T) mid (log_chk_sound log_T) 60
              f32_one c_log_hi log_run m Fm ltac:(rewrite f32_one_correct, Vh; exact Hm)) as [Ff Hf].
  split; [exact Ff|]. eapply Rle_trans; [exact Hf|]. unfold dR; cbn; lra.
Qed.

(** * Sine and cosine on [[-262144, 262144]] *)

Theorem red_bound : forall x : binary32, is_finite x = true -> -262144 <= B2R x <= 262144 ->
  red_rel red_T x.
Proof.
  intros x Fx Hx. destruct B2R_c262144 as [Fh Vh]. destruct B2R_c262144n as [Fl Vl].
  exact (erun_sound (red_rel red_T) (red_chk red_T) rsplit (red_chk_sound red_T) 60 _ _ red_run x Fx
           ltac:(rewrite Vl, Vh; exact Hx)).
Qed.

Lemma B2R_c_red : is_finite c_red = true /\ B2R c_red = 6492 / 2048.
Proof.
  split; [vm_compute; reflexivity|].
  rewrite Qb_correct.
  rewrite (Qeq_eqR _ (Qmake 6492 2048)) by (vm_compute; reflexivity).
  unfold Q2R; cbn [Qnum Qden]; unfold Rdiv; reflexivity.
Qed.

Lemma B2R_c_redn : is_finite (f32_neg c_red) = true /\ B2R (f32_neg c_red) = - (6492 / 2048).
Proof.
  destruct B2R_c_red as [F V]. unfold f32_neg. rewrite is_finite_Bopp, B2R_Bopp, V.
  split; [exact F | reflexivity].
Qed.

Lemma poly_bound : forall p T (poly : R -> R),
  forallb plain p = true ->
  (forall rr, snth (seval p [rr]) 37 = poly rr) ->
  erun 60 (poly_chk p T) mid (f32_neg c_red) c_red = true ->
  forall r : binary32, is_finite r = true -> Rabs (B2R r) <= 6492 / 2048 ->
  is_finite (fnth (feval p [r]) 37) = true /\
  Rabs (B2R (fnth (feval p [r]) 37) - poly (B2R r)) <= dR (fst T) (snd T).
Proof.
  intros p T poly Hp Hs R r Fr Hr.
  destruct B2R_c_red as [Fh Vh]. destruct B2R_c_redn as [Fl Vl].
  apply Rabs_le_inv in Hr.
  destruct (erun_sound _ (poly_chk p T) mid (fun a b => poly_chk_sound p T a b Hp) 60 _ _ R r Fr
              ltac:(rewrite Vl, Vh; lra)) as [F H].
  rewrite Hs in H. exact (conj F H).
Qed.

(** The reduced argument against [x - 2 pi n], with [n] the integer the
    reduction selected. *)
Lemma red_to_true : forall (xr rr : R) (n : Z) (T : R), (-41722 <= n <= 41722)%Z ->
  Rabs (rr - (xr - IZR n * (201 / 32) - IZR n * (8312081 / 4294967296))) <= T ->
  Rabs (rr - (xr + 2 * IZR (- n) * PI)) <= T + 41722 * (2 / 100000000000).
Proof.
  intros xr rr n T Hn H. pose proof two_pi_split_bound as Tw.
  assert (Hk : Rabs (IZR n) <= 41722).
  { apply Rabs_le. change (-(41722)) with (IZR (-41722)). split; apply IZR_le; lia. }
  replace (rr - (xr + 2 * IZR (- n) * PI))
    with ((rr - (xr - IZR n * (201 / 32) - IZR n * (8312081 / 4294967296)))
          + IZR n * (2 * PI - (201 / 32 + 8312081 / 4294967296)))
    by (rewrite opp_IZR; ring).
  eapply Rle_trans; [apply Rabs_triang|]. apply Rplus_le_compat; [exact H|].
  rewrite Rabs_mult.
  replace (2 * PI - (201 / 32 + 8312081 / 4294967296))
    with (- (201 / 32 + 8312081 / 4294967296 - 2 * PI)) by ring.
  rewrite Rabs_Ropp. apply Rmult_le_compat; try apply Rabs_pos; [exact Hk | exact Tw].
Qed.

(** The triangle inequalities that close the bounds, on plain variables. *)
Lemma chain4 : forall F A B C D r Tp c1 Tr c2 : R,
  Rabs (F - A) <= Tp -> Rabs (A - B) <= c1 -> Rabs (B - C) <= Rabs (r - D) ->
  Rabs (r - D) <= Tr + c2 -> Rabs (F - C) <= Tp + c1 + (Tr + c2).
Proof.
  intros F A B C D r Tp c1 Tr c2 H1 H2 H3 H4.
  replace (F - C) with ((F - A) + (A - B) + (B - C)) by ring.
  eapply Rle_trans; [apply Rabs_triang|]. apply Rplus_le_compat; [|lra].
  eapply Rle_trans; [apply Rabs_triang|]. lra.
Qed.

Lemma chain2 : forall Y A B T c : R,
  Rabs (Y - A) <= T -> Rabs (A - B) <= c -> Rabs (Y - B) <= T + c.
Proof.
  intros Y A B T c H1 H2. replace (Y - B) with ((Y - A) + (A - B)) by ring.
  eapply Rle_trans; [apply Rabs_triang|]. lra.
Qed.

Definition sin_B : R :=
  dR (fst sinp_T) (snd sinp_T) + 2 / 1000000000
  + (dR (fst red_T) (snd red_T) + 41722 * (2 / 100000000000)).
Definition cos_B : R :=
  dR (fst cosp_T) (snd cosp_T) + 2 / 100000000
  + (dR (fst red_T) (snd red_T) + 41722 * (2 / 100000000000)).

Theorem sin_bound : forall x : binary32, is_finite x = true -> -262144 <= B2R x <= 262144 ->
  is_finite (f32_sin x) = true /\ Rabs (B2R (f32_sin x) - sin (B2R x)) <= sin_B.
Proof.
  intros x Fx Hx.
  destruct (red_bound x Fx Hx) as (Fr & Hr & n & Hn & Hd).
  rewrite sin_prog_eq. set (r := f32_reduce_2pi x) in *.
  destruct (poly_bound sin_prog sinp_T sin_poly eq_refl sin_prog_shadow sinp_run r Fr Hr) as [Fp Hp].
  split; [exact Fp|].
  assert (Hr' : -3.17 <= B2R r <= 3.17) by (apply Rabs_le_inv in Hr; lra).
  pose proof (sin_series_bound (B2R r) Hr') as Hs. fold (sin_poly (B2R r)) in Hs.
  pose proof (red_to_true (B2R x) (B2R r) n _ Hn Hd) as Ht.
  pose proof (sin_lipschitz (B2R r) (B2R x + 2 * IZR (- n) * PI)) as Hl.
  rewrite sin_period_Z in Hl.
  exact (chain4 _ _ _ _ _ _ _ _ _ _ Hp Hs Hl Ht).
Qed.

Theorem cos_bound : forall x : binary32, is_finite x = true -> -262144 <= B2R x <= 262144 ->
  is_finite (f32_cos x) = true /\ Rabs (B2R (f32_cos x) - cos (B2R x)) <= cos_B.
Proof.
  intros x Fx Hx.
  destruct (red_bound x Fx Hx) as (Fr & Hr & n & Hn & Hd).
  rewrite cos_prog_eq. set (r := f32_reduce_2pi x) in *.
  destruct (poly_bound cos_prog cosp_T cos_poly eq_refl cos_prog_shadow cosp_run r Fr Hr) as [Fp Hp].
  split; [exact Fp|].
  assert (Hr' : -3.17 <= B2R r <= 3.17) by (apply Rabs_le_inv in Hr; lra).
  pose proof (cos_series_bound (B2R r) Hr') as Hs. fold (cos_poly (B2R r)) in Hs.
  pose proof (red_to_true (B2R x) (B2R r) n _ Hn Hd) as Ht.
  pose proof (cos_lipschitz (B2R r) (B2R x + 2 * IZR (- n) * PI)) as Hl.
  rewrite cos_period_Z in Hl.
  exact (chain4 _ _ _ _ _ _ _ _ _ _ Hp Hs Hl Ht).
Qed.

(** * Sigmoid, tanh, softplus and GELU *)

Lemma comp_bound : forall p res T lo hi (f : binary32 -> binary32) (g : R -> R),
  (forall x, f x = fnth (feval p [x]) res) ->
  (forall xr, snth (seval p [xr]) res = g xr) ->
  erun 60 (comp_chk p res exp_c log_c T lo hi) mid lo hi = true ->
  forall x : binary32, is_finite x = true -> B2R lo <= B2R x <= B2R hi ->
  is_finite (f x) = true /\ Rabs (B2R (f x) - g (B2R x)) <= dR (fst T) (snd T).
Proof.
  intros p res T lo hi f g Hf Hg R x Fx Hx.
  destruct (erun_sound _ (comp_chk p res exp_c log_c T lo hi) mid
              (comp_chk_sound p res exp_c log_c T lo hi exp_bound log_bound) 60 _ _ R x Fx Hx)
    as (_ & F & H).
  rewrite Hg in H. rewrite Hf. exact (conj F H).
Qed.

Lemma B2R_small_int : forall z, (Z.abs z < 16777216)%Z -> B2R (f32_of_Z z) = IZR z.
Proof. intros z H. exact (proj2 (f32_of_Z_exact z H)). Qed.

Theorem sigmoid_bound : forall x : binary32, is_finite x = true -> -88 <= B2R x <= 88 ->
  is_finite (f32_sigmoid x) = true /\
  Rabs (B2R (f32_sigmoid x) - sigmoid (B2R x)) <= dR (fst sig_T) (snd sig_T).
Proof.
  intros x Fx Hx. destruct B2R_88 as [Fh Vh]. destruct B2R_m88 as [Fl Vl].
  apply (comp_bound sig_prog 5 sig_T f32_exp_lo f32_exp_hi f32_sigmoid sigmoid
           sig_prog_eq sig_prog_shadow sig_run x Fx).
  rewrite Vl, Vh. exact Hx.
Qed.

Theorem tanh_bound : forall x : binary32, is_finite x = true -> -44 <= B2R x <= 44 ->
  is_finite (f32_tanh x) = true /\
  Rabs (B2R (f32_tanh x) - tanh (B2R x)) <= dR (fst tanh_T) (snd tanh_T).
Proof.
  intros x Fx Hx.
  apply (comp_bound tanh_prog 9 tanh_T c44n c44 f32_tanh tanh
           tanh_prog_eq tanh_prog_shadow tanh_run x Fx).
  unfold c44n, c44. rewrite !B2R_small_int by (cbn; lia). exact Hx.
Qed.

Theorem softplus_bound : forall x : binary32, is_finite x = true -> -88 <= B2R x <= 88 ->
  is_finite (f32_softplus x) = true /\
  Rabs (B2R (f32_softplus x) - ln (1 + exp (B2R x)))
  <= dR (fst softplus_T) (snd softplus_T) + 2 / 100000000.
Proof.
  intros x Fx Hx. destruct B2R_88 as [Fh Vh]. destruct B2R_m88 as [Fl Vl].
  destruct (comp_bound softplus_prog 38 softplus_T f32_exp_lo f32_exp_hi f32_softplus
              (fun xr => Rmax xr 0 + log_poly ((1 + exp (- Rabs xr) - 1) / (1 + exp (- Rabs xr) + 1)))
              softplus_prog_eq softplus_prog_shadow softplus_run x Fx
              ltac:(rewrite Vl, Vh; exact Hx)) as [F H].
  split; [exact F|].
  exact (chain2 _ _ _ _ _ H (softplus_shadow_true (B2R x))).
Qed.

(** GELU against the formula with [sqrt (2/pi)] and [0.044715], through the
    Lipschitz constant of [tanh]. *)
Lemma gelu_shadow_true : forall xr, -8 <= xr <= 8 ->
  Rabs (gelu_shadow xr - gelu xr) <= 56 / 10000000.
Proof.
  intros xr Hr. assert (Ha : Rabs xr <= 8) by (apply Rabs_le; lra).
  pose proof (gelu_inner_close xr Ha) as Hin. unfold Rgelu_inner in Hin.
  rewrite B2R_gelu_c1, B2R_gelu_c2 in Hin.
  unfold gelu_shadow, gelu.
  set (s := 6693141 / 8388608 * (xr + 12003091 / 268435456 * (xr * (xr * xr)))) in *.
  set (t := sqrt (2 / PI) * (xr + 44715 / 1000000 * xr ^ 3)) in *.
  pose proof (tanh_close s t) as Tc.
  set (TS := tanh s) in *. set (TT := tanh t) in *. clearbody s t TS TT.
  replace (1 / 2 * xr * (1 + TS) - xr / 2 * (1 + TT)) with (xr / 2 * (TS - TT)) by field.
  rewrite Rabs_mult.
  assert (H2 : Rabs (xr / 2) <= 4).
  { unfold Rdiv. rewrite Rabs_mult, (Rabs_pos_eq (/ 2)) by lra. lra. }
  apply Rle_trans with (4 * (14 / 10000000)); [|lra].
  apply Rmult_le_compat; try apply Rabs_pos; [exact H2 | lra].
Qed.

Theorem gelu_bound : forall x : binary32, is_finite x = true -> -8 <= B2R x <= 8 ->
  is_finite (f32_gelu x) = true /\
  Rabs (B2R (f32_gelu x) - gelu (B2R x)) <= dR (fst gelu_T) (snd gelu_T) + 56 / 10000000.
Proof.
  intros x Fx Hx.
  destruct (comp_bound gelu_prog 20 gelu_T c8n c8 f32_gelu gelu_shadow
              gelu_prog_eq gelu_prog_shadow gelu_run x Fx
              ltac:(unfold c8n, c8; rewrite !B2R_small_int by (cbn; lia); exact Hx)) as [F H].
  split; [exact F|].
  exact (chain2 _ _ _ _ _ H (gelu_shadow_true (B2R x) Hx)).
Qed.
