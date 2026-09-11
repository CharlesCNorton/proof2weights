(** * Satisfiable premises for the elementwise stages

    The composed bounds rest on records of side conditions, one record per
    stage. This file exhibits arguments meeting the records of the
    exponential, sigmoid, tanh and GELU, with [M = 256], [m = 1] and [L = 512],
    and composes them into the bounds themselves. Two points are covered: an
    exact zero, where rounding is exact and a relative-only error model has no
    purchase, and the argument [-1000], which the exponential saturates to
    [-88] and whose result is subnormal, as for a masked attention score.

    Every condition on a binary32 quantity is decided by computation: the
    quantity is read as the rational it denotes ([Qb], [Qb_correct]) and the
    comparison is evaluated. Conditions on the real evaluation are proved from
    the range the reduction guarantees. *)

From Stdlib Require Import ZArith QArith Qreals Reals Lra Lia.
From Flocq Require Import Core IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete Float_error.

Open Scope R_scope.

(** * Deciding a condition on binary32 quantities *)

(** Fold a real expression built from [B2R] of concrete floats by [+], [*]
    and [/] into [Q2R] of one rational. *)
Ltac fold_Q :=
  repeat rewrite Qb_correct;
  repeat first
    [ rewrite <- Q2R_mult
    | rewrite <- Q2R_plus
    | rewrite <- Q2R_div by (let Hq := fresh in intro Hq; vm_compute in Hq; discriminate) ].

Ltac qcheck :=
  fold_Q;
  first
    [ replace 256 with (Q2R (256 # 1)) by (unfold Q2R; simpl; lra);
      first [ apply regz_Q | apply abs_le_Q ]; vm_compute; reflexivity
    | replace 1 with (Q2R (1 # 1)) by (unfold Q2R; simpl; lra);
      apply le_abs_Q; vm_compute; reflexivity ].

Lemma amp_ok_witness : amp_ok 256 1 512.
Proof.
  unfold amp_ok. rewrite sqrt_1.
  split; [lra|]. split; [lra|]. split; [lra|]. split; [lra|].
  split.
  - replace (1 / 1 + 256 / (1 * 1)) with 257 by field. lra.
  - replace (1 / (2 * 1)) with (/ 2) by field. lra.
Qed.

Lemma budget_lt_emax : 256 < bpow radix2 emax32.
Proof.
  replace 256 with (bpow radix2 8) by reflexivity.
  apply bpow_lt. unfold emax32. lia.
Qed.

(** * The real evaluation at the two points *)

Lemma Rsat_0 : Rsat 0 = 0.
Proof. apply Rsat_id. lra. Qed.

Lemma Rsat_m1000 : Rsat (-1000) = -88.
Proof.
  destruct (Rsat_cases (-1000)) as [[H _]|[[_ H]|[H _]]]; [lra | exact H | lra].
Qed.

Lemma Re_k_0 : Re_k 0 = 0%Z.
Proof. unfold Re_k. apply Znearest_imp. apply Rabs_def1; lra. Qed.

Lemma Re_k_m88 : Re_k (-88) = (-127)%Z.
Proof.
  unfold Re_k. rewrite B2R_inv_ln2. apply Znearest_imp.
  apply Rabs_def1; lra.
Qed.

Lemma Re_r_0 : Re_r 0 = 0.
Proof. unfold Re_r, Re_r1, Re_khi, Re_klo. rewrite Re_k_0. ring. Qed.

(** The powers of the reduced argument stay below one wherever the saturated
    argument can be. *)
Lemma Re_pows_small : forall rx, -88 <= rx <= 88 ->
  Rabs (Re_r rx) <= 1 /\ Rabs (Re_r2 rx) <= 1 /\ Rabs (Re_r3 rx) <= 1
  /\ Rabs (Re_r4 rx) <= 1 /\ Rabs (Re_r5 rx) <= 1 /\ Rabs (Re_r6 rx) <= 1
  /\ Rabs (Re_r7 rx) <= 1.
Proof.
  intros rx Hrx.
  pose proof (Re_r_range rx Hrx) as H1.
  assert (Hr : Rabs (Re_r rx) <= 1) by lra.
  assert (P : forall a, Rabs a <= 1 -> Rabs (a * Re_r rx) <= 1).
  { intros a Ha. rewrite Rabs_mult.
    apply Rle_trans with (1 * 1); [|lra].
    apply Rmult_le_compat; try apply Rabs_pos; assumption. }
  assert (H2 : Rabs (Re_r2 rx) <= 1) by (unfold Re_r2; apply P, Hr).
  assert (H3 : Rabs (Re_r3 rx) <= 1) by (unfold Re_r3; apply P, H2).
  assert (H4 : Rabs (Re_r4 rx) <= 1) by (unfold Re_r4; apply P, H3).
  assert (H5 : Rabs (Re_r5 rx) <= 1) by (unfold Re_r5; apply P, H4).
  assert (H6 : Rabs (Re_r6 rx) <= 1) by (unfold Re_r6; apply P, H5).
  assert (H7 : Rabs (Re_r7 rx) <= 1) by (unfold Re_r7; apply P, H6).
  repeat split; assumption.
Qed.

Lemma Re_exp_approx_0 : Re_exp_approx 0 = 1.
Proof.
  unfold Re_exp_approx. rewrite Rsat_0. unfold Re_exp_core. rewrite Re_k_0.
  unfold Re_poly, Re_p6, Re_p5, Re_p4, Re_p3, Re_p2, Re_p1,
         Re_d7, Re_d6, Re_d5, Re_d4, Re_d3, Re_d2,
         Re_r7, Re_r6, Re_r5, Re_r4, Re_r3, Re_r2.
  rewrite Re_r_0. simpl bpow. unfold Rdiv. ring.
Qed.

Lemma Rsigmoid_0 : Rsigmoid 0 = / 2.
Proof. unfold Rsigmoid. rewrite Ropp_0, Re_exp_approx_0. field. Qed.

Lemma B2R_two_w : B2R f32_two = 2.
Proof.
  rewrite Qb_correct.
  rewrite (Qeq_eqR _ (inject_Z 2)) by (vm_compute; reflexivity).
  apply Q2R_inject.
Qed.

Lemma Rtanh_0 : Rtanh 0 = 0.
Proof. unfold Rtanh. rewrite Rmult_0_r, Rsigmoid_0, B2R_two_w. field. Qed.

Lemma Rgelu_inner_0 : Rgelu_inner 0 = 0.
Proof. unfold Rgelu_inner. ring. Qed.

(** * The exponential *)

(** A bound on a power of the real reduced argument. *)
Ltac rpow_bound :=
  match goal with
  | |- Rabs (_ ?r) <= 256 =>
      let H := fresh in
      assert (H : -88 <= r <= 88) by lra;
      destruct (Re_pows_small r H) as (? & ? & ? & ? & ? & ? & ?); lra
  end.

Ltac exp_core_witness :=
  constructor;
  rewrite ?Re_k_0, ?Re_k_m88;
  match goal with
  | |- B2SF _ = B2SF _ => vm_compute; reflexivity
  | |- Rabs (bpow radix2 _) <= _ =>
      rewrite Rabs_pos_eq by (left; apply bpow_gt_0);
      apply Rle_trans with (bpow radix2 0);
      [apply bpow_le; lia | simpl; lra]
  | |- Rabs (_ ?r) <= 256 =>
      first [ rpow_bound | qcheck ]
  | |- _ => qcheck
  end.

Lemma exp_reg_zero : exp_reg 256 1 f32_zero 0.
Proof. unfold exp_reg. rewrite Rsat_0. exp_core_witness. Qed.

Lemma exp_reg_masked : exp_reg 256 1 (f32_of_Z (-1000)) (-1000).
Proof. unfold exp_reg. rewrite Rsat_m1000. exp_core_witness. Qed.

(** * Sigmoid, tanh and GELU at zero *)

Ltac sig_witness :=
  constructor;
  [ unfold exp_reg; rewrite Ropp_0, Rsat_0; exp_core_witness
  | qcheck
  | rewrite Rabs_R1; lra
  | qcheck
  | rewrite Ropp_0, Re_exp_approx_0, Rabs_pos_eq by lra; lra
  | qcheck ].

Lemma sig_reg_zero : sig_reg 256 1 f32_zero 0.
Proof. sig_witness. Qed.

Ltac tanh_witness :=
  constructor;
  [ qcheck
  | rewrite Rabs_R0; lra
  | qcheck
  | rewrite Rmult_0_r; sig_witness
  | rewrite Rmult_0_r, Rsigmoid_0, Rabs_pos_eq by lra; lra
  | qcheck
  | qcheck ].

Lemma tanh_reg_zero : tanh_reg 256 1 f32_zero 0.
Proof. tanh_witness. Qed.

Lemma gelu_reg_zero : gelu_reg 256 1 f32_zero 0.
Proof.
  constructor.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - vm_compute. reflexivity.
  - qcheck.
  - rewrite Rabs_R0. lra.
  - qcheck.
  - rewrite Rmult_0_r, Rabs_R0. lra.
  - qcheck.
  - qcheck.
  - rewrite !Rmult_0_r, Rabs_R0. lra.
  - qcheck.
  - qcheck.
  - qcheck.
  - rewrite !Rmult_0_r, Rplus_0_r, Rabs_R0. lra.
  - qcheck.
  - rewrite Rgelu_inner_0. tanh_witness.
  - qcheck.
  - qcheck.
  - qcheck.
  - qcheck.
  - rewrite Rgelu_inner_0, Rtanh_0, Rplus_0_r, Rabs_R1. lra.
  - qcheck.
Qed.

(** * The bounds at those points

    The float results are within the composed error of the real evaluations,
    starting from exact inputs. *)

Lemma ok_input_zero : ok (errN 256 512 0) f32_zero 0.
Proof. apply ok_f32_zero. Qed.

Lemma ok_input_m1000 : ok (errN 256 512 0) (f32_of_Z (-1000)) (-1000).
Proof.
  split; [vm_compute; reflexivity|].
  assert (H : B2R (f32_of_Z (-1000)) = -1000).
  { rewrite Qb_correct.
    rewrite (Qeq_eqR _ (inject_Z (-1000))) by (vm_compute; reflexivity).
    apply Q2R_inject. }
  rewrite H. replace (-1000 - -1000) with 0 by ring. rewrite Rabs_R0.
  simpl. lra.
Qed.

Theorem exp_at_zero :
  ok (errN 256 512 21) (f32_exp_approx f32_zero) (Re_exp_approx 0).
Proof.
  apply (ok_exp_approx 256 1 512 0 f32_zero 0 budget_lt_emax amp_ok_witness
           ok_input_zero exp_reg_zero).
Qed.

Theorem exp_at_masked :
  ok (errN 256 512 21) (f32_exp_approx (f32_of_Z (-1000))) (Re_exp_approx (-1000)).
Proof.
  apply (ok_exp_approx 256 1 512 0 (f32_of_Z (-1000)) (-1000) budget_lt_emax
           amp_ok_witness ok_input_m1000 exp_reg_masked).
Qed.

Theorem sigmoid_at_zero :
  ok (errN 256 512 23) (f32_sigmoid f32_zero) (Rsigmoid 0).
Proof.
  apply (ok_sigmoid 256 1 512 0 f32_zero 0 budget_lt_emax amp_ok_witness
           ok_input_zero sig_reg_zero).
Qed.

Theorem gelu_at_zero :
  ok (errN 256 512 33) (f32_gelu f32_zero) (Rgelu 0).
Proof.
  apply (ok_gelu 256 1 512 0 f32_zero 0 budget_lt_emax amp_ok_witness
           ok_input_zero gelu_reg_zero).
Qed.
