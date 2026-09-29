(** * The logarithm and the absolute value in the enclosure arithmetic

    Qwen3.5's softplus takes both. The logarithm goes through CoqInterval's
    interval logarithm when the argument's interval lies above zero and is the
    whole line otherwise; the absolute value is exact on the endpoints. *)

From Stdlib Require Import ZArith Reals Lra Lia.
From Flocq Require Import Core.
From Interval Require Import Specific_stdz Specific_ops Float_full Interval Xreal Basic.
Require Import Enclose.

Open Scope R_scope.

Section Ext.

Variable FB : Z.
Variable prec : Z.

Definition enc_ln (a : enc) : enc :=
  match iv FB a with
  | Some (lo, hi) =>
      if (0 <? lo)%Z then of_interval FB (EI.ln prec (to_interval FB lo hi)) else EFail
  | None => EFail
  end.

Definition enc_abs (a : enc) : enc :=
  match a with
  | EFail => EFail
  | EPt m e => EPt (Z.abs m) e
  | EIv lo hi =>
      if (0 <=? lo)%Z then EIv lo hi
      else if (hi <=? 0)%Z then EIv (- hi) (- lo)
      else EIv 0 (Z.max (- lo) hi)
  end.

Lemma sc_pos : forall z, (0 < z)%Z -> 0 < sc FB z.
Proof.
  intros z Hz. unfold sc. apply Rmult_lt_0_compat; [apply IZR_lt; exact Hz | apply bpow_gt_0].
Qed.

Lemma sc_nonneg : forall z, (0 <= z)%Z -> 0 <= sc FB z.
Proof.
  intros z Hz. unfold sc. apply Rmult_le_pos; [apply IZR_le; exact Hz | apply bpow_ge_0].
Qed.

Lemma sc_opp' : forall z, sc FB (- z) = - sc FB z.
Proof. intros z. unfold sc. rewrite opp_IZR. ring. Qed.

Lemma sc_zmax' : forall a b, sc FB (Z.max a b) = Rmax (sc FB a) (sc FB b).
Proof.
  intros a b. unfold sc. destruct (Z.max_spec a b) as [[H E]|[H E]]; rewrite E.
  - rewrite Rmax_right; [reflexivity|].
    apply Rmult_le_compat_r; [apply bpow_ge_0 | apply IZR_le; lia].
  - rewrite Rmax_left; [reflexivity|].
    apply Rmult_le_compat_r; [apply bpow_ge_0 | apply IZR_le; lia].
Qed.

Lemma enc_ln_in : forall a r, enc_in FB a r -> enc_in FB (enc_ln a) (ln r).
Proof.
  intros a r Ha. unfold enc_ln.
  destruct (iv FB a) as [[lo hi]|] eqn:Ea; [|exact I].
  destruct (0 <? lo)%Z eqn:El; [|exact I].
  apply Z.ltb_lt in El.
  destruct (iv_in FB a r lo hi Ha Ea) as [A1 A2].
  assert (Hpos : 0 < r) by (eapply Rlt_le_trans; [apply sc_pos; exact El | exact A1]).
  apply of_interval_in.
  generalize (EI.ln_correct prec (to_interval FB lo hi) (Xreal r)).
  cbn [Xbind]. unfold Xln'. rewrite is_positive_true by exact Hpos.
  intros H. apply H.
  unfold to_interval. cbn [EI.convert]. rewrite !toX_Float. cbn.
  fold (sc FB lo). fold (sc FB hi). split; assumption.
Qed.

Lemma enc_abs_in : forall a r, enc_in FB a r -> enc_in FB (enc_abs a) (Rabs r).
Proof.
  intros [|m e|lo hi] r Ha; cbn [enc_abs enc_in] in *.
  - exact I.
  - subst r. rewrite Rabs_mult, (Rabs_pos_eq (bpow radix2 e)) by apply bpow_ge_0.
    rewrite <- abs_IZR. reflexivity.
  - destruct Ha as [A1 A2].
    destruct (0 <=? lo)%Z eqn:E1.
    + apply Z.leb_le in E1. pose proof (sc_nonneg lo E1).
      rewrite Rabs_pos_eq by lra. cbn [enc_in]. lra.
    + destruct (hi <=? 0)%Z eqn:E2.
      * apply Z.leb_le in E2.
        assert (sc FB hi <= 0) by (rewrite <- (Ropp_involutive (sc FB hi)), <- sc_opp';
                                    pose proof (sc_nonneg (- hi) ltac:(lia)); lra).
        rewrite Rabs_left1 by lra. cbn [enc_in]. rewrite !sc_opp'. lra.
      * cbn [enc_in]. rewrite sc_zmax', sc_opp'. unfold sc at 1. rewrite Rmult_0_l.
        split; [apply Rabs_pos|].
        unfold Rabs. destruct (Rcase_abs r).
        -- apply Rle_trans with (- sc FB lo); [lra | apply Rmax_l].
        -- apply Rle_trans with (sc FB hi); [lra | apply Rmax_r].
Qed.

End Ext.
