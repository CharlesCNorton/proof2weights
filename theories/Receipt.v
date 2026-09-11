(** Proof-carrying inference receipts.

    A receipt binds one generation to the exact weights (by checksum), the
    prompt, the output, and the IEEE-754 semantics under which it was produced.
    It is checkable without trusting the producer: recompute the weight checksum
    from the file and re-run the deterministic verified generation, then compare.
    Because the forward pass is a pure verified function, the regeneration is
    reproducible, and because generation always extends the prompt
    (gpt2_generation_preserves_prompt), the prompt is a prefix of the recorded
    output. This file defines the receipt, the checker, and the soundness
    lemmas; the checksum reuses the development's gpt2_compute_checksum. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import List.
From Stdlib Require Import Bool.
From Stdlib Require Import Lia.
From Stdlib Require Import Znumtheory.
From Stdlib Require Import Zpow_facts.
Require Import Phases1_15_complete.

Import ListNotations.
Open Scope Z_scope.

Record inference_receipt := mk_inference_receipt {
  ir_model_checksum : Z;     (* gpt2_compute_checksum of the weight bytes *)
  ir_n_layer : nat;          (* model depth, part of the bound configuration *)
  ir_prompt : list nat;      (* input token ids *)
  ir_output : list nat       (* full token sequence: prompt followed by generated *)
}.

Fixpoint nat_list_eqb (a b : list nat) : bool :=
  match a, b with
  | [], [] => true
  | x :: a', y :: b' => Nat.eqb x y && nat_list_eqb a' b'
  | _, _ => false
  end.

Lemma nat_list_eqb_sound : forall a b, nat_list_eqb a b = true -> a = b.
Proof.
  induction a as [|x a IH]; intros [|y b] H; simpl in H; try discriminate; [reflexivity|].
  apply andb_prop in H. destruct H as [Hh Ht].
  apply Nat.eqb_eq in Hh. subst y. f_equal. apply IH. exact Ht.
Qed.

Lemma nat_list_eqb_refl : forall a, nat_list_eqb a a = true.
Proof.
  induction a as [|x a IH]; simpl; [reflexivity|].
  rewrite Nat.eqb_refl. exact IH.
Qed.

(** A receipt verifies against a freshly recomputed weight checksum and a fresh
    regeneration of the full token sequence. *)
Definition verify_receipt (recomputed_checksum : Z) (regenerated : list nat)
                          (r : inference_receipt) : bool :=
  Z.eqb recomputed_checksum (ir_model_checksum r)
  && nat_list_eqb regenerated (ir_output r).

(** Soundness: a passing check certifies that the recorded weights match the
    recomputed checksum and the recorded output is exactly the regenerated
    deterministic output. *)
Theorem verify_receipt_sound : forall c regen r,
  verify_receipt c regen r = true ->
  ir_model_checksum r = c /\ ir_output r = regen.
Proof.
  intros c regen r H. unfold verify_receipt in H.
  apply andb_prop in H. destruct H as [Hc Ho].
  apply Z.eqb_eq in Hc. apply nat_list_eqb_sound in Ho.
  split; [symmetry; exact Hc | symmetry; exact Ho].
Qed.

(** Completeness: an honestly produced receipt verifies against its own inputs. *)
Theorem verify_receipt_complete : forall r,
  verify_receipt (ir_model_checksum r) (ir_output r) r = true.
Proof.
  intros r. unfold verify_receipt.
  rewrite Z.eqb_refl, nat_list_eqb_refl. reflexivity.
Qed.

(** The recorded output preserves the prompt as a prefix, checkable directly. *)
Definition receipt_prompt_ok (r : inference_receipt) : bool :=
  gpt2_check_prompt_preserved (ir_prompt r) (ir_output r).

(** For any output produced by the verified greedy generation, the prompt is a
    prefix: the receipt's prompt-preservation is guaranteed by the proven
    generation property, not merely checked. *)
Theorem receipt_output_extends_prompt : forall forward cfg prompt,
  List.firstn (List.length prompt) (gpt2_generate forward cfg prompt) = prompt.
Proof.
  intros forward cfg prompt. apply gpt2_generation_preserves_prompt.
Qed.

(** * What the checksum detects

    The receipt records a checksum of the weight file: a rolling polynomial
    evaluation modulo [2^32]. Altering any single byte changes it, so a receipt
    distinguishes the file it was issued for from every file differing from it
    in one position. The function is not a cryptographic hash: an adversary
    free to choose both files can exhibit a collision, so what the checksum
    establishes is that the recorded weights are the weights that ran, not that
    no other file could have been substituted. *)

(** The same rolling evaluation without the intermediate reduction. *)
Fixpoint ck_raw (bs : list Z) (acc : Z) : Z :=
  match bs with
  | [] => acc
  | b :: r => ck_raw r (acc * 31 + b)
  end.

Lemma ck_raw_app : forall xs ys a,
  ck_raw (xs ++ ys) a = ck_raw ys (ck_raw xs a).
Proof.
  induction xs as [|x xs IH]; intros ys a; simpl; [reflexivity | apply IH].
Qed.

Lemma ck_raw_shift : forall bs a,
  ck_raw bs a = a * 31 ^ Z.of_nat (List.length bs) + ck_raw bs 0.
Proof.
  induction bs as [|b bs IH]; intros a.
  - cbn [ck_raw List.length Z.of_nat]. rewrite Z.pow_0_r. ring.
  - cbn [ck_raw List.length].
    rewrite (IH (a * 31 + b)), (IH (0 * 31 + b)).
    rewrite Nat2Z.inj_succ, Z.pow_succ_r by apply Nat2Z.is_nonneg.
    ring.
Qed.

Lemma ck_raw_mod : forall bs a a',
  a mod 4294967296 = a' mod 4294967296 ->
  ck_raw bs a mod 4294967296 = ck_raw bs a' mod 4294967296.
Proof.
  intros bs a a' H.
  rewrite (ck_raw_shift bs a), (ck_raw_shift bs a').
  rewrite Zplus_mod, (Zplus_mod (a' * _)).
  do 2 f_equal.
  rewrite Zmult_mod, (Zmult_mod a'), H. reflexivity.
Qed.

Lemma ck_fold_congr : forall bs a,
  List.fold_left (fun acc b => (acc * 31 + b) mod 4294967296) bs a mod 4294967296
  = ck_raw bs a mod 4294967296.
Proof.
  induction bs as [|b bs IH]; intros a; simpl; [reflexivity|].
  rewrite IH. apply ck_raw_mod. rewrite Zmod_mod. reflexivity.
Qed.

Lemma ck_fold_range : forall l a,
  0 <= a < 4294967296 ->
  0 <= List.fold_left (fun acc b => (acc * 31 + b) mod 4294967296) l a < 4294967296.
Proof.
  induction l as [|x l IH]; intros a Ha; simpl; [exact Ha|].
  apply IH. apply Z.mod_pos_bound. lia.
Qed.

(** The checksum is the unreduced evaluation, taken modulo [2^32]. *)
Lemma checksum_is_raw : forall bs,
  gpt2_compute_checksum bs = ck_raw bs 0 mod 4294967296.
Proof.
  intros bs. unfold gpt2_compute_checksum.
  rewrite <- (ck_fold_congr bs 0).
  symmetry. apply Zmod_small. apply ck_fold_range. lia.
Qed.

(** Two byte strings of equal length differing in exactly one position have
    different checksums. *)
Theorem checksum_detects_single_byte : forall pre post b b',
  0 <= b < 256 -> 0 <= b' < 256 -> b <> b' ->
  gpt2_compute_checksum (pre ++ b :: post)
  <> gpt2_compute_checksum (pre ++ b' :: post).
Proof.
  intros pre post b b' Hb Hb' Hne Heq.
  rewrite !checksum_is_raw in Heq.
  assert (Hn : 0 <= Z.of_nat (List.length post)) by apply Nat2Z.is_nonneg.
  assert (Hexp : forall c, ck_raw (pre ++ c :: post) 0
                 = (ck_raw pre 0 * 31 + c) * 31 ^ Z.of_nat (List.length post)
                   + ck_raw post 0).
  { intros c. rewrite ck_raw_app. simpl. apply ck_raw_shift. }
  rewrite !Hexp in Heq.
  assert (Hdiv : (4294967296 | (b - b') * 31 ^ Z.of_nat (List.length post))).
  { apply Zmod_divide; [lia|].
    replace ((b - b') * 31 ^ Z.of_nat (List.length post))
      with (((ck_raw pre 0 * 31 + b) * 31 ^ Z.of_nat (List.length post)
             + ck_raw post 0)
            - ((ck_raw pre 0 * 31 + b') * 31 ^ Z.of_nat (List.length post)
               + ck_raw post 0)) by ring.
    rewrite Zminus_mod, Heq, Z.sub_diag. reflexivity. }
  assert (Hrp : rel_prime 4294967296 (31 ^ Z.of_nat (List.length post))).
  { apply rel_prime_Zpower_r; [exact Hn|].
    apply Zgcd_1_rel_prime. reflexivity. }
  assert (Hdb : (4294967296 | b - b')).
  { apply Gauss with (31 ^ Z.of_nat (List.length post));
      [rewrite Z.mul_comm; exact Hdiv | exact Hrp]. }
  assert (Hnz : b - b' <> 0) by lia.
  pose proof (Zdivide_bounds _ _ Hdb Hnz) as Hle.
  rewrite (Z.abs_eq 4294967296) in Hle by lia.
  assert (Habs : Z.abs (b - b') < 4294967296).
  { rewrite Z.abs_lt. lia. }
  lia.
Qed.
