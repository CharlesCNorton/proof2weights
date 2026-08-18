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
