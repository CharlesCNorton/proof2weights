(** * A defect in the trigonometric argument reduction

    [f32_round_int] rounds to the nearest integer by adding and subtracting a
    magic constant. Llama.v uses [2^23]. That works for a non-negative
    argument, where the sum lands in [[2^23, 2^24)] and the unit in the last
    place is exactly one. It does not work for a negative argument: the sum
    lands in [[2^22, 2^23)], where the unit in the last place is one half, so
    the result is the nearest *half*-integer.

    The consequence is that [f32_sin] and [f32_cos] reduce a negative argument
    by a half-integer multiple of two pi, which shifts the reduced argument by
    pi and flips the sign of the result.

    The examples below are checked by computation. Comparison is on [B2SF],
    the proof-free projection, because two [binary_float] values with the same
    payload can carry different bound proofs.

    Llama.v now uses [1.5 * 2^23 = 12582912], which keeps the sum in the
    correct binade for arguments of either sign. The superseded constant is
    reproduced here as [f32_round_int_old] so that the defect stays checked
    rather than merely described. *)

From Stdlib Require Import ZArith.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.
Require Import Llama.

Open Scope Z_scope.

Definition q (a b : Z) : binary32 := f32_div (f32_of_Z a) (f32_of_Z b).

(** The superseded definition, kept as a regression witness. *)
Definition f32_magic_old : binary32 := f32_of_Z 8388608.

Definition f32_round_int_old (y : binary32) : binary32 :=
  f32_minus (f32_plus y f32_magic_old) f32_magic_old.

(** It rounds non-negative arguments correctly. *)
Example old_pos_04 : B2SF (f32_round_int_old (q 2 5)) = B2SF f32_zero.
Proof. vm_compute. reflexivity. Qed.

Example old_pos_06 : B2SF (f32_round_int_old (q 3 5)) = B2SF f32_one.
Proof. vm_compute. reflexivity. Qed.

(** And negative arguments to the nearest half-integer: [-0.4] and [-0.6]
    both return [-0.5], and [-1.4] returns [-1.5]. *)
Example old_neg_04 : B2SF (f32_round_int_old (q (-2) 5)) = B2SF (q (-1) 2).
Proof. vm_compute. reflexivity. Qed.

Example old_neg_06 : B2SF (f32_round_int_old (q (-3) 5)) = B2SF (q (-1) 2).
Proof. vm_compute. reflexivity. Qed.

Example old_neg_14 : B2SF (f32_round_int_old (q (-7) 5)) = B2SF (q (-3) 2).
Proof. vm_compute. reflexivity. Qed.

Example old_neg_04_not_zero :
  B2SF (f32_round_int_old (q (-2) 5)) <> B2SF f32_zero.
Proof. vm_compute. discriminate. Qed.

(** The definition in force rounds correctly for either sign. *)
Example round_pos_04 : B2SF (f32_round_int (q 2 5)) = B2SF f32_zero.
Proof. vm_compute. reflexivity. Qed.

Example round_pos_06 : B2SF (f32_round_int (q 3 5)) = B2SF f32_one.
Proof. vm_compute. reflexivity. Qed.

Example round_neg_04 : B2SF (f32_round_int (q (-2) 5)) = B2SF f32_zero.
Proof. vm_compute. reflexivity. Qed.

Example round_neg_06 : B2SF (f32_round_int (q (-3) 5)) = B2SF (f32_neg f32_one).
Proof. vm_compute. reflexivity. Qed.

Example round_neg_14 : B2SF (f32_round_int (q (-7) 5)) = B2SF (f32_neg f32_one).
Proof. vm_compute. reflexivity. Qed.

(** The two disagree, which is the content of the repair. *)
Example old_and_new_disagree :
  B2SF (f32_round_int_old (q (-2) 5)) <> B2SF (f32_round_int (q (-2) 5)).
Proof. vm_compute. discriminate. Qed.
