(** * Rounding to the nearest integer, checked by computation

    [f32_round_int] adds and subtracts [f32_magic = 1.5 * 2^23]. For
    [|y| < 2^22] the sum lies in [[2^23, 2^24)], where the unit in the last
    place is one, so the result is the nearest integer to [y] for arguments of
    either sign. The first examples check that at representative points.

    The binade condition is what makes the constant work. With [2^23] in its
    place a negative argument lands in [[2^22, 2^23)], where the unit in the
    last place is one half, and rounds to the nearest half-integer; the last
    examples check that too.

    Comparison is on [B2SF], the proof-free projection, because two
    [binary_float] values with the same payload can carry different bound
    proofs. *)

From Stdlib Require Import ZArith.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.

Open Scope Z_scope.

Definition q (a b : Z) : binary32 := f32_div (f32_of_Z a) (f32_of_Z b).

(** * The definition in force *)

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

Example round_neg_126 : B2SF (f32_round_int (q (-1269) 10)) = B2SF (f32_of_Z (-127)).
Proof. vm_compute. reflexivity. Qed.

(** * The same scheme with [2^23] *)

Definition f32_round_2p23 (y : binary32) : binary32 :=
  f32_minus (f32_plus y (f32_of_Z 8388608)) (f32_of_Z 8388608).

(** Non-negative arguments still round to the nearest integer. *)
Example round_2p23_pos_06 : B2SF (f32_round_2p23 (q 3 5)) = B2SF f32_one.
Proof. vm_compute. reflexivity. Qed.

(** Negative ones round to the nearest half-integer: [-0.4] and [-0.6] both
    return [-0.5], and [-1.4] returns [-1.5]. *)
Example round_2p23_neg_04 : B2SF (f32_round_2p23 (q (-2) 5)) = B2SF (q (-1) 2).
Proof. vm_compute. reflexivity. Qed.

Example round_2p23_neg_06 : B2SF (f32_round_2p23 (q (-3) 5)) = B2SF (q (-1) 2).
Proof. vm_compute. reflexivity. Qed.

Example round_2p23_neg_14 : B2SF (f32_round_2p23 (q (-7) 5)) = B2SF (q (-3) 2).
Proof. vm_compute. reflexivity. Qed.

Example round_2p23_differs :
  B2SF (f32_round_2p23 (q (-2) 5)) <> B2SF (f32_round_int (q (-2) 5)).
Proof. vm_compute. discriminate. Qed.
