(** Llama-architecture float primitives, on top of the proof2weights binary32
    development: RMSNorm, SiLU, and the trigonometric functions RoPE needs.

    RMSNorm and SiLU compose existing verified f32 operations. f32_sin and
    f32_cos are defined here by argument reduction modulo 2*pi (nearest-integer
    rounding via the add-then-subtract magic-constant trick) followed by a
    Taylor polynomial; this is the specification. The native extraction
    (Llama_native.v) replaces f32_sin and f32_cos with the host's libm, in the
    same trusted manner as the basic arithmetic. The remaining primitives are
    bit-identical between the two builds. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import List.
From Flocq Require Import IEEE754.BinarySingleNaN.
Require Import Phases1_15_complete.

Import ListNotations.
Open Scope Z_scope.

(** SiLU: x * sigmoid(x). *)
Definition f32_silu (x : binary32) : binary32 := f32_mult x (f32_sigmoid x).
Definition f32_silu_vec (v : list binary32) : list binary32 := List.map f32_silu v.

(** RMSNorm: out_i = w_i * (x_i * rsqrt(mean(x^2) + eps)). *)
Definition f32_rmsnorm (w : list binary32) (eps : binary32) (x : list binary32) : list binary32 :=
  let n := f32_of_Z (Z.of_nat (List.length x)) in
  let ss := f32_sum (List.map (fun xi => f32_mult xi xi) x) in
  let ms := f32_div ss n in
  let rs := f32_div f32_one (f32_sqrt (f32_plus ms eps)) in
  List.map (fun '(wi, xi) => f32_mult wi (f32_mult xi rs)) (List.combine w x).

Lemma f32_rmsnorm_length : forall w eps x,
  List.length w = List.length x ->
  List.length (f32_rmsnorm w eps x) = List.length x.
Proof.
  intros w eps x H. unfold f32_rmsnorm.
  rewrite List.length_map, List.length_combine, H, Nat.min_id. reflexivity.
Qed.

(** pi and derived constants as binary32 values. *)
Definition f32_pi : binary32 := f32_div (f32_of_Z 31415927) (f32_of_Z 10000000).
Definition f32_2pi : binary32 := f32_mult f32_two f32_pi.
Definition f32_inv2pi : binary32 := f32_div f32_one f32_2pi.
Definition f32_magic : binary32 := f32_of_Z 8388608.

(** Nearest integer of y, as a binary32, valid for |y| < 2^22. *)
Definition f32_round_int (y : binary32) : binary32 :=
  f32_minus (f32_plus y f32_magic) f32_magic.

(** Reduce x into [-pi, pi]. *)
Definition f32_reduce_2pi (x : binary32) : binary32 :=
  let k := f32_round_int (f32_mult x f32_inv2pi) in
  f32_minus x (f32_mult k f32_2pi).

Definition f32_fac2 : binary32 := f32_of_Z 2.
Definition f32_fac3 : binary32 := f32_of_Z 6.
Definition f32_fac4 : binary32 := f32_of_Z 24.
Definition f32_fac5 : binary32 := f32_of_Z 120.
Definition f32_fac6 : binary32 := f32_of_Z 720.
Definition f32_fac7 : binary32 := f32_of_Z 5040.
Definition f32_fac8 : binary32 := f32_of_Z 40320.
Definition f32_fac9 : binary32 := f32_of_Z 362880.
Definition f32_fac10 : binary32 := f32_of_Z 3628800.
Definition f32_fac11 : binary32 := f32_of_Z 39916800.

(** sin via Taylor on the reduced argument:
    r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880 - r^11/39916800. *)
Definition f32_sin (x : binary32) : binary32 :=
  let r := f32_reduce_2pi x in
  let r2 := f32_mult r r in
  let r3 := f32_mult r2 r in
  let r5 := f32_mult r3 r2 in
  let r7 := f32_mult r5 r2 in
  let r9 := f32_mult r7 r2 in
  let r11 := f32_mult r9 r2 in
  f32_minus
    (f32_plus
      (f32_minus
        (f32_plus
          (f32_minus r (f32_div r3 f32_fac3))
          (f32_div r5 f32_fac5))
        (f32_div r7 f32_fac7))
      (f32_div r9 f32_fac9))
    (f32_div r11 f32_fac11).

(** cos via Taylor on the reduced argument:
    1 - r^2/2 + r^4/24 - r^6/720 + r^8/40320 - r^10/3628800. *)
Definition f32_cos (x : binary32) : binary32 :=
  let r := f32_reduce_2pi x in
  let r2 := f32_mult r r in
  let r4 := f32_mult r2 r2 in
  let r6 := f32_mult r4 r2 in
  let r8 := f32_mult r6 r2 in
  let r10 := f32_mult r8 r2 in
  f32_minus
    (f32_plus
      (f32_minus
        (f32_plus
          (f32_minus f32_one (f32_div r2 f32_fac2))
          (f32_div r4 f32_fac4))
        (f32_div r6 f32_fac6))
      (f32_div r8 f32_fac8))
    (f32_div r10 f32_fac10).
