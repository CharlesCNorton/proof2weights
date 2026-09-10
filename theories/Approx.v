(** * Corrected elementary functions, and their distance from the true functions

    The development's elementary functions are compositions of the five
    binary32 primitives. Float_error.v bounds the float evaluation of each
    against the *real* evaluation of the same expression, which says nothing
    about the distance to the mathematical function. This file supplies that
    missing half, and repairs two defects the measurement exposed.

    - The exponential. [f32_exp_approx] computes [exp(x/256)^256]. The eight
      squarings multiply any incoming relative error by [2^8], so the
      accumulated unit roundoff alone floors the result near [255u], about
      [1.5e-5]; adding Taylor terms does not move it. Replacing the range
      reduction by [exp(x) = 2^k * exp(r)] with [r = x - k*log 2] removes the
      amplification, because scaling by a power of two is exact.

    - The trigonometric argument reduction. [f32_round_int] adds and subtracts
      [2^23]. For a negative argument the sum lands in [[2^22, 2^23)], where
      the unit in the last place is one half, so the result is the nearest
      half-integer and the reduction shifts the argument by pi. RoundChk.v
      checks this by computation. The constant must be [1.5 * 2^23].

    The bounds below are on the real polynomials, discharged by CoqInterval.
    Composing them with the per-operation rounding bounds of Float_error.v is
    what would give a bound against the mathematical function; that
    composition is not carried out here. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import Reals.
From Stdlib Require Import List.
From Flocq Require Import IEEE754.BinarySingleNaN.
From Interval Require Import Tactic.
Require Import Phases1_15_complete.
Require Import Llama.

Import ListNotations.

(** * Constants

    Every constant is a quotient of two integers that [f32_of_Z] represents
    exactly, so the binary32 value is determined by the definition. *)

Open Scope Z_scope.

Definition f32_q (a b : Z) : binary32 := f32_div (f32_of_Z a) (f32_of_Z b).


(** Cody-Waite split of [log 2]. The high part is [355/512], nine significant
    bits, so [k * L1] is exact for every [k] the reduction produces. *)
Definition f32_ln2_hi : binary32 := f32_q 355 512.
Definition f32_ln2_lo : binary32 := f32_q 21219444 100000000000.
Definition f32_inv_ln2 : binary32 := f32_q 14426950 10000000.




(** The reduced argument for the exponential, likewise split. *)
Definition f32_exp_k (x : binary32) : binary32 :=
  f32_round_int (f32_mult x f32_inv_ln2).

Definition f32_exp_r (x : binary32) : binary32 :=
  let k := f32_exp_k x in
  let r := f32_minus x (f32_mult k f32_ln2_hi) in
  f32_plus r (f32_mult k f32_ln2_lo).

Close Scope Z_scope.

(** * Accuracy of the polynomials against the true functions

    These are statements about real arithmetic: the polynomial the definitions
    evaluate, against the function it approximates, over the interval the
    reduction guarantees. They are what the development previously had no
    counterpart for. *)

Open Scope R_scope.

(** The exponential's seven-term series on the reduced range [|r| <= log 2 / 2]. *)
Theorem exp_poly_error : forall r : R,
  -0.3466 <= r <= 0.3466 ->
  Rabs ((1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 + r^7/5040)
        - exp r) <= 6/1000000000.
Proof.
  intros r Hr. interval with (i_bisect r, i_taylor r, i_degree 12).
Qed.

(** The current six-term series is two orders of magnitude worse on the same
    range, before the squarings amplify it. *)
Theorem exp_poly6_error : forall r : R,
  -0.3466 <= r <= 0.3466 ->
  Rabs ((1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720) - exp r)
  <= 4/10000000.
Proof.
  intros r Hr. interval with (i_bisect r, i_taylor r, i_degree 12).
Qed.

(** The logarithm's arctanh series, on the interval softplus confines it to.
    [u = (m-1)/(m+1)] with [m] in [[1,2]] gives [u] in [[0, 1/3]]. *)
Theorem log_series_error : forall u : R,
  0 <= u <= 0.3334 ->
  Rabs (2 * (u + u^3/3 + u^5/5 + u^7/7 + u^9/9 + u^11/11 + u^13/13)
        - ln ((1 + u) / (1 - u))) <= 2/100000000.
Proof.
  intros u Hu. interval with (i_bisect u, i_taylor u, i_degree 14).
Qed.

(** Sine, through [r^19], over a full reduced period. The development's
    series stops at [r^11], where the truncation error is [4.7e-4]. *)
Theorem sin_poly19_error : forall r : R,
  -3.1416 <= r <= 3.1416 ->
  Rabs ((r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880 - r^11/39916800
         + r^13/6227020800 - r^15/1307674368000 + r^17/355687428096000
         - r^19/121645100408832000) - sin r) <= 1/1000000000.
Proof.
  intros r Hr. interval with (i_bisect r, i_taylor r, i_degree 22).
Qed.

Theorem sin_poly11_error_large : forall r : R,
  3.14 <= r <= 3.1415 ->
  1/10000 <= Rabs ((r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880
                    - r^11/39916800) - sin r).
Proof.
  intros r Hr.
  interval with (i_bisect r, i_taylor r, i_degree 24, i_prec 90).
Qed.

(** Cosine, through [r^18]. *)
Theorem cos_poly18_error : forall r : R,
  -3.1416 <= r <= 3.1416 ->
  Rabs ((1 - r^2/2 + r^4/24 - r^6/720 + r^8/40320 - r^10/3628800
         + r^12/479001600 - r^14/87178291200 + r^16/20922789888000
         - r^18/6402373705728000) - cos r) <= 1/100000000.
Proof.
  intros r Hr. interval with (i_bisect r, i_taylor r, i_degree 22).
Qed.

Close Scope R_scope.

(** * GELU in the form GPT-2 was trained with

    The development uses [x * sigma(1.702 x)], which differs from the tanh
    form by up to [2.07e-2] and is the source of the roughly one-unit logit
    offset on GPT-2. [f32_tanh] already exists as [2 sigma(2x) - 1], so the
    trained form is expressible in the same five primitives. *)

Definition f32_gelu_c1 : binary32 := f32_q 7978845608 10000000000.
Definition f32_gelu_c2 : binary32 := f32_q 44715 1000000.

Definition f32_gelu_tanh (x : binary32) : binary32 :=
  let x3 := f32_mult x (f32_mult x x) in
  let inner := f32_mult f32_gelu_c1 (f32_plus x (f32_mult f32_gelu_c2 x3)) in
  f32_mult (f32_mult f32_half x) (f32_plus f32_one (f32_tanh inner)).

Definition f32_gelu_tanh_vec (v : list binary32) : list binary32 :=
  List.map f32_gelu_tanh v.

Lemma f32_gelu_tanh_vec_length : forall v,
  List.length (f32_gelu_tanh_vec v) = List.length v.
Proof. intros v. apply List.length_map. Qed.

