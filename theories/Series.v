(** * Truncation error of the series the code evaluates

    The distance from each polynomial to the function it approximates, over the
    interval the argument reduction guarantees. The divisors are the constants
    the binary32 definitions hold. For the six highest trigonometric terms those
    are not the factorials: [f32_of_Z 15!] is 1307674411008, not 1307674368000,
    because 15! needs more than twenty-four significant bits.

    This file imports only the reals and the interval tactic; Truth.v consumes
    its statements. *)

From Stdlib Require Import Reals.
From Interval Require Import Tactic.

Open Scope R_scope.

(** * Sine and cosine, on the reduced argument

    The reduction lands within [3.17] of zero for every argument of magnitude
    up to [262144] (Truth.v), which covers the largest rotary angle any of the
    three checkpoints can form: the angle is the position times the largest
    inverse frequency, which is one, so the bound is the context length. The
    interval is wider than [pi] because the stored [1/(2 pi)] and the rounding
    of the product select an integer that can sit a little off the nearest one;
    the slack grows with the argument, and at [262144] it is [0.027]. *)

Theorem sin_series_bound : forall r : R,
  -3.17 <= r <= 3.17 ->
  Rabs ((r - r^3/6 + r^5/120 - r^7/5040 + r^9/362880 - r^11/39916800
         + r^13/6227020800 - r^15/1307674411008 + r^17/355687414628352
         - r^19/121645104594157568) - sin r) <= 2/1000000000.
Proof.
  intros r Hr. interval with (i_bisect r, i_taylor r, i_degree 22).
Qed.

Theorem cos_series_bound : forall r : R,
  -3.17 <= r <= 3.17 ->
  Rabs ((1 - r^2/2 + r^4/24 - r^6/720 + r^8/40320 - r^10/3628800
         + r^12/479001600 - r^14/87178289152 + r^16/20922790576128
         - r^18/6402373530419200) - cos r) <= 2/100000000.
Proof.
  intros r Hr. interval with (i_bisect r, i_taylor r, i_degree 22).
Qed.

(** * The logarithm

    The arctanh series, with [u = (m-1)/(m+1)]. For [m] in [[1, 2.0001]], [u]
    stays below 0.3334. All six divisors are exact in binary32. *)

Theorem log_series_bound : forall u : R,
  0 <= u <= 0.3334 ->
  Rabs (2 * (u + u^3/3 + u^5/5 + u^7/7 + u^9/9 + u^11/11 + u^13/13)
        - ln ((1 + u) / (1 - u))) <= 2/100000000.
Proof.
  intros u Hu. interval with (i_bisect u, i_taylor u, i_degree 14).
Qed.

(** * The exponential

    The reduction writes [x = k log 2 + r] with [k] the nearest integer to
    [x / log 2], so [|r|] stays below 0.35. The series runs through [r^7]; all
    six divisors are exact in binary32. *)

Theorem exp_series_bound : forall r : R,
  -0.35 <= r <= 0.35 ->
  Rabs ((1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + r^6/720 + r^7/5040) - exp r)
  <= 1/100000000.
Proof.
  intros r Hr. interval with (i_bisect r, i_taylor r, i_degree 12).
Qed.

Theorem exp_lower_bound : forall r : R,
  -0.36 <= r <= 0.36 -> 69/100 <= exp r.
Proof. intros r Hr. interval. Qed.

Theorem exp_upper_bound : forall r : R,
  -0.36 <= r <= 0.36 -> exp r <= 144/100.
Proof. intros r Hr. interval. Qed.

(** The split of [log 2] the reduction uses, [355/512 - 14581891/2^36]. *)
Theorem ln2_split_bound :
  Rabs (355/512 - 14581891/68719476736 - ln 2) <= 2/1000000000000.
Proof. interval with (i_prec 80). Qed.

(** An exponent perturbed by that much moves the exponential by a comparable
    relative amount. *)
Theorem exp_near_zero : forall t : R,
  -3/10000000000 <= t <= 3/10000000000 -> Rabs (exp t - 1) <= 4/10000000000.
Proof. intros t Ht. interval with (i_prec 80). Qed.

(** * The GELU constant

    [f32_gelu_c1] holds [6693141/2^23] in place of [sqrt (2/pi)]. *)

Theorem gelu_c1_bound : Rabs (6693141/8388608 - sqrt (2 / PI)) <= 23/1000000000.
Proof. interval with (i_prec 60). Qed.

Theorem sqrt_2_pi_le : sqrt (2 / PI) <= 4/5.
Proof. interval. Qed.

(** * The constants of the trigonometric reduction

    [f32_2pi_hi + f32_2pi_lo] as stored, and the stored [1/(2 pi)]. *)

Theorem two_pi_split_bound :
  Rabs (201/32 + 8312081/4294967296 - 2 * PI) <= 2/100000000000.
Proof. interval with (i_prec 80). Qed.

Theorem inv_two_pi_bound : Rabs (10680707/67108864 - / (2 * PI)) <= 1/100000000.
Proof. interval with (i_prec 60). Qed.

Theorem pi_bounds : 314159/100000 <= PI <= 31416/10000.
Proof. split; interval. Qed.
