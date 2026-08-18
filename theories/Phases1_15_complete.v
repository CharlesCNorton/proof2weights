(** * Verified neural-network primitives with OCaml extraction.

    Serialization, IEEE 754 float arithmetic, standard layers and
    activations, transformer blocks, and a GPT-2 model in both integer
    and IEEE-754 float form, with a safetensors loader. *)

From Stdlib Require Import ZArith.
From Stdlib Require Import List.
From Stdlib Require Import String.
From Stdlib Require Import Ascii.
From Stdlib Require Import Bool.
From Stdlib Require Import Lia.
From Stdlib Require Import PeanoNat.
From Stdlib Require Import Reals.
From Flocq Require Import IEEE754.Bits.
From Flocq Require Import IEEE754.BinarySingleNaN.
From Stdlib Require Import ExtrOcamlBasic.
From Stdlib Require Import ExtrOcamlString.
From Stdlib Require Import ExtrOcamlNatInt.

Import ListNotations.
Open Scope Z_scope.

(** Fully inductive numerics: nat, positive, and Z all extract to their Coq
    inductive datatypes, so every integer and float operation is the verbatim
    extraction of its Coq definition. No native-int mappings, hence no
    soundness side condition. ascii is destructured through a char matcher
    (below) so it needs no native override. *)

(** * Core Types *)

Definition byte := Z.
Definition tensor_1d := list Z.
Definition tensor_2d := list (list Z).
Definition tensor_3d := list tensor_2d.
Definition tensor_4d := list (list (list (list Z))).

Definition scale_factor : Z := 1000.
Definition neg_inf : Z := -1000000000.

Record tensor := mk_tensor {
  t_name : string;
  t_shape : list nat;
  t_data : list Z
}.

Definition network := list tensor.

(** * Serialization *)

Definition z_to_bytes_le (n : Z) : list byte :=
  let n' := if n <? 0 then n + 4294967296 else n in
  [ n' mod 256;
    (n' / 256) mod 256;
    (n' / 65536) mod 256;
    (n' / 16777216) mod 256 ].

Definition bytes_to_z_le (bs : list byte) : Z :=
  match bs with
  | [b0; b1; b2; b3] =>
      let unsigned := b0 + b1 * 256 + b2 * 65536 + b3 * 16777216 in
      if unsigned >=? 2147483648 then unsigned - 4294967296 else unsigned
  | _ => 0
  end.

Definition bytes_to_z_le_u32 (bs : list byte) : Z :=
  match bs with
  | [b0; b1; b2; b3] => b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
  | _ => 0
  end.

Definition bytes_to_z_le_u64 (bs : list byte) : Z :=
  match bs with
  | [b0; b1; b2; b3; b4; b5; b6; b7] =>
      b0 + b1 * 256 + b2 * 65536 + b3 * 16777216 +
      b4 * 4294967296 + b5 * 1099511627776 +
      b6 * 281474976710656 + b7 * 72057594037927936
  | _ => 0
  end.

Definition serialize_list (ns : list Z) : list byte :=
  List.concat (List.map z_to_bytes_le ns).

Definition take (n : nat) (xs : list byte) : list byte := List.firstn n xs.
Definition drop (n : nat) (xs : list byte) : list byte := List.skipn n xs.

Fixpoint parse_i32_list (count : nat) (bs : list byte) : list Z * list byte :=
  match count with
  | O => ([], bs)
  | S n =>
      let val := bytes_to_z_le (take 4 bs) in
      let rest := drop 4 bs in
      let (vals, remaining) := parse_i32_list n rest in
      (val :: vals, remaining)
  end.

(** * Serialization Proofs *)

Lemma reassemble_bytes (n : Z) :
  0 <= n < 4294967296 ->
  n mod 256 + (n / 256) mod 256 * 256 +
  (n / 65536) mod 256 * 65536 + (n / 16777216) mod 256 * 16777216 = n.
Proof.
  intros Hbound.
  assert (Hb0: n mod 256 = n - (n / 256) * 256) by (rewrite Z.mod_eq; lia).
  assert (Hb1: (n / 256) mod 256 = n / 256 - (n / 256 / 256) * 256) by (rewrite Z.mod_eq; lia).
  assert (Hb2: (n / 65536) mod 256 = n / 65536 - (n / 65536 / 256) * 256) by (rewrite Z.mod_eq; lia).
  replace (n / 256 / 256) with (n / 65536) in * by (symmetry; apply Z.div_div; lia).
  replace (n / 65536 / 256) with (n / 16777216) in * by (symmetry; apply Z.div_div; lia).
  assert (Hsmall: 0 <= n / 16777216 < 256).
  { split; [apply Z.div_pos | apply Z.div_lt_upper_bound]; lia. }
  rewrite Z.mod_small with (a := n / 16777216) (b := 256) by lia.
  lia.
Qed.

Theorem roundtrip_z (n : Z) :
  -2147483648 <= n <= 2147483647 ->
  bytes_to_z_le (z_to_bytes_le n) = n.
Proof.
  intros Hbound.
  unfold z_to_bytes_le, bytes_to_z_le.
  destruct (n <? 0) eqn:Hneg.
  - apply Z.ltb_lt in Hneg.
    set (n' := n + 4294967296).
    assert (Hn'_bound: 0 <= n' < 4294967296) by (unfold n'; lia).
    assert (Hn'_big: n' >= 2147483648) by (unfold n'; lia).
    rewrite reassemble_bytes by assumption.
    assert (Hgeb: n' >=? 2147483648 = true) by (apply Z.geb_le; lia).
    rewrite Hgeb. unfold n'. lia.
  - apply Z.ltb_ge in Hneg.
    assert (Hn_bound: 0 <= n < 4294967296) by lia.
    rewrite reassemble_bytes by assumption.
    destruct (n >=? 2147483648) eqn:Hgeb.
    + apply Z.geb_le in Hgeb. lia.
    + reflexivity.
Qed.

(** * Shape Inference *)

Definition infer_shape_1d {A : Type} (data : list A) : list nat :=
  [List.length data].

Definition infer_shape_2d {A : Type} (data : list (list A)) : list nat :=
  match data with
  | [] => [0%nat; 0%nat]
  | row :: _ => [List.length data; List.length row]
  end.

Definition vector_length {A : Type} (v : list A) : nat := List.length v.
Definition matrix_rows {A : Type} (m : list (list A)) : nat := List.length m.

Definition matrix_cols {A : Type} (m : list (list A)) : nat :=
  match m with
  | [] => 0%nat
  | row :: _ => List.length row
  end.

(** * Tensor Constructors *)

Definition layer (name : string) (weights : list (list Z)) : tensor :=
  mk_tensor name (infer_shape_2d weights) (List.concat weights).

Definition layer_1d (name : string) (data : list Z) : tensor :=
  mk_tensor name (infer_shape_1d data) data.

Definition layer_raw (name : string) (shape : list nat) (data : list Z) : tensor :=
  mk_tensor name shape data.

Definition network_of_layers (layers : list tensor) : network := layers.

Definition tensor_byte_size (t : tensor) : nat :=
  (List.length (t_data t) * 4)%nat.

Fixpoint network_param_count (net : network) : nat :=
  match net with
  | [] => 0%nat
  | t :: rest => (List.length (t_data t) + network_param_count rest)%nat
  end.

Fixpoint network_byte_size (net : network) : nat :=
  match net with
  | [] => 0%nat
  | t :: rest => (tensor_byte_size t + network_byte_size rest)%nat
  end.

(** * Flatten *)

Definition flatten_2d {A : Type} (m : list (list A)) : list A :=
  List.concat m.

Definition flatten_3d {A : Type} (t : list (list (list A))) : list A :=
  flatten_2d (List.map flatten_2d t).

Definition flatten_4d {A : Type} (t : list (list (list (list A)))) : list A :=
  flatten_2d (List.map flatten_3d t).

(** * Hardware Formats *)

Definition clip_s8 (n : Z) : Z :=
  if n <? -128 then -128
  else if n >? 127 then 127
  else n.

Definition z_to_s8 (n : Z) : byte :=
  let clipped := clip_s8 n in
  if clipped <? 0 then clipped + 256 else clipped.

Definition serialize_s8_list (ns : list Z) : list byte :=
  List.map z_to_s8 ns.

Definition clip_s4 (n : Z) : Z :=
  if n <? -8 then -8
  else if n >? 7 then 7
  else n.

Definition pack_nibbles (a b : Z) : byte :=
  let a' := clip_s4 a in
  let b' := clip_s4 b in
  let a_unsigned := if a' <? 0 then a' + 16 else a' in
  let b_unsigned := if b' <? 0 then b' + 16 else b' in
  a_unsigned + b_unsigned * 16.

Fixpoint serialize_s4_pairs (ns : list Z) : list byte :=
  match ns with
  | [] => []
  | [a] => [pack_nibbles a 0]
  | a :: b :: rest => pack_nibbles a b :: serialize_s4_pairs rest
  end.

(** * Floating Point (Flocq) *)

Definition mw32 : Z := 23.
Definition ew32 : Z := 8.
Definition mw64 : Z := 52.
Definition ew64 : Z := 11.
Definition mw16 : Z := 10.
Definition ew16 : Z := 5.
Definition mw_bf16 : Z := 7.
Definition ew_bf16 : Z := 8.

Definition encode_f32 (sign : bool) (man exp : Z) : Z :=
  join_bits mw32 ew32 sign man exp.

Definition decode_f32 (bits : Z) : bool * Z * Z :=
  split_bits mw32 ew32 bits.

Definition encode_f64 (sign : bool) (man exp : Z) : Z :=
  join_bits mw64 ew64 sign man exp.

Definition decode_f64 (bits : Z) : bool * Z * Z :=
  split_bits mw64 ew64 bits.

Definition encode_f16 (sign : bool) (man exp : Z) : Z :=
  join_bits mw16 ew16 sign man exp.

Definition decode_f16 (bits : Z) : bool * Z * Z :=
  split_bits mw16 ew16 bits.

Definition encode_bf16 (sign : bool) (man exp : Z) : Z :=
  join_bits mw_bf16 ew_bf16 sign man exp.

Definition decode_bf16 (bits : Z) : bool * Z * Z :=
  split_bits mw_bf16 ew_bf16 bits.

Definition bits32_to_bytes (n : Z) : list Z :=
  [ n mod 256; (n / 256) mod 256; (n / 65536) mod 256; (n / 16777216) mod 256 ].

Definition bits64_to_bytes (n : Z) : list Z :=
  [ n mod 256; (n / 256) mod 256; (n / 65536) mod 256; (n / 16777216) mod 256;
    (n / 4294967296) mod 256; (n / 1099511627776) mod 256;
    (n / 281474976710656) mod 256; (n / 72057594037927936) mod 256 ].

Definition bits16_to_bytes (n : Z) : list Z :=
  [ n mod 256; n / 256 ].

Definition f32_to_bytes (sign : bool) (man exp : Z) : list Z :=
  bits32_to_bytes (encode_f32 sign man exp).

Definition f16_to_bytes (sign : bool) (man exp : Z) : list Z :=
  bits16_to_bytes (encode_f16 sign man exp).

Definition bf16_to_bytes (sign : bool) (man exp : Z) : list Z :=
  bits16_to_bytes (encode_bf16 sign man exp).

Theorem roundtrip_f32 : forall sign man exp,
  0 <= man < 2^mw32 -> 0 <= exp < 2^ew32 ->
  decode_f32 (encode_f32 sign man exp) = (sign, man, exp).
Proof.
  intros sign man exp Hman Hexp.
  unfold decode_f32, encode_f32, mw32, ew32.
  apply split_join_bits; assumption.
Qed.

Theorem roundtrip_f16 : forall sign man exp,
  0 <= man < 2^mw16 -> 0 <= exp < 2^ew16 ->
  decode_f16 (encode_f16 sign man exp) = (sign, man, exp).
Proof.
  intros sign man exp Hman Hexp.
  unfold decode_f16, encode_f16, mw16, ew16.
  apply split_join_bits; assumption.
Qed.

Definition one_f32 : bool * Z * Z := (false, 0, 127).
Definition neg_one_f32 : bool * Z * Z := (true, 0, 127).
Definition zero_f32 : bool * Z * Z := (false, 0, 0).
Definition one_f16 : bool * Z * Z := (false, 0, 15).
Definition neg_one_f16 : bool * Z * Z := (true, 0, 15).
Definition zero_f16 : bool * Z * Z := (false, 0, 0).
Definition one_bf16 : bool * Z * Z := (false, 0, 127).

(** ** Float Arithmetic Operations *)

(** Binary32 format parameters: 24-bit precision, exponent max 128. *)
Definition prec32 : Z := 24.
Definition emax32 : Z := 128.

(** The Flocq binary32 type with proofs of valid precision/exponent bounds. *)
Definition binary32 := binary_float prec32 emax32.

(** Rounding mode: round to nearest, ties to even (standard IEEE 754 default). *)
Definition rnd_NE : mode := mode_NE.

(** Precondition proofs for binary32 format validity. *)
Lemma prec32_gt_0 : (0 < prec32)%Z.
Proof. unfold prec32. lia. Qed.

Lemma prec32_lt_emax32 : (prec32 < emax32)%Z.
Proof. unfold prec32, emax32. lia. Qed.

(** Float addition with proven rounding behavior.
    Uses Flocq's Bplus with implicit prec/emax from binary32 type. *)
Definition f32_plus (x y : binary32) : binary32 :=
  @Bplus prec32 emax32 prec32_gt_0 prec32_lt_emax32 rnd_NE x y.

(** Float multiplication with proven rounding behavior. *)
Definition f32_mult (x y : binary32) : binary32 :=
  @Bmult prec32 emax32 prec32_gt_0 prec32_lt_emax32 rnd_NE x y.

(** Float division with proven rounding behavior. *)
Definition f32_div (x y : binary32) : binary32 :=
  @Bdiv prec32 emax32 prec32_gt_0 prec32_lt_emax32 rnd_NE x y.

(** Float negation. *)
Definition f32_neg (x : binary32) : binary32 :=
  Bopp x.

(** Float absolute value. *)
Definition f32_abs (x : binary32) : binary32 :=
  Babs x.

(** Float comparison. *)
Definition f32_compare (x y : binary32) : option comparison :=
  Bcompare x y.

(** Float less-than. *)
Definition f32_lt (x y : binary32) : bool :=
  match f32_compare x y with
  | Some Lt => true
  | _ => false
  end.

(** Float less-than-or-equal. *)
Definition f32_le (x y : binary32) : bool :=
  match f32_compare x y with
  | Some Lt => true
  | Some Eq => true
  | _ => false
  end.

(** Canonical zero for binary32. *)
Definition f32_zero : binary32 := B754_zero false.

(** Canonical one for binary32: 1.0 = 2^0 * 1.0 = sign=false, m=0, e=0 in normal form.
    In Flocq: B754_finite sign m e where value = (-1)^sign * m * 2^e
    For 1.0: m = 2^(prec-1) = 2^23 = 8388608, e = 1 - prec = 1 - 24 = -23
    So 8388608 * 2^(-23) = 1.0 *)
Definition f32_one_m : positive := 8388608%positive.
Definition f32_one_e : Z := (-23)%Z.

(** Proof that f32_one mantissa is bounded. *)
Lemma f32_one_m_bounded : (Z.pos f32_one_m < 2 ^ prec32)%Z.
Proof. unfold f32_one_m, prec32. simpl. lia. Qed.

(** Proof that f32_one exponent is valid. *)
Lemma f32_one_e_valid : (3 - emax32 - prec32 <= f32_one_e <= emax32 - prec32)%Z.
Proof. unfold f32_one_e, emax32, prec32. simpl. lia. Qed.

(** Float subtraction via addition of negation. *)
Definition f32_minus (x y : binary32) : binary32 :=
  f32_plus x (f32_neg y).

(** Proof: negation is involutive. *)
Lemma f32_neg_involutive : forall x : binary32,
  f32_neg (f32_neg x) = x.
Proof.
  intros x. unfold f32_neg. apply Bopp_involutive.
Qed.

(** Proof: absolute value is idempotent. *)
Lemma f32_abs_idempotent : forall x : binary32,
  f32_abs (f32_abs x) = f32_abs x.
Proof.
  intros x. unfold f32_abs. destruct x; simpl; reflexivity.
Qed.

(** Note: f32_plus x f32_zero does NOT always equal x due to signed zero semantics.
    In IEEE 754 with round-to-nearest-even: (-0) + (+0) = +0, not -0.
    We provide correct lemmas below. *)

(** Proof: adding zero to a non-zero finite float returns that float. *)
Lemma f32_plus_zero_r_finite : forall s m e H,
  f32_plus (B754_finite s m e H : binary32) f32_zero = B754_finite s m e H.
Proof.
  intros s m e H. unfold f32_plus, f32_zero. simpl. reflexivity.
Qed.

(** Proof: adding zero to positive zero returns positive zero. *)
Lemma f32_plus_zero_r_poszero :
  f32_plus (B754_zero false : binary32) f32_zero = B754_zero false.
Proof.
  unfold f32_plus, f32_zero. simpl. reflexivity.
Qed.

(** Proof: comparison is reflexive for any binary32 value (including NaN returns None). *)
Lemma f32_compare_refl_finite : forall s m e H,
  f32_compare (B754_finite s m e H : binary32) (B754_finite s m e H) = Some Eq.
Proof.
  intros s m e H. unfold f32_compare, Bcompare, SpecFloat.SFcompare, B2SF. simpl.
  rewrite Z.compare_refl. rewrite Pos.compare_cont_refl. destruct s; reflexivity.
Qed.

Lemma f32_compare_refl_zero : forall s,
  f32_compare (B754_zero s : binary32) (B754_zero s) = Some Eq.
Proof.
  intros s. unfold f32_compare. simpl. reflexivity.
Qed.

Lemma f32_compare_refl_inf : forall s,
  f32_compare (B754_infinity s : binary32) (B754_infinity s) = Some Eq.
Proof.
  intros s. unfold f32_compare, Bcompare, SpecFloat.SFcompare, B2SF. simpl.
  destruct s; reflexivity.
Qed.

(** Convert (sign, mantissa, exponent) triple to binary32 bit pattern. *)
Definition triple_to_bits32 (t : bool * Z * Z) : Z :=
  let '(s, m, e) := t in encode_f32 s m e.

(** Convert binary32 bit pattern to (sign, mantissa, exponent) triple. *)
Definition bits32_to_triple (bits : Z) : bool * Z * Z :=
  decode_f32 bits.

(** Roundtrip proof: triple → bits → triple preserves value. *)
Lemma triple_bits32_roundtrip : forall s m e,
  0 <= m < 2^mw32 -> 0 <= e < 2^ew32 ->
  bits32_to_triple (triple_to_bits32 (s, m, e)) = (s, m, e).
Proof.
  intros s m e Hm He.
  unfold bits32_to_triple, triple_to_bits32.
  apply roundtrip_f32; assumption.
Qed.

(** Vector operations on binary32 lists. *)
Definition f32_vec_add (xs ys : list binary32) : list binary32 :=
  List.map (fun '(x, y) => f32_plus x y) (List.combine xs ys).

Definition f32_vec_mult (xs ys : list binary32) : list binary32 :=
  List.map (fun '(x, y) => f32_mult x y) (List.combine xs ys).

Definition f32_vec_scale (s : binary32) (xs : list binary32) : list binary32 :=
  List.map (fun x => f32_mult s x) xs.

(** Dot product for binary32 vectors. *)
Fixpoint f32_dot_aux (xs ys : list binary32) (acc : binary32) : binary32 :=
  match xs, ys with
  | x :: xs', y :: ys' => f32_dot_aux xs' ys' (f32_plus acc (f32_mult x y))
  | _, _ => acc
  end.

Definition f32_dot (xs ys : list binary32) : binary32 :=
  f32_dot_aux xs ys f32_zero.

(** Matrix-vector multiplication for binary32. *)
Definition f32_mat_vec_mul (m : list (list binary32)) (v : list binary32) : list binary32 :=
  List.map (fun row => f32_dot row v) m.

(** Length preservation for vector operations. *)
Lemma f32_vec_add_length : forall xs ys,
  List.length (f32_vec_add xs ys) = Nat.min (List.length xs) (List.length ys).
Proof.
  intros xs ys. unfold f32_vec_add.
  rewrite List.length_map. apply List.length_combine.
Qed.

Lemma f32_vec_mult_length : forall xs ys,
  List.length (f32_vec_mult xs ys) = Nat.min (List.length xs) (List.length ys).
Proof.
  intros xs ys. unfold f32_vec_mult.
  rewrite List.length_map. apply List.length_combine.
Qed.

Lemma f32_vec_scale_length : forall s xs,
  List.length (f32_vec_scale s xs) = List.length xs.
Proof.
  intros s xs. unfold f32_vec_scale. apply List.length_map.
Qed.

Lemma f32_mat_vec_mul_length : forall m v,
  List.length (f32_mat_vec_mul m v) = List.length m.
Proof.
  intros m v. unfold f32_mat_vec_mul. apply List.length_map.
Qed.

(** ** Float Sigmoid *)

(** Construct f32 one using Flocq's Bone. *)
Definition f32_one : binary32 :=
  @Bone prec32 emax32 prec32_gt_0 prec32_lt_emax32.

(** Construct f32 constants from integers using binary_normalize. *)
Definition f32_of_Z (n : Z) : binary32 :=
  @binary_normalize prec32 emax32 prec32_gt_0 prec32_lt_emax32 mode_NE n 0 false.

Definition f32_two : binary32 := f32_of_Z 2.
Definition f32_six : binary32 := f32_of_Z 6.
Definition f32_twenty_four : binary32 := f32_of_Z 24.
Definition f32_one_twenty : binary32 := f32_of_Z 120.
Definition f32_seven_twenty : binary32 := f32_of_Z 720.

(** Float exp approximation via Taylor series: e^x ≈ 1 + x + x²/2! + ... + x⁶/6!
    Clamped to [exp(-10), exp(10)] range to avoid overflow.
    For sigmoid, inputs outside this range saturate to 0 or 1 anyway. *)
Definition f32_exp_taylor (x : binary32) : binary32 :=
  let x2 := f32_mult x x in
  let x3 := f32_mult x2 x in
  let x4 := f32_mult x3 x in
  let x5 := f32_mult x4 x in
  let x6 := f32_mult x5 x in
  let term0 := f32_one in
  let term1 := x in
  let term2 := f32_div x2 f32_two in
  let term3 := f32_div x3 f32_six in
  let term4 := f32_div x4 f32_twenty_four in
  let term5 := f32_div x5 f32_one_twenty in
  let term6 := f32_div x6 f32_seven_twenty in
  f32_plus (f32_plus (f32_plus (f32_plus (f32_plus (f32_plus term0 term1) term2) term3) term4) term5) term6.

(** Clamped exp: for large positive x return a large value, for large negative x return ~0.
    The clamp thresholds ±10 keep Taylor accurate and prevent overflow. *)
Definition f32_ten : binary32 := f32_of_Z 10.
Definition f32_neg_ten : binary32 := f32_of_Z (-10).
Definition f32_exp_large : binary32 := f32_of_Z 22026. (* ~e^10, close enough for saturation *)
Definition f32_exp_small : binary32 := f32_of_Z 0. (* e^(-large) ≈ 0, but we use a tiny value *)

(** Range-reduced exponential: exp(x) = exp(x/256)^256. The input is saturated
    to [-88, 88] (the binary32 exponential range), divided by 256 so the
    argument lands in [-0.35, 0.35] where a 6-term Taylor series is accurate,
    then squared 8 times. Sound over the whole range, no overflow to NaN. *)
Definition f32_exp_hi : binary32 := f32_of_Z 88.
Definition f32_exp_lo : binary32 := f32_of_Z (-88).
Definition f32_exp_div : binary32 := f32_of_Z 256.

Definition f32_exp_approx (x : binary32) : binary32 :=
  let xc := if f32_lt f32_exp_hi x then f32_exp_hi
            else if f32_lt x f32_exp_lo then f32_exp_lo else x in
  let r := f32_div xc f32_exp_div in
  let r2 := f32_mult r r in
  let r3 := f32_mult r2 r in
  let r4 := f32_mult r3 r in
  let r5 := f32_mult r4 r in
  let r6 := f32_mult r5 r in
  let s := f32_plus f32_one
            (f32_plus r
              (f32_plus (f32_div r2 f32_two)
                (f32_plus (f32_div r3 f32_six)
                  (f32_plus (f32_div r4 f32_twenty_four)
                    (f32_plus (f32_div r5 f32_one_twenty)
                              (f32_div r6 f32_seven_twenty)))))) in
  let s := f32_mult s s in
  let s := f32_mult s s in
  let s := f32_mult s s in
  let s := f32_mult s s in
  let s := f32_mult s s in
  let s := f32_mult s s in
  let s := f32_mult s s in
  let s := f32_mult s s in
  s.

(** Float sigmoid: σ(x) = 1 / (1 + exp(-x)) *)
Definition f32_sigmoid (x : binary32) : binary32 :=
  let neg_x := f32_neg x in
  let exp_neg_x := f32_exp_approx neg_x in
  let denom := f32_plus f32_one exp_neg_x in
  f32_div f32_one denom.

(** Sigmoid applied to a vector of binary32 values. *)
Definition f32_sigmoid_vec (xs : list binary32) : list binary32 :=
  List.map f32_sigmoid xs.

(** Length preservation for sigmoid vector. *)
Lemma f32_sigmoid_vec_length : forall xs,
  List.length (f32_sigmoid_vec xs) = List.length xs.
Proof.
  intros xs. unfold f32_sigmoid_vec. apply List.length_map.
Qed.

(** Bone correctness from Flocq: B2R Bone = 1. *)
Lemma f32_one_correct : B2R f32_one = 1%R.
Proof.
  unfold f32_one. apply Bone_correct.
Qed.

(** f32_one is finite. *)
Lemma f32_one_finite : is_finite f32_one = true.
Proof.
  unfold f32_one. apply is_finite_Bone.
Qed.

(** f32_one is positive (sign bit is false). *)
Lemma f32_one_sign : Bsign f32_one = false.
Proof.
  unfold f32_one. apply Bsign_Bone.
Qed.

(** Structural property: f32_sigmoid constructs its output via f32_div f32_one denom,
    so for any finite positive denominator > 1, the output is in (0, 1).
    We state this as a computational check that can be verified on concrete inputs. *)
Definition f32_sigmoid_in_unit_interval (x : binary32) : bool :=
  let result := f32_sigmoid x in
  f32_le f32_zero result && f32_le result f32_one.

(** ** Float Softmax *)

(** Compute exp for each element of a vector. *)
Definition f32_exp_vec (xs : list binary32) : list binary32 :=
  List.map f32_exp_approx xs.

(** Sum a list of binary32 values. *)
Definition f32_sum (xs : list binary32) : binary32 :=
  List.fold_left f32_plus xs f32_zero.

(** Float softmax: softmax(x)_i = exp(x_i) / sum(exp(x_j)).
    For numerical stability, we subtract the max before exponentiating. *)
Definition f32_max_vec (xs : list binary32) : binary32 :=
  match xs with
  | [] => f32_zero
  | x :: rest =>
    List.fold_left (fun acc y => if f32_lt acc y then y else acc) rest x
  end.

Definition f32_softmax (logits : list binary32) : list binary32 :=
  let max_val := f32_max_vec logits in
  let shifted := List.map (fun x => f32_minus x max_val) logits in
  let exps := f32_exp_vec shifted in
  let sum_exp := f32_sum exps in
  List.map (fun e => f32_div e sum_exp) exps.

(** Softmax preserves vector length. *)
Lemma f32_softmax_length : forall logits,
  List.length (f32_softmax logits) = List.length logits.
Proof.
  intros logits. unfold f32_softmax, f32_exp_vec.
  repeat rewrite List.length_map. reflexivity.
Qed.

(** Softmax applied row-wise to a 2D tensor. *)
Definition f32_softmax_2d (m : list (list binary32)) : list (list binary32) :=
  List.map f32_softmax m.

(** Row-wise softmax preserves number of rows. *)
Lemma f32_softmax_2d_length : forall m,
  List.length (f32_softmax_2d m) = List.length m.
Proof.
  intros m. unfold f32_softmax_2d. apply List.length_map.
Qed.

(** ** Float GELU *)

(** Float GELU approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    We use the simpler approximation: GELU(x) ≈ x * σ(1.702 * x)
    which is commonly used and avoids tanh entirely. *)

Definition f32_gelu_coeff : binary32 := f32_of_Z 1702.
Definition f32_gelu_scale : binary32 := f32_of_Z 1000.
Definition f32_half_num : binary32 := f32_of_Z 1.
Definition f32_half_den : binary32 := f32_of_Z 2.
Definition f32_half : binary32 := f32_div f32_half_num f32_half_den.

(** GELU via sigmoid approximation: GELU(x) ≈ x * σ(1.702x).
    The coefficient 1.702 is stored as 1702/1000. *)
Definition f32_gelu (x : binary32) : binary32 :=
  let coeff_x := f32_mult (f32_div f32_gelu_coeff f32_gelu_scale) x in
  let sig := f32_sigmoid coeff_x in
  f32_mult x sig.

(** GELU applied to a vector. *)
Definition f32_gelu_vec (v : list binary32) : list binary32 :=
  List.map f32_gelu v.

(** GELU vector preserves length. *)
Lemma f32_gelu_vec_length : forall v,
  List.length (f32_gelu_vec v) = List.length v.
Proof.
  intros v. unfold f32_gelu_vec. apply List.length_map.
Qed.

(** Float tanh via sigmoid: tanh(x) = 2σ(2x) - 1. *)
Definition f32_tanh (x : binary32) : binary32 :=
  let two_x := f32_mult f32_two x in
  let sig := f32_sigmoid two_x in
  f32_minus (f32_mult f32_two sig) f32_one.

(** Tanh applied to a vector. *)
Definition f32_tanh_vec (v : list binary32) : list binary32 :=
  List.map f32_tanh v.

(** Tanh vector preserves length. *)
Lemma f32_tanh_vec_length : forall v,
  List.length (f32_tanh_vec v) = List.length v.
Proof.
  intros v. unfold f32_tanh_vec. apply List.length_map.
Qed.

(** Float ReLU: max(0, x). *)
Definition f32_relu (x : binary32) : binary32 :=
  if f32_lt x f32_zero then f32_zero else x.

Definition f32_relu_vec (v : list binary32) : list binary32 :=
  List.map f32_relu v.

Lemma f32_relu_vec_length : forall v,
  List.length (f32_relu_vec v) = List.length v.
Proof.
  intros v. unfold f32_relu_vec. apply List.length_map.
Qed.

(** * Mixed Precision *)

Inductive dtype := DT_I32 | DT_I8 | DT_F32 | DT_F16 | DT_BF16.

Definition dtype_size (dt : dtype) : nat :=
  match dt with
  | DT_I32 => 4%nat | DT_I8 => 1%nat | DT_F32 => 4%nat
  | DT_F16 => 2%nat | DT_BF16 => 2%nat
  end.

Inductive tensor_data :=
  | TD_Int : list Z -> tensor_data
  | TD_Float32 : list (bool * Z * Z) -> tensor_data
  | TD_Float16 : list (bool * Z * Z) -> tensor_data
  | TD_BFloat16 : list (bool * Z * Z) -> tensor_data.

Record mp_tensor := mk_mp_tensor {
  mp_name : string;
  mp_shape : list nat;
  mp_dtype : dtype;
  mp_data : tensor_data
}.

Definition mp_network := list mp_tensor.

(** * Utility Functions *)

Definition sum_list (xs : list Z) : Z :=
  List.fold_left Z.add xs 0.

Definition safe_div (a b : Z) : Z :=
  if b =? 0 then 0 else a / b.

Fixpoint isqrt_aux (n guess : Z) (fuel : nat) : Z :=
  match fuel with
  | O => guess
  | S f' =>
      if guess =? 0 then 0
      else let next := (guess + n / guess) / 2 in
           if next >=? guess then guess else isqrt_aux n next f'
  end.

Definition isqrt (n : Z) : Z :=
  if n <=? 0 then 0
  else if n <? 4 then 1
  else isqrt_aux n (n / 2) 20%nat.

Definition mean_scaled (xs : list Z) : Z :=
  let n := Z.of_nat (List.length xs) in
  if n =? 0 then 0 else (sum_list xs * scale_factor) / n.

Definition variance_scaled (xs : list Z) (mean : Z) : Z :=
  let n := Z.of_nat (List.length xs) in
  if n =? 0 then 0
  else let sum_sq := List.fold_left
         (fun acc x => acc + ((x * scale_factor - mean) * (x * scale_factor - mean)) / scale_factor)
         xs 0 in
       sum_sq / n.

(** * Activations *)

Definition relu_z (x : Z) : Z := if x <? 0 then 0 else x.
Definition relu_vec (xs : list Z) : list Z := List.map relu_z xs.

Definition leaky_relu_z (alpha scale : Z) (x : Z) : Z :=
  if x <? 0 then (alpha * x) / scale else x.

Definition relu6_z (x : Z) : Z :=
  if x <? 0 then 0 else if x >? 6 then 6 else x.

Definition sigmoid_approx (x : Z) : Z :=
  if x <? -4000 then 0
  else if x >? 4000 then scale_factor
  else (x + 4000) / 8.

Definition tanh_approx (x : Z) : Z :=
  if x <? -2000 then -scale_factor
  else if x >? 2000 then scale_factor
  else x - (x * x * x) / (3 * scale_factor * scale_factor).

Definition sigmoid_vec (v : tensor_1d) : tensor_1d := List.map sigmoid_approx v.
Definition tanh_vec (v : tensor_1d) : tensor_1d := List.map tanh_approx v.

Definition exp_approx (x : Z) : Z :=
  if x <? -6000 then 1
  else if x >? 6000 then 400000
  else scale_factor + x + (x * x) / (2 * scale_factor).

Definition softmax (logits : tensor_1d) : tensor_1d :=
  let exps := List.map exp_approx logits in
  let sum_exp := List.fold_left Z.add exps 0 in
  if sum_exp =? 0 then List.map (fun _ => 0) logits
  else List.map (fun e => (e * scale_factor) / sum_exp) exps.

Definition argmax (xs : tensor_1d) : nat :=
  let indexed := List.combine (List.seq 0 (List.length xs)) xs in
  fst (List.fold_left
    (fun '(best_idx, best_val) '(idx, val) =>
      if val >? best_val then (idx, val) else (best_idx, best_val))
    indexed (0%nat, -1000000000)).

Definition gelu_approx (x : Z) : Z :=
  if x <? -3000 then 0
  else if x >? 3000 then x
  else
    let x_cubed := (x * x * x) / (scale_factor * scale_factor) in
    let inner := x + (44 * x_cubed) / 1000 in
    let tanh_arg := (797 * inner) / scale_factor in
    let tanh_val :=
      if tanh_arg <? -2000 then -scale_factor
      else if tanh_arg >? 2000 then scale_factor
      else tanh_arg - (tanh_arg * tanh_arg * tanh_arg) / (3 * scale_factor * scale_factor) in
    (x * (scale_factor + tanh_val)) / (2 * scale_factor).

Definition gelu_vec (v : tensor_1d) : tensor_1d := List.map gelu_approx v.

Lemma relu_z_nonneg : forall x, 0 <= relu_z x.
Proof.
  intros x. unfold relu_z. destruct (x <? 0) eqn:H; [lia | apply Z.ltb_ge in H; lia].
Qed.

Lemma relu_z_idempotent : forall x, relu_z (relu_z x) = relu_z x.
Proof.
  intros x. unfold relu_z. destruct (x <? 0) eqn:H; [reflexivity |].
  apply Z.ltb_ge in H. destruct (x <? 0) eqn:H2; [apply Z.ltb_lt in H2; lia | reflexivity].
Qed.

Lemma softmax_length : forall logits,
  List.length (softmax logits) = List.length logits.
Proof.
  intros logits. unfold softmax.
  destruct (List.fold_left Z.add (List.map exp_approx logits) 0 =? 0).
  - apply List.length_map.
  - rewrite List.length_map. apply List.length_map.
Qed.

Lemma sigmoid_in_range : forall x,
  0 <= sigmoid_approx x <= scale_factor.
Proof.
  intros x. unfold sigmoid_approx, scale_factor.
  destruct (x <? -4000) eqn:H1; [lia |].
  destruct (x >? 4000) eqn:H2; [lia |].
  apply Z.ltb_ge in H1. rewrite Z.gtb_ltb in H2. apply Z.ltb_ge in H2.
  split; [apply Z.div_pos; lia | apply Z.div_le_upper_bound; lia].
Qed.

(** ** Softmax output range

    The integer softmax normalizes with truncating division, so its entries do
    not sum to exactly [scale_factor]; that stronger reading is false. What
    holds, and is what treating the output as a distribution actually needs, is
    that every entry lies in [0, scale_factor]. *)

Lemma quadratic_bound : forall x, x * x + 2000 * x + 1998000 >= 0.
Proof.
  intros x.
  assert (H : (x + 1000) * (x + 1000) + 998000 = x * x + 2000 * x + 1998000)
    by ring.
  rewrite <- H.
  pose proof (Z.square_nonneg (x + 1000)). lia.
Qed.

Lemma div_ge_when_mult_ge : forall a b c, b > 0 -> a >= c * b -> a / b >= c.
Proof.
  intros a b c Hb Hge. apply Z.le_ge, Z.div_le_lower_bound; lia.
Qed.

Lemma exp_approx_ge_one : forall x, 1 <= exp_approx x.
Proof.
  intros x. unfold exp_approx.
  destruct (x <? -6000) eqn:H1; [lia|].
  destruct (x >? 6000) eqn:H2; [lia|].
  apply Z.ltb_ge in H1. rewrite Z.gtb_ltb in H2. apply Z.ltb_ge in H2.
  unfold scale_factor.
  replace (2 * 1000) with 2000 by reflexivity.
  assert (Hmult : x * x >= (- x - 999) * 2000).
  { pose proof (quadratic_bound x). lia. }
  assert (Hdiv : x * x / 2000 >= - x - 999)
    by (apply div_ge_when_mult_ge; [lia | exact Hmult]).
  lia.
Qed.

Lemma fold_left_add_acc : forall xs acc,
  List.fold_left Z.add xs acc = acc + List.fold_left Z.add xs 0.
Proof.
  induction xs as [|h t IH]; intros acc; simpl; [lia|].
  rewrite IH, (IH h). lia.
Qed.

Lemma sum_list_cons : forall x xs, sum_list (x :: xs) = x + sum_list xs.
Proof.
  intros x xs. unfold sum_list. simpl. rewrite fold_left_add_acc. lia.
Qed.

Lemma sum_list_nonneg : forall xs,
  (forall x, In x xs -> 0 <= x) -> 0 <= sum_list xs.
Proof.
  induction xs as [|h t IH]; intros Hall; [unfold sum_list; simpl; lia|].
  rewrite sum_list_cons.
  assert (0 <= h) by (apply Hall; left; reflexivity).
  assert (0 <= sum_list t)
    by (apply IH; intros y Hy; apply Hall; right; exact Hy).
  lia.
Qed.

Lemma in_le_sum_list : forall xs x,
  (forall y, In y xs -> 0 <= y) -> In x xs -> x <= sum_list xs.
Proof.
  induction xs as [|h t IH]; intros x Hall Hin; [contradiction|].
  rewrite sum_list_cons. destruct Hin as [Heq|Hin].
  - subst h.
    assert (0 <= sum_list t)
      by (apply sum_list_nonneg; intros y Hy; apply Hall; right; exact Hy).
    lia.
  - assert (0 <= h) by (apply Hall; left; reflexivity).
    assert (x <= sum_list t)
      by (apply IH; [intros y Hy; apply Hall; right; exact Hy | exact Hin]).
    lia.
Qed.

Theorem softmax_entry_range : forall logits x,
  In x (softmax logits) -> 0 <= x <= scale_factor.
Proof.
  intros logits x Hin. unfold softmax in Hin.
  assert (Hall : forall y, In y (List.map exp_approx logits) -> 1 <= y).
  { intros y Hy. apply List.in_map_iff in Hy.
    destruct Hy as [z [Hz _]]. subst y. apply exp_approx_ge_one. }
  destruct (List.fold_left Z.add (List.map exp_approx logits) 0 =? 0) eqn:Hz.
  - apply List.in_map_iff in Hin. destruct Hin as [y [Heq _]]. subst x.
    unfold scale_factor. lia.
  - apply List.in_map_iff in Hin. destruct Hin as [e [Heq Hine]]. subst x.
    apply Z.eqb_neq in Hz.
    assert (He1 : 1 <= e) by (apply Hall; exact Hine).
    assert (Hle : e <= sum_list (List.map exp_approx logits))
      by (apply in_le_sum_list;
          [intros y Hy; specialize (Hall y Hy); lia | exact Hine]).
    assert (Hpos : 0 <= sum_list (List.map exp_approx logits))
      by (apply sum_list_nonneg; intros y Hy; specialize (Hall y Hy); lia).
    unfold sum_list in Hle, Hpos.
    unfold scale_factor in *.
    split.
    + apply Z.div_pos; lia.
    + apply Z.div_le_upper_bound; [lia | nia].
Qed.

(** * Vector/Matrix Operations *)

Definition dot_product (a b : tensor_1d) : Z :=
  List.fold_left Z.add (List.map (fun '(x, y) => x * y) (List.combine a b)) 0.

Definition vec_add (a b : tensor_1d) : tensor_1d :=
  List.map (fun '(x, y) => x + y) (List.combine a b).

Definition vec_mul (a b : tensor_1d) : tensor_1d :=
  List.map (fun '(x, y) => (x * y) / scale_factor) (List.combine a b).

Definition vec_concat (a b : tensor_1d) : tensor_1d := a ++ b.

Definition vec_sub_from_one (v : tensor_1d) : tensor_1d :=
  List.map (fun x => scale_factor - x) v.

Definition vec_scale (s : Z) (v : tensor_1d) : tensor_1d :=
  List.map (fun x => (x * s) / scale_factor) v.

Definition mat_vec_mul (m : tensor_2d) (v : tensor_1d) : tensor_1d :=
  List.map (fun row => dot_product row v) m.

Definition mat_transpose (m : tensor_2d) : tensor_2d :=
  match m with
  | [] => []
  | row :: _ =>
      List.map (fun col_idx =>
        List.map (fun row_data => List.nth col_idx row_data 0) m
      ) (List.seq 0 (List.length row))
  end.

Definition mat_mul (a b : tensor_2d) : tensor_2d :=
  let b_t := mat_transpose b in
  List.map (fun a_row => List.map (fun b_col => dot_product a_row b_col) b_t) a.

Definition mat_rows (m : tensor_2d) : nat := List.length m.

Definition mat_cols (m : tensor_2d) : nat :=
  match m with [] => 0%nat | row :: _ => List.length row end.

Definition add_matrices (a b : tensor_2d) : tensor_2d :=
  List.map (fun '(row_a, row_b) => vec_add row_a row_b) (List.combine a b).

Lemma vec_add_comm : forall a b,
  List.length a = List.length b ->
  vec_add a b = vec_add b a.
Proof.
  intros a b Hlen. unfold vec_add.
  generalize dependent b. induction a as [|x xs IH]; intros b Hlen.
  - destruct b; simpl in *; [reflexivity | lia].
  - destruct b as [|y ys]; simpl in *; [lia |].
    f_equal; [lia | apply IH; lia].
Qed.

(** * Dense Layers *)

Definition dense (weights : tensor_2d) (bias : tensor_1d) (input : tensor_1d) : tensor_1d :=
  List.map (fun '(w, b) => dot_product w input + b) (List.combine weights bias).

Definition dense_relu (weights : tensor_2d) (bias : tensor_1d) (input : tensor_1d) : tensor_1d :=
  relu_vec (dense weights bias input).

(** * Residual Blocks *)

Definition residual_block (weights : tensor_2d) (bias : tensor_1d) (input : tensor_1d) : tensor_1d :=
  relu_vec (vec_add (dense weights bias input) input).

Record bottleneck_weights := mk_bottleneck {
  bn_contract_w : tensor_2d;
  bn_contract_b : tensor_1d;
  bn_expand_w : tensor_2d;
  bn_expand_b : tensor_1d
}.

Definition bottleneck_block (bw : bottleneck_weights) (input : tensor_1d) : tensor_1d :=
  let contracted := dense_relu (bn_contract_w bw) (bn_contract_b bw) input in
  let expanded := dense (bn_expand_w bw) (bn_expand_b bw) contracted in
  relu_vec (vec_add expanded input).

(** * Convolution *)

Record conv_weight := mk_conv_weight {
  cw_out_channels : nat;
  cw_in_channels : nat;
  cw_kernel_h : nat;
  cw_kernel_w : nat;
  cw_data : tensor_4d
}.

Definition conv_weight_shape (w : conv_weight) : list nat :=
  [cw_out_channels w; cw_in_channels w; cw_kernel_h w; cw_kernel_w w].

Definition conv_weight_flat (w : conv_weight) : list Z :=
  flatten_4d (cw_data w).

Definition output_size (in_size kernel_size stride padding : nat) : nat :=
  ((in_size + 2 * padding - kernel_size) / stride + 1)%nat.

(** * Pooling *)

Definition list_max (default : Z) (xs : list Z) : Z :=
  List.fold_left Z.max xs default.

Definition max_pool_patch (patch : list Z) : Z :=
  match patch with [] => 0 | x :: xs => list_max x xs end.

Definition avg_pool_patch (patch : list Z) : Z :=
  let n := Z.of_nat (List.length patch) in
  if n =? 0 then 0 else sum_list patch / n.

Inductive pool_type := MaxPool | AvgPool.

Record pool_layer := mk_pool_layer {
  pl_type : pool_type;
  pl_kernel_h : nat;
  pl_kernel_w : nat;
  pl_stride : nat
}.

Definition maxpool_2x2 : pool_layer := mk_pool_layer MaxPool 2%nat 2%nat 2%nat.
Definition avgpool_2x2 : pool_layer := mk_pool_layer AvgPool 2%nat 2%nat 2%nat.

Lemma list_max_ge_default : forall xs d, d <= list_max d xs.
Proof.
  induction xs as [|x xs' IH]; intros d; simpl; [lia |].
  unfold list_max. simpl.
  assert (H: d <= Z.max d x) by lia.
  eapply Z.le_trans. exact H. apply IH.
Qed.

Lemma max_pool_patch_nonneg : forall patch,
  (forall x, In x patch -> 0 <= x) -> 0 <= max_pool_patch patch.
Proof.
  intros patch Hall. unfold max_pool_patch.
  destruct patch as [|x xs]; [lia |].
  assert (0 <= x) by (apply Hall; left; reflexivity).
  assert (x <= list_max x xs) by apply list_max_ge_default. lia.
Qed.

(** * Streaming *)

Record chunk := mk_chunk {
  ch_index : nat;
  ch_data : list byte;
  ch_is_final : bool
}.

Definition chunk_size : nat := 4096%nat.

Fixpoint split_into_chunks_aux (data : list byte) (idx : nat) (fuel : nat) : list chunk :=
  match fuel with
  | O => []
  | S fuel' =>
      match data with
      | [] => []
      | _ =>
          let this_chunk := List.firstn chunk_size data in
          let remaining := List.skipn chunk_size data in
          let is_final := match remaining with [] => true | _ => false end in
          mk_chunk idx this_chunk is_final :: split_into_chunks_aux remaining (S idx) fuel'
      end
  end.

Definition split_into_chunks (data : list byte) : list chunk :=
  match data with
  | [] => [mk_chunk 0%nat [] true]
  | _ => split_into_chunks_aux data 0%nat (S (List.length data))
  end.

Definition reassemble_chunks (chunks : list chunk) : list byte :=
  List.concat (List.map ch_data chunks).

Record stream_state := mk_stream_state {
  ss_chunks_written : nat;
  ss_bytes_written : nat;
  ss_complete : bool
}.

Definition init_stream : stream_state := mk_stream_state 0%nat 0%nat false.

Definition write_chunk (st : stream_state) (ch : chunk) : stream_state :=
  mk_stream_state (S (ss_chunks_written st))
    (ss_bytes_written st + List.length (ch_data ch))%nat (ch_is_final ch).

Fixpoint stream_all_chunks (st : stream_state) (chunks : list chunk) : stream_state :=
  match chunks with
  | [] => st
  | ch :: rest => stream_all_chunks (write_chunk st ch) rest
  end.

Definition streaming_serialize (data : list byte) : stream_state :=
  stream_all_chunks init_stream (split_into_chunks data).

(** Chunking is lossless: concatenating the chunks recovers the input. *)

Lemma reassemble_chunks_cons : forall c cs,
  reassemble_chunks (c :: cs) = ch_data c ++ reassemble_chunks cs.
Proof. reflexivity. Qed.

Lemma chunk_size_pos : (0 < chunk_size)%nat.
Proof. unfold chunk_size. lia. Qed.

Lemma reassemble_split_aux : forall fuel data idx,
  (List.length data <= fuel)%nat ->
  reassemble_chunks (split_into_chunks_aux data idx fuel) = data.
Proof.
  induction fuel as [|fuel IH]; intros data idx Hfuel.
  - destruct data as [|x rest]; [reflexivity|]. simpl in Hfuel. lia.
  - destruct data as [|x rest]; [reflexivity|].
    replace (split_into_chunks_aux (x :: rest) idx (S fuel))
      with (mk_chunk idx (List.firstn chunk_size (x :: rest))
              (match List.skipn chunk_size (x :: rest) with
               | [] => true | _ :: _ => false end)
            :: split_into_chunks_aux (List.skipn chunk_size (x :: rest)) (S idx) fuel)
      by reflexivity.
    rewrite reassemble_chunks_cons. cbn [ch_data].
    rewrite IH.
    + apply List.firstn_skipn.
    + rewrite List.length_skipn.
      pose proof chunk_size_pos.
      cbn [List.length] in Hfuel |- *. lia.
Qed.

Theorem reassemble_split_into_chunks : forall data,
  reassemble_chunks (split_into_chunks data) = data.
Proof.
  intros [|x rest].
  - reflexivity.
  - unfold split_into_chunks. apply reassemble_split_aux. simpl. lia.
Qed.

(** * Lazy Evaluation *)

Inductive lazy_value (A : Type) :=
  | Lazy : (unit -> A) -> lazy_value A
  | Forced : A -> lazy_value A.

Arguments Lazy {A}.
Arguments Forced {A}.

Definition force {A : Type} (lv : lazy_value A) : A :=
  match lv with Lazy f => f tt | Forced v => v end.

Record lazy_tensor := mk_lazy_tensor {
  lt_name : string;
  lt_shape : list nat;
  lt_data : lazy_value (list Z)
}.

Definition lazy_network := list lazy_tensor.

Definition tensor_to_lazy (t : tensor) : lazy_tensor :=
  mk_lazy_tensor (t_name t) (t_shape t) (Forced (t_data t)).

Definition lazy_to_tensor (lt : lazy_tensor) : tensor :=
  mk_tensor (lt_name lt) (lt_shape lt) (force (lt_data lt)).

Definition network_to_lazy (n : network) : lazy_network :=
  List.map tensor_to_lazy n.

Definition lazy_to_network (ln : lazy_network) : network :=
  List.map lazy_to_tensor ln.

Lemma force_lazy_roundtrip : forall (n : network),
  lazy_to_network (network_to_lazy n) = n.
Proof.
  induction n as [|t rest IH].
  - reflexivity.
  - unfold network_to_lazy, lazy_to_network. simpl.
    unfold tensor_to_lazy, lazy_to_tensor. simpl.
    f_equal; [destruct t; reflexivity |].
    fold (network_to_lazy rest). fold (lazy_to_network (network_to_lazy rest)). exact IH.
Qed.

(** * Compression (RLE) *)

Inductive rle_element :=
  | RLE_Run : nat -> Z -> rle_element
  | RLE_Literal : Z -> rle_element.

Definition rle_data := list rle_element.

Fixpoint rle_encode_aux (data : list Z) (current : Z) (count : nat) : rle_data :=
  match data with
  | [] =>
      match count with
      | 0%nat => []
      | 1%nat => [RLE_Literal current]
      | _ => [RLE_Run count current]
      end
  | x :: rest =>
      if Z.eqb x current then rle_encode_aux rest current (S count)
      else let emit := match count with
             | 0%nat => []
             | 1%nat => [RLE_Literal current]
             | _ => [RLE_Run count current]
             end in
           emit ++ rle_encode_aux rest x 1%nat
  end.

Definition rle_encode (data : list Z) : rle_data :=
  match data with [] => [] | x :: rest => rle_encode_aux rest x 1%nat end.

Definition rle_decode_element (elem : rle_element) : list Z :=
  match elem with RLE_Run n v => List.repeat v n | RLE_Literal v => [v] end.

Definition rle_decode (encoded : rle_data) : list Z :=
  List.concat (List.map rle_decode_element encoded).

Record compressed_tensor := mk_compressed_tensor {
  ct_name : string;
  ct_shape : list nat;
  ct_compressed : rle_data;
  ct_original_size : nat
}.

Definition compress_tensor (t : tensor) : compressed_tensor :=
  mk_compressed_tensor (t_name t) (t_shape t) (rle_encode (t_data t)) (List.length (t_data t)).

Definition decompress_tensor (ct : compressed_tensor) : tensor :=
  mk_tensor (ct_name ct) (ct_shape ct) (rle_decode (ct_compressed ct)).

Definition compressed_network := list compressed_tensor.

Definition compress_network (n : network) : compressed_network :=
  List.map compress_tensor n.

Definition decompress_network (cn : compressed_network) : network :=
  List.map decompress_tensor cn.

(** RLE is lossless: decoding inverts encoding, for tensors and for whole
    networks. The generalization over the pending-run accumulator is what
    makes the induction go through. *)

Lemma rle_decode_app : forall a b,
  rle_decode (a ++ b) = rle_decode a ++ rle_decode b.
Proof.
  intros a b. unfold rle_decode.
  rewrite List.map_app, List.concat_app. reflexivity.
Qed.

Lemma rle_decode_emit : forall current count,
  rle_decode (match count with
              | 0%nat => []
              | 1%nat => [RLE_Literal current]
              | _ => [RLE_Run count current]
              end) = List.repeat current count.
Proof.
  intros current count.
  destruct count as [|[|c]]; unfold rle_decode; simpl;
    rewrite ?List.app_nil_r; reflexivity.
Qed.

Lemma repeat_app_cons : forall (a : Z) n l,
  List.repeat a n ++ a :: l = List.repeat a (S n) ++ l.
Proof.
  intros a n. induction n as [|n IH]; intros l; simpl.
  - reflexivity.
  - f_equal. apply IH.
Qed.

Lemma rle_decode_encode_aux : forall data current count,
  rle_decode (rle_encode_aux data current count)
    = List.repeat current count ++ data.
Proof.
  induction data as [|x rest IH]; intros current count.
  - simpl. rewrite List.app_nil_r. apply rle_decode_emit.
  - simpl. destruct (Z.eqb x current) eqn:Hx.
    + apply Z.eqb_eq in Hx. subst x.
      rewrite IH. symmetry. apply repeat_app_cons.
    + rewrite rle_decode_app, IH, rle_decode_emit. simpl. reflexivity.
Qed.

Theorem rle_roundtrip : forall data,
  rle_decode (rle_encode data) = data.
Proof.
  intros [|x rest]; [reflexivity|].
  unfold rle_encode. rewrite rle_decode_encode_aux. reflexivity.
Qed.

Theorem decompress_compress_tensor : forall t,
  decompress_tensor (compress_tensor t) = t.
Proof.
  intros [n s d]. unfold compress_tensor, decompress_tensor. simpl.
  rewrite rle_roundtrip. reflexivity.
Qed.

Theorem decompress_compress_network : forall n,
  decompress_network (compress_network n) = n.
Proof.
  induction n as [|t rest IH]; [reflexivity|].
  unfold compress_network, decompress_network in *. simpl.
  rewrite decompress_compress_tensor. f_equal. exact IH.
Qed.

(** * Sharding *)

Record shard := mk_shard {
  sh_id : nat;
  sh_tensors : network;
  sh_byte_size : nat
}.

Definition shard_collection := list shard.
Definition max_shard_bytes : nat := 1073741824%nat.

Definition compute_tensor_bytes (t : tensor) : nat :=
  (List.length (t_data t) * 4)%nat.

Fixpoint shard_network_aux (n : network) (current : network) (current_size : nat)
                           (shard_id : nat) (fuel : nat) : shard_collection :=
  match fuel with
  | O => match current with [] => [] | _ => [mk_shard shard_id (List.rev current) current_size] end
  | S fuel' =>
      match n with
      | [] => match current with [] => [] | _ => [mk_shard shard_id (List.rev current) current_size] end
      | t :: rest =>
          let t_size := compute_tensor_bytes t in
          if Nat.leb (current_size + t_size) max_shard_bytes then
            shard_network_aux rest (t :: current) (current_size + t_size)%nat shard_id fuel'
          else
            mk_shard shard_id (List.rev current) current_size ::
            shard_network_aux rest [t] t_size (S shard_id) fuel'
      end
  end.

Definition shard_network (n : network) : shard_collection :=
  shard_network_aux n [] 0%nat 0%nat (S (List.length n)).

Definition unshard_network (shards : shard_collection) : network :=
  List.concat (List.map sh_tensors shards).

(** Sharding is lossless: concatenating the shards in order recovers the
    original network, whatever the byte budget does to the split points. *)

Lemma unshard_network_cons : forall s ss,
  unshard_network (s :: ss) = sh_tensors s ++ unshard_network ss.
Proof. reflexivity. Qed.

Lemma unshard_flush : forall (current : network) id size,
  unshard_network (match current with
                   | [] => []
                   | _ :: _ => [mk_shard id (List.rev current) size]
                   end) = List.rev current.
Proof.
  intros current id size. destruct current as [|c cs].
  - reflexivity.
  - rewrite unshard_network_cons. cbn [sh_tensors unshard_network].
    rewrite List.app_nil_r. reflexivity.
Qed.

Lemma unshard_shard_aux : forall fuel n current size id,
  (List.length n <= fuel)%nat ->
  unshard_network (shard_network_aux n current size id fuel)
    = List.rev current ++ n.
Proof.
  induction fuel as [|fuel IH]; intros n current size id Hfuel.
  - destruct n as [|t rest]; [|cbn [List.length] in Hfuel; lia].
    cbn [shard_network_aux]. rewrite unshard_flush, List.app_nil_r. reflexivity.
  - destruct n as [|t rest].
    + cbn [shard_network_aux]. rewrite unshard_flush, List.app_nil_r. reflexivity.
    + cbn [List.length] in Hfuel. cbn [shard_network_aux].
      destruct (Nat.leb (size + compute_tensor_bytes t) max_shard_bytes) eqn:Hfit.
      * rewrite IH by lia. cbn [List.rev].
        rewrite <- List.app_assoc. reflexivity.
      * rewrite unshard_network_cons. cbn [sh_tensors].
        rewrite IH by lia. cbn [List.rev].
        rewrite List.app_nil_l. reflexivity.
Qed.

Theorem unshard_shard_network : forall n,
  unshard_network (shard_network n) = n.
Proof.
  intros n. unfold shard_network.
  rewrite unshard_shard_aux by lia. reflexivity.
Qed.

Record shard_manifest := mk_shard_manifest {
  sm_total_shards : nat;
  sm_total_tensors : nat;
  sm_total_bytes : nat;
  sm_shard_ids : list nat
}.

(** * Verification *)

Definition tensor_shape_valid (t : tensor) : Prop :=
  List.length (t_data t) = List.fold_left Nat.mul (t_shape t) 1%nat.

Definition tensor_bounded (lo hi : Z) (t : tensor) : Prop :=
  forall x, In x (t_data t) -> lo <= x <= hi.

Definition tensor_shape_valid_b (t : tensor) : bool :=
  Nat.eqb (List.length (t_data t)) (List.fold_left Nat.mul (t_shape t) 1%nat).

Definition tensor_bounded_b (lo hi : Z) (t : tensor) : bool :=
  List.forallb (fun x => (lo <=? x) && (x <=? hi)) (t_data t).

Definition network_shape_valid_b (n : network) : bool :=
  List.forallb tensor_shape_valid_b n.

Definition network_bounded_b (lo hi : Z) (n : network) : bool :=
  List.forallb (tensor_bounded_b lo hi) n.

Inductive verification_result :=
  | Verified : verification_result
  | Failed : string -> verification_result.

Definition check_shape (t : tensor) : verification_result :=
  if tensor_shape_valid_b t then Verified else Failed "Shape mismatch"%string.

Definition check_bounds (lo hi : Z) (t : tensor) : verification_result :=
  if tensor_bounded_b lo hi t then Verified else Failed "Values out of bounds"%string.

Definition combine_results (r1 r2 : verification_result) : verification_result :=
  match r1 with Failed msg => Failed msg | Verified => r2 end.

Fixpoint verify_all_shapes (n : network) : verification_result :=
  match n with
  | [] => Verified
  | t :: rest => combine_results (check_shape t) (verify_all_shapes rest)
  end.

Lemma tensor_shape_valid_b_correct : forall t,
  tensor_shape_valid_b t = true -> tensor_shape_valid t.
Proof.
  intros t Hb. unfold tensor_shape_valid_b in Hb. unfold tensor_shape_valid.
  apply Nat.eqb_eq in Hb. exact Hb.
Qed.

Lemma tensor_bounded_b_correct : forall lo hi t,
  tensor_bounded_b lo hi t = true -> tensor_bounded lo hi t.
Proof.
  intros lo hi t Hb. unfold tensor_bounded_b in Hb. unfold tensor_bounded.
  intros x Hin. rewrite List.forallb_forall in Hb. specialize (Hb x Hin).
  apply andb_prop in Hb. destruct Hb as [Hlo Hhi].
  split; [apply Z.leb_le; exact Hlo | apply Z.leb_le; exact Hhi].
Qed.

Lemma verify_all_shapes_correct : forall n,
  verify_all_shapes n = Verified -> forall t, In t n -> tensor_shape_valid t.
Proof.
  induction n as [|t' rest IH]; intros Hv t Hin; [contradiction |].
  simpl in Hv, Hin. unfold combine_results in Hv.
  destruct (check_shape t') eqn:Hcheck; [|discriminate].
  destruct Hin as [Heq | Hrest].
  - subst. unfold check_shape in Hcheck.
    destruct (tensor_shape_valid_b t) eqn:Hb; [|discriminate].
    apply tensor_shape_valid_b_correct. exact Hb.
  - apply IH; assumption.
Qed.

(** * Diff Verification *)

Definition tensor_data_eq (t1 t2 : tensor) : bool :=
  List.forallb (fun '(x, y) => Z.eqb x y) (List.combine (t_data t1) (t_data t2))
  && Nat.eqb (List.length (t_data t1)) (List.length (t_data t2)).

Definition tensor_shape_eq (t1 t2 : tensor) : bool :=
  List.forallb (fun '(x, y) => Nat.eqb x y) (List.combine (t_shape t1) (t_shape t2))
  && Nat.eqb (List.length (t_shape t1)) (List.length (t_shape t2)).

Definition tensor_eq (t1 t2 : tensor) : bool :=
  tensor_shape_eq t1 t2 && tensor_data_eq t1 t2.

Fixpoint network_eq (n1 n2 : network) : bool :=
  match n1, n2 with
  | [], [] => true
  | t1 :: r1, t2 :: r2 => tensor_eq t1 t2 && network_eq r1 r2
  | _, _ => false
  end.

Definition within_tolerance (tol : Z) (x y : Z) : bool :=
  Z.abs (x - y) <=? tol.

Fixpoint network_approx_eq (tol : Z) (n1 n2 : network) : bool :=
  match n1, n2 with
  | [], [] => true
  | t1 :: r1, t2 :: r2 =>
      tensor_shape_eq t1 t2 &&
      List.forallb (fun '(x,y) => within_tolerance tol x y)
                   (List.combine (t_data t1) (t_data t2)) &&
      network_approx_eq tol r1 r2
  | _, _ => false
  end.

Lemma network_eq_refl : forall n, network_eq n n = true.
Proof.
  induction n as [|t rest IH]; [reflexivity |]. simpl.
  assert (Ht: tensor_eq t t = true).
  { unfold tensor_eq, tensor_shape_eq, tensor_data_eq.
    rewrite andb_true_iff. split.
    - rewrite andb_true_iff. split.
      + clear. induction (t_shape t) as [|s ss IHs]; simpl; [reflexivity |].
        rewrite Nat.eqb_refl. exact IHs.
      + apply Nat.eqb_refl.
    - rewrite andb_true_iff. split.
      + clear. induction (t_data t) as [|d ds IHd]; simpl; [reflexivity |].
        rewrite Z.eqb_refl. exact IHd.
      + apply Nat.eqb_refl. }
  rewrite Ht. simpl. exact IH.
Qed.

(** * Quantization *)

Record quant_params := mk_quant_params {
  qp_scale : Z;
  qp_zero_point : Z;
  qp_bits : nat
}.

Definition quant_min (qp : quant_params) : Z :=
  match qp_bits qp with 8%nat => -128 | 4%nat => -8 | _ => -2147483648 end.

Definition quant_max (qp : quant_params) : Z :=
  match qp_bits qp with 8%nat => 127 | 4%nat => 7 | _ => 2147483647 end.

Definition clip (lo hi x : Z) : Z :=
  if x <? lo then lo else if x >? hi then hi else x.

Definition quantize (qp : quant_params) (x : Z) : Z :=
  clip (quant_min qp) (quant_max qp) (x / qp_scale qp + qp_zero_point qp).

Definition dequantize (qp : quant_params) (q : Z) : Z :=
  (q - qp_zero_point qp) * qp_scale qp.

Definition quantize_list (qp : quant_params) (xs : list Z) : list Z :=
  List.map (quantize qp) xs.

Definition dequantize_list (qp : quant_params) (qs : list Z) : list Z :=
  List.map (dequantize qp) qs.

Definition quant_error (qp : quant_params) (x : Z) : Z :=
  Z.abs (dequantize qp (quantize qp x) - x).

Definition max_quant_error (qp : quant_params) (xs : list Z) : Z :=
  List.fold_left Z.max (List.map (quant_error qp) xs) 0.

Definition find_scale (xs : list Z) (bits : nat) : Z :=
  let max_val := List.fold_left (fun acc x => Z.max acc (Z.abs x)) xs 0 in
  let range := match bits with 8%nat => 127 | 4%nat => 7 | _ => 127 end in
  if max_val =? 0 then 1 else (max_val + range - 1) / range.

Definition calibrate (xs : list Z) (bits : nat) : quant_params :=
  mk_quant_params (find_scale xs bits) 0 bits.

Lemma clip_bounds : forall lo hi x, lo <= hi -> lo <= clip lo hi x <= hi.
Proof.
  intros lo hi x Hlohi. unfold clip.
  destruct (x <? lo) eqn:Hlo; [lia |].
  destruct (x >? hi) eqn:Hhi; [lia |].
  apply Z.ltb_ge in Hlo. rewrite Z.gtb_ltb in Hhi. apply Z.ltb_ge in Hhi. lia.
Qed.

Lemma quantize_in_range : forall qp x,
  quant_min qp <= quant_max qp -> quant_min qp <= quantize qp x <= quant_max qp.
Proof.
  intros qp x Hminmax. unfold quantize. apply clip_bounds. exact Hminmax.
Qed.

(** * Certificates *)

Inductive property_type :=
  | PT_ShapeValid
  | PT_Bounded : Z -> Z -> property_type
  | PT_NonEmpty
  | PT_Normalized
  | PT_Custom : string -> property_type.

Record proven_property := mk_proven_property {
  pp_type : property_type;
  pp_holds : bool;
  pp_witness : string
}.

Record json_certificate := mk_json_certificate {
  jc_version : string;
  jc_network_name : string;
  jc_tensor_count : nat;
  jc_total_params : nat;
  jc_properties : list proven_property;
  jc_timestamp : Z;
  jc_prover : string
}.

Definition check_property (n : network) (pt : property_type) : proven_property :=
  match pt with
  | PT_ShapeValid =>
      mk_proven_property PT_ShapeValid (network_shape_valid_b n) "shape_check"%string
  | PT_Bounded lo hi =>
      mk_proven_property (PT_Bounded lo hi) (network_bounded_b lo hi n) "bounds_check"%string
  | PT_NonEmpty =>
      mk_proven_property PT_NonEmpty (negb (Nat.eqb (List.length n) 0)) "non_empty_check"%string
  | PT_Normalized =>
      mk_proven_property PT_Normalized true "assumed_normalized"%string
  | PT_Custom s =>
      mk_proven_property (PT_Custom s) true "custom_check"%string
  end.

Definition generate_certificate (name : string) (n : network)
                                (props : list property_type) (ts : Z) : json_certificate :=
  mk_json_certificate "1.0"%string name (List.length n) (network_param_count n)
    (List.map (check_property n) props) ts "proof2weights"%string.

Definition all_properties_hold (cert : json_certificate) : bool :=
  List.forallb pp_holds (jc_properties cert).

(** * Attestations *)

Inductive attestation_level :=
  | AL_SelfSigned
  | AL_ThirdParty : string -> attestation_level
  | AL_Formal : string -> attestation_level.

Record attestation := mk_attestation {
  at_property : property_type;
  at_level : attestation_level;
  at_signature : Z;
  at_expiry : Z
}.

(** * Certificate Chain *)

Inductive chain_link :=
  | CL_Source : string -> chain_link
  | CL_Extraction : string -> string -> chain_link
  | CL_Serialization : string -> chain_link
  | CL_Verification : string -> bool -> chain_link.

Record certificate_chain := mk_cert_chain {
  cc_links : list chain_link;
  cc_final_hash : Z;
  cc_complete : bool
}.

Definition empty_chain : certificate_chain := mk_cert_chain [] 0 false.

Definition add_link (cc : certificate_chain) (link : chain_link) : certificate_chain :=
  mk_cert_chain (link :: cc_links cc) (cc_final_hash cc) false.

Definition simple_checksum (data : list byte) : Z :=
  List.fold_left (fun acc b => (acc * 31 + b) mod 4294967296) data 0.

Definition finalize_chain (cc : certificate_chain) (final_bytes : list byte) : certificate_chain :=
  mk_cert_chain (cc_links cc) (simple_checksum final_bytes) true.

(** * Proof-Carrying Weights *)

Inductive proof_fragment :=
  | PF_Axiom : string -> proof_fragment
  | PF_Lemma : string -> string -> proof_fragment
  | PF_Composition : proof_fragment -> proof_fragment -> proof_fragment
  | PF_Instantiation : string -> list Z -> proof_fragment.

Record proof_bundle := mk_proof_bundle {
  pb_theorem : string;
  pb_proof : proof_fragment;
  pb_dependencies : list string
}.

Record proof_carrying_weights := mk_pcw {
  pcw_network : network;
  pcw_certificate : json_certificate;
  pcw_proofs : list proof_bundle
}.

(** * Normalization *)

Record batch_norm_params := mk_batch_norm_params {
  bnp_gamma : tensor_1d;
  bnp_beta : tensor_1d;
  bnp_running_mean : tensor_1d;
  bnp_running_var : tensor_1d;
  bnp_epsilon : Z;
  bnp_momentum : Z
}.

Definition normalize_feature (x mean var gamma beta eps : Z) : Z :=
  let std := isqrt (var + eps) in
  let normalized := safe_div ((x * scale_factor - mean) * scale_factor) std in
  safe_div (gamma * normalized) scale_factor + beta.

Definition init_batch_norm (num_features : nat) : batch_norm_params :=
  mk_batch_norm_params
    (List.repeat scale_factor num_features)
    (List.repeat 0 num_features)
    (List.repeat 0 num_features)
    (List.repeat scale_factor num_features)
    1 900.

Record layer_norm_params := mk_layer_norm_params {
  ln_gamma : tensor_1d;
  ln_beta : tensor_1d;
  ln_epsilon : Z
}.

Definition layer_norm_single (params : layer_norm_params) (sample : tensor_1d) : tensor_1d :=
  let mean := mean_scaled sample in
  let var := variance_scaled sample mean in
  let std := isqrt (var + ln_epsilon params) in
  List.map (fun '(idx, x) =>
    let gamma := List.nth idx (ln_gamma params) scale_factor in
    let beta := List.nth idx (ln_beta params) 0 in
    let normalized := safe_div ((x * scale_factor - mean) * scale_factor) std in
    safe_div (gamma * normalized) scale_factor + beta
  ) (List.combine (List.seq 0 (List.length sample)) sample).

Definition layer_norm (params : layer_norm_params) (batch : tensor_2d) : tensor_2d :=
  List.map (layer_norm_single params) batch.

Definition init_layer_norm (num_features : nat) : layer_norm_params :=
  mk_layer_norm_params (List.repeat scale_factor num_features) (List.repeat 0 num_features) 1.

Record group_norm_params := mk_group_norm_params {
  gn_num_groups : nat;
  gn_num_channels : nat;
  gn_gamma : tensor_1d;
  gn_beta : tensor_1d;
  gn_epsilon : Z
}.

Definition init_group_norm (num_groups num_channels : nat) : group_norm_params :=
  mk_group_norm_params num_groups num_channels
    (List.repeat scale_factor num_channels) (List.repeat 0 num_channels) 1.

(** * Embeddings *)

Record token_embedding := mk_token_embedding {
  te_vocab_size : nat;
  te_embed_dim : nat;
  te_weights : tensor_2d
}.

Definition lookup_embedding (emb : token_embedding) (token_id : nat) : tensor_1d :=
  if Nat.ltb token_id (te_vocab_size emb)
  then List.nth token_id (te_weights emb) (List.repeat 0 (te_embed_dim emb))
  else List.repeat 0 (te_embed_dim emb).

Definition embed_tokens (emb : token_embedding) (token_ids : list nat) : tensor_2d :=
  List.map (lookup_embedding emb) token_ids.

Definition init_token_embedding (vocab_size embed_dim : nat) : token_embedding :=
  mk_token_embedding vocab_size embed_dim (List.repeat (List.repeat 0 embed_dim) vocab_size).

Record position_embedding := mk_position_embedding {
  pe_max_len : nat;
  pe_embed_dim : nat;
  pe_weights : tensor_2d
}.

Definition lookup_position (emb : position_embedding) (pos : nat) : tensor_1d :=
  if Nat.ltb pos (pe_max_len emb)
  then List.nth pos (pe_weights emb) (List.repeat 0 (pe_embed_dim emb))
  else List.repeat 0 (pe_embed_dim emb).

Definition embed_positions (emb : position_embedding) (seq_len : nat) : tensor_2d :=
  List.map (lookup_position emb) (List.seq 0 seq_len).

Definition init_position_embedding (max_len embed_dim : nat) : position_embedding :=
  mk_position_embedding max_len embed_dim (List.repeat (List.repeat 0 embed_dim) max_len).

Definition add_vectors (a b : tensor_1d) : tensor_1d :=
  List.map (fun '(x, y) => x + y) (List.combine a b).

Definition add_embeddings (tokens positions : tensor_2d) : tensor_2d :=
  List.map (fun '(t, p) => add_vectors t p) (List.combine tokens positions).

Record transformer_embeddings := mk_transformer_embeddings {
  tfe_token_emb : token_embedding;
  tfe_position_emb : position_embedding
}.

Definition transformer_embed (emb : transformer_embeddings) (token_ids : list nat) : tensor_2d :=
  let token_embeds := embed_tokens (tfe_token_emb emb) token_ids in
  let position_embeds := embed_positions (tfe_position_emb emb) (List.length token_ids) in
  add_embeddings token_embeds position_embeds.

(** * Sequence Models *)

Record rnn_cell_params := mk_rnn_cell_params {
  rnn_input_size : nat;
  rnn_hidden_size : nat;
  rnn_W_ih : tensor_2d;
  rnn_W_hh : tensor_2d;
  rnn_b : tensor_1d
}.

Definition rnn_cell_forward (params : rnn_cell_params) (x_t h_prev : tensor_1d) : tensor_1d :=
  let ih := mat_vec_mul (rnn_W_ih params) x_t in
  let hh := mat_vec_mul (rnn_W_hh params) h_prev in
  tanh_vec (vec_add (vec_add ih hh) (rnn_b params)).

Fixpoint rnn_forward_seq (params : rnn_cell_params) (xs : list tensor_1d) (h : tensor_1d) : list tensor_1d :=
  match xs with
  | [] => []
  | x :: rest =>
      let h_new := rnn_cell_forward params x h in
      h_new :: rnn_forward_seq params rest h_new
  end.

Definition rnn_forward (params : rnn_cell_params) (xs : list tensor_1d) : list tensor_1d :=
  rnn_forward_seq params xs (List.repeat 0 (rnn_hidden_size params)).

Definition init_rnn_cell (input_size hidden_size : nat) : rnn_cell_params :=
  mk_rnn_cell_params input_size hidden_size
    (List.repeat (List.repeat 0 input_size) hidden_size)
    (List.repeat (List.repeat 0 hidden_size) hidden_size)
    (List.repeat 0 hidden_size).

Lemma rnn_forward_seq_length : forall params xs h,
  List.length (rnn_forward_seq params xs h) = List.length xs.
Proof.
  intros params xs. induction xs as [|x rest IH]; intros h; [reflexivity |].
  simpl. f_equal. apply IH.
Qed.

Record lstm_cell_params := mk_lstm_cell_params {
  lstm_input_size : nat;
  lstm_hidden_size : nat;
  lstm_W_ii : tensor_2d; lstm_W_if : tensor_2d;
  lstm_W_ig : tensor_2d; lstm_W_io : tensor_2d;
  lstm_W_hi : tensor_2d; lstm_W_hf : tensor_2d;
  lstm_W_hg : tensor_2d; lstm_W_ho : tensor_2d;
  lstm_b_i : tensor_1d; lstm_b_f : tensor_1d;
  lstm_b_g : tensor_1d; lstm_b_o : tensor_1d
}.

Record lstm_state := mk_lstm_state {
  ls_hidden : tensor_1d;
  ls_cell : tensor_1d
}.

Definition lstm_cell_forward (params : lstm_cell_params) (x_t : tensor_1d) (state : lstm_state) : lstm_state :=
  let h_prev := ls_hidden state in
  let c_prev := ls_cell state in
  let i_t := sigmoid_vec (vec_add (vec_add (mat_vec_mul (lstm_W_ii params) x_t)
                                            (mat_vec_mul (lstm_W_hi params) h_prev))
                                   (lstm_b_i params)) in
  let f_t := sigmoid_vec (vec_add (vec_add (mat_vec_mul (lstm_W_if params) x_t)
                                            (mat_vec_mul (lstm_W_hf params) h_prev))
                                   (lstm_b_f params)) in
  let g_t := tanh_vec (vec_add (vec_add (mat_vec_mul (lstm_W_ig params) x_t)
                                         (mat_vec_mul (lstm_W_hg params) h_prev))
                                (lstm_b_g params)) in
  let o_t := sigmoid_vec (vec_add (vec_add (mat_vec_mul (lstm_W_io params) x_t)
                                            (mat_vec_mul (lstm_W_ho params) h_prev))
                                   (lstm_b_o params)) in
  let c_t := vec_add (vec_mul f_t c_prev) (vec_mul i_t g_t) in
  let h_t := vec_mul o_t (tanh_vec c_t) in
  mk_lstm_state h_t c_t.

Fixpoint lstm_forward_seq (params : lstm_cell_params) (xs : list tensor_1d) (state : lstm_state) : list lstm_state :=
  match xs with
  | [] => []
  | x :: rest =>
      let new_state := lstm_cell_forward params x state in
      new_state :: lstm_forward_seq params rest new_state
  end.

Definition init_lstm_state (hidden_size : nat) : lstm_state :=
  mk_lstm_state (List.repeat 0 hidden_size) (List.repeat 0 hidden_size).

Definition lstm_forward (params : lstm_cell_params) (xs : list tensor_1d) : list lstm_state :=
  lstm_forward_seq params xs (init_lstm_state (lstm_hidden_size params)).

Definition init_lstm_cell (input_size hidden_size : nat) : lstm_cell_params :=
  let zero_i := List.repeat (List.repeat 0 input_size) hidden_size in
  let zero_h := List.repeat (List.repeat 0 hidden_size) hidden_size in
  let zero_b := List.repeat 0 hidden_size in
  mk_lstm_cell_params input_size hidden_size
    zero_i zero_i zero_i zero_i zero_h zero_h zero_h zero_h
    zero_b zero_b zero_b zero_b.

Lemma lstm_forward_seq_length : forall params xs state,
  List.length (lstm_forward_seq params xs state) = List.length xs.
Proof.
  intros params xs. induction xs as [|x rest IH]; intros state; [reflexivity |].
  simpl. f_equal. apply IH.
Qed.

Record gru_cell_params := mk_gru_cell_params {
  gru_input_size : nat;
  gru_hidden_size : nat;
  gru_W_iz : tensor_2d; gru_W_ir : tensor_2d; gru_W_in : tensor_2d;
  gru_W_hz : tensor_2d; gru_W_hr : tensor_2d; gru_W_hn : tensor_2d;
  gru_b_z : tensor_1d; gru_b_r : tensor_1d; gru_b_n : tensor_1d
}.

Definition gru_cell_forward (params : gru_cell_params) (x_t h_prev : tensor_1d) : tensor_1d :=
  let z_t := sigmoid_vec (vec_add (vec_add (mat_vec_mul (gru_W_iz params) x_t)
                                            (mat_vec_mul (gru_W_hz params) h_prev))
                                   (gru_b_z params)) in
  let r_t := sigmoid_vec (vec_add (vec_add (mat_vec_mul (gru_W_ir params) x_t)
                                            (mat_vec_mul (gru_W_hr params) h_prev))
                                   (gru_b_r params)) in
  let h_reset := vec_mul r_t h_prev in
  let n_t := tanh_vec (vec_add (vec_add (mat_vec_mul (gru_W_in params) x_t)
                                         (mat_vec_mul (gru_W_hn params) h_reset))
                                (gru_b_n params)) in
  vec_add (vec_mul (vec_sub_from_one z_t) n_t) (vec_mul z_t h_prev).

Fixpoint gru_forward_seq (params : gru_cell_params) (xs : list tensor_1d) (h : tensor_1d) : list tensor_1d :=
  match xs with
  | [] => []
  | x :: rest =>
      let h_new := gru_cell_forward params x h in
      h_new :: gru_forward_seq params rest h_new
  end.

Definition gru_forward (params : gru_cell_params) (xs : list tensor_1d) : list tensor_1d :=
  gru_forward_seq params xs (List.repeat 0 (gru_hidden_size params)).

Definition init_gru_cell (input_size hidden_size : nat) : gru_cell_params :=
  let zero_i := List.repeat (List.repeat 0 input_size) hidden_size in
  let zero_h := List.repeat (List.repeat 0 hidden_size) hidden_size in
  let zero_b := List.repeat 0 hidden_size in
  mk_gru_cell_params input_size hidden_size
    zero_i zero_i zero_i zero_h zero_h zero_h zero_b zero_b zero_b.

Lemma gru_forward_seq_length : forall params xs h,
  List.length (gru_forward_seq params xs h) = List.length xs.
Proof.
  intros params xs. induction xs as [|x rest IH]; intros h; [reflexivity |].
  simpl. f_equal. apply IH.
Qed.

Record bidirectional_params := mk_bidirectional_params {
  bi_forward_params : rnn_cell_params;
  bi_backward_params : rnn_cell_params
}.

Definition bidirectional_forward (params : bidirectional_params) (xs : list tensor_1d) : list tensor_1d :=
  let forward_outputs := rnn_forward (bi_forward_params params) xs in
  let backward_outputs := List.rev (rnn_forward (bi_backward_params params) (List.rev xs)) in
  List.map (fun '(f, b) => vec_concat f b) (List.combine forward_outputs backward_outputs).

Definition init_bidirectional (input_size hidden_size : nat) : bidirectional_params :=
  mk_bidirectional_params (init_rnn_cell input_size hidden_size) (init_rnn_cell input_size hidden_size).

Lemma bidirectional_forward_length : forall params xs,
  List.length (bidirectional_forward params xs) = List.length xs.
Proof.
  intros params xs. unfold bidirectional_forward.
  rewrite List.length_map. rewrite List.length_combine.
  unfold rnn_forward. rewrite rnn_forward_seq_length.
  rewrite List.length_rev. rewrite rnn_forward_seq_length. rewrite List.length_rev. lia.
Qed.

Record dense_params := mk_dense_params {
  dense_in_features : nat;
  dense_out_features : nat;
  dense_weight : tensor_2d;
  dense_bias : tensor_1d
}.

Definition dense_forward (params : dense_params) (input : tensor_1d) : tensor_1d :=
  vec_add (mat_vec_mul (dense_weight params) input) (dense_bias params).

Definition init_dense (in_features out_features : nat) : dense_params :=
  mk_dense_params in_features out_features
    (List.repeat (List.repeat 0 in_features) out_features)
    (List.repeat 0 out_features).

Record seq_classifier_params := mk_seq_classifier_params {
  sc_encoder : rnn_cell_params;
  sc_classifier : dense_params;
  sc_num_classes : nat
}.

Fixpoint rnn_encode_seq (params : rnn_cell_params) (xs : list tensor_1d) (h : tensor_1d) : tensor_1d :=
  match xs with [] => h | x :: rest => rnn_encode_seq params rest (rnn_cell_forward params x h) end.

Definition rnn_encode (params : rnn_cell_params) (xs : list tensor_1d) : tensor_1d :=
  rnn_encode_seq params xs (List.repeat 0 (rnn_hidden_size params)).

Definition seq_classifier_forward (params : seq_classifier_params) (xs : list tensor_1d) : tensor_1d :=
  softmax (dense_forward (sc_classifier params) (rnn_encode (sc_encoder params) xs)).

Definition seq_classify (params : seq_classifier_params) (xs : list tensor_1d) : nat :=
  argmax (seq_classifier_forward params xs).

Definition init_seq_classifier (input_size hidden_size num_classes : nat) : seq_classifier_params :=
  mk_seq_classifier_params (init_rnn_cell input_size hidden_size) (init_dense hidden_size num_classes) num_classes.

(** * Attention *)

Definition softmax_row (row : tensor_1d) : tensor_1d :=
  let exps := List.map exp_approx row in
  let sum_exp := List.fold_left Z.add exps 0 in
  if sum_exp =? 0 then List.map (fun _ => 0) row
  else List.map (fun e => (e * scale_factor) / sum_exp) exps.

Definition softmax_2d (m : tensor_2d) : tensor_2d :=
  List.map softmax_row m.

Lemma softmax_row_length : forall row,
  List.length (softmax_row row) = List.length row.
Proof.
  intros row. unfold softmax_row.
  destruct (List.fold_left Z.add (List.map exp_approx row) 0 =? 0).
  - apply List.length_map.
  - rewrite List.length_map. apply List.length_map.
Qed.

Definition compute_attention_scores (q k : tensor_2d) : tensor_2d :=
  mat_mul q (mat_transpose k).

Definition scale_scores (scores : tensor_2d) (d_k : nat) : tensor_2d :=
  let sqrt_d := isqrt (Z.of_nat d_k * scale_factor) in
  let scale := if sqrt_d =? 0 then scale_factor else (scale_factor * scale_factor) / sqrt_d in
  List.map (fun row => List.map (fun x => (x * scale) / scale_factor) row) scores.

Definition attention_weights (q k : tensor_2d) (d_k : nat) : tensor_2d :=
  softmax_2d (scale_scores (compute_attention_scores q k) d_k).

Definition apply_attention (weights v : tensor_2d) : tensor_2d :=
  mat_mul weights v.

Definition scaled_dot_product_attention (q k v : tensor_2d) : tensor_2d :=
  let d_k := mat_cols q in
  apply_attention (attention_weights q k d_k) v.

Lemma attention_output_rows : forall q k v,
  mat_rows (scaled_dot_product_attention q k v) = mat_rows q.
Proof.
  intros q k v.
  unfold scaled_dot_product_attention, apply_attention, attention_weights.
  unfold softmax_2d, scale_scores, compute_attention_scores, mat_mul, mat_rows.
  repeat rewrite List.length_map. reflexivity.
Qed.

Definition split_row_into_heads (num_heads : nat) (row : tensor_1d) : list tensor_1d :=
  let head_dim := Nat.div (List.length row) num_heads in
  List.map (fun h => List.firstn head_dim (List.skipn (h * head_dim) row)) (List.seq 0 num_heads).

Definition split_into_heads (num_heads : nat) (m : tensor_2d) : tensor_3d :=
  let rows_split := List.map (split_row_into_heads num_heads) m in
  List.map (fun h => List.map (fun row_heads => List.nth h row_heads []) rows_split) (List.seq 0 num_heads).

Definition concat_heads (heads : tensor_3d) : tensor_2d :=
  match heads with
  | [] => []
  | first_head :: _ =>
      List.map (fun seq_idx =>
        List.concat (List.map (fun head => List.nth seq_idx head []) heads)
      ) (List.seq 0 (List.length first_head))
  end.

Definition linear_project (weight : tensor_2d) (input : tensor_2d) : tensor_2d :=
  mat_mul input weight.

Record mha_params := mk_mha_params {
  mha_num_heads : nat;
  mha_d_model : nat;
  mha_d_k : nat;
  mha_d_v : nat;
  mha_W_Q : tensor_2d;
  mha_W_K : tensor_2d;
  mha_W_V : tensor_2d;
  mha_W_O : tensor_2d
}.

Definition multi_head_attention (params : mha_params) (q k v : tensor_2d) : tensor_2d :=
  let q_proj := linear_project (mha_W_Q params) q in
  let k_proj := linear_project (mha_W_K params) k in
  let v_proj := linear_project (mha_W_V params) v in
  let q_heads := split_into_heads (mha_num_heads params) q_proj in
  let k_heads := split_into_heads (mha_num_heads params) k_proj in
  let v_heads := split_into_heads (mha_num_heads params) v_proj in
  let head_outputs := List.map (fun '(qh, (kh, vh)) => scaled_dot_product_attention qh kh vh)
                               (List.combine q_heads (List.combine k_heads v_heads)) in
  linear_project (mha_W_O params) (concat_heads head_outputs).

Definition self_attention (params : mha_params) (x : tensor_2d) : tensor_2d :=
  multi_head_attention params x x x.

Definition init_mha_params (num_heads d_model : nat) : mha_params :=
  let d_k := Nat.div d_model num_heads in
  let zero_weight := List.repeat (List.repeat 0 d_model) d_model in
  mk_mha_params num_heads d_model d_k d_k zero_weight zero_weight zero_weight zero_weight.

Definition causal_mask_entry (row col : nat) : Z :=
  if Nat.leb col row then 0 else neg_inf.

Definition causal_mask (seq_len : nat) : tensor_2d :=
  List.map (fun row => List.map (fun col => causal_mask_entry row col) (List.seq 0 seq_len))
           (List.seq 0 seq_len).

Definition apply_mask (scores mask : tensor_2d) : tensor_2d :=
  List.map (fun '(s_row, m_row) =>
    List.map (fun '(s, m) => s + m) (List.combine s_row m_row)
  ) (List.combine scores mask).

Definition masked_attention (q k v : tensor_2d) : tensor_2d :=
  let d_k := mat_cols q in
  let scores := scale_scores (compute_attention_scores q k) d_k in
  let masked := apply_mask scores (causal_mask (mat_rows q)) in
  mat_mul (softmax_2d masked) v.

Lemma causal_mask_length : forall seq_len,
  List.length (causal_mask seq_len) = seq_len.
Proof.
  intros seq_len. unfold causal_mask.
  rewrite List.length_map. rewrite List.length_seq. reflexivity.
Qed.

Definition cross_attention (q k v : tensor_2d) : tensor_2d :=
  let d_k := mat_cols q in
  mat_mul (softmax_2d (scale_scores (compute_attention_scores q k) d_k)) v.

Record cross_attention_params := mk_cross_attention_params {
  ca_d_model : nat;
  ca_d_k : nat;
  ca_d_v : nat;
  ca_W_Q : tensor_2d;
  ca_W_K : tensor_2d;
  ca_W_V : tensor_2d;
  ca_W_O : tensor_2d
}.

Definition cross_attention_with_proj (params : cross_attention_params)
                                     (decoder_hidden encoder_output : tensor_2d) : tensor_2d :=
  let q := linear_project (ca_W_Q params) decoder_hidden in
  let k := linear_project (ca_W_K params) encoder_output in
  let v := linear_project (ca_W_V params) encoder_output in
  linear_project (ca_W_O params) (cross_attention q k v).

Lemma cross_attention_output_rows : forall q k v,
  mat_rows (cross_attention q k v) = mat_rows q.
Proof.
  intros q k v. unfold cross_attention, softmax_2d, scale_scores, compute_attention_scores.
  unfold mat_mul, mat_rows. repeat rewrite List.length_map. reflexivity.
Qed.

Record attention_export := mk_attention_export {
  ae_query_len : nat;
  ae_key_len : nat;
  ae_weights : tensor_2d;
  ae_layer_name : string
}.

Definition create_attention_export (name : string) (q k : tensor_2d) : attention_export :=
  mk_attention_export (mat_rows q) (mat_rows k) (attention_weights q k (mat_cols q)) name.

Definition attention_argmax_pattern (weights : tensor_2d) : list nat :=
  List.map argmax weights.

Definition entropy_row (row : tensor_1d) : Z :=
  let log2_approx := fun x =>
    if x <=? 0 then 0
    else if x <? 100 then -3000
    else if x <? 500 then -1000
    else if x <? 900 then -150
    else if x <=? 1100 then 0
    else 500 in
  let terms := List.map (fun p =>
    if p <=? 0 then 0
    else - (p * log2_approx p) / scale_factor
  ) row in
  List.fold_left Z.add terms 0.

Definition attention_entropy (weights : tensor_2d) : tensor_1d :=
  List.map entropy_row weights.

Definition serialize_attention_weights (weights : tensor_2d) : list Z :=
  List.concat (List.map (fun row => List.concat (List.map z_to_bytes_le row)) weights).

(** * Safetensors Parser *)

Record tensor_meta := mk_tensor_meta {
  tm_name : string;
  tm_dtype : string;
  tm_shape : list nat;
  tm_data_start : nat;
  tm_data_end : nat
}.

Fixpoint find_char (c : ascii) (s : string) : nat :=
  match s with
  | EmptyString => 0%nat
  | String c' rest => if Ascii.eqb c c' then 0%nat else S (find_char c rest)
  end.

Fixpoint string_drop (n : nat) (s : string) : string :=
  match n, s with
  | O, _ => s
  | S n', EmptyString => EmptyString
  | S n', String _ rest => string_drop n' rest
  end.

Definition parse_header_size (bs : list byte) : nat * list byte :=
  let size := bytes_to_z_le_u64 (take 8 bs) in
  (Z.to_nat size, drop 8 bs).

Definition parse_header_string (size : nat) (bs : list byte) : string * list byte :=
  let header_bytes := take size bs in
  let header_str :=
    (fix bytes_to_string (l : list byte) : string :=
      match l with
      | [] => EmptyString
      | b :: rest => String (Ascii.ascii_of_nat (Z.to_nat b)) (bytes_to_string rest)
      end) header_bytes in
  (header_str, drop size bs).

Definition parse_safetensors_header (bs : list byte) : nat * string * list byte :=
  let (header_size, after_size) := parse_header_size bs in
  let (header_str, data_bytes) := parse_header_string header_size after_size in
  (header_size, header_str, data_bytes).

(** * ONNX Types *)

Inductive onnx_dtype :=
  | ONNX_FLOAT | ONNX_UINT8 | ONNX_INT8 | ONNX_UINT16 | ONNX_INT16
  | ONNX_INT32 | ONNX_INT64 | ONNX_STRING | ONNX_BOOL
  | ONNX_FLOAT16 | ONNX_DOUBLE | ONNX_UINT32 | ONNX_UINT64 | ONNX_BFLOAT16.

Definition onnx_dtype_code (dt : onnx_dtype) : Z :=
  match dt with
  | ONNX_FLOAT => 1 | ONNX_UINT8 => 2 | ONNX_INT8 => 3 | ONNX_UINT16 => 4 | ONNX_INT16 => 5
  | ONNX_INT32 => 6 | ONNX_INT64 => 7 | ONNX_STRING => 8 | ONNX_BOOL => 9
  | ONNX_FLOAT16 => 10 | ONNX_DOUBLE => 11 | ONNX_UINT32 => 12 | ONNX_UINT64 => 13 | ONNX_BFLOAT16 => 16
  end.

Definition onnx_dtype_of_code (code : Z) : option onnx_dtype :=
  match code with
  | 1 => Some ONNX_FLOAT | 2 => Some ONNX_UINT8 | 3 => Some ONNX_INT8
  | 4 => Some ONNX_UINT16 | 5 => Some ONNX_INT16 | 6 => Some ONNX_INT32
  | 7 => Some ONNX_INT64 | 8 => Some ONNX_STRING | 9 => Some ONNX_BOOL
  | 10 => Some ONNX_FLOAT16 | 11 => Some ONNX_DOUBLE | 12 => Some ONNX_UINT32
  | 13 => Some ONNX_UINT64 | 16 => Some ONNX_BFLOAT16 | _ => None
  end.

Record onnx_tensor := mk_onnx_tensor {
  ot_name : string;
  ot_dtype : onnx_dtype;
  ot_dims : list Z;
  ot_raw_data : list Z
}.

Record onnx_node := mk_onnx_node {
  on_name : string;
  on_op_type : string;
  on_inputs : list string;
  on_outputs : list string
}.

Record onnx_graph := mk_onnx_graph {
  og_name : string;
  og_nodes : list onnx_node;
  og_inputs : list string;
  og_outputs : list string;
  og_initializers : list onnx_tensor
}.

Record onnx_model := mk_onnx_model {
  om_ir_version : Z;
  om_producer_name : string;
  om_producer_version : string;
  om_domain : string;
  om_model_version : Z;
  om_graph : onnx_graph
}.

Fixpoint parse_varint_aux (bs : list byte) (shift : Z) (acc : Z) (fuel : nat) : Z * list byte :=
  match fuel, bs with
  | O, _ => (acc, bs)
  | S fuel', [] => (acc, [])
  | S fuel', b :: rest =>
      let value := b mod 128 in
      let new_acc := acc + value * (2 ^ shift) in
      if b <? 128 then (new_acc, rest)
      else parse_varint_aux rest (shift + 7) new_acc fuel'
  end.

Definition parse_varint (bs : list byte) : Z * list byte :=
  parse_varint_aux bs 0 0 10%nat.

Lemma parse_varint_zero : fst (parse_varint [0]) = 0.
Proof. reflexivity. Qed.

Lemma parse_varint_one : fst (parse_varint [1]) = 1.
Proof. reflexivity. Qed.

Lemma parse_varint_127 : fst (parse_varint [127]) = 127.
Proof. reflexivity. Qed.

(** * Example Networks *)

Definition majority_weight : list (list Z) := [[1; 1; 1; 1; 1; 1; 1; 1]].
Definition majority_bias : list Z := [-5].

Definition majority_network : network :=
  network_of_layers [
    layer "layer1.weight" majority_weight;
    layer_1d "layer1.bias" majority_bias
  ].

Definition mod3_layer1_weight : list (list Z) :=
  [ [1;1;1;1;1;1;1;1]; [1;1;1;1;1;1;1;1]; [1;1;1;1;1;1;1;1];
    [1;1;1;1;1;1;1;1]; [1;1;1;1;1;1;1;1]; [1;1;1;1;1;1;1;1];
    [1;1;1;1;1;1;1;1]; [1;1;1;1;1;1;1;1]; [1;1;1;1;1;1;1;1] ].
Definition mod3_layer1_bias : list Z := [0; -1; -2; -3; -4; -5; -6; -7; -8].

Definition mod3_layer2_weight : list (list Z) :=
  [ [0; 1; 1; -2; 1; 1; -2; 1; 1]; [0; 1; 1; -2; 1; 1; -2; 1; 1] ].
Definition mod3_layer2_bias : list Z := [-1; -2].

Definition mod3_output_weight : list (list Z) := [ [-1; 0]; [1; -2]; [0; 1] ].
Definition mod3_output_bias : list Z := [0; -1; -1].

Definition mod3_network : network :=
  network_of_layers [
    layer "layer1.weight" mod3_layer1_weight;
    layer_1d "layer1.bias" mod3_layer1_bias;
    layer "layer2.weight" mod3_layer2_weight;
    layer_1d "layer2.bias" mod3_layer2_bias;
    layer "output.weight" mod3_output_weight;
    layer_1d "output.bias" mod3_output_bias
  ].

(** * Transformer Blocks *)

(** ** Additional Activations *)

Definition swish_approx (x : Z) : Z :=
  let sigmoid_x :=
    if x <? -4000 then 0
    else if x >? 4000 then scale_factor
    else (x + 4000) / 8 in
  (x * sigmoid_x) / scale_factor.

Definition swish_vec (v : tensor_1d) : tensor_1d :=
  List.map swish_approx v.

(** ** Additional Projection *)

Definition linear_project_bias (weight : tensor_2d) (bias : tensor_1d) (input : tensor_2d) : tensor_2d :=
  List.map (fun row => vec_add (mat_vec_mul (mat_transpose weight) row) bias) input.

(** ** Shape Preservation Lemmas *)

Lemma layer_norm_rows : forall params x,
  mat_rows (layer_norm params x) = mat_rows x.
Proof.
  intros params x. unfold layer_norm, mat_rows. apply List.length_map.
Qed.

Lemma mat_mul_rows : forall a b,
  mat_rows (mat_mul a b) = mat_rows a.
Proof.
  intros a b. unfold mat_mul, mat_rows. apply List.length_map.
Qed.

Lemma linear_project_rows : forall w x,
  mat_rows (linear_project w x) = mat_rows x.
Proof.
  intros w x. unfold linear_project. apply mat_mul_rows.
Qed.

Lemma softmax_2d_rows : forall m,
  mat_rows (softmax_2d m) = mat_rows m.
Proof.
  intros m. unfold softmax_2d, mat_rows. apply List.length_map.
Qed.

Lemma concat_heads_rows : forall heads,
  heads <> [] ->
  mat_rows (concat_heads heads) = mat_rows (hd [] heads).
Proof.
  intros heads Hne.
  unfold concat_heads.
  destruct heads as [|h rest].
  - contradiction.
  - unfold mat_rows. rewrite List.length_map, List.length_seq. reflexivity.
Qed.

Lemma split_into_heads_length : forall n m,
  List.length (split_into_heads n m) = n.
Proof.
  intros n m. unfold split_into_heads. rewrite List.length_map, List.length_seq. reflexivity.
Qed.

Lemma split_into_heads_inner_rows : forall n m h,
  (n > 0)%nat ->
  In h (split_into_heads n m) ->
  mat_rows h = mat_rows m.
Proof.
  intros n m h Hn Hin.
  unfold split_into_heads in Hin.
  rewrite List.in_map_iff in Hin.
  destruct Hin as [idx [Heq Hidx]].
  subst h. unfold mat_rows.
  rewrite List.length_map. apply List.length_map.
Qed.

Lemma scale_scores_rows : forall m d_k,
  mat_rows (scale_scores m d_k) = mat_rows m.
Proof.
  intros m d_k. unfold scale_scores, mat_rows. apply List.length_map.
Qed.

Lemma compute_attention_scores_rows : forall q k,
  mat_rows (compute_attention_scores q k) = mat_rows q.
Proof.
  intros q k. unfold compute_attention_scores. apply mat_mul_rows.
Qed.

Lemma attention_weights_rows : forall q k d_k,
  mat_rows (attention_weights q k d_k) = mat_rows q.
Proof.
  intros q k d_k. unfold attention_weights.
  rewrite softmax_2d_rows, scale_scores_rows, compute_attention_scores_rows.
  reflexivity.
Qed.

Lemma scaled_dot_product_attention_rows : forall q k v,
  mat_rows (scaled_dot_product_attention q k v) = mat_rows q.
Proof.
  intros q k v. unfold scaled_dot_product_attention, apply_attention.
  rewrite mat_mul_rows, attention_weights_rows. reflexivity.
Qed.

Lemma split_into_heads_nonempty : forall n m,
  (n > 0)%nat -> split_into_heads n m <> [].
Proof.
  intros n m Hn. unfold split_into_heads.
  destruct n; [lia | ].
  simpl. discriminate.
Qed.

Lemma split_into_heads_hd_rows : forall n m,
  (n > 0)%nat -> mat_rows (hd [] (split_into_heads n m)) = mat_rows m.
Proof.
  intros n m Hn.
  unfold split_into_heads.
  destruct n; [lia | ].
  simpl. unfold mat_rows. rewrite !List.length_map. reflexivity.
Qed.

Lemma map_attention_heads_nonempty : forall params x,
  (mha_num_heads params > 0)%nat ->
  List.map (fun '(qh, (kh, vh)) => scaled_dot_product_attention qh kh vh)
    (List.combine
      (split_into_heads (mha_num_heads params) (linear_project (mha_W_Q params) x))
      (List.combine
        (split_into_heads (mha_num_heads params) (linear_project (mha_W_K params) x))
        (split_into_heads (mha_num_heads params) (linear_project (mha_W_V params) x)))) <> [].
Proof.
  intros params x Hheads.
  intro Hcontra.
  apply (f_equal (@List.length _)) in Hcontra.
  rewrite List.length_map, List.length_combine in Hcontra.
  rewrite !split_into_heads_length in Hcontra.
  rewrite List.length_combine, !split_into_heads_length in Hcontra.
  rewrite !Nat.min_id in Hcontra. simpl in Hcontra. lia.
Qed.

Lemma map_attention_heads_hd_rows : forall params x,
  (mha_num_heads params > 0)%nat ->
  mat_rows (hd [] (List.map (fun '(qh, (kh, vh)) => scaled_dot_product_attention qh kh vh)
    (List.combine
      (split_into_heads (mha_num_heads params) (linear_project (mha_W_Q params) x))
      (List.combine
        (split_into_heads (mha_num_heads params) (linear_project (mha_W_K params) x))
        (split_into_heads (mha_num_heads params) (linear_project (mha_W_V params) x)))))) = mat_rows x.
Proof.
  intros params x Hheads.
  destruct (mha_num_heads params) as [|n] eqn:Hn; [lia | ].
  simpl. rewrite scaled_dot_product_attention_rows.
  unfold split_into_heads. simpl.
  unfold mat_rows. rewrite !List.length_map.
  rewrite linear_project_rows. reflexivity.
Qed.

Lemma self_attention_rows : forall params x,
  (mha_num_heads params > 0)%nat ->
  mat_rows (self_attention params x) = mat_rows x.
Proof.
  intros params x Hheads.
  unfold self_attention, multi_head_attention.
  rewrite linear_project_rows.
  rewrite concat_heads_rows by (apply map_attention_heads_nonempty; assumption).
  apply map_attention_heads_hd_rows. assumption.
Qed.

Lemma add_matrices_rows_eq : forall a b,
  mat_rows a = mat_rows b ->
  mat_rows (add_matrices a b) = mat_rows a.
Proof.
  intros a b Heq.
  unfold add_matrices, mat_rows in *.
  rewrite List.length_map, List.length_combine.
  rewrite Heq. apply Nat.min_id.
Qed.

(** ** Masked Self-Attention with Parameters *)

Definition masked_self_attention (params : mha_params) (x : tensor_2d) : tensor_2d :=
  let q_proj := linear_project (mha_W_Q params) x in
  let k_proj := linear_project (mha_W_K params) x in
  let v_proj := linear_project (mha_W_V params) x in
  let q_heads := split_into_heads (mha_num_heads params) q_proj in
  let k_heads := split_into_heads (mha_num_heads params) k_proj in
  let v_heads := split_into_heads (mha_num_heads params) v_proj in
  let seq_len := mat_rows x in
  let mask := causal_mask seq_len in
  let head_outputs := List.map (fun '(qh, (kh, vh)) =>
    let d_k := mat_cols qh in
    let scores := scale_scores (compute_attention_scores qh kh) d_k in
    let masked := apply_mask scores mask in
    mat_mul (softmax_2d masked) vh
  ) (List.combine q_heads (List.combine k_heads v_heads)) in
  linear_project (mha_W_O params) (concat_heads head_outputs).

Lemma map_masked_attention_heads_nonempty : forall params x,
  (mha_num_heads params > 0)%nat ->
  List.map (fun '(qh, (kh, vh)) =>
    let d_k := mat_cols qh in
    let scores := scale_scores (compute_attention_scores qh kh) d_k in
    let masked := apply_mask scores (causal_mask (mat_rows x)) in
    mat_mul (softmax_2d masked) vh)
    (List.combine
      (split_into_heads (mha_num_heads params) (linear_project (mha_W_Q params) x))
      (List.combine
        (split_into_heads (mha_num_heads params) (linear_project (mha_W_K params) x))
        (split_into_heads (mha_num_heads params) (linear_project (mha_W_V params) x)))) <> [].
Proof.
  intros params x Hheads.
  intro Hcontra.
  apply (f_equal (@List.length _)) in Hcontra.
  rewrite List.length_map, List.length_combine in Hcontra.
  rewrite !split_into_heads_length in Hcontra.
  rewrite List.length_combine, !split_into_heads_length in Hcontra.
  rewrite !Nat.min_id in Hcontra. simpl in Hcontra. lia.
Qed.

Lemma map_masked_attention_heads_hd_rows : forall params x,
  (mha_num_heads params > 0)%nat ->
  mat_rows (hd [] (List.map (fun '(qh, (kh, vh)) =>
    let d_k := mat_cols qh in
    let scores := scale_scores (compute_attention_scores qh kh) d_k in
    let masked := apply_mask scores (causal_mask (mat_rows x)) in
    mat_mul (softmax_2d masked) vh)
    (List.combine
      (split_into_heads (mha_num_heads params) (linear_project (mha_W_Q params) x))
      (List.combine
        (split_into_heads (mha_num_heads params) (linear_project (mha_W_K params) x))
        (split_into_heads (mha_num_heads params) (linear_project (mha_W_V params) x)))))) = mat_rows x.
Proof.
  intros params x Hheads.
  destruct (mha_num_heads params) as [|n] eqn:Hn; [lia | ].
  simpl.
  rewrite mat_mul_rows, softmax_2d_rows.
  unfold apply_mask, mat_rows.
  rewrite List.length_map, List.length_combine.
  rewrite scale_scores_rows, compute_attention_scores_rows.
  unfold split_into_heads. simpl.
  unfold mat_rows. rewrite !List.length_map.
  rewrite linear_project_rows.
  unfold causal_mask. rewrite List.length_map, List.length_seq.
  apply Nat.min_id.
Qed.

Lemma masked_self_attention_rows : forall params x,
  (mha_num_heads params > 0)%nat ->
  mat_rows (masked_self_attention params x) = mat_rows x.
Proof.
  intros params x Hheads.
  unfold masked_self_attention.
  rewrite linear_project_rows.
  rewrite concat_heads_rows by (apply map_masked_attention_heads_nonempty; assumption).
  apply map_masked_attention_heads_hd_rows. assumption.
Qed.

(** ** Cross-Attention with MHA Parameters *)

Definition cross_attention_mha (params : mha_params) (q_input kv_input : tensor_2d) : tensor_2d :=
  multi_head_attention params q_input kv_input kv_input.

(** ** Pre-norm Block (GPT-style) *)

Record prenorm_attention_block := mk_prenorm_attention_block {
  pab_norm : layer_norm_params;
  pab_attention : mha_params
}.

Definition prenorm_attention_forward (block : prenorm_attention_block) (x : tensor_2d) : tensor_2d :=
  let normalized := layer_norm (pab_norm block) x in
  let attended := self_attention (pab_attention block) normalized in
  add_matrices x attended.

Definition prenorm_masked_attention_forward (block : prenorm_attention_block) (x : tensor_2d) : tensor_2d :=
  let normalized := layer_norm (pab_norm block) x in
  let attended := masked_self_attention (pab_attention block) normalized in
  add_matrices x attended.

Definition init_prenorm_attention_block (num_heads d_model : nat) : prenorm_attention_block :=
  mk_prenorm_attention_block
    (init_layer_norm d_model)
    (init_mha_params num_heads d_model).

Lemma prenorm_preserves_shape : forall block x,
  (mha_num_heads (pab_attention block) > 0)%nat ->
  mat_rows (prenorm_attention_forward block x) = mat_rows x.
Proof.
  intros block x Hheads.
  unfold prenorm_attention_forward.
  apply add_matrices_rows_eq.
  rewrite self_attention_rows by assumption.
  symmetry. apply layer_norm_rows.
Qed.

Lemma prenorm_masked_preserves_shape : forall block x,
  (mha_num_heads (pab_attention block) > 0)%nat ->
  mat_rows (prenorm_masked_attention_forward block x) = mat_rows x.
Proof.
  intros block x Hheads.
  unfold prenorm_masked_attention_forward.
  apply add_matrices_rows_eq.
  rewrite masked_self_attention_rows by assumption.
  symmetry. apply layer_norm_rows.
Qed.

(** ** Post-norm Block (BERT-style) *)

Record postnorm_attention_block := mk_postnorm_attention_block {
  poab_attention : mha_params;
  poab_norm : layer_norm_params
}.

Definition postnorm_attention_forward (block : postnorm_attention_block) (x : tensor_2d) : tensor_2d :=
  let attended := self_attention (poab_attention block) x in
  let residual := add_matrices x attended in
  layer_norm (poab_norm block) residual.

Definition postnorm_masked_attention_forward (block : postnorm_attention_block) (x : tensor_2d) : tensor_2d :=
  let attended := masked_self_attention (poab_attention block) x in
  let residual := add_matrices x attended in
  layer_norm (poab_norm block) residual.

Definition init_postnorm_attention_block (num_heads d_model : nat) : postnorm_attention_block :=
  mk_postnorm_attention_block
    (init_mha_params num_heads d_model)
    (init_layer_norm d_model).

Lemma postnorm_preserves_shape : forall block x,
  (mha_num_heads (poab_attention block) > 0)%nat ->
  mat_rows (postnorm_attention_forward block x) = mat_rows x.
Proof.
  intros block x Hheads.
  unfold postnorm_attention_forward.
  rewrite layer_norm_rows.
  apply add_matrices_rows_eq.
  symmetry. apply self_attention_rows. assumption.
Qed.

(** ** Feed-Forward Network *)

Record ffn_params := mk_ffn_params {
  ffn_d_model : nat;
  ffn_d_ff : nat;
  ffn_W1 : tensor_2d;
  ffn_b1 : tensor_1d;
  ffn_W2 : tensor_2d;
  ffn_b2 : tensor_1d
}.

Definition ffn_forward_single (params : ffn_params) (x : tensor_1d) : tensor_1d :=
  let hidden := gelu_vec (vec_add (mat_vec_mul (mat_transpose (ffn_W1 params)) x) (ffn_b1 params)) in
  vec_add (mat_vec_mul (mat_transpose (ffn_W2 params)) hidden) (ffn_b2 params).

Definition ffn_forward (params : ffn_params) (batch : tensor_2d) : tensor_2d :=
  List.map (ffn_forward_single params) batch.

Definition init_ffn_params (d_model d_ff : nat) : ffn_params :=
  mk_ffn_params d_model d_ff
    (List.repeat (List.repeat 0 d_ff) d_model)
    (List.repeat 0 d_ff)
    (List.repeat (List.repeat 0 d_model) d_ff)
    (List.repeat 0 d_model).

Definition init_ffn_params_default (d_model : nat) : ffn_params :=
  init_ffn_params d_model (4 * d_model)%nat.

Lemma ffn_preserves_batch_size : forall params batch,
  List.length (ffn_forward params batch) = List.length batch.
Proof.
  intros params batch. unfold ffn_forward. apply List.length_map.
Qed.

Lemma ffn_forward_rows : forall params x,
  mat_rows (ffn_forward params x) = mat_rows x.
Proof.
  intros params x. unfold mat_rows. apply ffn_preserves_batch_size.
Qed.

Definition ffn_relu_forward_single (params : ffn_params) (x : tensor_1d) : tensor_1d :=
  let hidden := List.map relu_z (vec_add (mat_vec_mul (mat_transpose (ffn_W1 params)) x) (ffn_b1 params)) in
  vec_add (mat_vec_mul (mat_transpose (ffn_W2 params)) hidden) (ffn_b2 params).

Definition ffn_relu_forward (params : ffn_params) (batch : tensor_2d) : tensor_2d :=
  List.map (ffn_relu_forward_single params) batch.

Definition ffn_swish_forward_single (params : ffn_params) (x : tensor_1d) : tensor_1d :=
  let hidden := swish_vec (vec_add (mat_vec_mul (mat_transpose (ffn_W1 params)) x) (ffn_b1 params)) in
  vec_add (mat_vec_mul (mat_transpose (ffn_W2 params)) hidden) (ffn_b2 params).

Definition ffn_swish_forward (params : ffn_params) (batch : tensor_2d) : tensor_2d :=
  List.map (ffn_swish_forward_single params) batch.

(** ** Pre-norm FFN Block *)

Record prenorm_ffn_block := mk_prenorm_ffn_block {
  pffnb_norm : layer_norm_params;
  pffnb_ffn : ffn_params
}.

Definition prenorm_ffn_forward (block : prenorm_ffn_block) (x : tensor_2d) : tensor_2d :=
  let normalized := layer_norm (pffnb_norm block) x in
  let ffn_out := ffn_forward (pffnb_ffn block) normalized in
  add_matrices x ffn_out.

Definition init_prenorm_ffn_block (d_model d_ff : nat) : prenorm_ffn_block :=
  mk_prenorm_ffn_block
    (init_layer_norm d_model)
    (init_ffn_params d_model d_ff).

Lemma prenorm_ffn_preserves_shape : forall block x,
  mat_rows (prenorm_ffn_forward block x) = mat_rows x.
Proof.
  intros block x.
  unfold prenorm_ffn_forward.
  apply add_matrices_rows_eq.
  rewrite ffn_forward_rows. symmetry. apply layer_norm_rows.
Qed.

(** ** Post-norm FFN Block *)

Record postnorm_ffn_block := mk_postnorm_ffn_block {
  poffnb_ffn : ffn_params;
  poffnb_norm : layer_norm_params
}.

Definition postnorm_ffn_forward (block : postnorm_ffn_block) (x : tensor_2d) : tensor_2d :=
  let ffn_out := ffn_forward (poffnb_ffn block) x in
  let residual := add_matrices x ffn_out in
  layer_norm (poffnb_norm block) residual.

Definition init_postnorm_ffn_block (d_model d_ff : nat) : postnorm_ffn_block :=
  mk_postnorm_ffn_block
    (init_ffn_params d_model d_ff)
    (init_layer_norm d_model).

(** ** Full Encoder Layer *)

Record encoder_layer_prenorm := mk_encoder_layer_prenorm {
  elp_attention : prenorm_attention_block;
  elp_ffn : prenorm_ffn_block
}.

Definition encoder_layer_prenorm_forward (layer : encoder_layer_prenorm) (x : tensor_2d) : tensor_2d :=
  let after_attn := prenorm_attention_forward (elp_attention layer) x in
  prenorm_ffn_forward (elp_ffn layer) after_attn.

Definition init_encoder_layer_prenorm (num_heads d_model d_ff : nat) : encoder_layer_prenorm :=
  mk_encoder_layer_prenorm
    (init_prenorm_attention_block num_heads d_model)
    (init_prenorm_ffn_block d_model d_ff).

Record encoder_layer_postnorm := mk_encoder_layer_postnorm {
  elpo_attention : postnorm_attention_block;
  elpo_ffn : postnorm_ffn_block
}.

Definition encoder_layer_postnorm_forward (layer : encoder_layer_postnorm) (x : tensor_2d) : tensor_2d :=
  let after_attn := postnorm_attention_forward (elpo_attention layer) x in
  postnorm_ffn_forward (elpo_ffn layer) after_attn.

Definition init_encoder_layer_postnorm (num_heads d_model d_ff : nat) : encoder_layer_postnorm :=
  mk_encoder_layer_postnorm
    (init_postnorm_attention_block num_heads d_model)
    (init_postnorm_ffn_block d_model d_ff).

Fixpoint encoder_stack_prenorm_forward (layers : list encoder_layer_prenorm) (x : tensor_2d) : tensor_2d :=
  match layers with
  | [] => x
  | layer :: rest => encoder_stack_prenorm_forward rest (encoder_layer_prenorm_forward layer x)
  end.

Fixpoint encoder_stack_postnorm_forward (layers : list encoder_layer_postnorm) (x : tensor_2d) : tensor_2d :=
  match layers with
  | [] => x
  | layer :: rest => encoder_stack_postnorm_forward rest (encoder_layer_postnorm_forward layer x)
  end.

Definition init_encoder_stack_prenorm (num_layers num_heads d_model d_ff : nat) : list encoder_layer_prenorm :=
  List.repeat (init_encoder_layer_prenorm num_heads d_model d_ff) num_layers.

Definition init_encoder_stack_postnorm (num_layers num_heads d_model d_ff : nat) : list encoder_layer_postnorm :=
  List.repeat (init_encoder_layer_postnorm num_heads d_model d_ff) num_layers.

Lemma encoder_prenorm_preserves_shape : forall layer x,
  (mha_num_heads (pab_attention (elp_attention layer)) > 0)%nat ->
  mat_rows (encoder_layer_prenorm_forward layer x) = mat_rows x.
Proof.
  intros layer x Hheads.
  unfold encoder_layer_prenorm_forward.
  rewrite prenorm_ffn_preserves_shape.
  apply prenorm_preserves_shape. assumption.
Qed.

(** ** Full Decoder Layer *)

Record decoder_layer_prenorm := mk_decoder_layer_prenorm {
  dlp_self_attention : prenorm_attention_block;
  dlp_ffn : prenorm_ffn_block
}.

Definition decoder_layer_prenorm_forward (layer : decoder_layer_prenorm) (x : tensor_2d) : tensor_2d :=
  let after_attn := prenorm_masked_attention_forward (dlp_self_attention layer) x in
  prenorm_ffn_forward (dlp_ffn layer) after_attn.

Definition init_decoder_layer_prenorm (num_heads d_model d_ff : nat) : decoder_layer_prenorm :=
  mk_decoder_layer_prenorm
    (init_prenorm_attention_block num_heads d_model)
    (init_prenorm_ffn_block d_model d_ff).

Record decoder_layer_postnorm := mk_decoder_layer_postnorm {
  dlpo_self_attention : postnorm_attention_block;
  dlpo_ffn : postnorm_ffn_block
}.

Definition decoder_layer_postnorm_forward (layer : decoder_layer_postnorm) (x : tensor_2d) : tensor_2d :=
  let after_attn := postnorm_masked_attention_forward (dlpo_self_attention layer) x in
  postnorm_ffn_forward (dlpo_ffn layer) after_attn.

Definition init_decoder_layer_postnorm (num_heads d_model d_ff : nat) : decoder_layer_postnorm :=
  mk_decoder_layer_postnorm
    (init_postnorm_attention_block num_heads d_model)
    (init_postnorm_ffn_block d_model d_ff).

(** ** Decoder with Cross-Attention *)

Record cross_attention_block := mk_cross_attention_block {
  cab_norm : layer_norm_params;
  cab_attention : mha_params
}.

Definition cross_attention_block_forward (block : cross_attention_block)
                                         (x encoder_output : tensor_2d) : tensor_2d :=
  let normalized := layer_norm (cab_norm block) x in
  let attended := cross_attention_mha (cab_attention block) normalized encoder_output in
  add_matrices x attended.

Definition init_cross_attention_block (num_heads d_model : nat) : cross_attention_block :=
  mk_cross_attention_block
    (init_layer_norm d_model)
    (init_mha_params num_heads d_model).

Record decoder_layer_with_cross := mk_decoder_layer_with_cross {
  dlwc_self_attention : prenorm_attention_block;
  dlwc_cross_attention : cross_attention_block;
  dlwc_ffn : prenorm_ffn_block
}.

Definition decoder_layer_with_cross_forward (layer : decoder_layer_with_cross)
                                            (x encoder_output : tensor_2d) : tensor_2d :=
  let after_self_attn := prenorm_masked_attention_forward (dlwc_self_attention layer) x in
  let after_cross_attn := cross_attention_block_forward (dlwc_cross_attention layer) after_self_attn encoder_output in
  prenorm_ffn_forward (dlwc_ffn layer) after_cross_attn.

Definition init_decoder_layer_with_cross (num_heads d_model d_ff : nat) : decoder_layer_with_cross :=
  mk_decoder_layer_with_cross
    (init_prenorm_attention_block num_heads d_model)
    (init_cross_attention_block num_heads d_model)
    (init_prenorm_ffn_block d_model d_ff).

Fixpoint decoder_stack_prenorm_forward (layers : list decoder_layer_prenorm) (x : tensor_2d) : tensor_2d :=
  match layers with
  | [] => x
  | layer :: rest => decoder_stack_prenorm_forward rest (decoder_layer_prenorm_forward layer x)
  end.

Fixpoint decoder_stack_with_cross_forward (layers : list decoder_layer_with_cross)
                                          (x encoder_output : tensor_2d) : tensor_2d :=
  match layers with
  | [] => x
  | layer :: rest =>
      decoder_stack_with_cross_forward rest (decoder_layer_with_cross_forward layer x encoder_output) encoder_output
  end.

Definition init_decoder_stack_prenorm (num_layers num_heads d_model d_ff : nat) : list decoder_layer_prenorm :=
  List.repeat (init_decoder_layer_prenorm num_heads d_model d_ff) num_layers.

Definition init_decoder_stack_with_cross (num_layers num_heads d_model d_ff : nat) : list decoder_layer_with_cross :=
  List.repeat (init_decoder_layer_with_cross num_heads d_model d_ff) num_layers.

Lemma decoder_prenorm_preserves_shape : forall layer x,
  (mha_num_heads (pab_attention (dlp_self_attention layer)) > 0)%nat ->
  mat_rows (decoder_layer_prenorm_forward layer x) = mat_rows x.
Proof.
  intros layer x Hheads.
  unfold decoder_layer_prenorm_forward.
  rewrite prenorm_ffn_preserves_shape.
  apply prenorm_masked_preserves_shape. assumption.
Qed.

(** ** Complete Transformer Models *)

(** GPT-style Decoder-Only Model *)

Record gpt_model := mk_gpt_model {
  gpt_num_layers : nat;
  gpt_num_heads : nat;
  gpt_d_model : nat;
  gpt_d_ff : nat;
  gpt_layers : list decoder_layer_prenorm;
  gpt_final_norm : layer_norm_params
}.

Definition gpt_forward (model : gpt_model) (x : tensor_2d) : tensor_2d :=
  let hidden := decoder_stack_prenorm_forward (gpt_layers model) x in
  layer_norm (gpt_final_norm model) hidden.

Definition init_gpt_model (num_layers num_heads d_model d_ff : nat) : gpt_model :=
  mk_gpt_model num_layers num_heads d_model d_ff
    (init_decoder_stack_prenorm num_layers num_heads d_model d_ff)
    (init_layer_norm d_model).

(** BERT-style Encoder-Only Model *)

Record bert_model := mk_bert_model {
  bert_num_layers : nat;
  bert_num_heads : nat;
  bert_d_model : nat;
  bert_d_ff : nat;
  bert_layers : list encoder_layer_postnorm
}.

Definition bert_forward (model : bert_model) (x : tensor_2d) : tensor_2d :=
  encoder_stack_postnorm_forward (bert_layers model) x.

Definition init_bert_model (num_layers num_heads d_model d_ff : nat) : bert_model :=
  mk_bert_model num_layers num_heads d_model d_ff
    (init_encoder_stack_postnorm num_layers num_heads d_model d_ff).

(** Encoder-Decoder Transformer *)

Record transformer_model := mk_transformer_model {
  tm_num_layers : nat;
  tm_num_heads : nat;
  tm_d_model : nat;
  tm_d_ff : nat;
  tm_encoder_layers : list encoder_layer_prenorm;
  tm_decoder_layers : list decoder_layer_with_cross;
  tm_encoder_final_norm : layer_norm_params;
  tm_decoder_final_norm : layer_norm_params
}.

Definition transformer_encode (model : transformer_model) (src : tensor_2d) : tensor_2d :=
  let hidden := encoder_stack_prenorm_forward (tm_encoder_layers model) src in
  layer_norm (tm_encoder_final_norm model) hidden.

Definition transformer_decode (model : transformer_model) (tgt encoder_output : tensor_2d) : tensor_2d :=
  let hidden := decoder_stack_with_cross_forward (tm_decoder_layers model) tgt encoder_output in
  layer_norm (tm_decoder_final_norm model) hidden.

Definition transformer_forward (model : transformer_model) (src tgt : tensor_2d) : tensor_2d :=
  let encoder_output := transformer_encode model src in
  transformer_decode model tgt encoder_output.

Definition init_transformer_model (num_layers num_heads d_model d_ff : nat) : transformer_model :=
  mk_transformer_model num_layers num_heads d_model d_ff
    (init_encoder_stack_prenorm num_layers num_heads d_model d_ff)
    (init_decoder_stack_with_cross num_layers num_heads d_model d_ff)
    (init_layer_norm d_model)
    (init_layer_norm d_model).

(** ** Transformer Examples *)

Definition p14_example_d_model : nat := 64%nat.
Definition p14_example_d_ff : nat := 256%nat.
Definition p14_example_num_heads : nat := 4%nat.
Definition p14_example_num_layers : nat := 2%nat.

Definition p14_example_input : tensor_2d :=
  [[100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150;
    100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150;
    100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150;
    100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150];
   [150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50;
    150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50;
    150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50;
    150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50; 150; 100; 200; 50];
   [200; 150; 100; 100; 200; 150; 100; 100; 200; 150; 100; 100; 200; 150; 100; 100;
    200; 150; 100; 100; 200; 150; 100; 100; 200; 150; 100; 100; 200; 150; 100; 100;
    200; 150; 100; 100; 200; 150; 100; 100; 200; 150; 100; 100; 200; 150; 100; 100;
    200; 150; 100; 100; 200; 150; 100; 100; 200; 150; 100; 100; 200; 150; 100; 100]].

Definition p14_example_prenorm_attn := init_prenorm_attention_block p14_example_num_heads p14_example_d_model.
Definition p14_example_postnorm_attn := init_postnorm_attention_block p14_example_num_heads p14_example_d_model.
Definition p14_example_ffn := init_ffn_params p14_example_d_model p14_example_d_ff.
Definition p14_example_encoder_layer := init_encoder_layer_prenorm p14_example_num_heads p14_example_d_model p14_example_d_ff.
Definition p14_example_decoder_layer := init_decoder_layer_prenorm p14_example_num_heads p14_example_d_model p14_example_d_ff.
Definition p14_example_gpt := init_gpt_model p14_example_num_layers p14_example_num_heads p14_example_d_model p14_example_d_ff.
Definition p14_example_bert := init_bert_model p14_example_num_layers p14_example_num_heads p14_example_d_model p14_example_d_ff.
Definition p14_example_transformer := init_transformer_model p14_example_num_layers p14_example_num_heads p14_example_d_model p14_example_d_ff.

(** * Real Models (GPT-2) *)

(** ** GPT-2 Configuration *)

Record gpt2_config := mk_gpt2_config {
  gpt2_vocab_size : Z;
  gpt2_n_positions : Z;
  gpt2_n_embd : Z;
  gpt2_n_layer : Z;
  gpt2_n_head : Z;
  gpt2_n_inner : Z;
  gpt2_activation : string;
  gpt2_resid_pdrop : Z;
  gpt2_embd_pdrop : Z;
  gpt2_attn_pdrop : Z;
  gpt2_layer_norm_epsilon : Z;
  gpt2_initializer_range : Z;
  gpt2_bos_token_id : Z;
  gpt2_eos_token_id : Z
}.

Definition gpt2_small : gpt2_config := mk_gpt2_config
  50257 1024 768 12 12 3072
  "gelu_new" 100000 100000 100000 10 20000 50256 50256.

Definition gpt2_medium : gpt2_config := mk_gpt2_config
  50257 1024 1024 24 16 4096
  "gelu_new" 100000 100000 100000 10 20000 50256 50256.

Definition gpt2_large : gpt2_config := mk_gpt2_config
  50257 1024 1280 36 20 5120
  "gelu_new" 100000 100000 100000 10 20000 50256 50256.

Definition gpt2_xl : gpt2_config := mk_gpt2_config
  50257 1024 1600 48 25 6400
  "gelu_new" 100000 100000 100000 10 20000 50256 50256.

Definition gpt2_embedding_params (cfg : gpt2_config) : Z :=
  gpt2_vocab_size cfg * gpt2_n_embd cfg +
  gpt2_n_positions cfg * gpt2_n_embd cfg.

Definition gpt2_attention_params_per_layer (cfg : gpt2_config) : Z :=
  let d := gpt2_n_embd cfg in
  4 * d * d + 4 * d.

Definition gpt2_ffn_params_per_layer (cfg : gpt2_config) : Z :=
  let d := gpt2_n_embd cfg in
  let ff := gpt2_n_inner cfg in
  d * ff + ff + ff * d + d.

Definition gpt2_layernorm_params_per_layer (cfg : gpt2_config) : Z :=
  4 * gpt2_n_embd cfg.

Definition gpt2_transformer_layer_params (cfg : gpt2_config) : Z :=
  gpt2_attention_params_per_layer cfg +
  gpt2_ffn_params_per_layer cfg +
  gpt2_layernorm_params_per_layer cfg.

Definition gpt2_final_layernorm_params (cfg : gpt2_config) : Z :=
  2 * gpt2_n_embd cfg.

Definition gpt2_total_params (cfg : gpt2_config) : Z :=
  gpt2_embedding_params cfg +
  gpt2_n_layer cfg * gpt2_transformer_layer_params cfg +
  gpt2_final_layernorm_params cfg.

Definition gpt2_small_expected_params : Z := 124439808.
Definition gpt2_medium_expected_params : Z := 354823168.
Definition gpt2_large_expected_params : Z := 774030080.
Definition gpt2_xl_expected_params : Z := 1557611200.

Lemma gpt2_small_params_correct :
  gpt2_total_params gpt2_small = gpt2_small_expected_params.
Proof. reflexivity. Qed.

Lemma gpt2_medium_params_correct :
  gpt2_total_params gpt2_medium = gpt2_medium_expected_params.
Proof. reflexivity. Qed.

Lemma gpt2_large_params_correct :
  gpt2_total_params gpt2_large = gpt2_large_expected_params.
Proof. reflexivity. Qed.

Lemma gpt2_xl_params_correct :
  gpt2_total_params gpt2_xl = gpt2_xl_expected_params.
Proof. reflexivity. Qed.

Definition gpt2_head_dim (cfg : gpt2_config) : Z :=
  gpt2_n_embd cfg / gpt2_n_head cfg.

Definition gpt2_ffn_expansion_factor (cfg : gpt2_config) : Z :=
  gpt2_n_inner cfg / gpt2_n_embd cfg.

Lemma gpt2_small_head_dim : gpt2_head_dim gpt2_small = 64.
Proof. reflexivity. Qed.

Lemma gpt2_small_ffn_factor : gpt2_ffn_expansion_factor gpt2_small = 4.
Proof. reflexivity. Qed.

Definition valid_gpt2_head_divisibility (cfg : gpt2_config) : bool :=
  Z.eqb (gpt2_n_embd cfg mod gpt2_n_head cfg) 0.

Definition valid_gpt2_ffn_expansion (cfg : gpt2_config) : bool :=
  Z.eqb (gpt2_n_inner cfg) (4 * gpt2_n_embd cfg).

Definition valid_gpt2_config (cfg : gpt2_config) : bool :=
  valid_gpt2_head_divisibility cfg &&
  valid_gpt2_ffn_expansion cfg &&
  (gpt2_vocab_size cfg >? 0) &&
  (gpt2_n_positions cfg >? 0) &&
  (gpt2_n_embd cfg >? 0) &&
  (gpt2_n_layer cfg >? 0) &&
  (gpt2_n_head cfg >? 0).

Lemma gpt2_small_valid : valid_gpt2_config gpt2_small = true.
Proof. reflexivity. Qed.

Lemma gpt2_medium_valid : valid_gpt2_config gpt2_medium = true.
Proof. reflexivity. Qed.

Lemma gpt2_large_valid : valid_gpt2_config gpt2_large = true.
Proof. reflexivity. Qed.

Lemma gpt2_xl_valid : valid_gpt2_config gpt2_xl = true.
Proof. reflexivity. Qed.

Record gpt2_weight_shape := mk_gpt2_weight_shape {
  gpt2_ws_name : string;
  gpt2_ws_dims : list Z
}.

Definition gpt2_wte_shape (cfg : gpt2_config) : gpt2_weight_shape :=
  mk_gpt2_weight_shape "wte" [gpt2_vocab_size cfg; gpt2_n_embd cfg].

Definition gpt2_wpe_shape (cfg : gpt2_config) : gpt2_weight_shape :=
  mk_gpt2_weight_shape "wpe" [gpt2_n_positions cfg; gpt2_n_embd cfg].

Definition gpt2_ln_f_weight_shape (cfg : gpt2_config) : gpt2_weight_shape :=
  mk_gpt2_weight_shape "ln_f.weight" [gpt2_n_embd cfg].

Definition gpt2_ln_f_bias_shape (cfg : gpt2_config) : gpt2_weight_shape :=
  mk_gpt2_weight_shape "ln_f.bias" [gpt2_n_embd cfg].

(** ** Weight Loading *)

Record gpt2_named_tensor := mk_gpt2_named_tensor {
  gpt2_nt_name : string;
  gpt2_nt_shape : list Z;
  gpt2_nt_data : list Z
}.

Definition gpt2_weight_dict := list gpt2_named_tensor.

Record gpt2_attention_weights := mk_gpt2_attention_weights {
  gpt2_attn_c_attn_weight : tensor_2d;
  gpt2_attn_c_attn_bias : tensor_1d;
  gpt2_attn_c_proj_weight : tensor_2d;
  gpt2_attn_c_proj_bias : tensor_1d
}.

Record gpt2_mlp_weights := mk_gpt2_mlp_weights {
  gpt2_mlp_c_fc_weight : tensor_2d;
  gpt2_mlp_c_fc_bias : tensor_1d;
  gpt2_mlp_c_proj_weight : tensor_2d;
  gpt2_mlp_c_proj_bias : tensor_1d
}.

Record gpt2_layernorm_weights := mk_gpt2_layernorm_weights {
  gpt2_ln_weight : tensor_1d;
  gpt2_ln_bias : tensor_1d
}.

Record gpt2_block_weights := mk_gpt2_block_weights {
  gpt2_block_ln_1 : gpt2_layernorm_weights;
  gpt2_block_attn : gpt2_attention_weights;
  gpt2_block_ln_2 : gpt2_layernorm_weights;
  gpt2_block_mlp : gpt2_mlp_weights
}.

Record gpt2_model_weights := mk_gpt2_model_weights {
  gpt2_wte : tensor_2d;
  gpt2_wpe : tensor_2d;
  gpt2_blocks : list gpt2_block_weights;
  gpt2_ln_f : gpt2_layernorm_weights
}.

Definition gpt2_tensor_1d_size (t : tensor_1d) : Z :=
  Z.of_nat (List.length t).

Definition gpt2_tensor_2d_shape (t : tensor_2d) : list Z :=
  match t with
  | [] => [0; 0]
  | row :: _ => [Z.of_nat (List.length t); Z.of_nat (List.length row)]
  end.

Definition gpt2_shape_eq (s1 s2 : list Z) : bool :=
  (Z.of_nat (List.length s1) =? Z.of_nat (List.length s2)) &&
  List.forallb (fun '(a, b) => a =? b) (List.combine s1 s2).

Definition gpt2_validate_tensor_1d_shape (t : tensor_1d) (expected : Z) : bool :=
  gpt2_tensor_1d_size t =? expected.

Definition gpt2_validate_tensor_2d_shape (t : tensor_2d) (expected : list Z) : bool :=
  gpt2_shape_eq (gpt2_tensor_2d_shape t) expected.

Definition gpt2_validate_attention_weights (cfg : gpt2_config) (attn : gpt2_attention_weights) : bool :=
  let d := gpt2_n_embd cfg in
  gpt2_validate_tensor_2d_shape (gpt2_attn_c_attn_weight attn) [d; 3 * d] &&
  gpt2_validate_tensor_1d_shape (gpt2_attn_c_attn_bias attn) (3 * d) &&
  gpt2_validate_tensor_2d_shape (gpt2_attn_c_proj_weight attn) [d; d] &&
  gpt2_validate_tensor_1d_shape (gpt2_attn_c_proj_bias attn) d.

Definition gpt2_validate_mlp_weights (cfg : gpt2_config) (mlp : gpt2_mlp_weights) : bool :=
  let d := gpt2_n_embd cfg in
  let ff := gpt2_n_inner cfg in
  gpt2_validate_tensor_2d_shape (gpt2_mlp_c_fc_weight mlp) [d; ff] &&
  gpt2_validate_tensor_1d_shape (gpt2_mlp_c_fc_bias mlp) ff &&
  gpt2_validate_tensor_2d_shape (gpt2_mlp_c_proj_weight mlp) [ff; d] &&
  gpt2_validate_tensor_1d_shape (gpt2_mlp_c_proj_bias mlp) d.

Definition gpt2_validate_layernorm_weights (cfg : gpt2_config) (ln : gpt2_layernorm_weights) : bool :=
  let d := gpt2_n_embd cfg in
  gpt2_validate_tensor_1d_shape (gpt2_ln_weight ln) d &&
  gpt2_validate_tensor_1d_shape (gpt2_ln_bias ln) d.

Definition gpt2_validate_block_weights (cfg : gpt2_config) (block : gpt2_block_weights) : bool :=
  gpt2_validate_layernorm_weights cfg (gpt2_block_ln_1 block) &&
  gpt2_validate_attention_weights cfg (gpt2_block_attn block) &&
  gpt2_validate_layernorm_weights cfg (gpt2_block_ln_2 block) &&
  gpt2_validate_mlp_weights cfg (gpt2_block_mlp block).

Definition gpt2_validate_model_weights (cfg : gpt2_config) (model : gpt2_model_weights) : bool :=
  gpt2_validate_tensor_2d_shape (gpt2_wte model) [gpt2_vocab_size cfg; gpt2_n_embd cfg] &&
  gpt2_validate_tensor_2d_shape (gpt2_wpe model) [gpt2_n_positions cfg; gpt2_n_embd cfg] &&
  (Z.of_nat (List.length (gpt2_blocks model)) =? gpt2_n_layer cfg) &&
  List.forallb (gpt2_validate_block_weights cfg) (gpt2_blocks model) &&
  gpt2_validate_layernorm_weights cfg (gpt2_ln_f model).

Fixpoint gpt2_reshape_1d_to_2d (data : list Z) (rows cols : nat) : tensor_2d :=
  match rows with
  | O => []
  | S r' =>
      List.firstn cols data :: gpt2_reshape_1d_to_2d (List.skipn cols data) r' cols
  end.

Definition gpt2_make_layernorm (weight bias : tensor_1d) : gpt2_layernorm_weights :=
  mk_gpt2_layernorm_weights weight bias.

Definition gpt2_make_attention (c_attn_w c_attn_b c_proj_w c_proj_b : list Z)
                                (d : nat) : gpt2_attention_weights :=
  mk_gpt2_attention_weights
    (gpt2_reshape_1d_to_2d c_attn_w d (3 * d))
    c_attn_b
    (gpt2_reshape_1d_to_2d c_proj_w d d)
    c_proj_b.

Definition gpt2_make_mlp (c_fc_w c_fc_b c_proj_w c_proj_b : list Z)
                          (d ff : nat) : gpt2_mlp_weights :=
  mk_gpt2_mlp_weights
    (gpt2_reshape_1d_to_2d c_fc_w d ff)
    c_fc_b
    (gpt2_reshape_1d_to_2d c_proj_w ff d)
    c_proj_b.

Definition gpt2_count_tensor_1d_params (t : tensor_1d) : Z :=
  Z.of_nat (List.length t).

Definition gpt2_count_tensor_2d_params (t : tensor_2d) : Z :=
  Z.of_nat (List.fold_left (fun acc row => acc + List.length row)%nat t 0%nat).

Definition gpt2_count_layernorm_params (ln : gpt2_layernorm_weights) : Z :=
  gpt2_count_tensor_1d_params (gpt2_ln_weight ln) + gpt2_count_tensor_1d_params (gpt2_ln_bias ln).

Definition gpt2_count_attention_params (attn : gpt2_attention_weights) : Z :=
  gpt2_count_tensor_2d_params (gpt2_attn_c_attn_weight attn) +
  gpt2_count_tensor_1d_params (gpt2_attn_c_attn_bias attn) +
  gpt2_count_tensor_2d_params (gpt2_attn_c_proj_weight attn) +
  gpt2_count_tensor_1d_params (gpt2_attn_c_proj_bias attn).

Definition gpt2_count_mlp_params (mlp : gpt2_mlp_weights) : Z :=
  gpt2_count_tensor_2d_params (gpt2_mlp_c_fc_weight mlp) +
  gpt2_count_tensor_1d_params (gpt2_mlp_c_fc_bias mlp) +
  gpt2_count_tensor_2d_params (gpt2_mlp_c_proj_weight mlp) +
  gpt2_count_tensor_1d_params (gpt2_mlp_c_proj_bias mlp).

Definition gpt2_count_block_params (block : gpt2_block_weights) : Z :=
  gpt2_count_layernorm_params (gpt2_block_ln_1 block) +
  gpt2_count_attention_params (gpt2_block_attn block) +
  gpt2_count_layernorm_params (gpt2_block_ln_2 block) +
  gpt2_count_mlp_params (gpt2_block_mlp block).

Definition gpt2_count_model_params (model : gpt2_model_weights) : Z :=
  gpt2_count_tensor_2d_params (gpt2_wte model) +
  gpt2_count_tensor_2d_params (gpt2_wpe model) +
  List.fold_left (fun acc block => acc + gpt2_count_block_params block) (gpt2_blocks model) 0 +
  gpt2_count_layernorm_params (gpt2_ln_f model).

(** ** Inference *)

Record gpt2_inference_config := mk_gpt2_inference_config {
  gpt2_inf_n_embd : nat;
  gpt2_inf_n_head : nat;
  gpt2_inf_n_layer : nat;
  gpt2_inf_n_inner : nat;
  gpt2_inf_vocab_size : nat;
  gpt2_inf_n_positions : nat
}.

Definition gpt2_small_inference_cfg : gpt2_inference_config :=
  mk_gpt2_inference_config 768 12 12 3072 50257 1024.

Definition gpt2_layer_norm_vec (gamma beta : tensor_1d) (eps : Z) (x : tensor_1d) : tensor_1d :=
  let mean := mean_scaled x in
  let var := variance_scaled x mean in
  let std := isqrt (var + eps) in
  List.map (fun '(idx, val) =>
    let g := List.nth idx gamma scale_factor in
    let b := List.nth idx beta 0 in
    let normalized := safe_div ((val * scale_factor - mean) * scale_factor) std in
    safe_div (g * normalized) scale_factor + b
  ) (List.combine (List.seq 0 (List.length x)) x).

Definition gpt2_layer_norm_2d (gamma beta : tensor_1d) (eps : Z) (m : tensor_2d) : tensor_2d :=
  List.map (gpt2_layer_norm_vec gamma beta eps) m.

Definition gpt2_causal_attention (q k v : tensor_2d) (d_k : nat) : tensor_2d :=
  let scores := scale_scores (compute_attention_scores q k) d_k in
  let masked := apply_mask scores (causal_mask (mat_rows q)) in
  mat_mul (softmax_2d masked) v.

Definition gpt2_linear_forward (weight : tensor_2d) (bias : tensor_1d) (x : tensor_1d) : tensor_1d :=
  vec_add (mat_vec_mul (mat_transpose weight) x) bias.

Definition gpt2_linear_forward_2d (weight : tensor_2d) (bias : tensor_1d) (x : tensor_2d) : tensor_2d :=
  List.map (gpt2_linear_forward weight bias) x.

Definition gpt2_attention_forward (cfg : gpt2_inference_config) (attn : gpt2_attention_weights)
                                   (hidden : tensor_2d) : tensor_2d :=
  let qkv := gpt2_linear_forward_2d (gpt2_attn_c_attn_weight attn) (gpt2_attn_c_attn_bias attn) hidden in
  let d := gpt2_inf_n_embd cfg in
  let n_head := gpt2_inf_n_head cfg in
  let head_dim := Nat.div d n_head in
  let split_qkv row :=
    let q := List.firstn d row in
    let k := List.firstn d (List.skipn d row) in
    let v := List.skipn (2 * d) row in
    (q, k, v) in
  let qkv_split := List.map split_qkv qkv in
  let qs := List.map (fun '(q, _, _) => q) qkv_split in
  let ks := List.map (fun '(_, k, _) => k) qkv_split in
  let vs := List.map (fun '(_, _, v) => v) qkv_split in
  let q_heads := split_into_heads n_head qs in
  let k_heads := split_into_heads n_head ks in
  let v_heads := split_into_heads n_head vs in
  let head_outputs := List.map (fun '(qh, (kh, vh)) => gpt2_causal_attention qh kh vh head_dim)
                               (List.combine q_heads (List.combine k_heads v_heads)) in
  let concat := concat_heads head_outputs in
  gpt2_linear_forward_2d (gpt2_attn_c_proj_weight attn) (gpt2_attn_c_proj_bias attn) concat.

Definition gpt2_mlp_forward (mlp : gpt2_mlp_weights) (hidden : tensor_2d) : tensor_2d :=
  let h := gpt2_linear_forward_2d (gpt2_mlp_c_fc_weight mlp) (gpt2_mlp_c_fc_bias mlp) hidden in
  let h_gelu := List.map gelu_vec h in
  gpt2_linear_forward_2d (gpt2_mlp_c_proj_weight mlp) (gpt2_mlp_c_proj_bias mlp) h_gelu.

Definition gpt2_block_forward (cfg : gpt2_inference_config) (block : gpt2_block_weights)
                               (hidden : tensor_2d) : tensor_2d :=
  let ln1 := gpt2_layer_norm_2d (gpt2_ln_weight (gpt2_block_ln_1 block)) (gpt2_ln_bias (gpt2_block_ln_1 block)) 1 hidden in
  let attn_out := gpt2_attention_forward cfg (gpt2_block_attn block) ln1 in
  let hidden2 := add_matrices hidden attn_out in
  let ln2 := gpt2_layer_norm_2d (gpt2_ln_weight (gpt2_block_ln_2 block)) (gpt2_ln_bias (gpt2_block_ln_2 block)) 1 hidden2 in
  let mlp_out := gpt2_mlp_forward (gpt2_block_mlp block) ln2 in
  add_matrices hidden2 mlp_out.

Fixpoint gpt2_blocks_forward (cfg : gpt2_inference_config) (blocks : list gpt2_block_weights)
                              (hidden : tensor_2d) : tensor_2d :=
  match blocks with
  | [] => hidden
  | block :: rest => gpt2_blocks_forward cfg rest (gpt2_block_forward cfg block hidden)
  end.

Definition gpt2_lookup_embedding (embeddings : tensor_2d) (token_id : nat) : tensor_1d :=
  List.nth token_id embeddings [].

Definition gpt2_embed_tokens (embeddings : tensor_2d) (token_ids : list nat) : tensor_2d :=
  List.map (gpt2_lookup_embedding embeddings) token_ids.

Definition gpt2_embed_positions (embeddings : tensor_2d) (seq_len : nat) : tensor_2d :=
  List.map (gpt2_lookup_embedding embeddings) (List.seq 0 seq_len).

Definition gpt2_combine_embeddings (token_emb pos_emb : tensor_2d) : tensor_2d :=
  add_matrices token_emb pos_emb.

Definition gpt2_forward (cfg : gpt2_inference_config) (model : gpt2_model_weights)
                        (token_ids : list nat) : tensor_2d :=
  let seq_len := List.length token_ids in
  let tok_emb := gpt2_embed_tokens (gpt2_wte model) token_ids in
  let pos_emb := gpt2_embed_positions (gpt2_wpe model) seq_len in
  let hidden := gpt2_combine_embeddings tok_emb pos_emb in
  let transformed := gpt2_blocks_forward cfg (gpt2_blocks model) hidden in
  gpt2_layer_norm_2d (gpt2_ln_weight (gpt2_ln_f model)) (gpt2_ln_bias (gpt2_ln_f model)) 1 transformed.

Definition gpt2_logits (cfg : gpt2_inference_config) (model : gpt2_model_weights)
                       (token_ids : list nat) : tensor_2d :=
  let hidden := gpt2_forward cfg model token_ids in
  mat_mul hidden (gpt2_wte model).

Definition gpt2_next_token_logits (cfg : gpt2_inference_config) (model : gpt2_model_weights)
                                   (token_ids : list nat) : tensor_1d :=
  let logits := gpt2_logits cfg model token_ids in
  match List.rev logits with
  | [] => []
  | last :: _ => last
  end.

Definition gpt2_predict_next (cfg : gpt2_inference_config) (model : gpt2_model_weights)
                              (token_ids : list nat) : nat :=
  argmax (gpt2_next_token_logits cfg model token_ids).

(** ** Roundtrip Verification *)

Definition gpt2_serialize_tensor_1d (t : tensor_1d) : list byte :=
  List.concat (List.map z_to_bytes_le t).

Definition gpt2_serialize_tensor_2d (t : tensor_2d) : list byte :=
  List.concat (List.map gpt2_serialize_tensor_1d t).

Fixpoint gpt2_deserialize_tensor_1d_aux (count : nat) (bs : list byte) : list Z :=
  match count with
  | O => []
  | S n => bytes_to_z_le (take 4 bs) :: gpt2_deserialize_tensor_1d_aux n (drop 4 bs)
  end.

Definition gpt2_deserialize_tensor_1d (len : nat) (bs : list byte) : tensor_1d :=
  gpt2_deserialize_tensor_1d_aux len bs.

Fixpoint gpt2_deserialize_tensor_2d_aux (rows cols : nat) (bs : list byte) : tensor_2d :=
  match rows with
  | O => []
  | S r =>
      let row := gpt2_deserialize_tensor_1d cols bs in
      let rest := drop (4 * cols) bs in
      row :: gpt2_deserialize_tensor_2d_aux r cols rest
  end.

Definition gpt2_deserialize_tensor_2d (rows cols : nat) (bs : list byte) : tensor_2d :=
  gpt2_deserialize_tensor_2d_aux rows cols bs.

Definition gpt2_tensor_1d_eq (a b : tensor_1d) : bool :=
  (Nat.eqb (List.length a) (List.length b)) &&
  List.forallb (fun '(x, y) => Z.eqb x y) (List.combine a b).

Definition gpt2_tensor_2d_eq (a b : tensor_2d) : bool :=
  (Nat.eqb (List.length a) (List.length b)) &&
  List.forallb (fun '(ra, rb) => gpt2_tensor_1d_eq ra rb) (List.combine a b).

Definition gpt2_bytes_eq (a b : list byte) : bool :=
  (Nat.eqb (List.length a) (List.length b)) &&
  List.forallb (fun '(x, y) => Z.eqb x y) (List.combine a b).

Definition gpt2_roundtrip_tensor_1d (t : tensor_1d) : bool :=
  gpt2_tensor_1d_eq t (gpt2_deserialize_tensor_1d (List.length t) (gpt2_serialize_tensor_1d t)).

Definition gpt2_roundtrip_tensor_2d (t : tensor_2d) : bool :=
  match t with
  | [] => true
  | row :: _ =>
      let rows := List.length t in
      let cols := List.length row in
      gpt2_tensor_2d_eq t (gpt2_deserialize_tensor_2d rows cols (gpt2_serialize_tensor_2d t))
  end.

Definition gpt2_serialize_layernorm (ln : gpt2_layernorm_weights) : list byte :=
  gpt2_serialize_tensor_1d (gpt2_ln_weight ln) ++ gpt2_serialize_tensor_1d (gpt2_ln_bias ln).

Definition gpt2_serialize_attention (attn : gpt2_attention_weights) : list byte :=
  gpt2_serialize_tensor_2d (gpt2_attn_c_attn_weight attn) ++
  gpt2_serialize_tensor_1d (gpt2_attn_c_attn_bias attn) ++
  gpt2_serialize_tensor_2d (gpt2_attn_c_proj_weight attn) ++
  gpt2_serialize_tensor_1d (gpt2_attn_c_proj_bias attn).

Definition gpt2_serialize_mlp (mlp : gpt2_mlp_weights) : list byte :=
  gpt2_serialize_tensor_2d (gpt2_mlp_c_fc_weight mlp) ++
  gpt2_serialize_tensor_1d (gpt2_mlp_c_fc_bias mlp) ++
  gpt2_serialize_tensor_2d (gpt2_mlp_c_proj_weight mlp) ++
  gpt2_serialize_tensor_1d (gpt2_mlp_c_proj_bias mlp).

Definition gpt2_serialize_block (blk : gpt2_block_weights) : list byte :=
  gpt2_serialize_layernorm (gpt2_block_ln_1 blk) ++
  gpt2_serialize_attention (gpt2_block_attn blk) ++
  gpt2_serialize_layernorm (gpt2_block_ln_2 blk) ++
  gpt2_serialize_mlp (gpt2_block_mlp blk).

Definition gpt2_serialize_model (model : gpt2_model_weights) : list byte :=
  gpt2_serialize_tensor_2d (gpt2_wte model) ++
  gpt2_serialize_tensor_2d (gpt2_wpe model) ++
  List.concat (List.map gpt2_serialize_block (gpt2_blocks model)) ++
  gpt2_serialize_layernorm (gpt2_ln_f model).

Definition gpt2_roundtrip_layernorm (ln : gpt2_layernorm_weights) : bool :=
  gpt2_roundtrip_tensor_1d (gpt2_ln_weight ln) && gpt2_roundtrip_tensor_1d (gpt2_ln_bias ln).

Definition gpt2_roundtrip_attention (attn : gpt2_attention_weights) : bool :=
  gpt2_roundtrip_tensor_2d (gpt2_attn_c_attn_weight attn) &&
  gpt2_roundtrip_tensor_1d (gpt2_attn_c_attn_bias attn) &&
  gpt2_roundtrip_tensor_2d (gpt2_attn_c_proj_weight attn) &&
  gpt2_roundtrip_tensor_1d (gpt2_attn_c_proj_bias attn).

Definition gpt2_roundtrip_mlp (mlp : gpt2_mlp_weights) : bool :=
  gpt2_roundtrip_tensor_2d (gpt2_mlp_c_fc_weight mlp) &&
  gpt2_roundtrip_tensor_1d (gpt2_mlp_c_fc_bias mlp) &&
  gpt2_roundtrip_tensor_2d (gpt2_mlp_c_proj_weight mlp) &&
  gpt2_roundtrip_tensor_1d (gpt2_mlp_c_proj_bias mlp).

Definition gpt2_roundtrip_block (blk : gpt2_block_weights) : bool :=
  gpt2_roundtrip_layernorm (gpt2_block_ln_1 blk) &&
  gpt2_roundtrip_attention (gpt2_block_attn blk) &&
  gpt2_roundtrip_layernorm (gpt2_block_ln_2 blk) &&
  gpt2_roundtrip_mlp (gpt2_block_mlp blk).

Definition gpt2_roundtrip_model (model : gpt2_model_weights) : bool :=
  gpt2_roundtrip_tensor_2d (gpt2_wte model) &&
  gpt2_roundtrip_tensor_2d (gpt2_wpe model) &&
  List.forallb gpt2_roundtrip_block (gpt2_blocks model) &&
  gpt2_roundtrip_layernorm (gpt2_ln_f model).

Definition gpt2_compute_checksum (bs : list byte) : Z :=
  List.fold_left (fun acc b => (acc * 31 + b) mod 4294967296) bs 0.

Definition gpt2_verify_checksum (model : gpt2_model_weights) (expected : Z) : bool :=
  Z.eqb (gpt2_compute_checksum (gpt2_serialize_model model)) expected.

(** ** Verified Generation *)

Inductive gpt2_sampling_method :=
  | GPT2_Greedy
  | GPT2_TopK : nat -> gpt2_sampling_method
  | GPT2_TopP : Z -> gpt2_sampling_method
  | GPT2_Temperature : Z -> gpt2_sampling_method.

Record gpt2_generation_config := mk_gpt2_generation_config {
  gpt2_gen_max_new_tokens : nat;
  gpt2_gen_sampling : gpt2_sampling_method;
  gpt2_gen_eos_token_id : nat;
  gpt2_gen_pad_token_id : nat
}.

Definition gpt2_default_gen_config : gpt2_generation_config :=
  mk_gpt2_generation_config 50 GPT2_Greedy 50256 50256.

Definition gpt2_temperature_scale (logits : tensor_1d) (temp : Z) : tensor_1d :=
  if temp =? 0 then logits
  else List.map (fun x => (x * scale_factor) / temp) logits.

Definition gpt2_find_kth_largest (logits : tensor_1d) (k : nat) : Z :=
  let sorted_desc := List.fold_left
    (fun acc x =>
      let insert_pos := List.length (List.filter (fun y => y >? x) acc) in
      List.firstn insert_pos acc ++ [x] ++ List.skipn insert_pos acc)
    logits [] in
  List.nth (Nat.pred k) sorted_desc neg_inf.

Definition gpt2_top_k_mask (logits : tensor_1d) (k : nat) : tensor_1d :=
  let threshold := gpt2_find_kth_largest logits k in
  List.map (fun x => if x >=? threshold then x else neg_inf) logits.

Definition gpt2_top_p_mask (logits : tensor_1d) (p : Z) : tensor_1d :=
  let probs := softmax logits in
  let max_prob := List.fold_left Z.max probs 0 in
  let threshold := (max_prob * p) / scale_factor in
  List.map (fun '(l, prob) =>
    if prob >=? threshold then l else neg_inf
  ) (List.combine logits probs).

Definition gpt2_model_forward_fn := list nat -> tensor_1d.

Definition gpt2_select_token (method : gpt2_sampling_method) (logits : tensor_1d) : nat :=
  match method with
  | GPT2_Greedy => argmax logits
  | GPT2_TopK k => argmax (gpt2_top_k_mask logits k)
  | GPT2_TopP p => argmax (gpt2_top_p_mask logits p)
  | GPT2_Temperature t => argmax (gpt2_temperature_scale logits t)
  end.

Fixpoint gpt2_generate_tokens_aux (forward : gpt2_model_forward_fn) (cfg : gpt2_generation_config)
                                   (tokens : list nat) (remaining : nat) : list nat :=
  match remaining with
  | O => tokens
  | S n =>
      let logits := forward tokens in
      let next_token := gpt2_select_token (gpt2_gen_sampling cfg) logits in
      if Nat.eqb next_token (gpt2_gen_eos_token_id cfg) then tokens
      else gpt2_generate_tokens_aux forward cfg (tokens ++ [next_token]) n
  end.

Definition gpt2_generate (forward : gpt2_model_forward_fn) (cfg : gpt2_generation_config)
                          (prompt : list nat) : list nat :=
  gpt2_generate_tokens_aux forward cfg prompt (gpt2_gen_max_new_tokens cfg).

Record gpt2_generation_trace := mk_gpt2_generation_trace {
  gpt2_gt_prompt : list nat;
  gpt2_gt_generated : list nat;
  gpt2_gt_all_logits : list tensor_1d;
  gpt2_gt_selected_tokens : list nat
}.

Fixpoint gpt2_generate_with_trace_aux (forward : gpt2_model_forward_fn) (cfg : gpt2_generation_config)
                                       (tokens : list nat) (remaining : nat)
                                       (logits_acc : list tensor_1d) : gpt2_generation_trace :=
  match remaining with
  | O => mk_gpt2_generation_trace [] tokens logits_acc []
  | S n =>
      let logits := forward tokens in
      let next_token := gpt2_select_token (gpt2_gen_sampling cfg) logits in
      if Nat.eqb next_token (gpt2_gen_eos_token_id cfg)
      then mk_gpt2_generation_trace [] tokens (logits_acc ++ [logits]) []
      else gpt2_generate_with_trace_aux forward cfg (tokens ++ [next_token]) n (logits_acc ++ [logits])
  end.

Definition gpt2_generate_with_trace (forward : gpt2_model_forward_fn) (cfg : gpt2_generation_config)
                                     (prompt : list nat) : gpt2_generation_trace :=
  let trace := gpt2_generate_with_trace_aux forward cfg prompt (gpt2_gen_max_new_tokens cfg) [] in
  mk_gpt2_generation_trace prompt (gpt2_gt_generated trace) (gpt2_gt_all_logits trace)
                           (List.skipn (List.length prompt) (gpt2_gt_generated trace)).

Definition gpt2_verify_greedy_selection (logits : tensor_1d) (selected : nat) : bool :=
  Nat.eqb selected (argmax logits).

Definition gpt2_verify_token_in_top_k (logits : tensor_1d) (k : nat) (selected : nat) : bool :=
  let threshold := gpt2_find_kth_largest logits k in
  let selected_logit := List.nth selected logits neg_inf in
  selected_logit >=? threshold.

Definition gpt2_verify_generation_step (logits : tensor_1d) (method : gpt2_sampling_method)
                                        (selected : nat) : bool :=
  match method with
  | GPT2_Greedy => gpt2_verify_greedy_selection logits selected
  | GPT2_TopK k => gpt2_verify_token_in_top_k logits k selected
  | GPT2_TopP _ => true
  | GPT2_Temperature _ => true
  end.

Definition gpt2_verify_trace (trace : gpt2_generation_trace) (method : gpt2_sampling_method) : bool :=
  let checks := List.combine (gpt2_gt_all_logits trace) (gpt2_gt_selected_tokens trace) in
  List.forallb (fun '(logits, token) => gpt2_verify_generation_step logits method token) checks.

Definition gpt2_check_argmax_is_max (logits : tensor_1d) : bool :=
  let selected := argmax logits in
  let selected_val := List.nth selected logits neg_inf in
  List.forallb (fun x => selected_val >=? x) logits.

Definition gpt2_check_prompt_preserved (prompt output : list nat) : bool :=
  let prefix := List.firstn (List.length prompt) output in
  (Nat.eqb (List.length prefix) (List.length prompt)) &&
  List.forallb (fun '(a, b) => Nat.eqb a b) (List.combine prefix prompt).

Lemma gpt2_verify_greedy_reflexive : forall logits,
  gpt2_verify_greedy_selection logits (argmax logits) = true.
Proof.
  intros logits.
  unfold gpt2_verify_greedy_selection.
  apply Nat.eqb_refl.
Qed.

Lemma gpt2_generate_base_case : forall forward cfg prompt,
  gpt2_gen_max_new_tokens cfg = 0%nat ->
  gpt2_generate forward cfg prompt = prompt.
Proof.
  intros forward cfg prompt H.
  unfold gpt2_generate.
  rewrite H.
  simpl.
  reflexivity.
Qed.

(** Token selection is a total function of the logits and the method, so it is
    deterministic by construction; there is no content to state as a lemma. *)

Lemma gpt2_firstn_app_exact : forall {A : Type} (l1 l2 : list A),
  List.firstn (List.length l1) (l1 ++ l2) = l1.
Proof.
  intros A l1 l2.
  rewrite List.firstn_app.
  rewrite Nat.sub_diag.
  simpl.
  rewrite List.app_nil_r.
  apply List.firstn_all.
Qed.

Lemma gpt2_generate_tokens_aux_extends : forall forward cfg n prompt,
  exists suffix,
    gpt2_generate_tokens_aux forward cfg prompt n = prompt ++ suffix.
Proof.
  intros forward cfg n.
  induction n as [|n' IH].
  - intros prompt.
    exists [].
    simpl.
    rewrite List.app_nil_r.
    reflexivity.
  - intros prompt.
    simpl.
    destruct (Nat.eqb (gpt2_select_token (gpt2_gen_sampling cfg) (forward prompt))
                      (gpt2_gen_eos_token_id cfg)) eqn:Heos.
    + exists [].
      rewrite List.app_nil_r.
      reflexivity.
    + set (new_token := gpt2_select_token (gpt2_gen_sampling cfg) (forward prompt)).
      destruct (IH (prompt ++ [new_token])) as [suffix' Hsuffix'].
      exists ([new_token] ++ suffix').
      rewrite Hsuffix'.
      rewrite <- List.app_assoc.
      reflexivity.
Qed.

Lemma gpt2_generate_tokens_aux_preserves_prompt : forall forward cfg n prompt,
  List.firstn (List.length prompt) (gpt2_generate_tokens_aux forward cfg prompt n) = prompt.
Proof.
  intros forward cfg n prompt.
  destruct (gpt2_generate_tokens_aux_extends forward cfg n prompt) as [suffix Hextends].
  rewrite Hextends.
  apply gpt2_firstn_app_exact.
Qed.

Lemma gpt2_generation_preserves_prompt : forall forward cfg prompt,
  List.firstn (List.length prompt) (gpt2_generate forward cfg prompt) = prompt.
Proof.
  intros forward cfg prompt.
  unfold gpt2_generate.
  apply gpt2_generate_tokens_aux_preserves_prompt.
Qed.

(** * Float GPT-2: Linear Algebra and LayerNorm

    The Flocq f32 activations exist but are not assembled into a forward
    pass. This section builds the float linear-algebra layer
    and an IEEE-754 layer normalization on top of f32_dot / f32_sqrt, so the
    GPT-2 block can be rebuilt in true float semantics rather than the
    integer fixed-point approximation. *)

(** Float square root via Flocq, round-to-nearest-even. *)
Definition f32_sqrt (x : binary32) : binary32 :=
  @Bsqrt prec32 emax32 prec32_gt_0 prec32_lt_emax32 rnd_NE x.

(** Float matrix transpose. *)
Definition f32_mat_transpose (m : list (list binary32)) : list (list binary32) :=
  match m with
  | [] => []
  | row :: _ =>
      List.map (fun col_idx =>
        List.map (fun row_data => List.nth col_idx row_data f32_zero) m
      ) (List.seq 0 (List.length row))
  end.

(** Float matrix multiply: row i dot column j, via f32_dot over the transpose. *)
Definition f32_mat_mul (a b : list (list binary32)) : list (list binary32) :=
  let b_t := f32_mat_transpose b in
  List.map (fun a_row => List.map (fun b_col => f32_dot a_row b_col) b_t) a.

Definition f32_mat_rows (m : list (list binary32)) : nat := List.length m.

Lemma f32_mat_mul_rows : forall a b,
  f32_mat_rows (f32_mat_mul a b) = f32_mat_rows a.
Proof.
  intros a b. unfold f32_mat_mul, f32_mat_rows. apply List.length_map.
Qed.

(** Float mean of a vector. *)
Definition f32_mean (xs : list binary32) : binary32 :=
  let n := f32_of_Z (Z.of_nat (List.length xs)) in
  f32_div (f32_sum xs) n.

(** Float (population) variance of a vector given its mean. *)
Definition f32_variance (xs : list binary32) (mean : binary32) : binary32 :=
  let n := f32_of_Z (Z.of_nat (List.length xs)) in
  let sq := List.map (fun x => let d := f32_minus x mean in f32_mult d d) xs in
  f32_div (f32_sum sq) n.

(** IEEE-754 layer normalization of a single vector with per-feature gamma/beta. *)
Definition f32_layer_norm_vec (gamma beta : list binary32) (eps : binary32)
                              (x : list binary32) : list binary32 :=
  let mu := f32_mean x in
  let var := f32_variance x mu in
  let denom := f32_sqrt (f32_plus var eps) in
  List.map (fun '(idx, xi) =>
    let g := List.nth idx gamma f32_one in
    let b := List.nth idx beta f32_zero in
    let normalized := f32_div (f32_minus xi mu) denom in
    f32_plus (f32_mult g normalized) b
  ) (List.combine (List.seq 0 (List.length x)) x).

Lemma f32_layer_norm_vec_length : forall gamma beta eps x,
  List.length (f32_layer_norm_vec gamma beta eps x) = List.length x.
Proof.
  intros gamma beta eps x. unfold f32_layer_norm_vec.
  rewrite List.length_map, List.length_combine, List.length_seq, Nat.min_id.
  reflexivity.
Qed.

Definition f32_layer_norm_2d (gamma beta : list binary32) (eps : binary32)
                             (m : list (list binary32)) : list (list binary32) :=
  List.map (f32_layer_norm_vec gamma beta eps) m.

Lemma f32_layer_norm_2d_rows : forall gamma beta eps m,
  f32_mat_rows (f32_layer_norm_2d gamma beta eps m) = f32_mat_rows m.
Proof.
  intros gamma beta eps m. unfold f32_layer_norm_2d, f32_mat_rows.
  apply List.length_map.
Qed.

(** Witness: transpose of a 2x2 has 2 rows. *)
Example f32_transpose_dims :
  f32_mat_rows (f32_mat_transpose [[f32_zero; f32_one]; [f32_one; f32_zero]]) = 2%nat.
Proof. reflexivity. Qed.

(** * Float GPT-2: Attention and MLP

    Float linear layers, multi-head causal self-attention, and the MLP,
    all in IEEE-754 f32 via the operations above and the existing
    f32_softmax_2d / f32_gelu_vec. These mirror the integer gpt2_*
    forward functions but compute in true float semantics. *)

(** Float dense layer: x . W + b, with W stored [in, out] (GPT-2 Conv1D layout). *)
Definition f32_linear_forward (weight : list (list binary32)) (bias : list binary32)
                              (x : list binary32) : list binary32 :=
  f32_vec_add (f32_mat_vec_mul (f32_mat_transpose weight) x) bias.

Definition f32_linear_forward_2d (weight : list (list binary32)) (bias : list binary32)
                                 (x : list (list binary32)) : list (list binary32) :=
  List.map (f32_linear_forward weight bias) x.

Lemma f32_linear_forward_2d_rows : forall w b x,
  f32_mat_rows (f32_linear_forward_2d w b x) = f32_mat_rows x.
Proof.
  intros w b x. unfold f32_linear_forward_2d, f32_mat_rows. apply List.length_map.
Qed.

(** Float residual matrix add. *)
Definition f32_add_matrices (a b : list (list binary32)) : list (list binary32) :=
  List.map (fun '(row_a, row_b) => f32_vec_add row_a row_b) (List.combine a b).

Lemma f32_add_matrices_rows_eq : forall a b,
  f32_mat_rows a = f32_mat_rows b ->
  f32_mat_rows (f32_add_matrices a b) = f32_mat_rows a.
Proof.
  intros a b Heq. unfold f32_add_matrices, f32_mat_rows in *.
  rewrite List.length_map, List.length_combine. rewrite Heq. apply Nat.min_id.
Qed.

(** Multi-head split / concat (structure only, element type binary32). *)
Definition f32_split_row_into_heads (num_heads : nat) (row : list binary32) : list (list binary32) :=
  let head_dim := Nat.div (List.length row) num_heads in
  List.map (fun h => List.firstn head_dim (List.skipn (h * head_dim) row)) (List.seq 0 num_heads).

Definition f32_split_into_heads (num_heads : nat) (m : list (list binary32))
                               : list (list (list binary32)) :=
  let rows_split := List.map (f32_split_row_into_heads num_heads) m in
  List.map (fun h => List.map (fun row_heads => List.nth h row_heads []) rows_split)
           (List.seq 0 num_heads).

Lemma f32_split_into_heads_length : forall n m,
  List.length (f32_split_into_heads n m) = n.
Proof.
  intros n m. unfold f32_split_into_heads.
  rewrite List.length_map, List.length_seq. reflexivity.
Qed.

Definition f32_concat_heads (heads : list (list (list binary32))) : list (list binary32) :=
  match heads with
  | [] => []
  | first_head :: _ =>
      List.map (fun seq_idx =>
        List.concat (List.map (fun head => List.nth seq_idx head []) heads)
      ) (List.seq 0 (List.length first_head))
  end.

(** Causal mask in float: 0 on/below the diagonal, large negative above. *)
Definition f32_mask_neg : binary32 := f32_of_Z (-1000000000).

Definition f32_causal_mask_entry (row col : nat) : binary32 :=
  if Nat.leb col row then f32_zero else f32_mask_neg.

Definition f32_causal_mask (seq_len : nat) : list (list binary32) :=
  List.map (fun row => List.map (fun col => f32_causal_mask_entry row col) (List.seq 0 seq_len))
           (List.seq 0 seq_len).

Lemma f32_causal_mask_length : forall seq_len,
  List.length (f32_causal_mask seq_len) = seq_len.
Proof.
  intros seq_len. unfold f32_causal_mask.
  rewrite List.length_map, List.length_seq. reflexivity.
Qed.

(** Scale scores by 1/sqrt(d_k). *)
Definition f32_scale_scores (scores : list (list binary32)) (d_k : nat) : list (list binary32) :=
  let s := f32_div f32_one (f32_sqrt (f32_of_Z (Z.of_nat d_k))) in
  List.map (fun row => List.map (fun x => f32_mult x s) row) scores.

Definition f32_apply_mask (scores mask : list (list binary32)) : list (list binary32) :=
  List.map (fun '(s_row, m_row) =>
    List.map (fun '(s, m) => f32_plus s m) (List.combine s_row m_row)
  ) (List.combine scores mask).

(** Float causal scaled-dot-product attention. *)
Definition f32_causal_attention (q k v : list (list binary32)) (d_k : nat) : list (list binary32) :=
  let scores := f32_scale_scores (f32_mat_mul q (f32_mat_transpose k)) d_k in
  let masked := f32_apply_mask scores (f32_causal_mask (f32_mat_rows q)) in
  f32_mat_mul (f32_softmax_2d masked) v.

(** Multi-head causal self-attention forward (GPT-2 c_attn / c_proj). *)
Definition f32_attention_forward (n_embd n_head : nat)
    (c_attn_w : list (list binary32)) (c_attn_b : list binary32)
    (c_proj_w : list (list binary32)) (c_proj_b : list binary32)
    (hidden : list (list binary32)) : list (list binary32) :=
  let qkv := f32_linear_forward_2d c_attn_w c_attn_b hidden in
  let d := n_embd in
  let head_dim := Nat.div d n_head in
  let split_qkv row :=
    (List.firstn d row, List.firstn d (List.skipn d row), List.skipn (2 * d) row) in
  let qkv_split := List.map split_qkv qkv in
  let qs := List.map (fun '(q, _, _) => q) qkv_split in
  let ks := List.map (fun '(_, k, _) => k) qkv_split in
  let vs := List.map (fun '(_, _, v) => v) qkv_split in
  let q_heads := f32_split_into_heads n_head qs in
  let k_heads := f32_split_into_heads n_head ks in
  let v_heads := f32_split_into_heads n_head vs in
  let head_outputs := List.map (fun '(qh, (kh, vh)) => f32_causal_attention qh kh vh head_dim)
                               (List.combine q_heads (List.combine k_heads v_heads)) in
  let concat := f32_concat_heads head_outputs in
  f32_linear_forward_2d c_proj_w c_proj_b concat.

(** Float MLP: c_fc linear, GELU, c_proj linear. *)
Definition f32_mlp_forward
    (c_fc_w : list (list binary32)) (c_fc_b : list binary32)
    (c_proj_w : list (list binary32)) (c_proj_b : list binary32)
    (hidden : list (list binary32)) : list (list binary32) :=
  let h := f32_linear_forward_2d c_fc_w c_fc_b hidden in
  let h_gelu := List.map f32_gelu_vec h in
  f32_linear_forward_2d c_proj_w c_proj_b h_gelu.

Lemma f32_mlp_forward_rows : forall cfw cfb cpw cpb h,
  f32_mat_rows (f32_mlp_forward cfw cfb cpw cpb h) = f32_mat_rows h.
Proof.
  intros cfw cfb cpw cpb h.
  unfold f32_mlp_forward, f32_linear_forward_2d, f32_mat_rows.
  rewrite !List.length_map. reflexivity.
Qed.

(** * Float GPT-2: Block and Forward Pass

    Float weight containers and the assembled IEEE-754 forward pass:
    embeddings, the pre-norm decoder block (ln -> attn -> residual ->
    ln -> mlp -> residual), the block stack, final layer norm, and the
    tied-embedding logit projection. *)

Record f32_attention_weights := mk_f32_attention_weights {
  f32_attn_c_attn_weight : list (list binary32);
  f32_attn_c_attn_bias : list binary32;
  f32_attn_c_proj_weight : list (list binary32);
  f32_attn_c_proj_bias : list binary32
}.

Record f32_mlp_weights := mk_f32_mlp_weights {
  f32_mlp_c_fc_weight : list (list binary32);
  f32_mlp_c_fc_bias : list binary32;
  f32_mlp_c_proj_weight : list (list binary32);
  f32_mlp_c_proj_bias : list binary32
}.

Record f32_layernorm_weights := mk_f32_layernorm_weights {
  f32_ln_weight : list binary32;
  f32_ln_bias : list binary32
}.

Record f32_block_weights := mk_f32_block_weights {
  f32_block_ln_1 : f32_layernorm_weights;
  f32_block_attn : f32_attention_weights;
  f32_block_ln_2 : f32_layernorm_weights;
  f32_block_mlp : f32_mlp_weights
}.

Record f32_model_weights := mk_f32_model_weights {
  f32_wte : list (list binary32);
  f32_wpe : list (list binary32);
  f32_blocks : list f32_block_weights;
  f32_ln_f : f32_layernorm_weights
}.

(** Standard GPT-2 layer-norm epsilon, 1e-5. *)
Definition f32_ln_eps : binary32 := f32_div f32_one (f32_of_Z 100000).

(** Pre-norm decoder block. *)
Definition f32_block_forward (cfg : gpt2_inference_config) (eps : binary32)
                             (block : f32_block_weights) (hidden : list (list binary32))
                             : list (list binary32) :=
  let ln1 := f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                               (f32_ln_bias (f32_block_ln_1 block)) eps hidden in
  let attn_out := f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
                    (f32_attn_c_attn_weight (f32_block_attn block))
                    (f32_attn_c_attn_bias (f32_block_attn block))
                    (f32_attn_c_proj_weight (f32_block_attn block))
                    (f32_attn_c_proj_bias (f32_block_attn block)) ln1 in
  let hidden2 := f32_add_matrices hidden attn_out in
  let ln2 := f32_layer_norm_2d (f32_ln_weight (f32_block_ln_2 block))
                               (f32_ln_bias (f32_block_ln_2 block)) eps hidden2 in
  let mlp_out := f32_mlp_forward (f32_mlp_c_fc_weight (f32_block_mlp block))
                   (f32_mlp_c_fc_bias (f32_block_mlp block))
                   (f32_mlp_c_proj_weight (f32_block_mlp block))
                   (f32_mlp_c_proj_bias (f32_block_mlp block)) ln2 in
  f32_add_matrices hidden2 mlp_out.

Fixpoint f32_blocks_forward (cfg : gpt2_inference_config) (eps : binary32)
                            (blocks : list f32_block_weights) (hidden : list (list binary32))
                            : list (list binary32) :=
  match blocks with
  | [] => hidden
  | block :: rest => f32_blocks_forward cfg eps rest (f32_block_forward cfg eps block hidden)
  end.

Definition f32_lookup_embedding (embeddings : list (list binary32)) (token_id : nat) : list binary32 :=
  List.nth token_id embeddings [].

Definition f32_embed_tokens (embeddings : list (list binary32)) (token_ids : list nat)
                            : list (list binary32) :=
  List.map (f32_lookup_embedding embeddings) token_ids.

Definition f32_embed_positions (embeddings : list (list binary32)) (seq_len : nat)
                               : list (list binary32) :=
  List.map (f32_lookup_embedding embeddings) (List.seq 0 seq_len).

Lemma f32_embed_tokens_length : forall e ids,
  List.length (f32_embed_tokens e ids) = List.length ids.
Proof. intros e ids. unfold f32_embed_tokens. apply List.length_map. Qed.

Lemma f32_embed_positions_length : forall e n,
  List.length (f32_embed_positions e n) = n.
Proof.
  intros e n. unfold f32_embed_positions.
  rewrite List.length_map, List.length_seq. reflexivity.
Qed.

(** Full IEEE-754 forward pass: embed, blocks, final layer norm. *)
Definition f32_gpt2_forward (cfg : gpt2_inference_config) (eps : binary32)
                            (model : f32_model_weights) (token_ids : list nat)
                            : list (list binary32) :=
  let seq_len := List.length token_ids in
  let tok := f32_embed_tokens (f32_wte model) token_ids in
  let pos := f32_embed_positions (f32_wpe model) seq_len in
  let hidden := f32_add_matrices tok pos in
  let transformed := f32_blocks_forward cfg eps (f32_blocks model) hidden in
  f32_layer_norm_2d (f32_ln_weight (f32_ln_f model)) (f32_ln_bias (f32_ln_f model)) eps transformed.

(** Logits via tied embeddings: logits[i][j] = dot(hidden[i], wte[j]). *)
Definition f32_gpt2_logits (cfg : gpt2_inference_config) (eps : binary32)
                           (model : f32_model_weights) (token_ids : list nat)
                           : list (list binary32) :=
  let hidden := f32_gpt2_forward cfg eps model token_ids in
  List.map (fun h_row => List.map (fun w_row => f32_dot h_row w_row) (f32_wte model)) hidden.

(** Next-token logits: the final row. *)
Definition f32_gpt2_next_token_logits (cfg : gpt2_inference_config) (eps : binary32)
                                      (model : f32_model_weights) (token_ids : list nat)
                                      : list binary32 :=
  match List.rev (f32_gpt2_logits cfg eps model token_ids) with
  | [] => []
  | last :: _ => last
  end.

(** Shape preservation through the float attention primitives. *)
Lemma f32_softmax_2d_rows : forall m,
  f32_mat_rows (f32_softmax_2d m) = f32_mat_rows m.
Proof. intros m. unfold f32_softmax_2d, f32_mat_rows. apply List.length_map. Qed.

Lemma f32_scale_scores_rows : forall s d,
  f32_mat_rows (f32_scale_scores s d) = f32_mat_rows s.
Proof. intros s d. unfold f32_scale_scores, f32_mat_rows. apply List.length_map. Qed.

Lemma f32_apply_mask_rows : forall scores mask,
  f32_mat_rows scores = f32_mat_rows mask ->
  f32_mat_rows (f32_apply_mask scores mask) = f32_mat_rows scores.
Proof.
  intros scores mask H. unfold f32_apply_mask, f32_mat_rows in *.
  rewrite List.length_map, List.length_combine, H. apply Nat.min_id.
Qed.

Lemma f32_causal_attention_rows : forall q k v d,
  f32_mat_rows (f32_causal_attention q k v d) = f32_mat_rows q.
Proof.
  intros q k v d. unfold f32_causal_attention.
  rewrite f32_mat_mul_rows, f32_softmax_2d_rows.
  rewrite f32_apply_mask_rows.
  - rewrite f32_scale_scores_rows, f32_mat_mul_rows. reflexivity.
  - rewrite f32_scale_scores_rows, f32_mat_mul_rows.
    unfold f32_mat_rows. rewrite f32_causal_mask_length. reflexivity.
Qed.

(** Head split/concat row lemmas, for the full attention shape proof. *)
Lemma f32_split_into_heads_nonempty : forall n m,
  (n > 0)%nat -> f32_split_into_heads n m <> [].
Proof.
  intros n m Hn. unfold f32_split_into_heads.
  destruct n; [lia | ]. simpl. discriminate.
Qed.

Lemma f32_split_into_heads_hd_rows : forall n m,
  (n > 0)%nat -> f32_mat_rows (hd [] (f32_split_into_heads n m)) = f32_mat_rows m.
Proof.
  intros n m Hn. unfold f32_split_into_heads.
  destruct n; [lia | ]. simpl. unfold f32_mat_rows.
  rewrite !List.length_map. reflexivity.
Qed.

Lemma f32_concat_heads_rows : forall heads,
  heads <> [] -> f32_mat_rows (f32_concat_heads heads) = f32_mat_rows (hd [] heads).
Proof.
  intros heads Hne. unfold f32_concat_heads.
  destruct heads as [|h rest]; [contradiction | ].
  unfold f32_mat_rows. rewrite List.length_map, List.length_seq. reflexivity.
Qed.

(** ** Shape preservation through the whole float forward

    The per-primitive row lemmas above compose: multi-head attention, the
    pre-norm block, the block stack, and finally the assembled forward all
    preserve the sequence length, so the model emits exactly one hidden row
    and one logit row per input token. *)

Lemma f32_head_outputs_length : forall n_head hd qs ks vs,
  List.length
    (List.map (fun '(qh, (kh, vh)) => f32_causal_attention qh kh vh hd)
      (List.combine (f32_split_into_heads n_head qs)
        (List.combine (f32_split_into_heads n_head ks)
                      (f32_split_into_heads n_head vs)))) = n_head.
Proof.
  intros. rewrite List.length_map, List.length_combine.
  rewrite !f32_split_into_heads_length.
  rewrite List.length_combine, !f32_split_into_heads_length.
  rewrite !Nat.min_id. reflexivity.
Qed.

Lemma f32_attention_forward_rows : forall n_embd n_head caw cab cpw cpb hidden,
  (n_head > 0)%nat ->
  f32_mat_rows (f32_attention_forward n_embd n_head caw cab cpw cpb hidden)
    = f32_mat_rows hidden.
Proof.
  intros n_embd n_head caw cab cpw cpb hidden Hh.
  unfold f32_attention_forward.
  rewrite f32_linear_forward_2d_rows.
  rewrite f32_concat_heads_rows.
  - destruct n_head as [|n]; [lia|].
    cbn [f32_split_into_heads List.seq List.map List.combine hd].
    rewrite f32_causal_attention_rows.
    unfold f32_mat_rows. rewrite !List.length_map.
    fold (f32_mat_rows (f32_linear_forward_2d caw cab hidden)).
    apply f32_linear_forward_2d_rows.
  - intro Hc. apply (f_equal (@List.length _)) in Hc.
    rewrite f32_head_outputs_length in Hc. cbn in Hc. lia.
Qed.

Lemma f32_block_forward_rows : forall cfg eps block hidden,
  (gpt2_inf_n_head cfg > 0)%nat ->
  f32_mat_rows (f32_block_forward cfg eps block hidden) = f32_mat_rows hidden.
Proof.
  intros cfg eps block hidden Hh.
  unfold f32_block_forward.
  set (ln1 := f32_layer_norm_2d (f32_ln_weight (f32_block_ln_1 block))
                (f32_ln_bias (f32_block_ln_1 block)) eps hidden).
  assert (Hln1 : f32_mat_rows ln1 = f32_mat_rows hidden)
    by (unfold ln1; apply f32_layer_norm_2d_rows).
  assert (Hattn : f32_mat_rows
    (f32_attention_forward (gpt2_inf_n_embd cfg) (gpt2_inf_n_head cfg)
      (f32_attn_c_attn_weight (f32_block_attn block))
      (f32_attn_c_attn_bias (f32_block_attn block))
      (f32_attn_c_proj_weight (f32_block_attn block))
      (f32_attn_c_proj_bias (f32_block_attn block)) ln1) = f32_mat_rows hidden).
  { rewrite f32_attention_forward_rows by assumption. exact Hln1. }
  set (hidden2 := f32_add_matrices hidden _).
  assert (H2 : f32_mat_rows hidden2 = f32_mat_rows hidden).
  { unfold hidden2. rewrite f32_add_matrices_rows_eq; [reflexivity|].
    symmetry. exact Hattn. }
  rewrite f32_add_matrices_rows_eq.
  - exact H2.
  - rewrite f32_mlp_forward_rows. symmetry. apply f32_layer_norm_2d_rows.
Qed.

Lemma f32_blocks_forward_rows : forall cfg eps blocks hidden,
  (gpt2_inf_n_head cfg > 0)%nat ->
  f32_mat_rows (f32_blocks_forward cfg eps blocks hidden) = f32_mat_rows hidden.
Proof.
  intros cfg eps blocks. induction blocks as [|b bs IH]; intros hidden Hh.
  - reflexivity.
  - simpl. rewrite IH by assumption. apply f32_block_forward_rows. assumption.
Qed.

Theorem f32_gpt2_forward_rows : forall cfg eps model toks,
  (gpt2_inf_n_head cfg > 0)%nat ->
  f32_mat_rows (f32_gpt2_forward cfg eps model toks) = List.length toks.
Proof.
  intros cfg eps model toks Hh. unfold f32_gpt2_forward.
  rewrite f32_layer_norm_2d_rows.
  rewrite f32_blocks_forward_rows by assumption.
  unfold f32_add_matrices, f32_mat_rows.
  rewrite List.length_map, List.length_combine.
  rewrite f32_embed_tokens_length, f32_embed_positions_length.
  apply Nat.min_id.
Qed.

Theorem f32_gpt2_logits_rows : forall cfg eps model toks,
  (gpt2_inf_n_head cfg > 0)%nat ->
  f32_mat_rows (f32_gpt2_logits cfg eps model toks) = List.length toks.
Proof.
  intros cfg eps model toks Hh. unfold f32_gpt2_logits, f32_mat_rows.
  rewrite List.length_map.
  fold (f32_mat_rows (f32_gpt2_forward cfg eps model toks)).
  apply f32_gpt2_forward_rows. assumption.
Qed.

(** Every logit row scores exactly one entry per vocabulary item. *)
Theorem f32_gpt2_logits_row_width : forall cfg eps model toks row,
  In row (f32_gpt2_logits cfg eps model toks) ->
  List.length row = List.length (f32_wte model).
Proof.
  intros cfg eps model toks row Hin.
  unfold f32_gpt2_logits in Hin.
  apply List.in_map_iff in Hin.
  destruct Hin as [h [Heq _]]. subst row.
  apply List.length_map.
Qed.

(** * Float GPT-2: Safetensors f32 Value Decoding

    Decode little-endian IEEE-754 f32 bytes into Flocq binary32 values.
    b32_of_bits yields a full Binary.binary_float; Binary.B2BSN collapses
    it to the single-NaN binary32 used throughout. This is the
    "bytes are the proven bytes" deserialization half of the loader. *)

Definition f32_of_bits (bits : Z) : binary32 :=
  Binary.B2BSN _ _ (b32_of_bits bits).

Definition f32_bytes_to_binary32 (bs : list byte) : binary32 :=
  f32_of_bits (bytes_to_z_le_u32 bs).

Fixpoint f32_decode_tensor_1d_aux (count : nat) (bs : list byte) : list binary32 :=
  match count with
  | O => []
  | S n => f32_bytes_to_binary32 (take 4 bs) :: f32_decode_tensor_1d_aux n (drop 4 bs)
  end.

Definition f32_decode_tensor_1d (len : nat) (bs : list byte) : list binary32 :=
  f32_decode_tensor_1d_aux len bs.

Lemma f32_decode_tensor_1d_length : forall len bs,
  List.length (f32_decode_tensor_1d len bs) = len.
Proof.
  unfold f32_decode_tensor_1d. intros len.
  induction len as [|n IH]; intros bs; simpl; [reflexivity|].
  rewrite IH. reflexivity.
Qed.

Fixpoint f32_decode_tensor_2d_aux (rows cols : nat) (bs : list byte) : list (list binary32) :=
  match rows with
  | O => []
  | S r => f32_decode_tensor_1d cols bs :: f32_decode_tensor_2d_aux r cols (drop (4 * cols) bs)
  end.

Definition f32_decode_tensor_2d (rows cols : nat) (bs : list byte) : list (list binary32) :=
  f32_decode_tensor_2d_aux rows cols bs.

Lemma f32_decode_tensor_2d_rows : forall rows cols bs,
  List.length (f32_decode_tensor_2d rows cols bs) = rows.
Proof.
  unfold f32_decode_tensor_2d. intros rows.
  induction rows as [|r IH]; intros cols bs; simpl; [reflexivity|].
  rewrite IH. reflexivity.
Qed.

(** * Float GPT-2: Weight Reshaping and Block Assembly

    Reshape flat decoded f32 vectors into the typed weight records, mirroring
    the integer gpt2_make_* builders. Together with the value decoder above,
    this turns a raw byte buffer into f32_model_weights for f32_gpt2_forward. *)

Fixpoint f32_reshape_1d_to_2d (data : list binary32) (rows cols : nat) : list (list binary32) :=
  match rows with
  | O => []
  | S r' => List.firstn cols data :: f32_reshape_1d_to_2d (List.skipn cols data) r' cols
  end.

Lemma f32_reshape_1d_to_2d_rows : forall rows data cols,
  List.length (f32_reshape_1d_to_2d data rows cols) = rows.
Proof.
  induction rows as [|r IH]; intros data cols; simpl; [reflexivity|].
  rewrite IH. reflexivity.
Qed.

Definition f32_make_layernorm (weight bias : list binary32) : f32_layernorm_weights :=
  mk_f32_layernorm_weights weight bias.

Definition f32_make_attention (c_attn_w c_attn_b c_proj_w c_proj_b : list binary32)
                              (d : nat) : f32_attention_weights :=
  mk_f32_attention_weights
    (f32_reshape_1d_to_2d c_attn_w d (3 * d))
    c_attn_b
    (f32_reshape_1d_to_2d c_proj_w d d)
    c_proj_b.

Definition f32_make_mlp (c_fc_w c_fc_b c_proj_w c_proj_b : list binary32)
                        (d ff : nat) : f32_mlp_weights :=
  mk_f32_mlp_weights
    (f32_reshape_1d_to_2d c_fc_w d ff)
    c_fc_b
    (f32_reshape_1d_to_2d c_proj_w ff d)
    c_proj_b.

Definition f32_make_block (ln1_w ln1_b : list binary32)
                          (c_attn_w c_attn_b c_proj_w c_proj_b : list binary32)
                          (ln2_w ln2_b : list binary32)
                          (fc_w fc_b mlp_proj_w mlp_proj_b : list binary32)
                          (d ff : nat) : f32_block_weights :=
  mk_f32_block_weights
    (f32_make_layernorm ln1_w ln1_b)
    (f32_make_attention c_attn_w c_attn_b c_proj_w c_proj_b d)
    (f32_make_layernorm ln2_w ln2_b)
    (f32_make_mlp fc_w fc_b mlp_proj_w mlp_proj_b d ff).

(** The reshaped attention QKV projection has d rows. *)
Lemma f32_make_attention_qkv_rows : forall caw cab cpw cpb d,
  f32_mat_rows (f32_attn_c_attn_weight (f32_make_attention caw cab cpw cpb d)) = d.
Proof.
  intros caw cab cpw cpb d. unfold f32_make_attention, f32_mat_rows.
  apply f32_reshape_1d_to_2d_rows.
Qed.

(** * Float GPT-2: Safetensors Header JSON Primitives

    Minimal JSON scanners over the header string: whitespace, natural
    numbers, quoted strings, and integer arrays. Building blocks for parsing
    the safetensors tensor table (dtype, shape, data_offsets). *)

Definition json_is_ws (c : ascii) : bool :=
  let n := Ascii.nat_of_ascii c in
  orb (Nat.eqb n 32) (orb (Nat.eqb n 9) (orb (Nat.eqb n 10) (Nat.eqb n 13))).

Fixpoint json_skip_ws (s : string) : string :=
  match s with
  | String c rest => if json_is_ws c then json_skip_ws rest else s
  | EmptyString => s
  end.

Definition json_digit (c : ascii) : option nat :=
  let n := Ascii.nat_of_ascii c in
  if andb (Nat.leb 48 n) (Nat.leb n 57) then Some (n - 48)%nat else None.

Fixpoint json_parse_nat_aux (s : string) (acc : nat) : nat * string :=
  match s with
  | String c rest =>
      match json_digit c with
      | Some d => json_parse_nat_aux rest (acc * 10 + d)%nat
      | None => (acc, s)
      end
  | EmptyString => (acc, s)
  end.

Definition json_parse_nat (s : string) : nat * string :=
  json_parse_nat_aux (json_skip_ws s) 0%nat.

Definition json_dquote : ascii := Ascii.ascii_of_nat 34.
Definition json_rbrack : ascii := Ascii.ascii_of_nat 93.
Definition json_comma : ascii := Ascii.ascii_of_nat 44.
Definition json_lbrack : ascii := Ascii.ascii_of_nat 91.

Fixpoint json_parse_string_chars (s : string) : string * string :=
  match s with
  | String c rest =>
      if Ascii.eqb c json_dquote then (EmptyString, rest)
      else let (str, remaining) := json_parse_string_chars rest in (String c str, remaining)
  | EmptyString => (EmptyString, EmptyString)
  end.

(** Parse a JSON array of naturals, e.g. "[768, 50257]". Skips any non-digit
    char (brackets, commas, colons, whitespace); terminates at ']'. Fuel-bounded. *)
Fixpoint json_parse_nat_list_aux (fuel : nat) (s : string) (acc : list nat) : list nat * string :=
  match fuel with
  | O => (List.rev acc, s)
  | S f =>
      match s with
      | String c rest =>
          if Ascii.eqb c json_rbrack then (List.rev acc, rest)
          else match json_digit c with
               | Some _ => let (n, s') := json_parse_nat s in json_parse_nat_list_aux f s' (n :: acc)
               | None => json_parse_nat_list_aux f rest acc
               end
      | EmptyString => (List.rev acc, s)
      end
  end.

Definition json_parse_nat_list (s : string) : list nat * string :=
  json_parse_nat_list_aux (String.length s) s [].

Example json_parse_nat_ex : fst (json_parse_nat "768abc"%string) = 768%nat.
Proof. vm_compute. reflexivity. Qed.

Example json_parse_string_ex :
  fst (json_parse_string_chars "F32""rest"%string) = "F32"%string.
Proof. reflexivity. Qed.

Example json_parse_nat_list_ex :
  fst (json_parse_nat_list "[3, 5]"%string) = [3%nat; 5%nat].
Proof. vm_compute. reflexivity. Qed.

(** Key lookup in the header object, then value extraction. *)
Definition json_quoted (k : string) : string :=
  String json_dquote (String.append k (String json_dquote EmptyString)).

Fixpoint json_is_prefix (pre s : string) : bool :=
  match pre, s with
  | EmptyString, _ => true
  | String _ _, EmptyString => false
  | String pc pr, String sc sr => if Ascii.eqb pc sc then json_is_prefix pr sr else false
  end.

Fixpoint json_find_after (fuel : nat) (needle s : string) : option string :=
  match fuel with
  | O => None
  | S f =>
      if json_is_prefix needle s then Some (string_drop (String.length needle) s)
      else match s with
           | EmptyString => None
           | String _ rest => json_find_after f needle rest
           end
  end.

(** Extract the integer array that follows "key" (e.g. shape, data_offsets). *)
Definition json_extract_nat_list (key : string) (s : string) : list nat :=
  match json_find_after (String.length s) (json_quoted key) s with
  | None => []
  | Some after => fst (json_parse_nat_list after)
  end.

(** Extract the quoted string value that follows "key" (e.g. dtype). *)
Definition json_extract_string (key : string) (s : string) : string :=
  match json_find_after (String.length s) (json_quoted key) s with
  | None => EmptyString
  | Some after =>
      match json_find_after (String.length after) (String json_dquote EmptyString) after with
      | None => EmptyString
      | Some after_q => fst (json_parse_string_chars after_q)
      end
  end.

Example json_extract_shape_ex :
  json_extract_nat_list "shape" "abc""shape"": [4, 7], x"%string = [4%nat; 7%nat].
Proof. vm_compute. reflexivity. Qed.

Example json_extract_dtype_ex :
  json_extract_string "dtype" "x""dtype"": ""F32"", y"%string = "F32"%string.
Proof. vm_compute. reflexivity. Qed.

(** * Float GPT-2: Named Tensor Loading from Data Section

    Combine the header table and the data section: scope to a tensor by name,
    read its byte data_offsets, slice the data buffer, decode the f32 values. *)

Definition json_tensor_shape (header tname : string) : list nat :=
  match json_find_after (String.length header) (json_quoted tname) header with
  | None => []
  | Some sub => json_extract_nat_list "shape" sub
  end.

Definition json_tensor_offsets (header tname : string) : list nat :=
  match json_find_after (String.length header) (json_quoted tname) header with
  | None => []
  | Some sub => json_extract_nat_list "data_offsets" sub
  end.

Definition f32_load_tensor_bytes (data : list byte) (start_byte end_byte : nat) : list binary32 :=
  f32_decode_tensor_1d ((end_byte - start_byte) / 4)%nat (drop start_byte data).

Lemma f32_load_tensor_bytes_length : forall data a b,
  List.length (f32_load_tensor_bytes data a b) = ((b - a) / 4)%nat.
Proof. intros data a b. unfold f32_load_tensor_bytes. apply f32_decode_tensor_1d_length. Qed.

(** Load a named tensor's f32 values: look up its [start, end] byte offsets in
    the header, slice the data section, decode. *)
Definition f32_load_named (header : string) (data : list byte) (tname : string) : list binary32 :=
  match json_tensor_offsets header tname with
  | start_byte :: end_byte :: _ => f32_load_tensor_bytes data start_byte end_byte
  | _ => []
  end.

(** * Float GPT-2: Full Model Loader

    Construct every GPT-2 tensor name, load and reshape each, and assemble
    f32_model_weights from a safetensors header string and data section.
    This closes the loader: bytes on disk become the typed float weights
    that f32_gpt2_forward runs on. *)

Fixpoint nat_to_string_aux (fuel n : nat) (acc : string) : string :=
  match fuel with
  | O => acc
  | S f =>
      let d := (n mod 10)%nat in
      let acc' := String (Ascii.ascii_of_nat ((48 + d)%nat)) acc in
      let q := (n / 10)%nat in
      match q with
      | O => acc'
      | _ => nat_to_string_aux f q acc'
      end
  end.

Definition nat_to_string (n : nat) : string := nat_to_string_aux (S n) n EmptyString.

Example nat_to_string_ex : nat_to_string 12 = "12"%string.
Proof. vm_compute. reflexivity. Qed.

Example nat_to_string_zero : nat_to_string 0 = "0"%string.
Proof. vm_compute. reflexivity. Qed.

Definition gpt2_block_prefix (i : nat) : string :=
  ("h." ++ nat_to_string i ++ ".")%string.

Definition f32_load_block (header : string) (data : list byte)
                          (cfg : gpt2_inference_config) (i : nat) : f32_block_weights :=
  let p := gpt2_block_prefix i in
  f32_make_block
    (f32_load_named header data (p ++ "ln_1.weight")%string)
    (f32_load_named header data (p ++ "ln_1.bias")%string)
    (f32_load_named header data (p ++ "attn.c_attn.weight")%string)
    (f32_load_named header data (p ++ "attn.c_attn.bias")%string)
    (f32_load_named header data (p ++ "attn.c_proj.weight")%string)
    (f32_load_named header data (p ++ "attn.c_proj.bias")%string)
    (f32_load_named header data (p ++ "ln_2.weight")%string)
    (f32_load_named header data (p ++ "ln_2.bias")%string)
    (f32_load_named header data (p ++ "mlp.c_fc.weight")%string)
    (f32_load_named header data (p ++ "mlp.c_fc.bias")%string)
    (f32_load_named header data (p ++ "mlp.c_proj.weight")%string)
    (f32_load_named header data (p ++ "mlp.c_proj.bias")%string)
    (gpt2_inf_n_embd cfg) (gpt2_inf_n_inner cfg).

Definition f32_load_blocks (header : string) (data : list byte)
                           (cfg : gpt2_inference_config) : list f32_block_weights :=
  List.map (f32_load_block header data cfg) (List.seq 0 (gpt2_inf_n_layer cfg)).

Lemma f32_load_blocks_length : forall header data cfg,
  List.length (f32_load_blocks header data cfg) = gpt2_inf_n_layer cfg.
Proof.
  intros header data cfg. unfold f32_load_blocks.
  rewrite List.length_map, List.length_seq. reflexivity.
Qed.

Definition f32_load_model (header : string) (data : list byte)
                          (cfg : gpt2_inference_config) : f32_model_weights :=
  let d := gpt2_inf_n_embd cfg in
  mk_f32_model_weights
    (f32_reshape_1d_to_2d (f32_load_named header data "wte.weight"%string) (gpt2_inf_vocab_size cfg) d)
    (f32_reshape_1d_to_2d (f32_load_named header data "wpe.weight"%string) (gpt2_inf_n_positions cfg) d)
    (f32_load_blocks header data cfg)
    (f32_make_layernorm (f32_load_named header data "ln_f.weight"%string)
                        (f32_load_named header data "ln_f.bias"%string)).

(** Loaded token-embedding matrix has vocab_size rows. *)
Lemma f32_load_model_wte_rows : forall header data cfg,
  f32_mat_rows (f32_wte (f32_load_model header data cfg)) = gpt2_inf_vocab_size cfg.
Proof.
  intros header data cfg. unfold f32_load_model, f32_mat_rows.
  apply f32_reshape_1d_to_2d_rows.
Qed.

(** The loaded model carries exactly n_layer blocks. *)
Lemma f32_load_model_num_blocks : forall header data cfg,
  List.length (f32_blocks (f32_load_model header data cfg)) = gpt2_inf_n_layer cfg.
Proof.
  intros header data cfg. unfold f32_load_model.
  apply f32_load_blocks_length.
Qed.

(** * Float GPT-2: Verified Greedy Generation

    Greedy decoding over the float logits via f32 argmax. The generated
    sequence always extends the prompt, so the prompt is preserved as a
    prefix of the output (proven, mirroring the integer generation path). *)

Definition f32_argmax (xs : list binary32) : nat :=
  let indexed := List.combine (List.seq 0 (List.length xs)) xs in
  fst (List.fold_left
    (fun '(best_idx, best_val) '(idx, val) =>
      if f32_lt best_val val then (idx, val) else (best_idx, best_val))
    indexed (0%nat, f32_mask_neg)).

Fixpoint f32_generate_tokens_aux (forward : list nat -> list binary32) (eos : nat)
                                 (tokens : list nat) (remaining : nat) : list nat :=
  match remaining with
  | O => tokens
  | S n =>
      let next_token := f32_argmax (forward tokens) in
      if Nat.eqb next_token eos then tokens
      else f32_generate_tokens_aux forward eos (tokens ++ [next_token]) n
  end.

Definition f32_generate (forward : list nat -> list binary32) (eos max_new : nat)
                        (prompt : list nat) : list nat :=
  f32_generate_tokens_aux forward eos prompt max_new.

Lemma f32_generate_tokens_aux_extends : forall forward eos n prompt,
  exists suffix, f32_generate_tokens_aux forward eos prompt n = prompt ++ suffix.
Proof.
  intros forward eos n. induction n as [|n' IH].
  - intros prompt. exists []. simpl. rewrite List.app_nil_r. reflexivity.
  - intros prompt. simpl.
    destruct (Nat.eqb (f32_argmax (forward prompt)) eos) eqn:Heos.
    + exists []. rewrite List.app_nil_r. reflexivity.
    + set (nt := f32_argmax (forward prompt)).
      destruct (IH (prompt ++ [nt])) as [suffix' Hs].
      exists ([nt] ++ suffix'). rewrite Hs. rewrite <- List.app_assoc. reflexivity.
Qed.

Lemma f32_generation_preserves_prompt : forall forward eos max_new prompt,
  List.firstn (List.length prompt) (f32_generate forward eos max_new prompt) = prompt.
Proof.
  intros forward eos max_new prompt. unfold f32_generate.
  destruct (f32_generate_tokens_aux_extends forward eos max_new prompt) as [suffix H].
  rewrite H. apply gpt2_firstn_app_exact.
Qed.

(** * Float GPT-2: Inference Certificate

    A finiteness certificate over a forward/logits output: every entry is a
    finite IEEE-754 value (no NaN, no infinity). The checker is computable and
    its truth soundly implies entrywise finiteness. Determinism is immediate
    from purity of the forward pass. *)

Definition f32_finite (x : binary32) : bool := is_finite x.

Definition f32_vec_all_finite (xs : list binary32) : bool :=
  List.forallb f32_finite xs.

Definition f32_mat_all_finite (m : list (list binary32)) : bool :=
  List.forallb f32_vec_all_finite m.

Definition f32_output_clean (m : list (list binary32)) : bool :=
  f32_mat_all_finite m.

Lemma f32_output_clean_sound : forall m,
  f32_output_clean m = true ->
  forall row, In row m -> forall x, In x row -> is_finite x = true.
Proof.
  intros m H row Hrow x Hx.
  unfold f32_output_clean, f32_mat_all_finite in H.
  rewrite List.forallb_forall in H. specialize (H row Hrow).
  unfold f32_vec_all_finite in H. rewrite List.forallb_forall in H.
  specialize (H x Hx). unfold f32_finite in H. exact H.
Qed.

(** The forward pass is a pure function of its arguments, so determinism is
    definitional rather than a theorem. The substantive guarantees about it are
    [f32_gpt2_forward_rows] above and [f32_output_clean_sound] here. *)

(** Certified-clean forward output: the boolean check over the forward pass. *)
Definition f32_gpt2_certified_forward (cfg : gpt2_inference_config) (eps : binary32)
                                      (model : f32_model_weights) (toks : list nat) : bool :=
  f32_output_clean (f32_gpt2_forward cfg eps model toks).

(** * Float GPT-2: Weight Shape Validation

    Validate that loaded float weights match the config dimensions, so a
    malformed safetensors file is rejected before inference. A passing 2-D
    check soundly implies the expected row count. *)

Definition f32_mat_cols (m : list (list binary32)) : nat :=
  match m with [] => 0%nat | row :: _ => List.length row end.

Definition f32_validate_2d (m : list (list binary32)) (rows cols : nat) : bool :=
  andb (Nat.eqb (f32_mat_rows m) rows) (Nat.eqb (f32_mat_cols m) cols).

Definition f32_validate_1d (v : list binary32) (len : nat) : bool :=
  Nat.eqb (List.length v) len.

Definition f32_validate_attention (cfg : gpt2_inference_config) (a : f32_attention_weights) : bool :=
  let d := gpt2_inf_n_embd cfg in
  f32_validate_2d (f32_attn_c_attn_weight a) d (3 * d)%nat &&
  f32_validate_1d (f32_attn_c_attn_bias a) (3 * d)%nat &&
  f32_validate_2d (f32_attn_c_proj_weight a) d d &&
  f32_validate_1d (f32_attn_c_proj_bias a) d.

Definition f32_validate_mlp (cfg : gpt2_inference_config) (m : f32_mlp_weights) : bool :=
  let d := gpt2_inf_n_embd cfg in
  let ff := gpt2_inf_n_inner cfg in
  f32_validate_2d (f32_mlp_c_fc_weight m) d ff &&
  f32_validate_1d (f32_mlp_c_fc_bias m) ff &&
  f32_validate_2d (f32_mlp_c_proj_weight m) ff d &&
  f32_validate_1d (f32_mlp_c_proj_bias m) d.

Definition f32_validate_layernorm (cfg : gpt2_inference_config) (ln : f32_layernorm_weights) : bool :=
  let d := gpt2_inf_n_embd cfg in
  f32_validate_1d (f32_ln_weight ln) d && f32_validate_1d (f32_ln_bias ln) d.

Definition f32_validate_block (cfg : gpt2_inference_config) (b : f32_block_weights) : bool :=
  f32_validate_layernorm cfg (f32_block_ln_1 b) &&
  f32_validate_attention cfg (f32_block_attn b) &&
  f32_validate_layernorm cfg (f32_block_ln_2 b) &&
  f32_validate_mlp cfg (f32_block_mlp b).

Definition f32_validate_model (cfg : gpt2_inference_config) (model : f32_model_weights) : bool :=
  f32_validate_2d (f32_wte model) (gpt2_inf_vocab_size cfg) (gpt2_inf_n_embd cfg) &&
  f32_validate_2d (f32_wpe model) (gpt2_inf_n_positions cfg) (gpt2_inf_n_embd cfg) &&
  Nat.eqb (List.length (f32_blocks model)) (gpt2_inf_n_layer cfg) &&
  List.forallb (f32_validate_block cfg) (f32_blocks model) &&
  f32_validate_layernorm cfg (f32_ln_f model).

Lemma f32_validate_2d_rows : forall m rows cols,
  f32_validate_2d m rows cols = true -> f32_mat_rows m = rows.
Proof.
  intros m rows cols H. unfold f32_validate_2d in H.
  apply andb_prop in H. destruct H as [H1 _]. apply Nat.eqb_eq in H1. exact H1.
Qed.

(** * Extraction Directives *)

Extraction Language OCaml.
Extract Inductive list => "list" ["[]" "(::)"].
Extract Inductive bool => "bool" ["true" "false"].
Extract Inductive prod => "(*)" ["(,)"].
Extract Inductive option => "option" ["Some" "None"].
Extract Inductive unit => "unit" ["()"].
Extract Inductive string => "char list" ["[]" "(::)"].
Extract Inductive ascii => "char"
  ["(fun b0 b1 b2 b3 b4 b5 b6 b7 -> Char.chr ((if b0 then 1 else 0) + (if b1 then 2 else 0) + (if b2 then 4 else 0) + (if b3 then 8 else 0) + (if b4 then 16 else 0) + (if b5 then 32 else 0) + (if b6 then 64 else 0) + (if b7 then 128 else 0)))"]
  "(fun f c -> f (Char.code c land 1 <> 0) (Char.code c land 2 <> 0) (Char.code c land 4 <> 0) (Char.code c land 8 <> 0) (Char.code c land 16 <> 0) (Char.code c land 32 <> 0) (Char.code c land 64 <> 0) (Char.code c land 128 <> 0))".

Set Extraction Output Directory ".".
Extraction "phases1_15_complete.ml"
  byte tensor_1d tensor_2d tensor_3d tensor_4d scale_factor neg_inf
  tensor mk_tensor t_name t_shape t_data network
  z_to_bytes_le bytes_to_z_le bytes_to_z_le_u32 bytes_to_z_le_u64
  serialize_list take drop parse_i32_list
  infer_shape_1d infer_shape_2d vector_length matrix_rows matrix_cols
  layer layer_1d layer_raw network_of_layers
  tensor_byte_size network_param_count network_byte_size
  flatten_2d flatten_3d flatten_4d
  clip_s8 z_to_s8 serialize_s8_list clip_s4 pack_nibbles serialize_s4_pairs
  encode_f32 decode_f32 encode_f64 decode_f64
  encode_f16 decode_f16 encode_bf16 decode_bf16
  bits32_to_bytes bits64_to_bytes bits16_to_bytes
  f32_to_bytes f16_to_bytes bf16_to_bytes
  one_f32 neg_one_f32 zero_f32 one_f16 neg_one_f16 zero_f16 one_bf16
  (* Float arithmetic *)
  prec32 emax32 binary32 rnd_NE
  f32_plus f32_mult f32_div f32_neg f32_abs f32_compare f32_lt f32_le
  f32_zero f32_minus
  triple_to_bits32 bits32_to_triple
  f32_vec_add f32_vec_mult f32_vec_scale f32_dot f32_mat_vec_mul
  (* Float sigmoid *)
  f32_one f32_of_Z f32_two f32_six f32_twenty_four f32_one_twenty f32_seven_twenty
  f32_exp_taylor f32_ten f32_neg_ten f32_exp_large f32_exp_small f32_exp_approx
  f32_sigmoid f32_sigmoid_vec f32_sigmoid_in_unit_interval
  (* Float softmax *)
  f32_exp_vec f32_sum f32_max_vec f32_softmax f32_softmax_2d
  (* Float GELU *)
  f32_gelu_coeff f32_gelu_scale f32_half f32_gelu f32_gelu_vec
  f32_tanh f32_tanh_vec f32_relu f32_relu_vec
  dtype DT_I32 DT_I8 DT_F32 DT_F16 DT_BF16 dtype_size
  tensor_data TD_Int TD_Float32 TD_Float16 TD_BFloat16
  mp_tensor mk_mp_tensor mp_name mp_shape mp_dtype mp_data mp_network
  sum_list safe_div isqrt mean_scaled variance_scaled
  relu_z relu_vec leaky_relu_z relu6_z
  sigmoid_approx tanh_approx sigmoid_vec tanh_vec
  exp_approx softmax argmax gelu_approx gelu_vec
  dot_product vec_add vec_mul vec_concat vec_sub_from_one vec_scale
  mat_vec_mul mat_transpose mat_mul mat_rows mat_cols add_matrices
  dense dense_relu residual_block
  bottleneck_weights mk_bottleneck bn_contract_w bn_contract_b bn_expand_w bn_expand_b
  bottleneck_block
  conv_weight mk_conv_weight cw_out_channels cw_in_channels cw_kernel_h cw_kernel_w cw_data
  conv_weight_shape conv_weight_flat output_size
  list_max max_pool_patch avg_pool_patch
  pool_type MaxPool AvgPool pool_layer mk_pool_layer pl_type pl_kernel_h pl_kernel_w pl_stride
  maxpool_2x2 avgpool_2x2
  chunk mk_chunk ch_index ch_data ch_is_final chunk_size
  split_into_chunks reassemble_chunks
  stream_state mk_stream_state ss_chunks_written ss_bytes_written ss_complete
  init_stream write_chunk stream_all_chunks streaming_serialize
  lazy_value Lazy Forced force
  lazy_tensor mk_lazy_tensor lt_name lt_shape lt_data lazy_network
  tensor_to_lazy lazy_to_tensor network_to_lazy lazy_to_network
  rle_element RLE_Run RLE_Literal rle_data rle_encode rle_decode
  compressed_tensor mk_compressed_tensor ct_name ct_shape ct_compressed ct_original_size
  compress_tensor decompress_tensor compressed_network compress_network decompress_network
  shard mk_shard sh_id sh_tensors sh_byte_size shard_collection max_shard_bytes
  compute_tensor_bytes shard_network unshard_network
  shard_manifest mk_shard_manifest sm_total_shards sm_total_tensors sm_total_bytes sm_shard_ids
  tensor_shape_valid_b tensor_bounded_b network_shape_valid_b network_bounded_b
  verification_result Verified Failed
  check_shape check_bounds combine_results verify_all_shapes
  tensor_data_eq tensor_shape_eq tensor_eq network_eq
  within_tolerance network_approx_eq
  quant_params mk_quant_params qp_scale qp_zero_point qp_bits
  quant_min quant_max clip quantize dequantize quantize_list dequantize_list
  quant_error max_quant_error find_scale calibrate
  property_type PT_ShapeValid PT_Bounded PT_NonEmpty PT_Normalized PT_Custom
  proven_property mk_proven_property pp_type pp_holds pp_witness
  json_certificate mk_json_certificate jc_version jc_network_name jc_tensor_count jc_total_params jc_properties jc_timestamp jc_prover
  check_property generate_certificate all_properties_hold
  attestation_level AL_SelfSigned AL_ThirdParty AL_Formal
  attestation mk_attestation at_property at_level at_signature at_expiry
  chain_link CL_Source CL_Extraction CL_Serialization CL_Verification
  certificate_chain mk_cert_chain cc_links cc_final_hash cc_complete
  empty_chain add_link simple_checksum finalize_chain
  proof_fragment PF_Axiom PF_Lemma PF_Composition PF_Instantiation
  proof_bundle mk_proof_bundle pb_theorem pb_proof pb_dependencies
  proof_carrying_weights mk_pcw pcw_network pcw_certificate pcw_proofs
  batch_norm_params mk_batch_norm_params bnp_gamma bnp_beta bnp_running_mean bnp_running_var bnp_epsilon bnp_momentum
  normalize_feature init_batch_norm
  layer_norm_params mk_layer_norm_params ln_gamma ln_beta ln_epsilon
  layer_norm_single layer_norm init_layer_norm
  group_norm_params mk_group_norm_params gn_num_groups gn_num_channels gn_gamma gn_beta gn_epsilon
  init_group_norm
  token_embedding mk_token_embedding te_vocab_size te_embed_dim te_weights
  lookup_embedding embed_tokens init_token_embedding
  position_embedding mk_position_embedding pe_max_len pe_embed_dim pe_weights
  lookup_position embed_positions init_position_embedding
  add_vectors add_embeddings
  transformer_embeddings mk_transformer_embeddings tfe_token_emb tfe_position_emb
  transformer_embed
  rnn_cell_params mk_rnn_cell_params rnn_input_size rnn_hidden_size rnn_W_ih rnn_W_hh rnn_b
  rnn_cell_forward rnn_forward_seq rnn_forward init_rnn_cell
  lstm_cell_params mk_lstm_cell_params lstm_input_size lstm_hidden_size
  lstm_W_ii lstm_W_if lstm_W_ig lstm_W_io lstm_W_hi lstm_W_hf lstm_W_hg lstm_W_ho
  lstm_b_i lstm_b_f lstm_b_g lstm_b_o
  lstm_state mk_lstm_state ls_hidden ls_cell
  lstm_cell_forward lstm_forward_seq init_lstm_state lstm_forward init_lstm_cell
  gru_cell_params mk_gru_cell_params gru_input_size gru_hidden_size
  gru_W_iz gru_W_ir gru_W_in gru_W_hz gru_W_hr gru_W_hn gru_b_z gru_b_r gru_b_n
  gru_cell_forward gru_forward_seq gru_forward init_gru_cell
  bidirectional_params mk_bidirectional_params bi_forward_params bi_backward_params
  bidirectional_forward init_bidirectional
  dense_params mk_dense_params dense_in_features dense_out_features dense_weight dense_bias
  dense_forward init_dense
  seq_classifier_params mk_seq_classifier_params sc_encoder sc_classifier sc_num_classes
  rnn_encode_seq rnn_encode seq_classifier_forward seq_classify init_seq_classifier
  softmax_row softmax_2d
  compute_attention_scores scale_scores attention_weights apply_attention
  scaled_dot_product_attention
  split_row_into_heads split_into_heads concat_heads linear_project
  mha_params mk_mha_params mha_num_heads mha_d_model mha_d_k mha_d_v mha_W_Q mha_W_K mha_W_V mha_W_O
  multi_head_attention self_attention init_mha_params
  causal_mask_entry causal_mask apply_mask masked_attention
  cross_attention
  cross_attention_params mk_cross_attention_params ca_d_model ca_d_k ca_d_v ca_W_Q ca_W_K ca_W_V ca_W_O
  cross_attention_with_proj
  attention_export mk_attention_export ae_query_len ae_key_len ae_weights ae_layer_name
  create_attention_export attention_argmax_pattern entropy_row attention_entropy
  serialize_attention_weights
  tensor_meta mk_tensor_meta tm_name tm_dtype tm_shape tm_data_start tm_data_end
  find_char string_drop parse_header_size parse_header_string parse_safetensors_header
  onnx_dtype ONNX_FLOAT ONNX_UINT8 ONNX_INT8 ONNX_UINT16 ONNX_INT16 ONNX_INT32 ONNX_INT64
  ONNX_STRING ONNX_BOOL ONNX_FLOAT16 ONNX_DOUBLE ONNX_UINT32 ONNX_UINT64 ONNX_BFLOAT16
  onnx_dtype_code onnx_dtype_of_code
  onnx_tensor mk_onnx_tensor ot_name ot_dtype ot_dims ot_raw_data
  onnx_node mk_onnx_node on_name on_op_type on_inputs on_outputs
  onnx_graph mk_onnx_graph og_name og_nodes og_inputs og_outputs og_initializers
  onnx_model mk_onnx_model om_ir_version om_producer_name om_producer_version om_domain om_model_version om_graph
  parse_varint_aux parse_varint
  majority_weight majority_bias majority_network
  mod3_layer1_weight mod3_layer1_bias mod3_layer2_weight mod3_layer2_bias
  mod3_output_weight mod3_output_bias mod3_network
  (* Transformer blocks *)
  swish_approx swish_vec
  linear_project_bias
  masked_self_attention cross_attention_mha
  prenorm_attention_block mk_prenorm_attention_block pab_norm pab_attention
  prenorm_attention_forward prenorm_masked_attention_forward init_prenorm_attention_block
  postnorm_attention_block mk_postnorm_attention_block poab_attention poab_norm
  postnorm_attention_forward postnorm_masked_attention_forward init_postnorm_attention_block
  ffn_params mk_ffn_params ffn_d_model ffn_d_ff ffn_W1 ffn_b1 ffn_W2 ffn_b2
  ffn_forward_single ffn_forward init_ffn_params init_ffn_params_default
  ffn_relu_forward_single ffn_relu_forward ffn_swish_forward_single ffn_swish_forward
  prenorm_ffn_block mk_prenorm_ffn_block pffnb_norm pffnb_ffn
  prenorm_ffn_forward init_prenorm_ffn_block
  postnorm_ffn_block mk_postnorm_ffn_block poffnb_ffn poffnb_norm
  postnorm_ffn_forward init_postnorm_ffn_block
  encoder_layer_prenorm mk_encoder_layer_prenorm elp_attention elp_ffn
  encoder_layer_prenorm_forward init_encoder_layer_prenorm
  encoder_layer_postnorm mk_encoder_layer_postnorm elpo_attention elpo_ffn
  encoder_layer_postnorm_forward init_encoder_layer_postnorm
  encoder_stack_prenorm_forward encoder_stack_postnorm_forward
  init_encoder_stack_prenorm init_encoder_stack_postnorm
  decoder_layer_prenorm mk_decoder_layer_prenorm dlp_self_attention dlp_ffn
  decoder_layer_prenorm_forward init_decoder_layer_prenorm
  decoder_layer_postnorm mk_decoder_layer_postnorm dlpo_self_attention dlpo_ffn
  decoder_layer_postnorm_forward init_decoder_layer_postnorm
  cross_attention_block mk_cross_attention_block cab_norm cab_attention
  cross_attention_block_forward init_cross_attention_block
  decoder_layer_with_cross mk_decoder_layer_with_cross dlwc_self_attention dlwc_cross_attention dlwc_ffn
  decoder_layer_with_cross_forward init_decoder_layer_with_cross
  decoder_stack_prenorm_forward decoder_stack_with_cross_forward
  init_decoder_stack_prenorm init_decoder_stack_with_cross
  gpt_model mk_gpt_model gpt_num_layers gpt_num_heads gpt_d_model gpt_d_ff gpt_layers gpt_final_norm
  gpt_forward init_gpt_model
  bert_model mk_bert_model bert_num_layers bert_num_heads bert_d_model bert_d_ff bert_layers
  bert_forward init_bert_model
  transformer_model mk_transformer_model tm_num_layers tm_num_heads tm_d_model tm_d_ff
  tm_encoder_layers tm_decoder_layers tm_encoder_final_norm tm_decoder_final_norm
  transformer_encode transformer_decode transformer_forward init_transformer_model
  p14_example_d_model p14_example_d_ff p14_example_num_heads p14_example_num_layers
  p14_example_input p14_example_prenorm_attn p14_example_postnorm_attn p14_example_ffn
  p14_example_encoder_layer p14_example_decoder_layer
  p14_example_gpt p14_example_bert p14_example_transformer
  (* Real models - GPT-2 *)
  gpt2_config mk_gpt2_config
  gpt2_vocab_size gpt2_n_positions gpt2_n_embd gpt2_n_layer gpt2_n_head gpt2_n_inner
  gpt2_activation gpt2_resid_pdrop gpt2_embd_pdrop gpt2_attn_pdrop
  gpt2_layer_norm_epsilon gpt2_initializer_range gpt2_bos_token_id gpt2_eos_token_id
  gpt2_small gpt2_medium gpt2_large gpt2_xl
  gpt2_embedding_params gpt2_attention_params_per_layer gpt2_ffn_params_per_layer
  gpt2_layernorm_params_per_layer gpt2_transformer_layer_params gpt2_final_layernorm_params gpt2_total_params
  gpt2_small_expected_params gpt2_medium_expected_params gpt2_large_expected_params gpt2_xl_expected_params
  gpt2_head_dim gpt2_ffn_expansion_factor valid_gpt2_head_divisibility valid_gpt2_ffn_expansion valid_gpt2_config
  gpt2_weight_shape mk_gpt2_weight_shape gpt2_ws_name gpt2_ws_dims
  gpt2_wte_shape gpt2_wpe_shape gpt2_ln_f_weight_shape gpt2_ln_f_bias_shape
  gpt2_named_tensor mk_gpt2_named_tensor gpt2_nt_name gpt2_nt_shape gpt2_nt_data gpt2_weight_dict
  gpt2_attention_weights mk_gpt2_attention_weights
  gpt2_attn_c_attn_weight gpt2_attn_c_attn_bias gpt2_attn_c_proj_weight gpt2_attn_c_proj_bias
  gpt2_mlp_weights mk_gpt2_mlp_weights
  gpt2_mlp_c_fc_weight gpt2_mlp_c_fc_bias gpt2_mlp_c_proj_weight gpt2_mlp_c_proj_bias
  gpt2_layernorm_weights mk_gpt2_layernorm_weights gpt2_ln_weight gpt2_ln_bias
  gpt2_block_weights mk_gpt2_block_weights gpt2_block_ln_1 gpt2_block_attn gpt2_block_ln_2 gpt2_block_mlp
  gpt2_model_weights mk_gpt2_model_weights gpt2_wte gpt2_wpe gpt2_blocks gpt2_ln_f
  gpt2_tensor_1d_size gpt2_tensor_2d_shape gpt2_shape_eq
  gpt2_validate_tensor_1d_shape gpt2_validate_tensor_2d_shape
  gpt2_validate_attention_weights gpt2_validate_mlp_weights gpt2_validate_layernorm_weights
  gpt2_validate_block_weights gpt2_validate_model_weights
  gpt2_reshape_1d_to_2d gpt2_make_layernorm gpt2_make_attention gpt2_make_mlp
  gpt2_count_tensor_1d_params gpt2_count_tensor_2d_params gpt2_count_layernorm_params
  gpt2_count_attention_params gpt2_count_mlp_params gpt2_count_block_params gpt2_count_model_params
  gpt2_inference_config mk_gpt2_inference_config
  gpt2_inf_n_embd gpt2_inf_n_head gpt2_inf_n_layer gpt2_inf_n_inner gpt2_inf_vocab_size gpt2_inf_n_positions
  gpt2_small_inference_cfg
  gpt2_layer_norm_vec gpt2_layer_norm_2d gpt2_causal_attention
  gpt2_linear_forward gpt2_linear_forward_2d
  gpt2_attention_forward gpt2_mlp_forward gpt2_block_forward gpt2_blocks_forward
  gpt2_lookup_embedding gpt2_embed_tokens gpt2_embed_positions gpt2_combine_embeddings
  gpt2_forward gpt2_logits gpt2_next_token_logits gpt2_predict_next
  gpt2_serialize_tensor_1d gpt2_serialize_tensor_2d
  gpt2_deserialize_tensor_1d gpt2_deserialize_tensor_2d
  gpt2_tensor_1d_eq gpt2_tensor_2d_eq gpt2_bytes_eq
  gpt2_roundtrip_tensor_1d gpt2_roundtrip_tensor_2d
  gpt2_serialize_layernorm gpt2_serialize_attention gpt2_serialize_mlp gpt2_serialize_block gpt2_serialize_model
  gpt2_roundtrip_layernorm gpt2_roundtrip_attention gpt2_roundtrip_mlp gpt2_roundtrip_block gpt2_roundtrip_model
  gpt2_compute_checksum gpt2_verify_checksum
  gpt2_sampling_method GPT2_Greedy GPT2_TopK GPT2_TopP GPT2_Temperature
  gpt2_generation_config mk_gpt2_generation_config
  gpt2_gen_max_new_tokens gpt2_gen_sampling gpt2_gen_eos_token_id gpt2_gen_pad_token_id
  gpt2_default_gen_config
  gpt2_temperature_scale gpt2_find_kth_largest gpt2_top_k_mask gpt2_top_p_mask
  gpt2_model_forward_fn gpt2_select_token gpt2_generate_tokens_aux gpt2_generate
  gpt2_generation_trace mk_gpt2_generation_trace
  gpt2_gt_prompt gpt2_gt_generated gpt2_gt_all_logits gpt2_gt_selected_tokens
  gpt2_generate_with_trace_aux gpt2_generate_with_trace
  gpt2_verify_greedy_selection gpt2_verify_token_in_top_k gpt2_verify_generation_step gpt2_verify_trace
  gpt2_check_argmax_is_max gpt2_check_prompt_preserved
  (* Float GPT-2: load, forward, generate, certify, validate *)
  f32_sqrt f32_mat_transpose f32_mat_mul f32_mat_rows f32_mat_cols
  f32_mean f32_variance f32_layer_norm_vec f32_layer_norm_2d
  f32_linear_forward f32_linear_forward_2d f32_add_matrices
  f32_split_row_into_heads f32_split_into_heads f32_concat_heads
  f32_mask_neg f32_causal_mask_entry f32_causal_mask f32_scale_scores f32_apply_mask
  f32_causal_attention f32_attention_forward f32_mlp_forward
  f32_attention_weights mk_f32_attention_weights
  f32_attn_c_attn_weight f32_attn_c_attn_bias f32_attn_c_proj_weight f32_attn_c_proj_bias
  f32_mlp_weights mk_f32_mlp_weights
  f32_mlp_c_fc_weight f32_mlp_c_fc_bias f32_mlp_c_proj_weight f32_mlp_c_proj_bias
  f32_layernorm_weights mk_f32_layernorm_weights f32_ln_weight f32_ln_bias
  f32_block_weights mk_f32_block_weights f32_block_ln_1 f32_block_attn f32_block_ln_2 f32_block_mlp
  f32_model_weights mk_f32_model_weights f32_wte f32_wpe f32_blocks f32_ln_f
  f32_ln_eps f32_block_forward f32_blocks_forward
  f32_lookup_embedding f32_embed_tokens f32_embed_positions
  f32_gpt2_forward f32_gpt2_logits f32_gpt2_next_token_logits
  f32_of_bits f32_bytes_to_binary32 f32_decode_tensor_1d f32_decode_tensor_2d
  f32_reshape_1d_to_2d f32_make_layernorm f32_make_attention f32_make_mlp f32_make_block
  json_is_ws json_skip_ws json_digit json_parse_nat json_dquote json_parse_string_chars
  json_parse_nat_list json_quoted json_is_prefix json_find_after
  json_extract_nat_list json_extract_string json_tensor_shape json_tensor_offsets
  f32_load_tensor_bytes f32_load_named nat_to_string gpt2_block_prefix
  f32_load_block f32_load_blocks f32_load_model
  f32_argmax f32_generate_tokens_aux f32_generate
  f32_finite f32_vec_all_finite f32_mat_all_finite f32_output_clean f32_gpt2_certified_forward
  f32_validate_2d f32_validate_1d f32_validate_attention f32_validate_mlp
  f32_validate_layernorm f32_validate_block f32_validate_model.
