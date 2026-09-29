(** * Native extraction of the enclosure passes

    The GPT-2 pass of EncGPT2.v, the Llama pass of EncLlama.v and the Qwen3.5
    pass of EncQwen.v in the enclosure arithmetic of Enclose.v and EncExt.v,
    together with the checkpoint loaders and the rotary tables, extracted with
    [Z] as Zarith's integers, so the fixed-point endpoints and CoqInterval's
    floats are GMP numbers, and with binary32 as the host's float, as Extract.v
    maps it. The directives beyond Extract.v's: [b32_decomp] reads the mantissa
    and exponent of a binary32 through [Int32.bits_of_float]; [Z.pow] and
    [Z.sqrt] are Zarith's; CoqInterval's digit count and shifts on its radix-2
    mantissas are single Zarith operations, through the alias [Zz] of Zarith's
    [Z] (runners/zz.ml), since the extracted code has a module [Z] of its own;
    and Flocq's [Zfast_div_eucl] is [Z.div_eucl]. CoqInterval's module
    signatures bring the real-number library into the extracted code, where
    nothing calls it; the one axiom it needs is realized as a function that
    fails when called. *)

From Stdlib Require Import ZArith.
Require Import Phases1_15_complete.
Require Import Pretransposed.
Require Import Loadpre.
Require Import RunErr.
Require Import Enclose.
Require Import EncGPT2.
Require Import EncExt.
Require Import EncLlama.
Require Import EncQwen.
From Flocq Require Import IEEE754.BinarySingleNaN.
From Stdlib Require Import ExtrOcamlBasic.
From Stdlib Require Import ExtrOcamlNatInt.
From Stdlib Require Import ExtrOcamlZBigInt.

Extract Inductive binary_float => "float"
  [ "(fun s -> if s then (-0.0) else 0.0)"
    "(fun s -> if s then neg_infinity else infinity)"
    "nan"
    "(fun (s, m, e) -> let v = ldexp (Big_int_Z.float_of_big_int m) (Big_int_Z.int_of_big_int e) in if s then (-. v) else v)" ]
  "(fun _ _ _ _ _ -> failwith ""binary_float match in native build"")".

Extract Constant f32_plus => "(fun a b -> Int32.float_of_bits (Int32.bits_of_float (a +. b)))".
Extract Constant f32_mult => "(fun a b -> Int32.float_of_bits (Int32.bits_of_float (a *. b)))".
Extract Constant f32_div  => "(fun a b -> Int32.float_of_bits (Int32.bits_of_float (a /. b)))".
Extract Constant f32_sqrt => "(fun a -> Int32.float_of_bits (Int32.bits_of_float (sqrt a)))".
Extract Constant f32_neg  => "(fun a -> (-. a))".
Extract Constant f32_abs  => "abs_float".
Extract Constant f32_zero => "0.0".
Extract Constant f32_one  => "1.0".
Extract Constant f32_of_Z => "(fun n -> Int32.float_of_bits (Int32.bits_of_float (Big_int_Z.float_of_big_int n)))".
Extract Constant f32_lt   => "(fun (a:float) (b:float) -> a < b)".
Extract Constant f32_le   => "(fun (a:float) (b:float) -> a <= b)".

Extract Constant f32_bytes_to_binary32 => "(fun bs -> match bs with b0 :: b1 :: b2 :: b3 :: _ -> let g = Big_int_Z.int_of_big_int in Int32.float_of_bits (Int32.logor (Int32.of_int (g b0)) (Int32.logor (Int32.shift_left (Int32.of_int (g b1)) 8) (Int32.logor (Int32.shift_left (Int32.of_int (g b2)) 16) (Int32.shift_left (Int32.of_int (g b3)) 24)))) | _ -> 0.0)".

Extract Constant b32_decomp => "(fun x -> let bi = (Int32.to_int (Int32.bits_of_float x)) land 0xFFFFFFFF in let s = bi lsr 31 and ex = (bi lsr 23) land 0xFF and fr = bi land 0x7FFFFF in let z = Big_int_Z.big_int_of_int in if ex = 0xFF || (ex = 0 && fr = 0) then (z 0, z 0) else if ex = 0 then (z (if s = 1 then - fr else fr), z (-149)) else (let m = fr lor 0x800000 in (z (if s = 1 then - m else m), z (ex - 150))))".

From Stdlib Require ClassicalDedekindReals.
Extract Constant ClassicalDedekindReals.sig_forall_dec =>
  "(fun _ -> failwith ""real-number comparison in the enclosure pass"")".

(** CoqInterval's radix-2 mantissas on [Z]: digit counts and shifts in one
    Zarith operation rather than one bit at a time, each returning what the
    definition returns, the rounding position included. *)
From Interval Require Import Specific_stdz.
Extract Constant StdZRadix2.mantissa_digits =>
  "(fun m -> if Big_int_Z.le_big_int m Big_int_Z.unit_big_int then Big_int_Z.unit_big_int else Big_int_Z.big_int_of_int (Zz.numbits m))".
Extract Constant StdZRadix2.mantissa_shl =>
  "(fun m d -> if Big_int_Z.sign_big_int d > 0 then Big_int_Z.shift_left_big_int m (Big_int_Z.int_of_big_int d) else Big_int_Z.unit_big_int)".
Extract Constant StdZRadix2.mantissa_shr =>
  "(fun m d pos -> if Big_int_Z.sign_big_int d <= 0 then (Big_int_Z.unit_big_int, Pos_Eq) else let dd = Big_int_Z.int_of_big_int d in if Zz.numbits m <= dd then (Big_int_Z.unit_big_int, Pos_Eq) else let q = Big_int_Z.shift_right_big_int m dd in let top = Big_int_Z.sign_big_int (Big_int_Z.extract_big_int m (dd - 1) 1) <> 0 in let low0 = dd = 1 || Big_int_Z.sign_big_int (Big_int_Z.extract_big_int m 0 (dd - 1)) = 0 in let eqb = low0 && pos = Pos_Eq in (q, if top then (if eqb then Pos_Mi else Pos_Up) else (if eqb then Pos_Eq else Pos_Lo)))".
Extract Constant StdZRadix2.mantissa_shrp_aux =>
  "(fun m d -> if Big_int_Z.eq_big_int m (Big_int_Z.shift_left_big_int Big_int_Z.unit_big_int (Big_int_Z.int_of_big_int d - 1)) then Pos_Mi else Pos_Up)".

(** Flocq's [Zfast_div_eucl], through which every division of CoqInterval's
    floats passes, is [Z.div_eucl] ([Zfast_div_eucl_correct]) and is given
    ExtrOcamlZBigInt's realization of [Z.div_eucl]. *)
From Flocq Require Import Core.Zaux.
Extract Constant Zfast_div_eucl => "Big_int_Z.(fun x y ->
  match sign_big_int y with
  | 0 -> (zero_big_int, x)
  | 1 -> quomod_big_int x y
  | _ -> let (q, r) = quomod_big_int (add_int_big_int (-1) x) y in
          (add_int_big_int (-1) q, add_big_int (add_int_big_int 1 y) r))".

Extract Constant Z.pow => "(fun b e -> if Big_int_Z.sign_big_int e < 0 then Big_int_Z.zero_big_int else Big_int_Z.power_big_int_positive_int b (Big_int_Z.int_of_big_int e))".
Extract Constant Z.sqrt => "(fun n -> if Big_int_Z.sign_big_int n <= 0 then Big_int_Z.zero_big_int else Big_int_Z.sqrt_big_int n)".

Set Extraction Output Directory ".".

Extraction "enc_native.ml"
  f32_load_model_pre json_tensor_offsets f32_ln_eps
  enc_ops enc_const g_gpt2_logits_pre gmodel_map to_gmodel
  f32_load_llama f32_load_qwen f32_rope_cos f32_rope_sin f32_div f32_of_Z f32_one
  enc_ln enc_abs g_llama_logits glmodel_map to_glmodel g_qwen_logits gqmodel_map to_gqmodel.
