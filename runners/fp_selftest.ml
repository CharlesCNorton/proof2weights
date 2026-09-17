(* fp_selftest.ml - the assumptions the native build makes about the host.

   Narrow.v proves that narrowing the binary64 result of an operation on
   widened binary32 operands gives the binary32 result Flocq specifies, for
   every input. That proof is about Flocq's operations. The native extraction
   replaces them by the host's, so two things have to be true of the host for
   the proof to transfer:

     - its float arithmetic is IEEE-754 binary64, round-to-nearest-even, with
       no excess precision and no contraction of a multiply and an add into a
       fused multiply-add;
     - Int32.bits_of_float rounds to nearest-even, overflows to an infinity,
       keeps subnormals, and propagates NaNs and signed zeros.

   This program decides both. It exits non-zero on the first failure and is the
   first thing the referee target runs; a native number produced on a host
   where it fails means nothing. Everything here is a property of the host and
   of the extraction directives, not of the development. *)

open Phases1_15_native

let failures = ref 0

let check name ok =
  if ok then Printf.printf "  ok    %s\n" name
  else begin incr failures; Printf.printf "  FAIL  %s\n" name end

let bits x = Int32.bits_of_float x
let narrow x = Int32.float_of_bits (Int32.bits_of_float x)

(* Bit-level equality, so that the two zeros and the NaNs are distinguished. *)
let same a b = Int64.bits_of_float a = Int64.bits_of_float b
let same32 a b = Int32.bits_of_float a = Int32.bits_of_float b

let p2 k = ldexp 1.0 k

let () =
  Printf.printf "host arithmetic\n";

  (* No excess precision. With 80-bit registers the first sum is exact and the
     second lands a ulp above one; in binary64 both round back to one. *)
  let a = 1.0 +. p2 (-53) in
  check "binary64 significand, no excess precision" (same (a +. p2 (-53)) 1.0);

  (* No fused multiply-add. The exact product is the midpoint below one, which
     rounds to one; a contracted fma would keep the -2^-54 and the difference
     would not vanish. *)
  let u = 1.0 +. p2 (-27) and v = 1.0 -. p2 (-27) in
  check "no multiply-add contraction" (u *. v -. 1.0 = 0.0);

  (* Round-to-nearest, ties to even, at binary64. *)
  check "binary64 ties to even" (same (1.0 +. p2 (-53)) 1.0);
  check "binary64 ties to even, odd neighbour"
    (same (1.0 +. 3.0 *. p2 (-53)) (1.0 +. p2 (-51)));

  Printf.printf "Int32 narrowing\n";

  (* Ties to even at binary32: 1 + 2^-24 is the midpoint between 1 and the
     next binary32, and rounds down; 1 + 3*2^-24 rounds up. *)
  check "narrow ties to even, down" (narrow (1.0 +. p2 (-24)) = 1.0);
  check "narrow ties to even, up"
    (narrow (1.0 +. 3.0 *. p2 (-24)) = 1.0 +. p2 (-22));

  (* Subnormals survive; they are not flushed to zero. *)
  check "smallest subnormal survives" (narrow (p2 (-149)) = p2 (-149));
  check "half the smallest subnormal is a tie to even" (narrow (p2 (-150)) = 0.0);
  check "just above that tie rounds up"
    (narrow (p2 (-150) +. p2 (-160)) = p2 (-149));

  (* Overflow gives an infinity of the operand's sign. *)
  check "overflow to +inf" (narrow 1e39 = infinity);
  check "overflow to -inf" (narrow (-1e39) = neg_infinity);
  check "largest binary32 survives"
    (narrow (ldexp (2.0 -. p2 (-23)) 127) = ldexp (2.0 -. p2 (-23)) 127);

  (* NaNs and signed zeros. *)
  check "NaN narrows to a NaN" (Float.is_nan (narrow nan));
  check "positive zero keeps its sign" (same32 (narrow 0.0) 0.0);
  check "negative zero keeps its sign" (same32 (narrow (-0.0)) (-0.0));

  Printf.printf "extracted operations\n";

  (* The extracted primitives, on the same boundaries. The values below are
     binary32 values held in an OCaml float, which is what every extracted
     operation receives. *)
  let mx = ldexp (2.0 -. p2 (-23)) 127 in
  check "plus overflows to an infinity" (f32_plus mx mx = infinity);
  check "plus overflows with the sign" (f32_plus (-.mx) (-.mx) = neg_infinity);
  check "minus of equals is positive zero" (same32 (f32_minus mx mx) 0.0);
  check "sum of negative zeros is negative zero"
    (same32 (f32_plus (-0.0) (-0.0)) (-0.0));
  check "sum of opposite zeros is positive zero"
    (same32 (f32_plus 0.0 (-0.0)) 0.0);
  check "product of a zero and an infinity is a NaN"
    (Float.is_nan (f32_mult 0.0 infinity));
  check "quotient of zeros is a NaN" (Float.is_nan (f32_div 0.0 0.0));
  check "division by zero is an infinity" (f32_div 1.0 0.0 = infinity);
  check "square root of a negative is a NaN" (Float.is_nan (f32_sqrt (-1.0)));
  check "square root of a negative zero is a negative zero"
    (same32 (f32_sqrt (-0.0)) (-0.0));
  check "NaN propagates through plus" (Float.is_nan (f32_plus nan 1.0));
  check "underflow to zero keeps the sign"
    (same32 (f32_mult (-. p2 (-149)) (p2 (-2))) (-0.0));

  (* A binary32 operand is unchanged by narrowing, which is narrow_widen. *)
  let sample =
    [ 0.0; -0.0; 1.0; -1.0; p2 (-149); -. p2 (-149); p2 (-126); mx; -.mx;
      Int32.float_of_bits 0x3fc00000l; Int32.float_of_bits 0xbf800001l ] in
  check "narrowing a binary32 is the identity"
    (List.for_all (fun x -> same32 (narrow x) x) sample);

  (* Every operation returns a binary32: narrowing its result changes nothing. *)
  let pairs =
    List.concat_map (fun a -> List.map (fun b -> (a, b)) sample) sample in
  let closed op =
    List.for_all (fun (a, b) -> let r = op a b in same32 (narrow r) r) pairs in
  check "plus returns a binary32" (closed f32_plus);
  check "minus returns a binary32" (closed f32_minus);
  check "mult returns a binary32" (closed f32_mult);
  check "div returns a binary32" (closed f32_div);
  check "sqrt returns a binary32"
    (List.for_all (fun x -> let r = f32_sqrt x in same32 (narrow r) r) sample);

  ignore (bits 0.0);
  if !failures = 0 then begin
    Printf.printf "\nfp_selftest: all checks passed\n"; exit 0
  end else begin
    Printf.printf "\nfp_selftest: %d check(s) failed\n" !failures; exit 1
  end
