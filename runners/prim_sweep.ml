(* prim_sweep.ml - evaluate the extracted elementary functions on a sweep of
   inputs, against the INDUCTIVE extraction.

   The differential harness in scripts/experiment_arch.py never calls
   f32_sin or f32_cos: it hands the rotary tables to both sides as random
   weight matrices. Neither of the two defects in the trigonometric path was
   therefore reachable by any sweep row, which is how a sign flip survived 368
   samples with no next-token disagreement. This driver closes that gap by
   evaluating the primitives themselves.

   Input is one unsigned 32-bit little-endian bit pattern per line on stdin.
   Output is one line per input: the result of each primitive, printed as the
   exact binary64 rendering of the binary32 value, so the comparison in
   scripts/prim_check.py is not mediated by a reimplementation.

   usage: prim_sweep <sin|cos|exp|log|softplus|sigmoid|sqrt> < bits.txt *)

open Qwen_inductive

let rec pos_of_int n =
  if n <= 1 then XH
  else if n land 1 = 0 then XO (pos_of_int (n / 2)) else XI (pos_of_int (n / 2))
let z_of_int n =
  if n = 0 then Z0 else if n > 0 then Zpos (pos_of_int n) else Zneg (pos_of_int (-n))
let rec int_of_pos = function XH -> 1 | XO p -> 2 * int_of_pos p | XI p -> 2 * int_of_pos p + 1
let int_of_z = function Z0 -> 0 | Zpos p -> int_of_pos p | Zneg p -> - (int_of_pos p)

let b2f = function
  | B754_zero s -> if s then -0.0 else 0.0
  | B754_infinity s -> if s then neg_infinity else infinity
  | B754_nan -> nan
  | B754_finite (s, m, e) ->
      (if s then -1.0 else 1.0) *. float_of_int (int_of_pos m)
      *. (2.0 ** float_of_int (int_of_z e))

(* a 32-bit pattern to the binary32 it denotes, through the verified decoder *)
let of_bits u =
  f32_bytes_to_binary32
    [ z_of_int (u land 0xff); z_of_int ((u lsr 8) land 0xff);
      z_of_int ((u lsr 16) land 0xff); z_of_int ((u lsr 24) land 0xff) ]

let () =
  let which = Sys.argv.(1) in
  let f =
    match which with
    | "sin"      -> f32_sin
    | "cos"      -> f32_cos
    | "exp"      -> f32_exp_approx
    | "log"      -> f32_log_unit
    | "softplus" -> f32_softplus
    | "sigmoid"  -> f32_sigmoid
    | "sqrt"     -> f32_sqrt
    | s -> failwith ("unknown primitive " ^ s) in
  (try
     while true do
       let line = String.trim (input_line stdin) in
       if line <> "" then
         Printf.printf "%.17g\n" (b2f (f (of_bits (int_of_string line))))
     done
   with End_of_file -> ());
  flush stdout
