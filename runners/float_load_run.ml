(* float_load_run.ml - load a real .safetensors via the extracted verified
   loader and run the float forward; print logits for comparison with the
   numpy reference. nat extracts to int; positive/Z stay inductive, so file
   bytes (byte = Z) are converted to inductive Z on the way in and a result
   binary_float's inductive mantissa/exponent are read back to int on the
   way out. *)

open Phases1_15_complete

let rec pos_of_int n =
  if n <= 1 then XH
  else if n land 1 = 0 then XO (pos_of_int (n / 2)) else XI (pos_of_int (n / 2))
let z_of_int n = if n = 0 then Z0 else if n > 0 then Zpos (pos_of_int n) else Zneg (pos_of_int (-n))
let rec int_of_pos = function XH -> 1 | XO p -> 2 * int_of_pos p | XI p -> 2 * int_of_pos p + 1
let int_of_z = function Z0 -> 0 | Zpos p -> int_of_pos p | Zneg p -> - (int_of_pos p)

(* binary32 -> OCaml float, reconstructing (-1)^s * m * 2^e from Flocq's rep. *)
let b32_to_float = function
  | B754_zero _ -> 0.0
  | B754_infinity s -> if s then neg_infinity else infinity
  | B754_nan -> nan
  | B754_finite (s, m, e) ->
      (if s then -1.0 else 1.0) *. float_of_int (int_of_pos m) *. (2.0 ** float_of_int (int_of_z e))

let read_bytes path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in
  really_input ic b 0 n;
  close_in ic;
  List.init n (fun i -> z_of_int (Char.code (Bytes.get b i)))

let cfg = { gpt2_inf_n_embd = 4; gpt2_inf_n_head = 2; gpt2_inf_n_layer = 1;
            gpt2_inf_n_inner = 8; gpt2_inf_vocab_size = 5; gpt2_inf_n_positions = 4 }

let () =
  let data = read_bytes "tiny_gpt2.safetensors" in
  let (size, after) = parse_header_size data in
  let (header, body) = parse_header_string size after in
  let model = f32_load_model header body cfg in
  let logits = f32_gpt2_logits cfg f32_ln_eps model [0; 1; 2] in
  List.iter (fun row ->
    print_string "  ";
    List.iter (fun x -> Printf.printf "%.6f " (b32_to_float x)) row;
    print_newline ()) logits
