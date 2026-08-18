(* ref_logits.ml - canonical IEEE-754 reference logits from the extracted
   verified GPT-2. nat extracts to native int (indices, dims, tokens);
   positive/Z stay inductive (exact float arithmetic). byte = Z, so file
   bytes are converted to inductive Z at the boundary.
   usage: ref_logits <safetensors> <n_layer> <comma-separated token ids>
                     [n_embd n_head n_inner vocab n_positions] *)

open Phases1_15_complete

let rec pos_of_int n =
  if n <= 1 then XH
  else if n land 1 = 0 then XO (pos_of_int (n / 2)) else XI (pos_of_int (n / 2))
let z_of_int n = if n = 0 then Z0 else if n > 0 then Zpos (pos_of_int n) else Zneg (pos_of_int (-n))
let rec int_of_pos = function XH -> 1 | XO p -> 2 * int_of_pos p | XI p -> 2 * int_of_pos p + 1
let int_of_z = function Z0 -> 0 | Zpos p -> int_of_pos p | Zneg p -> - (int_of_pos p)

let b2f = function
  | B754_zero _ -> 0.0
  | B754_infinity s -> if s then neg_infinity else infinity
  | B754_nan -> nan
  | B754_finite (s, m, e) ->
      (if s then -1.0 else 1.0) *. float_of_int (int_of_pos m) *. (2.0 ** float_of_int (int_of_z e))

let read_bytes path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in really_input ic b 0 n; close_in ic;
  List.init n (fun i -> z_of_int (Char.code (Bytes.get b i)))

(* Optional dimension argument, defaulting to the depth sweep's model. *)
let arg i d = if Array.length Sys.argv > i then int_of_string Sys.argv.(i) else d

let () =
  let path = Sys.argv.(1) in
  let n_layer = int_of_string Sys.argv.(2) in
  let toks = List.map int_of_string (String.split_on_char ',' Sys.argv.(3)) in
  let cfg = { gpt2_inf_n_embd = arg 4 8; gpt2_inf_n_head = arg 5 2;
              gpt2_inf_n_layer = n_layer; gpt2_inf_n_inner = arg 6 32;
              gpt2_inf_vocab_size = arg 7 16; gpt2_inf_n_positions = arg 8 16 } in
  let data = read_bytes path in
  let (size, after) = parse_header_size data in
  let (header, body) = parse_header_string size after in
  let model = f32_load_model header body cfg in
  let logits = f32_gpt2_logits cfg f32_ln_eps model toks in
  List.iter (fun row ->
    print_string (String.concat " " (List.map (fun x -> Printf.sprintf "%.9g" (b2f x)) row));
    print_newline ()) logits
