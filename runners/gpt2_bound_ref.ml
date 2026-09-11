(* gpt2_bound_ref.ml - the annotated GPT-2 forward of RunErr.v against the
   inductive extraction, on a model loaded through the verified list-based
   loader. It prints one row per position, each entry the binary32 logit and its
   bound as value:bound; the theorem is gpt2_logits_bounded. With P2W_CHECK=1 it
   also computes f32_gpt2_logits and reports whether the values agree.

   usage: gpt2_bound_ref <safetensors> <n_layer> <tok,...>
                         [n_embd n_head n_inner vocab n_positions] *)

open Runerr_inductive

let rec pos_of_int n =
  if n <= 1 then XH
  else if n land 1 = 0 then XO (pos_of_int (n / 2)) else XI (pos_of_int (n / 2))
let z_of_int n = if n = 0 then Z0 else if n > 0 then Zpos (pos_of_int n) else Zneg (pos_of_int (-n))
let rec int_of_pos = function XH -> 1 | XO p -> 2 * int_of_pos p | XI p -> 2 * int_of_pos p + 1
let int_of_z = function Z0 -> 0 | Zpos p -> int_of_pos p | Zneg p -> - (int_of_pos p)

(* the value of an inductive binary_float of either format *)
let b2f = function
  | B754_zero s -> if s then (-0.0) else 0.0
  | B754_infinity s -> if s then neg_infinity else infinity
  | B754_nan -> nan
  | B754_finite (s, m, e) ->
      let v = ldexp (float_of_int (int_of_pos m)) (int_of_z e) in
      if s then (-. v) else v

let read_bytes path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in really_input ic b 0 n; close_in ic;
  List.init n (fun i -> z_of_int (Char.code (Bytes.get b i)))

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
  let out = g_gpt2_logits ann_ops cfg (ann_const f32_ln_eps) (ann_model model) toks in
  if Sys.getenv_opt "P2W_CHECK" = Some "1" then begin
    let plain = f32_gpt2_logits cfg f32_ln_eps model toks in
    let same = List.for_all2 (List.for_all2 (fun (v, _) x -> b2f v = b2f x)) out plain in
    Printf.printf "values equal to f32_gpt2_logits: %b\n" same
  end;
  List.iter (fun row ->
    print_string (String.concat " "
      (List.map (fun (v, d) -> Printf.sprintf "%.9g:%.3e" (b2f v) (b2f d)) row));
    print_newline ()) out
