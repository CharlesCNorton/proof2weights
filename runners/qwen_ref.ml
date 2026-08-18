(* qwen_ref.ml - canonical IEEE-754 reference logits from the verified Qwen3.5
   forward, against the INDUCTIVE extraction.

   qwen_talk_native.ml composes the extracted primitives itself and runs against
   the native build, where binary32 is the host's hardware float. This driver
   instead calls the Rocq-level composition directly, f32_qwen_wrap over
   f32_qwen_delta_mix and f32_qwen_attn_mix inside f32_qwen_forward, against the
   inductive extraction, where binary32 is Flocq's binary_float and Z is its
   inductive datatype. There is no trusted floating-point boundary at all here.
   That costs microseconds per operation, so the configuration is small.

   This is the composition the error bound in Float_error.v is stated about, so
   running it is what shows those definitions compute.

   Weights arrive as raw binary32 bit patterns, one tensor per line

     <name> <rows> <cols> <rows*cols unsigned 32-bit ints>

   so the values the reference and the numpy mirror see are bit-identical, and
   the decode itself goes through the verified f32_bytes_to_binary32.

   usage: qwen_ref <weights> d nh nkv hd rd ff vocab nl lnh lhd ck <kinds> <ids> *)

open Qwen_inductive

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
      (if s then -1.0 else 1.0) *. float_of_int (int_of_pos m)
      *. (2.0 ** float_of_int (int_of_z e))

let of_bits (u : int) =
  f32_bytes_to_binary32
    [ z_of_int (u land 0xff); z_of_int ((u lsr 8) land 0xff);
      z_of_int ((u lsr 16) land 0xff); z_of_int ((u lsr 24) land 0xff) ]

let read_tensors path =
  let ic = open_in path in
  let tbl = Hashtbl.create 64 in
  (try
    while true do
      let line = input_line ic in
      if String.trim line <> "" then begin
        let parts = List.filter (fun s -> s <> "") (String.split_on_char ' ' line) in
        match parts with
        | name :: rs :: cs :: vals ->
            let r = int_of_string rs and c = int_of_string cs in
            let v = Array.of_list (List.map (fun s -> of_bits (int_of_string s)) vals) in
            let rows = List.init r (fun i -> List.init c (fun j -> v.(i * c + j))) in
            Hashtbl.replace tbl name (r, c, rows)
        | _ -> ()
      end
    done
  with End_of_file -> ());
  close_in ic; tbl

let () =
  let path = Sys.argv.(1) in
  let _d = int_of_string Sys.argv.(2) in
  let nh = int_of_string Sys.argv.(3) in
  let nkv = int_of_string Sys.argv.(4) in
  let hd = int_of_string Sys.argv.(5) in
  let rd = int_of_string Sys.argv.(6) in
  let _ff = int_of_string Sys.argv.(7) in
  let _vocab = int_of_string Sys.argv.(8) in
  let nl = int_of_string Sys.argv.(9) in
  let lnh = int_of_string Sys.argv.(10) in
  let lhd = int_of_string Sys.argv.(11) in
  let ck = int_of_string Sys.argv.(12) in
  let kinds = Array.of_list (String.split_on_char ',' Sys.argv.(13)) in
  let ids = List.map int_of_string (String.split_on_char ',' Sys.argv.(14)) in

  let tbl = read_tensors path in
  let mat name = let (_, _, m) = Hashtbl.find tbl name in m in
  let vec name = match mat name with row :: _ -> row | [] -> [] in

  (* rms_norm_eps = 1e-6, as Qwen3.5 uses. *)
  let eps = f32_div (f32_of_Z (z_of_int 1)) (f32_of_Z (z_of_int 1000000)) in
  let cosv = mat "cos" and sinv = mat "sin" in
  let emb = mat "emb" in
  let normw = vec "norm" in

  let layer i h =
    let p = Printf.sprintf "L%d." i in
    let mlp = { qm_gate = mat (p ^ "gate"); qm_up = mat (p ^ "up");
                qm_down = mat (p ^ "down") } in
    let ln1 = vec (p ^ "ln1") and ln2 = vec (p ^ "ln2") in
    if kinds.(i) = "attn" then begin
      let aw = { qa_q = mat (p ^ "q"); qa_k = mat (p ^ "k"); qa_v = mat (p ^ "v");
                 qa_o = mat (p ^ "o"); qa_q_norm = vec (p ^ "q_norm");
                 qa_k_norm = vec (p ^ "k_norm") } in
      f32_qwen_wrap eps ln1 ln2 mlp
        (fun hn -> f32_qwen_attn_mix nh nkv hd rd eps aw cosv sinv hn) h
    end else begin
      let dw = { qd_in_qkv = mat (p ^ "in_qkv"); qd_in_z = mat (p ^ "in_z");
                 qd_in_a = mat (p ^ "in_a"); qd_in_b = mat (p ^ "in_b");
                 qd_conv_w = mat (p ^ "conv_w"); qd_a_log = vec (p ^ "a_log");
                 qd_dt_bias = vec (p ^ "dt_bias"); qd_norm_w = vec (p ^ "norm_w");
                 qd_out = mat (p ^ "out") } in
      f32_qwen_wrap eps ln1 ln2 mlp
        (fun hn -> f32_qwen_delta_mix lnh lhd ck eps dw hn) h
    end in

  let fs = List.init nl (fun i -> layer i) in
  let hidden = f32_qwen_forward eps normw fs emb ids in
  let logits = f32_qwen_logits emb hidden in
  List.iter (fun row ->
    print_string (String.concat " " (List.map (fun x -> Printf.sprintf "%.9g" (b2f x)) row));
    print_newline ()) logits
