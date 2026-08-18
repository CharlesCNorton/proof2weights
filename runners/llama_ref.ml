(* llama_ref.ml - canonical IEEE-754 reference logits from the verified Llama
   forward, against the INDUCTIVE extraction.

   binary32 is Flocq's binary_float and Z is its inductive datatype here, so
   every float operation is the computational content of its proof and no
   floating-point boundary is trusted. This is the oracle the differential
   harness measures a numpy float32 implementation of the same operations
   against.

   Weights arrive as raw binary32 bit patterns, one tensor per line

     <name> <rows> <cols> <rows*cols unsigned 32-bit ints>

   so the values the reference and the numpy mirror see are bit-identical, and
   the decode itself goes through the verified f32_bytes_to_binary32.

   usage: llama_ref <weights> d nh nkv hd ff vocab n_layer <tok,tok,...> *)

open Llama_inductive

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

(* A 32-bit pattern to the binary32 it denotes, through the verified decoder. *)
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
  let d = int_of_string Sys.argv.(2) in
  let nh = int_of_string Sys.argv.(3) in
  let nkv = int_of_string Sys.argv.(4) in
  let hd = int_of_string Sys.argv.(5) in
  let _ff = int_of_string Sys.argv.(6) in
  let _vocab = int_of_string Sys.argv.(7) in
  let nl = int_of_string Sys.argv.(8) in
  let ids = List.map int_of_string (String.split_on_char ',' Sys.argv.(9)) in
  ignore d;
  let tbl = read_tensors path in
  let mat name = let (_, _, m) = Hashtbl.find tbl name in m in
  let vec name = match mat name with row :: _ -> row | [] -> [] in

  (* rms_norm_eps = 1e-5, as the reference models use. *)
  let eps = f32_div (f32_of_Z (z_of_int 1)) (f32_of_Z (z_of_int 100000)) in
  let cosv = mat "cos" and sinv = mat "sin" in
  let emb = mat "emb" in
  let normw = vec "norm" in

  let layer i h =
    let p = Printf.sprintf "L%d." i in
    let aw = { la_q = mat (p ^ "q"); la_k = mat (p ^ "k");
               la_v = mat (p ^ "v"); la_o = mat (p ^ "o") } in
    let mw = { lm_gate = mat (p ^ "gate"); lm_up = mat (p ^ "up");
               lm_down = mat (p ^ "down") } in
    f32_llama_layer nh nkv hd eps (vec (p ^ "ln1")) (vec (p ^ "ln2")) aw mw
      cosv sinv h in

  let fs = List.init nl (fun i -> layer i) in
  let hidden = f32_llama_forward eps normw fs emb ids in
  let logits = f32_llama_logits emb hidden in
  List.iter (fun row ->
    print_string (String.concat " " (List.map (fun x -> Printf.sprintf "%.9g" (b2f x)) row));
    print_newline ()) logits
