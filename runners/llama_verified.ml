(* llama_verified.ml - a Llama-architecture checkpoint through the extracted
   forward pass itself.

   llama_talk_native.ml writes the layer loop by hand in OCaml and calls the
   extracted operators from it, so the arithmetic is verified but the
   composition around it is not. This runner does not write the loop. It reads
   the file, decodes each value with the verified f32_bytes_to_binary32, builds
   the layer maps with f32_llama_layer and hands them to f32_llama_forward and
   f32_llama_logits, which are the definitions the proofs are about, extracted.
   What remains unverified here is the file read, the offset lookup and the
   assembly of the weight records; the forward pass is extracted code.

   The rotary tables are built with the extracted f32_sin and f32_cos on the
   angles the checkpoint's stored inverse frequencies give, so no trigonometry
   is performed outside the development either.

   usage:
     llama_verified <path> d n_layer n_head n_kv ff vocab <tok,...>
     llama_verified <path> d n_layer n_head n_kv ff vocab dump <windows> <outdir> <first> <last>

   Without dump it prints the ten highest logits of the final position. With it
   it writes <outdir>/<index>.f32 per window: the logits of every position as
   little-endian binary32, positions in order. *)

open Llama_native

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in really_input ic b 0 n; close_in ic; b

let u64_le b off =
  let r = ref 0 in
  for i = 7 downto 0 do r := (!r * 256) + Char.code (Bytes.get b (off + i)) done; !r

let dec1 b o =
  f32_bytes_to_binary32
    [ Char.code (Bytes.get b o); Char.code (Bytes.get b (o + 1));
      Char.code (Bytes.get b (o + 2)); Char.code (Bytes.get b (o + 3)) ]

let dec_vec b start count = List.init count (fun i -> dec1 b (start + 4 * i))
let dec_mat b start rows cols =
  List.init rows (fun r -> dec_vec b (start + 4 * r * cols) cols)

let coqstr s = List.init (String.length s) (fun i -> s.[i])

let write_f32 oc (x : float) =
  let bits = Int32.bits_of_float x in
  for k = 0 to 3 do
    output_byte oc
      (Int32.to_int (Int32.logand (Int32.shift_right_logical bits (8 * k)) 0xffl))
  done

let read_lines path =
  let ic = open_in path in
  let rec go acc = match input_line ic with
    | l -> go (if String.trim l = "" then acc else String.trim l :: acc)
    | exception End_of_file -> close_in ic; List.rev acc in
  go []

let () =
  let path  = Sys.argv.(1) in
  let d     = int_of_string Sys.argv.(2) in
  let nl    = int_of_string Sys.argv.(3) in
  let nh    = int_of_string Sys.argv.(4) in
  let nkv   = int_of_string Sys.argv.(5) in
  let ff    = int_of_string Sys.argv.(6) in
  let vocab = int_of_string Sys.argv.(7) in
  let mode  = Sys.argv.(8) in
  let dump  = (mode = "dump") in
  let hd = d / nh in
  ignore ff;

  let b = read_file path in
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let off name =
    match json_tensor_offsets header (coqstr name) with
    | s :: _ :: _ -> base + s
    | _ -> failwith ("offsets not found for " ^ name) in

  let eps = f32_div (f32_of_Z 1) (f32_of_Z 100000) in
  Printf.eprintf "decoding weights...\n%!";
  let emb = dec_mat b (off "embed_tokens.weight") vocab d in
  let invf = dec_vec b (off "rope.inv_freq") (hd / 2) in
  let normw = dec_vec b (off "norm.weight") d in
  let layers = Array.init nl (fun i ->
    let p = Printf.sprintf "layers.%d." i in
    (dec_vec b (off (p ^ "input_layernorm.weight")) d,
     dec_vec b (off (p ^ "post_attention_layernorm.weight")) d,
     { la_q = dec_mat b (off (p ^ "self_attn.q_proj.weight")) d d;
       la_k = dec_mat b (off (p ^ "self_attn.k_proj.weight")) (nkv * hd) d;
       la_v = dec_mat b (off (p ^ "self_attn.v_proj.weight")) (nkv * hd) d;
       la_o = dec_mat b (off (p ^ "self_attn.o_proj.weight")) d d },
     { lm_gate = dec_mat b (off (p ^ "mlp.gate_proj.weight")) ff d;
       lm_up   = dec_mat b (off (p ^ "mlp.up_proj.weight")) ff d;
       lm_down = dec_mat b (off (p ^ "mlp.down_proj.weight")) d ff })) in
  Printf.eprintf "weights ready.\n%!";

  (* cos and sin of position * inverse frequency, in the half-split convention
     f32_partial_rope reads: a table of length hd per position, the first half
     repeated. Both are computed by the extracted f32_cos and f32_sin. *)
  let rope_tables npos =
    let rows = List.init npos (fun pos ->
      let pf = f32_of_Z pos in
      let ang = List.map (fun fj -> f32_mult pf fj) invf in
      (List.map f32_cos ang, List.map f32_sin ang)) in
    (List.map (fun (c, _) -> c @ c) rows, List.map (fun (_, s) -> s @ s) rows) in

  let logits_of ids =
    let npos = List.length ids in
    let (cosv, sinv) = rope_tables npos in
    let fs = Array.to_list (Array.map (fun (ln1, ln2, aw, mw) ->
        f32_llama_layer nh nkv hd eps ln1 ln2 aw mw cosv sinv) layers) in
    let h = f32_llama_forward eps normw fs emb ids in
    f32_llama_logits emb h in

  if dump then begin
    let windows = Sys.argv.(9) and outdir = Sys.argv.(10) in
    let first = int_of_string Sys.argv.(11) and last = int_of_string Sys.argv.(12) in
    (try Unix.mkdir outdir 0o755 with _ -> ());
    List.iteri (fun i line ->
      if i >= first && i < last then begin
        let ids = List.map int_of_string (String.split_on_char ',' line) in
        let rows = logits_of ids in
        let oc = open_out_bin (Filename.concat outdir (string_of_int i ^ ".f32")) in
        List.iter (fun row -> List.iter (write_f32 oc) row) rows;
        close_out oc;
        Printf.eprintf "window %d\n%!" i
      end) (read_lines windows)
  end else begin
    let ids = List.map int_of_string (String.split_on_char ',' mode) in
    let rows = logits_of ids in
    let last = List.nth rows (List.length rows - 1) in
    let arr = Array.of_list last in
    let idx = Array.init (Array.length arr) (fun i -> i) in
    Array.sort (fun a c -> compare arr.(c) arr.(a)) idx;
    Printf.printf "top 10 of the final position\n";
    for r = 0 to 9 do
      Printf.printf "  %2d  %7d  %.9g\n" (r + 1) idx.(r) arr.(idx.(r))
    done
  end
