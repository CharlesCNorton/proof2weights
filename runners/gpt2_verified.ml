(* gpt2_verified.ml - GPT-2 through the extracted forward pass itself.

   gpt2_talk_native.ml writes the layer loop by hand in OCaml. This runner does
   not. It calls f32_load_model_pre, which looks up each tensor's offset in the
   header, decodes each weight matrix in transposed order and assembles the
   weight record, and then f32_gpt2_logits_pre. The first is proved to return
   the model f32_load_model defines, transposed (f32_load_model_pre_correct);
   the second is proved to return, on a model whose matrices are stored
   transposed, exactly what f32_gpt2_logits returns on the model
   (f32_gpt2_logits_pre_correct). What remains native is reading the file and
   answering the byte fetch the loader reads it through.

   usage:
     gpt2_verified <path> d n_head n_layer ff vocab n_positions <tok,...>
     gpt2_verified <path> d n_head n_layer ff vocab n_positions dump <windows> <outdir> <first> <last> *)

open Phases1_15_native

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in really_input ic b 0 n; close_in ic; b

let u64_le b off =
  let r = ref 0 in
  for i = 7 downto 0 do r := (!r * 256) + Char.code (Bytes.get b (off + i)) done; !r

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
  let nh    = int_of_string Sys.argv.(3) in
  let nl    = int_of_string Sys.argv.(4) in
  let ff    = int_of_string Sys.argv.(5) in
  let vocab = int_of_string Sys.argv.(6) in
  let npos  = int_of_string Sys.argv.(7) in
  let mode  = Sys.argv.(8) in
  let dump  = (mode = "dump") in

  let b = read_file path in
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  (* the data section, read one byte at a time, which is all the loader wants *)
  let rd o = Char.code (Bytes.get b (base + o)) in
  let cfg = { gpt2_inf_n_embd = d; gpt2_inf_n_head = nh; gpt2_inf_n_layer = nl;
              gpt2_inf_n_inner = ff; gpt2_inf_vocab_size = vocab;
              gpt2_inf_n_positions = npos } in

  Printf.eprintf "decoding weights...\n%!";
  let model = f32_load_model_pre header rd cfg in
  Printf.eprintf "weights ready.\n%!";

  let logits_of ids = f32_gpt2_logits_pre cfg f32_ln_eps model ids in

  if dump then begin
    let windows = Sys.argv.(9) and outdir = Sys.argv.(10) in
    let first = int_of_string Sys.argv.(11) and last = int_of_string Sys.argv.(12) in
    (try Unix.mkdir outdir 0o755 with _ -> ());
    List.iteri (fun i line ->
      if i >= first && i < last then begin
        let ids = List.map int_of_string (String.split_on_char ',' line) in
        let oc = open_out_bin (Filename.concat outdir (string_of_int i ^ ".f32")) in
        List.iter (fun row -> List.iter (write_f32 oc) row) (logits_of ids);
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
