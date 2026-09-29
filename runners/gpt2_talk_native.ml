(* gpt2_talk_native.ml - GPT-2 through the extracted decode step (native build).

   f32_load_model_pre reads the checkpoint into the weight record, and gpt2_step
   emits the logits of one position per token from a key/value cache per head
   and layer. gpt2_decode_correct says that running it over a sequence gives the
   rows f32_gpt2_logits_pre computes for the whole sequence. What remains native
   is reading the file and answering the byte fetch the loader reads it through.

   usage: gpt2_talk_native <full|next> <path> d n_head n_layer ff vocab n_pos <tok,...>
          gpt2_talk_native dump <path> d n_head n_layer ff vocab n_pos <windows> <outdir> <first> <last>

   full prints the logits of every position, next the ten highest of the last.
   dump reads one comma-separated window per line and, for each line index in
   [first, last), writes <outdir>/<index>.f32: the logits of every position, as
   little-endian binary32 bit patterns, positions in order. The windows are
   shared among P2W_DOMAINS domains (default one), which read the one weight
   record, and a window whose file exists is skipped. *)

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
    output_byte oc (Int32.to_int (Int32.logand (Int32.shift_right_logical bits (8 * k)) 0xffl))
  done

let read_lines path =
  let ic = open_in path in
  let rec go acc = match input_line ic with
    | l -> go (if String.trim l = "" then acc else String.trim l :: acc)
    | exception End_of_file -> close_in ic; List.rev acc in
  go []

let dump_windows rows windows outdir first last =
  (try Unix.mkdir outdir 0o755 with _ -> ());
  let last = min last (Array.length windows) in
  let nd = try max 1 (int_of_string (Sys.getenv "P2W_DOMAINS")) with _ -> 1 in
  let work k =
    let w = ref (first + k) in
    while !w < last do
      let fin = Printf.sprintf "%s/%d.f32" outdir !w in
      if not (Sys.file_exists fin) then begin
        let toks = List.map int_of_string (String.split_on_char ',' windows.(!w)) in
        let oc = open_out_bin (fin ^ ".tmp") in
        List.iter (List.iter (write_f32 oc)) (rows toks);
        close_out oc;
        Sys.rename (fin ^ ".tmp") fin;
        Printf.eprintf "window %d done\n%!" !w
      end;
      w := !w + nd
    done in
  let ds = List.init (nd - 1) (fun k -> Domain.spawn (fun () -> work (k + 1))) in
  work 0;
  List.iter Domain.join ds

let () =
  let mode  = Sys.argv.(1) in
  let path  = Sys.argv.(2) in
  let d     = int_of_string Sys.argv.(3) in
  let nh    = int_of_string Sys.argv.(4) in
  let nl    = int_of_string Sys.argv.(5) in
  let ff    = int_of_string Sys.argv.(6) in
  let vocab = int_of_string Sys.argv.(7) in
  let npos  = int_of_string Sys.argv.(8) in
  let b = read_file path in
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let rd o = Char.code (Bytes.get b (base + o)) in
  let cfg = { gpt2_inf_n_embd = d; gpt2_inf_n_head = nh; gpt2_inf_n_layer = nl;
              gpt2_inf_n_inner = ff; gpt2_inf_vocab_size = vocab;
              gpt2_inf_n_positions = npos } in
  Printf.eprintf "decoding weights...\n%!";
  let model = f32_load_model_pre header rd cfg in
  Printf.eprintf "weights ready.\n%!";
  let rows ids = run (gpt2_step cfg f32_ln_eps model) (gpt2_init cfg f32_ln_eps model) ids in

  match mode with
  | "dump" ->
      let windows = Array.of_list (read_lines Sys.argv.(9)) in
      dump_windows rows windows Sys.argv.(10)
        (int_of_string Sys.argv.(11)) (int_of_string Sys.argv.(12))
  | "full" ->
      let toks = List.map int_of_string (String.split_on_char ',' Sys.argv.(9)) in
      List.iter (fun row ->
        print_string (String.concat " " (List.map (Printf.sprintf "%.9g") row));
        print_newline ()) (rows toks)
  | _ ->
      let toks = List.map int_of_string (String.split_on_char ',' Sys.argv.(9)) in
      let a = Array.of_list (List.nth (rows toks) (List.length toks - 1)) in
      let idx = Array.init (Array.length a) (fun i -> i) in
      Array.sort (fun i j -> compare a.(j) a.(i)) idx;
      Printf.printf "top-10 next-token logits:\n";
      for r = 0 to 9 do let i = idx.(r) in Printf.printf "  %6d  %.9g\n" i a.(i) done
