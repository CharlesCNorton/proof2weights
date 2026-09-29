(* llama_talk_native.ml - a Llama-architecture checkpoint (SmolLM2) through the
   extracted decode step (native build), with greedy generation and a
   persistent serve mode.

   f32_load_llama reads the checkpoint into the weight record, and llama_step
   emits the logits of one position per token from a key/value cache per head
   and layer, forming each position's rotation as it reaches it.
   llama_decode_correct says that running it over a sequence gives the rows
   f32_llama_logits_of computes for the whole sequence. Generation feeds each
   argmax back through the same step. What remains native is reading the file,
   answering the byte fetch the loader reads it through, and choosing the next
   token.

   In serve mode the weights stay resident and the process answers queries from
   stdin (one per line: "<comma ids> <max_new>"), streaming "TOK <id>" per token
   and "END <comma gen ids>" per query. Each query starts from the initial state.

   usage:
     llama_talk_native <path> d n_layer n_head n_kv ff vocab <tok,...>            (top-10)
     llama_talk_native <path> d n_layer n_head n_kv ff vocab <tok,...> <max_new> <eos>  (generate)
     llama_talk_native <path> d n_layer n_head n_kv ff vocab serve <eos>          (persistent server)
     llama_talk_native <path> d n_layer n_head n_kv ff vocab dump <windows> <outdir> <first> <last>

   dump reads one comma-separated window per line and, for each line index in
   [first, last), writes <outdir>/<index>.f32: the logits of every position, as
   little-endian binary32 bit patterns, positions in order. The windows are
   shared among P2W_DOMAINS domains (default one), which read the one weight
   record, and a window whose file exists is skipped. *)

open Llama_native

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

let argmax row =
  let a = Array.of_list row in
  let bi = ref 0 in
  for j = 1 to Array.length a - 1 do if a.(j) > a.(!bi) then bi := j done; !bi

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
  let path  = Sys.argv.(1) in
  let d     = int_of_string Sys.argv.(2) in
  let nl    = int_of_string Sys.argv.(3) in
  let nh    = int_of_string Sys.argv.(4) in
  let nkv   = int_of_string Sys.argv.(5) in
  let ff    = int_of_string Sys.argv.(6) in
  let vocab = int_of_string Sys.argv.(7) in
  let toks_arg = Sys.argv.(8) in
  let serve = (toks_arg = "serve") in
  let dump = (toks_arg = "dump") in
  let eos =
    if serve then (if Array.length Sys.argv > 9 then int_of_string Sys.argv.(9) else -1)
    else if dump then -1
    else (if Array.length Sys.argv > 10 then int_of_string Sys.argv.(10) else -1) in
  let max_new =
    if serve || dump then 0
    else (if Array.length Sys.argv > 9 then int_of_string Sys.argv.(9) else 0) in
  let hd = d / nh in
  let b = read_file path in
  let cksum =
    let r = ref 0 and n = Bytes.length b in
    for i = 0 to n - 1 do r := (!r * 31 + Char.code (Bytes.get b i)) mod 4294967296 done; !r in
  Printf.eprintf "checksum %d\n%!" cksum;
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let rdb o = Char.code (Bytes.get b (base + o)) in
  let eps = f32_div (f32_of_Z 1) (f32_of_Z 100000) in
  Printf.eprintf "decoding weights...\n%!";
  let model = f32_load_llama header rdb d nl nkv hd ff vocab in
  Printf.eprintf "weights ready.\n%!";
  let step = llama_step nh nkv hd eps model in
  let init = llama_init nh nkv hd eps model in

  (* Greedy continuation of a prompt: the prompt is read one token at a time,
     then each argmax is fed back, as run reads a sequence. *)
  let generate toks max_new stream =
    let rec feed s last = function
      | [] -> (s, last)
      | t :: rest -> let (s', row) = step s t in feed s' row rest in
    let (s, last) = feed init [] toks in
    let rec go s row n acc =
      let cur = argmax row in
      if n = 0 || cur = eos then List.rev acc
      else begin
        if stream then Printf.printf "TOK %d\n%!" cur;
        if n = 1 then List.rev (cur :: acc)
        else let (s', row') = step s cur in go s' row' (n - 1) (cur :: acc)
      end in
    go s last max_new [] in

  if dump then begin
    let windows = Array.of_list (read_lines Sys.argv.(9)) in
    dump_windows (run step init) windows Sys.argv.(10)
      (int_of_string Sys.argv.(11)) (int_of_string Sys.argv.(12))
  end
  else if serve then begin
    Printf.printf "CKSUM %d\nREADY\n%!" cksum;
    (try
      while true do
        let line = String.trim (input_line stdin) in
        if line <> "" then begin
          match String.split_on_char ' ' line with
          | ids_csv :: mn :: _ ->
              let qtoks = List.map int_of_string (String.split_on_char ',' ids_csv) in
              let g = generate qtoks (int_of_string mn) true in
              Printf.printf "END %s\n%!" (String.concat "," (List.map string_of_int g))
          | _ -> Printf.printf "END \n%!"
        end
      done
    with End_of_file -> ())
  end
  else begin
    let prompt = List.map int_of_string (String.split_on_char ',' toks_arg) in
    if max_new > 0 then begin
      let g = generate prompt max_new false in
      print_string (String.concat " " (List.map string_of_int g)); print_newline ()
    end else begin
      let rows = run step init prompt in
      let a = Array.of_list (List.nth rows (List.length rows - 1)) in
      let idx = Array.init (Array.length a) (fun i -> i) in
      Array.sort (fun i j -> compare a.(j) a.(i)) idx;
      Printf.printf "top-10 next-token logits:\n";
      for r = 0 to 9 do let i = idx.(r) in Printf.printf "  %6d  %.9g\n" i a.(i) done
    end
  end
