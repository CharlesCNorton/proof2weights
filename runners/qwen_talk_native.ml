(* qwen_talk_native.ml - Qwen3.5 through the extracted decode step (native
   build), with greedy generation and a persistent serve mode.

   f32_load_qwen reads the checkpoint into the weight record, and qwen_step
   emits the logits of one position per token. Its state holds a key/value cache
   per head of each gated attention layer, and the convolution window and a
   recurrent state per head of each DeltaNet layer. qwen_decode_correct says
   that running it over a sequence gives the rows f32_qwen_logits_of computes
   for the whole sequence. Generation feeds each argmax back through the same
   step. What remains native is reading the file, answering the byte fetch the
   loader reads it through, and choosing the next token.

   In serve mode the weights stay resident and the process answers queries from
   stdin (one per line: "<comma ids> <max_new>"), streaming "TOK <id>" per token
   and "END <comma gen ids>" per query, after announcing the checksum of the
   weight file. Each query starts from the initial state.

   usage:
     qwen_talk_native <path> d nl nh nkv hd rd ff vocab lnh lhd ck <tok,...>
       -> top-10 next-token logits
     qwen_talk_native <path> d nl nh nkv hd rd ff vocab lnh lhd ck <tok,...> <max_new> <eos>
       -> greedy continuation, one token id per line
     qwen_talk_native <path> d nl nh nkv hd rd ff vocab lnh lhd ck serve <eos>
       -> persistent server
     qwen_talk_native <path> d nl nh nkv hd rd ff vocab lnh lhd ck dump <windows> <outdir> <first> <last>
       -> for each line index in [first, last) of the windows file, writes
          <outdir>/<index>.f32: the logits of every position, as little-endian
          binary32 bit patterns, positions in order; the windows are shared
          among P2W_DOMAINS domains (default one), which read the one weight
          record, and a window whose file exists is skipped *)

open Qwen_native

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
  let d     = int_of_string Sys.argv.(2) in    (* hidden size            *)
  let nl    = int_of_string Sys.argv.(3) in    (* layers                 *)
  let nh    = int_of_string Sys.argv.(4) in    (* attention query heads  *)
  let nkv   = int_of_string Sys.argv.(5) in    (* attention kv heads     *)
  let hd    = int_of_string Sys.argv.(6) in    (* attention head dim     *)
  let rd    = int_of_string Sys.argv.(7) in    (* rotary dim (partial)   *)
  let ff    = int_of_string Sys.argv.(8) in    (* mlp intermediate       *)
  let vocab = int_of_string Sys.argv.(9) in
  let lnh   = int_of_string Sys.argv.(10) in   (* deltanet heads         *)
  let lhd   = int_of_string Sys.argv.(11) in   (* deltanet head dim      *)
  let ck    = int_of_string Sys.argv.(12) in   (* conv kernel            *)
  let toks_arg = Sys.argv.(13) in
  let serve = (toks_arg = "serve") in
  let dump = (toks_arg = "dump") in
  let max_new =
    if serve || dump then 0
    else (if Array.length Sys.argv > 14 then int_of_string Sys.argv.(14) else 0) in
  let eos =
    if serve then (if Array.length Sys.argv > 14 then int_of_string Sys.argv.(14) else -1)
    else if dump then -1
    else (if Array.length Sys.argv > 15 then int_of_string Sys.argv.(15) else -1) in

  let b = read_file path in
  let cksum =
    let r = ref 0 and n = Bytes.length b in
    for i = 0 to n - 1 do r := (!r * 31 + Char.code (Bytes.get b i)) mod 4294967296 done;
    !r in
  Printf.eprintf "checksum %d\n%!" cksum;
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let rdb o = Char.code (Bytes.get b (base + o)) in
  let eps = f32_div f32_one (f32_of_Z 1000000) in
  Printf.eprintf "decoding weights...\n%!";
  let model = f32_load_qwen header rdb d nl nh nkv hd rd ff vocab lnh lhd ck in
  Printf.eprintf "weights ready.\n%!";
  let step = qwen_step nh nkv hd rd lnh lhd ck eps model in
  let init = qwen_init nh nkv hd rd lnh lhd ck eps model in

  (* Greedy continuation of a prompt: the prompt is read one token at a time,
     then each argmax is fed back, as run reads a sequence. *)
  let generate toks mx stream =
    let rec feed s last = function
      | [] -> (s, last)
      | t :: rest -> let (s', row) = step s t in feed s' row rest in
    let (s, last) = feed init [] toks in
    let rec go s row n acc =
      let cur = argmax row in
      if n = 0 || cur = eos then List.rev acc
      else begin
        if stream then Printf.printf "TOK %d\n%!" cur else Printf.printf "%d\n%!" cur;
        if n = 1 then List.rev (cur :: acc)
        else let (s', row') = step s cur in go s' row' (n - 1) (cur :: acc)
      end in
    go s last mx [] in

  if dump then begin
    let windows = Array.of_list (read_lines Sys.argv.(14)) in
    dump_windows (run step init) windows Sys.argv.(15)
      (int_of_string Sys.argv.(16)) (int_of_string Sys.argv.(17))
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
    if max_new > 0 then ignore (generate prompt max_new false)
    else begin
      let rows = run step init prompt in
      let a = Array.of_list (List.nth rows (List.length rows - 1)) in
      let idx = Array.init (Array.length a) (fun i -> i) in
      Array.sort (fun i j -> compare a.(j) a.(i)) idx;
      Printf.printf "top-10 next-token logits:\n";
      for r = 0 to 9 do
        let i = idx.(r) in Printf.printf "  %7d  %.9g\n" i a.(i)
      done
    end
  end
