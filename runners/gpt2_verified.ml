(* gpt2_verified.ml - GPT-2 through the extracted forward pass itself.

   gpt2_talk_native.ml writes the layer loop by hand in OCaml. This runner does
   not. It decodes each weight in transposed order, which is the order
   Pretransposed.v's forward pass expects and the order Runner.v proves equal to
   the transpose of the reshape, assembles the model record, and calls
   f32_gpt2_logits_pre. That function is proved to return, on a model whose
   matrices are stored transposed, exactly what f32_gpt2_logits returns on the
   model (f32_gpt2_logits_pre_correct). The composition that runs is therefore
   extracted code; what remains native is the file read, the offset lookup and
   the assembly of the record.

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

let dec1 b o =
  f32_bytes_to_binary32
    [ Char.code (Bytes.get b o); Char.code (Bytes.get b (o + 1));
      Char.code (Bytes.get b (o + 2)); Char.code (Bytes.get b (o + 3)) ]

let dec_vec b start count = List.init count (fun i -> dec1 b (start + 4 * i))

(* the stored layout is [in, out] row-major, so this is its transpose *)
let dec_wt b start in_dim out_dim =
  List.init out_dim (fun o ->
    List.init in_dim (fun i -> dec1 b (start + 4 * (i * out_dim + o))))

let dec_rows b start rows cols =
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
  let off name =
    match json_tensor_offsets header (coqstr name) with
    | s :: _ :: _ -> base + s
    | _ -> failwith ("offsets not found for " ^ name) in

  Printf.eprintf "decoding weights...\n%!";
  let ln nm = { f32_ln_weight = dec_vec b (off (nm ^ ".weight")) d;
                f32_ln_bias   = dec_vec b (off (nm ^ ".bias")) d } in
  let blocks = List.init nl (fun i ->
    let p = Printf.sprintf "h.%d." i in
    { f32_block_ln_1 = ln (p ^ "ln_1");
      f32_block_attn =
        { f32_attn_c_attn_weight = dec_wt b (off (p ^ "attn.c_attn.weight")) d (3 * d);
          f32_attn_c_attn_bias   = dec_vec b (off (p ^ "attn.c_attn.bias")) (3 * d);
          f32_attn_c_proj_weight = dec_wt b (off (p ^ "attn.c_proj.weight")) d d;
          f32_attn_c_proj_bias   = dec_vec b (off (p ^ "attn.c_proj.bias")) d };
      f32_block_ln_2 = ln (p ^ "ln_2");
      f32_block_mlp =
        { f32_mlp_c_fc_weight   = dec_wt b (off (p ^ "mlp.c_fc.weight")) d ff;
          f32_mlp_c_fc_bias     = dec_vec b (off (p ^ "mlp.c_fc.bias")) ff;
          f32_mlp_c_proj_weight = dec_wt b (off (p ^ "mlp.c_proj.weight")) ff d;
          f32_mlp_c_proj_bias   = dec_vec b (off (p ^ "mlp.c_proj.bias")) d } }) in
  let model =
    { f32_wte = dec_rows b (off "wte.weight") vocab d;
      f32_wpe = dec_rows b (off "wpe.weight") npos d;
      f32_blocks = blocks;
      f32_ln_f = ln "ln_f" } in
  let cfg = { gpt2_inf_n_embd = d; gpt2_inf_n_head = nh; gpt2_inf_n_layer = nl;
              gpt2_inf_n_inner = ff; gpt2_inf_vocab_size = vocab;
              gpt2_inf_n_positions = npos } in
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
