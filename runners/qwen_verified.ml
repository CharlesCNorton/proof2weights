(* qwen_verified.ml - a Qwen3.5 checkpoint through the extracted forward pass.

   qwen_talk_native.ml writes the layer loop by hand and carries its own
   caches. This runner does not. It reads the file, decodes each value with the
   verified f32_bytes_to_binary32, builds one layer map per layer with
   f32_qwen_delta_mix or f32_qwen_attn_mix inside f32_qwen_wrap, and hands the
   list to f32_qwen_forward and f32_qwen_logits. The composition that runs is
   extracted code; what remains native is the file read, the offset lookup and
   the assembly of the weight records.

   usage:
     qwen_verified <path> d nl nh nkv hd rd ff vocab lnh lhd ck <tok,...>
     qwen_verified <path> d nl nh nkv hd rd ff vocab lnh lhd ck dump <windows> <outdir> <first> <last> *)

open Qwen_native

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
  let hd    = int_of_string Sys.argv.(6) in
  let rd    = int_of_string Sys.argv.(7) in
  let ff    = int_of_string Sys.argv.(8) in
  let vocab = int_of_string Sys.argv.(9) in
  let lnh   = int_of_string Sys.argv.(10) in
  let lhd   = int_of_string Sys.argv.(11) in
  let ck    = int_of_string Sys.argv.(12) in
  let mode  = Sys.argv.(13) in
  let dump  = (mode = "dump") in

  let kvd = nkv * hd in
  let ldim = lnh * lhd in
  let qkvd = 3 * ldim in

  let b = read_file path in
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let off name =
    match json_tensor_offsets header (coqstr name) with
    | s :: _ :: _ -> base + s
    | _ -> failwith ("offsets not found for " ^ name) in
  let lname i s = Printf.sprintf "layers.%d.%s" i s in

  let eps = f32_div f32_one (f32_of_Z 1000000) in
  Printf.eprintf "decoding weights...\n%!";
  let emb = dec_mat b (off "embed_tokens.weight") vocab d in
  let invf = dec_vec b (off "rope.inv_freq") (rd / 2) in
  let normw = dec_vec b (off "norm.weight") d in
  let layer_is_full i = (i mod 4) = 3 in

  (* per layer: the two norms, the mlp, and either mixer's weights *)
  let layers = Array.init nl (fun i ->
    let ln1 = dec_vec b (off (lname i "input_layernorm.weight")) d in
    let ln2 = dec_vec b (off (lname i "post_attention_layernorm.weight")) d in
    let mlp = { qm_gate = dec_mat b (off (lname i "mlp.gate_proj.weight")) ff d;
                qm_up   = dec_mat b (off (lname i "mlp.up_proj.weight")) ff d;
                qm_down = dec_mat b (off (lname i "mlp.down_proj.weight")) d ff } in
    if layer_is_full i then
      let aw =
        { qa_q = dec_mat b (off (lname i "self_attn.q_proj.weight")) (nh * hd * 2) d;
          qa_k = dec_mat b (off (lname i "self_attn.k_proj.weight")) kvd d;
          qa_v = dec_mat b (off (lname i "self_attn.v_proj.weight")) kvd d;
          qa_o = dec_mat b (off (lname i "self_attn.o_proj.weight")) d (nh * hd);
          qa_q_norm = dec_vec b (off (lname i "self_attn.q_norm.weight")) hd;
          qa_k_norm = dec_vec b (off (lname i "self_attn.k_norm.weight")) hd } in
      `Attn (ln1, ln2, mlp, aw)
    else
      let dw =
        { qd_in_qkv = dec_mat b (off (lname i "linear_attn.in_proj_qkv.weight")) qkvd d;
          qd_in_z   = dec_mat b (off (lname i "linear_attn.in_proj_z.weight")) ldim d;
          qd_in_a   = dec_mat b (off (lname i "linear_attn.in_proj_a.weight")) lnh d;
          qd_in_b   = dec_mat b (off (lname i "linear_attn.in_proj_b.weight")) lnh d;
          qd_conv_w = dec_mat b (off (lname i "linear_attn.conv1d.weight")) qkvd ck;
          qd_a_log  = dec_vec b (off (lname i "linear_attn.A_log")) lnh;
          qd_dt_bias = dec_vec b (off (lname i "linear_attn.dt_bias")) lnh;
          qd_norm_w = dec_vec b (off (lname i "linear_attn.norm.weight")) lhd;
          qd_out    = dec_mat b (off (lname i "linear_attn.out_proj.weight")) d ldim } in
      `Delta (ln1, ln2, mlp, dw)) in
  Printf.eprintf "weights ready.\n%!";

  (* the rotated prefix's cos and sin per position, in the half-split
     convention f32_partial_rope reads, through the extracted f32_cos/f32_sin *)
  let rope_tables npos =
    let rows = List.init npos (fun pos ->
      let pf = f32_of_Z pos in
      let ang = List.map (fun fj -> f32_mult pf fj) invf in
      (List.map f32_cos ang, List.map f32_sin ang)) in
    (List.map (fun (c, _) -> c @ c) rows, List.map (fun (_, s) -> s @ s) rows) in

  let logits_of ids =
    let (cosv, sinv) = rope_tables (List.length ids) in
    let fs = Array.to_list (Array.map (function
      | `Attn (ln1, ln2, mlp, aw) ->
          f32_qwen_wrap eps ln1 ln2 mlp
            (f32_qwen_attn_mix nh nkv hd rd eps aw cosv sinv)
      | `Delta (ln1, ln2, mlp, dw) ->
          f32_qwen_wrap eps ln1 ln2 mlp
            (f32_qwen_delta_mix lnh lhd ck eps dw)) layers) in
    f32_qwen_logits emb (f32_qwen_forward eps normw fs emb ids) in

  if dump then begin
    let windows = Sys.argv.(14) and outdir = Sys.argv.(15) in
    let first = int_of_string Sys.argv.(16) and last = int_of_string Sys.argv.(17) in
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
