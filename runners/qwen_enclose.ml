(* qwen_enclose.ml - an enclosure of every Qwen3.5 logit of the exact real
   reference, and each binary32 logit's distance from it.

   f32_load_qwen reads the checkpoint into the weight record, and the Qwen3.5
   pass of EncQwen.v in the enclosure arithmetic of Enclose.v and EncExt.v, at
   FB fractional bits, on the rotary tables the binary32 pass forms, returns
   for every logit an interval that contains the logit of the exact real
   reference (qwen_logits_enclosed). The binary32 logits are read from a dump
   of the decoder runner, whose rows are f32_qwen_logits_of
   (qwen_decode_correct), the same pass at binary32 (g_qwen_logits_f32).

   usage: qwen_enclose <path> d nl nh nkv hd rd ff vocab lnh lhd ck FB <windows> <f32dir> <outdir> <first> <last>

   Output as gpt2_enclose.ml's; the exponential and the logarithm run at FB + 32
   bits of relative precision, and RMSNorm's epsilon is the decoder runner's. *)

module ZZ = Z
open Enc_native
open Enc_common

let () =
  let path = Sys.argv.(1) in
  let a i = int_of_string Sys.argv.(i) in
  let d = a 2 and nl = a 3 and nh = a 4 and nkv = a 5 and hd = a 6 and rd = a 7 in
  let ff = a 8 and vocab = a 9 and lnh = a 10 and lhd = a 11 and ck = a 12 in
  let fb = a 13 in
  let windows = Array.of_list (read_lines Sys.argv.(14)) in
  let f32dir = Sys.argv.(15) and outdir = Sys.argv.(16) in
  let first = a 17 and last = a 18 in
  let (header, rdr) = checkpoint path in
  Printf.eprintf "decoding weights...\n%!";
  let model = f32_load_qwen header rdr d nl nh nkv hd rd ff vocab lnh lhd ck in
  let emodel = gqmodel_map enc_const (to_gqmodel model) in
  Printf.eprintf "weights ready.\n%!";
  let fbz = ZZ.of_int fb and prec = ZZ.of_int (fb + 32) in
  let ops = enc_ops fbz prec in
  let eps = enc_const (f32_div f32_one (f32_of_Z (ZZ.of_int 1000000))) in
  let tab t = List.map (List.map enc_const) t in
  let rows toks =
    let n = List.length toks in
    g_qwen_logits ops (enc_ln fbz prec) enc_abs nh nkv hd rd lnh lhd ck eps emodel
      (tab (f32_rope_cos model.qw_invf n)) (tab (f32_rope_sin model.qw_invf n)) toks in
  over_windows outdir first (min last (Array.length windows))
    (window fb vocab windows f32dir outdir rows)
