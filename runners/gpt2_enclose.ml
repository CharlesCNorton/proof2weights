(* gpt2_enclose.ml - an enclosure of every GPT-2 logit of the exact real
   reference, and each binary32 logit's distance from it.

   f32_load_model_pre reads the checkpoint into its transposed weight record,
   and the pre-transposed pass of EncGPT2.v in the enclosure arithmetic of
   Enclose.v, at FB fractional bits, returns for every logit an interval that
   contains the logit of the exact real reference (gpt2_logits_enclosed). The
   binary32 logits are read from a dump of the decoder runner, whose rows are
   f32_gpt2_logits (gpt2_decode_correct, f32_gpt2_logits_pre_correct).

   usage: gpt2_enclose <path> d n_head n_layer ff vocab n_pos FB <windows> <f32dir> <outdir> <first> <last>

   For each window it writes <outdir>/<index>.txt: per position the largest
   bound, the largest interval width, and whether the reference's top-1 token
   is certified to be the binary32 top-1 (enc_common.ml), and a summary line
   on stderr. The exponential runs at FB + 32 bits of relative precision. *)

module ZZ = Z
open Enc_native
open Enc_common

let () =
  let path = Sys.argv.(1) in
  let d = int_of_string Sys.argv.(2) and nh = int_of_string Sys.argv.(3) in
  let nl = int_of_string Sys.argv.(4) and ff = int_of_string Sys.argv.(5) in
  let vocab = int_of_string Sys.argv.(6) and npos = int_of_string Sys.argv.(7) in
  let fb = int_of_string Sys.argv.(8) in
  let windows = Array.of_list (read_lines Sys.argv.(9)) in
  let f32dir = Sys.argv.(10) and outdir = Sys.argv.(11) in
  let first = int_of_string Sys.argv.(12) and last = int_of_string Sys.argv.(13) in
  let (header, rd) = checkpoint path in
  let cfg = { gpt2_inf_n_embd = d; gpt2_inf_n_head = nh; gpt2_inf_n_layer = nl;
              gpt2_inf_n_inner = ff; gpt2_inf_vocab_size = vocab;
              gpt2_inf_n_positions = npos } in
  Printf.eprintf "decoding weights...\n%!";
  let emodel = gmodel_map enc_const (to_gmodel (f32_load_model_pre header rd cfg)) in
  Printf.eprintf "weights ready.\n%!";
  let ops = enc_ops (ZZ.of_int fb) (ZZ.of_int (fb + 32)) in
  let eps = enc_const f32_ln_eps in
  over_windows outdir first (min last (Array.length windows))
    (window fb vocab windows f32dir outdir (fun toks -> g_gpt2_logits_pre ops cfg eps emodel toks))
