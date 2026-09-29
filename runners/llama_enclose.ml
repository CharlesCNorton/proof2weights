(* llama_enclose.ml - an enclosure of every logit of a Llama-architecture
   checkpoint (SmolLM2) of the exact real reference, and each binary32 logit's
   distance from it.

   f32_load_llama reads the checkpoint into the weight record, and the Llama
   pass of EncLlama.v in the enclosure arithmetic of Enclose.v, at FB
   fractional bits, on the rotary tables the binary32 pass forms, returns for
   every logit an interval that contains the logit of the exact real reference
   (llama_logits_enclosed). The binary32 logits are read from a dump of the
   decoder runner, whose rows are f32_llama_logits_of (llama_decode_correct),
   the same pass at binary32 (g_llama_logits_f32).

   usage: llama_enclose <path> d n_layer n_head n_kv ff vocab FB <windows> <f32dir> <outdir> <first> <last>

   Output as gpt2_enclose.ml's; RMSNorm's epsilon is the decoder runner's. *)

module ZZ = Z
open Enc_native
open Enc_common

let () =
  let path = Sys.argv.(1) in
  let d = int_of_string Sys.argv.(2) and nl = int_of_string Sys.argv.(3) in
  let nh = int_of_string Sys.argv.(4) and nkv = int_of_string Sys.argv.(5) in
  let ff = int_of_string Sys.argv.(6) and vocab = int_of_string Sys.argv.(7) in
  let fb = int_of_string Sys.argv.(8) in
  let windows = Array.of_list (read_lines Sys.argv.(9)) in
  let f32dir = Sys.argv.(10) and outdir = Sys.argv.(11) in
  let first = int_of_string Sys.argv.(12) and last = int_of_string Sys.argv.(13) in
  let hd = d / nh in
  let (header, rd) = checkpoint path in
  Printf.eprintf "decoding weights...\n%!";
  let model = f32_load_llama header rd d nl nkv hd ff vocab in
  let emodel = glmodel_map enc_const (to_glmodel model) in
  Printf.eprintf "weights ready.\n%!";
  let ops = enc_ops (ZZ.of_int fb) (ZZ.of_int (fb + 32)) in
  let eps = enc_const (f32_div (f32_of_Z (ZZ.of_int 1)) (f32_of_Z (ZZ.of_int 100000))) in
  let tab t = List.map (List.map enc_const) t in
  let rows toks =
    let n = List.length toks in
    g_llama_logits ops nh nkv hd eps emodel
      (tab (f32_rope_cos model.lw_invf n)) (tab (f32_rope_sin model.lw_invf n)) toks in
  over_windows outdir first (min last (Array.length windows))
    (window fb vocab windows f32dir outdir rows)
