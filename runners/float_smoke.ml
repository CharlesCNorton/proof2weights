(* float_smoke.ml - run the extracted verified float GPT-2 on a tiny model.
   Proves the extraction executes end to end and yields a finite,
   correctly-dimensioned output. *)

open Phases1_15_complete

(* nat extracts to int; Z/positive stay inductive, so integer literals are
   converted to inductive Z at the boundary before f32_of_Z. *)
let rec pos_of_int n =
  if n <= 1 then XH
  else if n land 1 = 0 then XO (pos_of_int (n / 2)) else XI (pos_of_int (n / 2))
let z_of_int n = if n = 0 then Z0 else if n > 0 then Zpos (pos_of_int n) else Zneg (pos_of_int (-n))

let vec n v = List.init n (fun _ -> f32_of_Z (z_of_int v))
let mat r c v = List.init r (fun _ -> vec c v)

let d = 4
let ff = 8
let vocab = 5
let npos = 4

let cfg = { gpt2_inf_n_embd = d; gpt2_inf_n_head = 2; gpt2_inf_n_layer = 1;
            gpt2_inf_n_inner = ff; gpt2_inf_vocab_size = vocab;
            gpt2_inf_n_positions = npos }

let ln () = { f32_ln_weight = vec d 1; f32_ln_bias = vec d 0 }

let attn = { f32_attn_c_attn_weight = mat d (3 * d) 1; f32_attn_c_attn_bias = vec (3 * d) 0;
             f32_attn_c_proj_weight = mat d d 1; f32_attn_c_proj_bias = vec d 0 }

let mlp = { f32_mlp_c_fc_weight = mat d ff 1; f32_mlp_c_fc_bias = vec ff 0;
            f32_mlp_c_proj_weight = mat ff d 1; f32_mlp_c_proj_bias = vec d 0 }

let block = { f32_block_ln_1 = ln (); f32_block_attn = attn;
              f32_block_ln_2 = ln (); f32_block_mlp = mlp }

let model = { f32_wte = mat vocab d 1; f32_wpe = mat npos d 1;
              f32_blocks = [block]; f32_ln_f = ln () }

let () =
  let toks = [0; 1; 2] in
  let out = f32_gpt2_forward cfg f32_ln_eps model toks in
  Printf.printf "f32_gpt2_forward : rows=%d cols=%d clean=%b\n"
    (f32_mat_rows out) (f32_mat_cols out) (f32_output_clean out);
  let logits = f32_gpt2_logits cfg f32_ln_eps model toks in
  Printf.printf "f32_gpt2_logits  : rows=%d cols=%d clean=%b\n"
    (f32_mat_rows logits) (f32_mat_cols logits) (f32_output_clean logits)
