(* gpt2_talk.ml - run a real safetensors GPT-2 through the verified extracted
   IEEE-754 float operators.

   The top-level f32_gpt2_forward is faithful but intractable at GPT-2 scale:
   the verified f32_mat_transpose inside f32_linear_forward is O(in*out^2) in
   list-nth, and the verified loader would materialize a ~500 MB file as a
   linked list of boxed bignums. This driver instead reads bytes natively,
   decodes each f32 with the VERIFIED f32_bytes_to_binary32, and composes the
   VERIFIED arithmetic primitives (f32_mat_vec_mul, f32_dot, f32_layer_norm_2d,
   f32_causal_attention, f32_concat_heads, f32_gelu_vec, f32_add_matrices) in
   the exact order f32_block_forward / f32_gpt2_logits specify. Weights are
   decoded pre-transposed so f32_mat_vec_mul computes the same dot products
   f32_linear_forward would, without building the transpose via nth. Weights
   stream block by block, so memory stays ~GB. Every floating-point value is
   the verified extracted computation; only byte addressing is native.

   usage: gpt2_talk <full|next> <path> d n_head n_layer ff vocab n_pos <tok,tok,...> *)

open Phases1_15_complete

let rec pos_of_int n =
  if n <= 1 then XH
  else if n land 1 = 0 then XO (pos_of_int (n / 2)) else XI (pos_of_int (n / 2))
let z_of_int n = if n = 0 then Z0 else if n > 0 then Zpos (pos_of_int n) else Zneg (pos_of_int (-n))
let rec int_of_pos = function XH -> 1 | XO p -> 2 * int_of_pos p | XI p -> 2 * int_of_pos p + 1
let int_of_z = function Z0 -> 0 | Zpos p -> int_of_pos p | Zneg p -> - (int_of_pos p)

let b2f = function
  | B754_zero _ -> 0.0
  | B754_infinity s -> if s then neg_infinity else infinity
  | B754_nan -> nan
  | B754_finite (s, m, e) ->
      (if s then -1.0 else 1.0) *. float_of_int (int_of_pos m) *. (2.0 ** float_of_int (int_of_z e))

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in really_input ic b 0 n; close_in ic; b

let u64_le b off =
  let r = ref 0 in
  for i = 7 downto 0 do r := (!r * 256) + Char.code (Bytes.get b (off + i)) done; !r

(* verified per-value decode: 4 little-endian bytes -> Flocq binary32 *)
let dec1 b o =
  f32_bytes_to_binary32
    [ z_of_int (Char.code (Bytes.get b o));
      z_of_int (Char.code (Bytes.get b (o + 1)));
      z_of_int (Char.code (Bytes.get b (o + 2)));
      z_of_int (Char.code (Bytes.get b (o + 3))) ]

let dec_vec b start count = List.init count (fun i -> dec1 b (start + 4 * i))

(* decode a row-major [in_dim, out_dim] weight directly into transposed
   [out_dim, in_dim], so f32_mat_vec_mul on it reproduces f32_linear_forward. *)
let dec_wt b start in_dim out_dim =
  List.init out_dim (fun o -> List.init in_dim (fun i -> dec1 b (start + 4 * (i * out_dim + o))))

let coqstr s = List.init (String.length s) (fun i -> s.[i])
let rec ftake n l = if n <= 0 then [] else match l with [] -> [] | x :: r -> x :: ftake (n - 1) r
let rec fdrop n l = if n <= 0 then l else match l with [] -> [] | _ :: r -> fdrop (n - 1) r
let rec map3 f a b c =
  match a, b, c with x :: a', y :: b', z :: c' -> f x y z :: map3 f a' b' c' | _, _, _ -> []

let () =
  let mode  = Sys.argv.(1) in
  let path  = Sys.argv.(2) in
  let d     = int_of_string Sys.argv.(3) in
  let nh    = int_of_string Sys.argv.(4) in
  let nl    = int_of_string Sys.argv.(5) in
  let ff    = int_of_string Sys.argv.(6) in
  let vocab = int_of_string Sys.argv.(7) in
  let _npos = int_of_string Sys.argv.(8) in
  let toks  = List.map int_of_string (String.split_on_char ',' Sys.argv.(9)) in
  let head_dim = d / nh in
  let b = read_file path in
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let off name =
    match json_tensor_offsets header (coqstr name) with
    | s :: _ :: _ -> base + s
    | _ -> failwith ("offsets not found for " ^ name) in
  (* linear: wt is [out, in] (pre-transposed); rows of x are [in] -> [out] + bias.
     Identical arithmetic to the verified f32_linear_forward_2d. *)
  let lin2d wt bias x = List.map (fun row -> f32_vec_add (f32_mat_vec_mul wt row) bias) x in

  let seq = List.length toks in
  let wte_s = off "wte.weight" in
  let wpe_s = off "wpe.weight" in
  let tok_emb = List.map (fun t -> dec_vec b (wte_s + 4 * t * d) d) toks in
  let pos_emb = List.init seq (fun p -> dec_vec b (wpe_s + 4 * p * d) d) in
  let hidden = ref (f32_add_matrices tok_emb pos_emb) in

  for i = 0 to nl - 1 do
    let p = Printf.sprintf "h.%d." i in
    let ln1w = dec_vec b (off (p ^ "ln_1.weight")) d in
    let ln1b = dec_vec b (off (p ^ "ln_1.bias")) d in
    let ln2w = dec_vec b (off (p ^ "ln_2.weight")) d in
    let ln2b = dec_vec b (off (p ^ "ln_2.bias")) d in
    let caw = dec_wt  b (off (p ^ "attn.c_attn.weight")) d (3 * d) in
    let cab = dec_vec b (off (p ^ "attn.c_attn.bias")) (3 * d) in
    let cpw = dec_wt  b (off (p ^ "attn.c_proj.weight")) d d in
    let cpb = dec_vec b (off (p ^ "attn.c_proj.bias")) d in
    let fcw = dec_wt  b (off (p ^ "mlp.c_fc.weight")) d ff in
    let fcb = dec_vec b (off (p ^ "mlp.c_fc.bias")) ff in
    let mpw = dec_wt  b (off (p ^ "mlp.c_proj.weight")) ff d in
    let mpb = dec_vec b (off (p ^ "mlp.c_proj.bias")) d in
    let ln1 = f32_layer_norm_2d ln1w ln1b f32_ln_eps !hidden in
    let qkv = lin2d caw cab ln1 in
    let qs = List.map (fun row -> ftake d row) qkv in
    let ks = List.map (fun row -> ftake d (fdrop d row)) qkv in
    let vs = List.map (fun row -> fdrop (2 * d) row) qkv in
    let qh = f32_split_into_heads nh qs in
    let kh = f32_split_into_heads nh ks in
    let vh = f32_split_into_heads nh vs in
    let heads = map3 (fun q k v -> f32_causal_attention q k v head_dim) qh kh vh in
    let attn = lin2d cpw cpb (f32_concat_heads heads) in
    let hidden2 = f32_add_matrices !hidden attn in
    let ln2 = f32_layer_norm_2d ln2w ln2b f32_ln_eps hidden2 in
    let h = lin2d fcw fcb ln2 in
    let hg = List.map f32_gelu_vec h in
    let mlp = lin2d mpw mpb hg in
    hidden := f32_add_matrices hidden2 mlp;
    Printf.eprintf "block %d/%d done\n%!" (i + 1) nl
  done;

  let lnfw = dec_vec b (off "ln_f.weight") d in
  let lnfb = dec_vec b (off "ln_f.bias") d in
  let final = f32_layer_norm_2d lnfw lnfb f32_ln_eps !hidden in
  let logits_for_row hrow =
    Array.init vocab (fun j -> b2f (f32_dot hrow (dec_vec b (wte_s + 4 * j * d) d))) in

  match mode with
  | "full" ->
      List.iter (fun hrow ->
        let a = logits_for_row hrow in
        print_string (String.concat " " (Array.to_list (Array.map (Printf.sprintf "%.9g") a)));
        print_newline ()) final
  | _ ->
      let last = List.nth final (seq - 1) in
      Printf.eprintf "projecting %d logits...\n%!" vocab;
      let a = logits_for_row last in
      let idx = Array.init vocab (fun i -> i) in
      Array.sort (fun i j -> compare a.(j) a.(i)) idx;
      Printf.printf "top-10 next-token logits:\n";
      for r = 0 to 9 do let i = idx.(r) in Printf.printf "  %6d  %.4f\n" i a.(i) done
