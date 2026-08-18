(* gpt2_talk_native.ml - identical composition to gpt2_talk.ml, but against the
   native IEEE-754 extraction (phases1_15_native: binary32 = OCaml float, byte =
   int). Every float value is still produced by the extracted operators; those
   operators are now the host's hardware float rounded to binary32, the trusted
   CompCert-style boundary. ~1000x faster, so generation is feasible.

   usage: gpt2_talk_native <full|next> <path> d n_head n_layer ff vocab n_pos <tok,...> *)

open Phases1_15_native

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in really_input ic b 0 n; close_in ic; b

let u64_le b off =
  let r = ref 0 in
  for i = 7 downto 0 do r := (!r * 256) + Char.code (Bytes.get b (off + i)) done; !r

(* verified per-value decode: 4 little-endian bytes -> binary32 (native) *)
let dec1 b o =
  f32_bytes_to_binary32
    [ Char.code (Bytes.get b o); Char.code (Bytes.get b (o + 1));
      Char.code (Bytes.get b (o + 2)); Char.code (Bytes.get b (o + 3)) ]

let dec_vec b start count = List.init count (fun i -> dec1 b (start + 4 * i))
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
    Array.init vocab (fun j -> f32_dot hrow (dec_vec b (wte_s + 4 * j * d) d)) in

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
