(* gpt2_bound_native.ml - GPT-2 through the annotated forward pass of RunErr.v,
   against the native extraction. Every value is the binary32 value
   gpt2_talk_native computes, paired with a binary64 bound on its distance from
   the exact real evaluation of the same network on the same weights; the
   theorem is gpt2_logits_bounded. The composition is gpt2_talk_native's,
   operation for operation, over the annotated primitives.

   After each stage of each block (the two layer norms, the query/key/value
   projection, the attention heads, the output projection, the feed-forward
   projection, GELU, the feed-forward output and both residuals) it prints the
   largest magnitude, the largest bound and the number of infinite bounds. It
   then prints the top-10 next-token logits with their bounds, the largest
   bound over the whole row, and whether the bounds separate the top logit
   from every other.

   usage: gpt2_bound_native <path> d n_head n_layer ff vocab <tok,...> *)

open Runerr_native

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in really_input ic b 0 n; close_in ic; b

let u64_le b off =
  let r = ref 0 in
  for i = 7 downto 0 do r := (!r * 256) + Char.code (Bytes.get b (off + i)) done; !r

(* each weight enters exact: the binary32 value with a zero bound *)
let dec1 b o =
  ann_const
    (f32_bytes_to_binary32
       [ Char.code (Bytes.get b o); Char.code (Bytes.get b (o + 1));
         Char.code (Bytes.get b (o + 2)); Char.code (Bytes.get b (o + 3)) ])

let dec_vec b start count = List.init count (fun i -> dec1 b (start + 4 * i))
let dec_wt b start in_dim out_dim =
  List.init out_dim (fun o -> List.init in_dim (fun i -> dec1 b (start + 4 * (i * out_dim + o))))

let coqstr s = List.init (String.length s) (fun i -> s.[i])
let rec ftake n l = if n <= 0 then [] else match l with [] -> [] | x :: r -> x :: ftake (n - 1) r
let rec fdrop n l = if n <= 0 then l else match l with [] -> [] | _ :: r -> fdrop (n - 1) r
let rec map3 f a b c =
  match a, b, c with x :: a', y :: b', z :: c' -> f x y z :: map3 f a' b' c' | _, _, _ -> []

let stats name rows =
  let mx = ref 0.0 and vmax = ref 0.0 and inf = ref 0 in
  List.iter (List.iter (fun (v, d) ->
    if abs_float v > !vmax then vmax := abs_float v;
    if Float.is_finite d then (if d > !mx then mx := d) else incr inf)) rows;
  Printf.printf "  %-8s max |value| %.3e  max bound %.3e  infinite %d\n%!" name !vmax !mx !inf

let () =
  let path  = Sys.argv.(1) in
  let d     = int_of_string Sys.argv.(2) in
  let nh    = int_of_string Sys.argv.(3) in
  let nl    = int_of_string Sys.argv.(4) in
  let ff    = int_of_string Sys.argv.(5) in
  let vocab = int_of_string Sys.argv.(6) in
  let toks  = List.map int_of_string (String.split_on_char ',' Sys.argv.(7)) in
  let head_dim = d / nh in
  let b = read_file path in
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let off name =
    match json_tensor_offsets header (coqstr name) with
    | s :: _ :: _ -> base + s
    | _ -> failwith ("offsets not found for " ^ name) in
  let lin2d wt bias x =
    List.map (fun row -> g_vec_add ann_ops (g_mat_vec_mul ann_ops wt row) bias) x in
  let eps = ann_const f32_ln_eps in

  let seq = List.length toks in
  let wte_s = off "wte.weight" in
  let wpe_s = off "wpe.weight" in
  let tok_emb = List.map (fun t -> dec_vec b (wte_s + 4 * t * d) d) toks in
  let pos_emb = List.init seq (fun p -> dec_vec b (wpe_s + 4 * p * d) d) in
  let hidden = ref (g_add_matrices ann_ops tok_emb pos_emb) in
  Printf.printf "embedding\n";
  stats "input" !hidden;

  for i = 0 to nl - 1 do
    Printf.printf "block %d\n" i;
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
    let ln1 = g_layer_norm_2d ann_ops ln1w ln1b eps !hidden in
    stats "ln_1" ln1;
    let qkv = lin2d caw cab ln1 in
    stats "c_attn" qkv;
    let qs = List.map (fun row -> ftake d row) qkv in
    let ks = List.map (fun row -> ftake d (fdrop d row)) qkv in
    let vs = List.map (fun row -> fdrop (2 * d) row) qkv in
    let qh = g_split_into_heads nh qs in
    let kh = g_split_into_heads nh ks in
    let vh = g_split_into_heads nh vs in
    let heads = g_concat_heads
      (map3 (fun q k v -> g_causal_attention ann_ops q k v head_dim) qh kh vh) in
    stats "heads" heads;
    let attn = lin2d cpw cpb heads in
    stats "c_proj" attn;
    let hidden2 = g_add_matrices ann_ops !hidden attn in
    stats "resid_1" hidden2;
    let ln2 = g_layer_norm_2d ann_ops ln2w ln2b eps hidden2 in
    stats "ln_2" ln2;
    let h = lin2d fcw fcb ln2 in
    stats "c_fc" h;
    let hg = List.map (g_gelu_vec ann_ops) h in
    stats "gelu" hg;
    let mlp = lin2d mpw mpb hg in
    stats "mlp_proj" mlp;
    hidden := g_add_matrices ann_ops hidden2 mlp;
    stats "resid_2" !hidden
  done;

  let lnfw = dec_vec b (off "ln_f.weight") d in
  let lnfb = dec_vec b (off "ln_f.bias") d in
  let final = g_layer_norm_2d ann_ops lnfw lnfb eps !hidden in
  Printf.printf "final\n";
  stats "ln_f" final;
  let last = List.nth final (seq - 1) in
  let a = Array.init vocab (fun j -> ann_dot last (dec_vec b (wte_s + 4 * j * d) d)) in
  let idx = Array.init vocab (fun i -> i) in
  Array.sort (fun i j -> compare (fst a.(j)) (fst a.(i))) idx;
  let row_max = Array.fold_left (fun m (_, d) -> if Float.is_finite d && d > m then d else m) 0.0 a in
  let row_inf = Array.fold_left (fun n (_, d) -> if Float.is_finite d then n else n + 1) 0 a in
  Printf.printf "logits  max bound %.3e  infinite %d\n" row_max row_inf;
  Printf.printf "top-10 next-token logits (value, bound):\n";
  for r = 0 to min 9 (vocab - 1) do
    let i = idx.(r) in Printf.printf "  %6d  %.9g  %.3e\n" i (fst a.(i)) (snd a.(i))
  done;
  let t = idx.(0) in
  let (vt, dt) = a.(t) in
  let separated = ref (Float.is_finite dt) in
  Array.iteri (fun j (vj, dj) ->
    if j <> t && not (Float.is_finite dj && vj +. dj < vt -. dt) then separated := false) a;
  Printf.printf "top token %d separated from every other by the bounds: %b\n" t !separated
