(* llama_talk_inductive.ml - the SmolLM2 forward against the INDUCTIVE
   extraction, where binary32 is Flocq's binary_float and Z is its inductive
   datatype. No floating-point hardware is trusted anywhere: every arithmetic
   operation is the computational content of its proof.

   This is llama_talk_native.ml with two changes. Integers cross the boundary
   as inductive Z rather than as machine int, and weights are decoded one
   layer at a time rather than all at once. The second change is forced: an
   inductive binary_float carries its mantissa as a chain of positive
   constructors, roughly twenty-five times the footprint of an OCaml float,
   so materialising all 135M parameters at once would need tens of gigabytes.
   Streaming holds one layer, about 3.5M parameters, at a time.

   The embedding is never held. Rows are decoded on demand for the token
   lookup and streamed for the logit projection.

   usage: llama_talk_inductive <path> d n_layer n_head n_kv ff vocab <tok,...> [topk] *)

open Llama_inductive

let rec pos_of_int n =
  if n <= 1 then XH
  else if n land 1 = 0 then XO (pos_of_int (n / 2)) else XI (pos_of_int (n / 2))
let z_of_int n =
  if n = 0 then Z0 else if n > 0 then Zpos (pos_of_int n) else Zneg (pos_of_int (-n))
let rec int_of_pos = function XH -> 1 | XO p -> 2 * int_of_pos p | XI p -> 2 * int_of_pos p + 1
let int_of_z = function Z0 -> 0 | Zpos p -> int_of_pos p | Zneg p -> - (int_of_pos p)

let b2f = function
  | B754_zero _ -> 0.0
  | B754_infinity s -> if s then neg_infinity else infinity
  | B754_nan -> nan
  | B754_finite (s, m, e) ->
      (if s then -1.0 else 1.0) *. float_of_int (int_of_pos m)
      *. (2.0 ** float_of_int (int_of_z e))

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in really_input ic b 0 n; close_in ic; b

let u64_le b off =
  let r = ref 0 in
  for i = 7 downto 0 do r := (!r * 256) + Char.code (Bytes.get b (off + i)) done; !r

(* verified per-value decode: four little-endian bytes -> Flocq binary32 *)
let dec1 b o =
  f32_bytes_to_binary32
    [ z_of_int (Char.code (Bytes.get b o));
      z_of_int (Char.code (Bytes.get b (o + 1)));
      z_of_int (Char.code (Bytes.get b (o + 2)));
      z_of_int (Char.code (Bytes.get b (o + 3))) ]

let dec_vec b start count = List.init count (fun i -> dec1 b (start + 4 * i))
let dec_mat b start rows cols = List.init rows (fun r -> dec_vec b (start + 4 * r * cols) cols)

let coqstr s = List.init (String.length s) (fun i -> s.[i])
let rec ftake n l = if n <= 0 then [] else match l with [] -> [] | x :: r -> x :: ftake (n - 1) r
let rec fdrop n l = if n <= 0 then l else match l with [] -> [] | _ :: r -> fdrop (n - 1) r
let slice off len row = ftake len (fdrop off row)
let rec map3 f a b c =
  match a, b, c with x :: a', y :: b', z :: c' -> f x y z :: map3 f a' b' c' | _, _, _ -> []

let () =
  let path  = Sys.argv.(1) in
  let d     = int_of_string Sys.argv.(2) in
  let nl    = int_of_string Sys.argv.(3) in
  let nh    = int_of_string Sys.argv.(4) in
  let nkv   = int_of_string Sys.argv.(5) in
  let ff    = int_of_string Sys.argv.(6) in
  let vocab = int_of_string Sys.argv.(7) in
  let toks  = List.map int_of_string (String.split_on_char ',' Sys.argv.(8)) in
  let topk  = if Array.length Sys.argv > 9 then int_of_string Sys.argv.(9) else vocab in
  let hd = d / nh in
  let group = nh / nkv in
  let kvd = nkv * hd in
  let b = read_file path in
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let off name =
    match json_tensor_offsets header (coqstr name) with
    | s :: _ :: _ -> base + s
    | _ -> failwith ("offsets not found for " ^ name) in
  let eps = f32_div (f32_of_Z (z_of_int 1)) (f32_of_Z (z_of_int 100000)) in
  let embed_s = off "embed_tokens.weight" in
  let emb_row t = dec_vec b (embed_s + 4 * t * d) d in
  let invf = dec_vec b (off "rope.inv_freq") (hd / 2) in

  let rope pos x =
    let xa = ftake (hd / 2) x and xb = fdrop (hd / 2) x in
    let pf = f32_of_Z (z_of_int pos) in
    let cs = List.map (fun fj -> let a = f32_mult pf fj in (f32_cos a, f32_sin a)) invf in
    let outa = map3 (fun xaj xbj (c, s) -> f32_minus (f32_mult xaj c) (f32_mult xbj s)) xa xb cs in
    let outb = map3 (fun xaj xbj (c, s) -> f32_plus (f32_mult xbj c) (f32_mult xaj s)) xa xb cs in
    outa @ outb in

  Printf.eprintf "prefill over %d tokens, %d layers, streaming weights\n%!"
    (List.length toks) nl;
  let hidden = ref (List.map emb_row toks) in
  for i = 0 to nl - 1 do
    let p = Printf.sprintf "layers.%d." i in
    let ln1 = dec_vec b (off (p ^ "input_layernorm.weight")) d in
    let qw = dec_mat b (off (p ^ "self_attn.q_proj.weight")) d d in
    let kw = dec_mat b (off (p ^ "self_attn.k_proj.weight")) kvd d in
    let vw = dec_mat b (off (p ^ "self_attn.v_proj.weight")) kvd d in
    let ow = dec_mat b (off (p ^ "self_attn.o_proj.weight")) d d in
    let h = List.map (fun row -> f32_rmsnorm ln1 eps row) !hidden in
    let q = List.map (fun row -> f32_mat_vec_mul qw row) h in
    let k = List.map (fun row -> f32_mat_vec_mul kw row) h in
    let v = List.map (fun row -> f32_mat_vec_mul vw row) h in
    let qheads = List.init nh  (fun hh -> List.mapi (fun pos row -> rope pos (slice (hh * hd) hd row)) q) in
    let kheads = List.init nkv (fun kk -> List.mapi (fun pos row -> rope pos (slice (kk * hd) hd row)) k) in
    let vheads = List.init nkv (fun kk -> List.map  (fun row -> slice (kk * hd) hd row) v) in
    let headouts = List.init nh (fun hh ->
      f32_causal_attention (List.nth qheads hh) (List.nth kheads (hh / group))
                           (List.nth vheads (hh / group)) hd) in
    let attn = List.map (fun row -> f32_mat_vec_mul ow row) (f32_concat_heads headouts) in
    let hidden2 = List.map2 f32_vec_add !hidden attn in
    let ln2 = dec_vec b (off (p ^ "post_attention_layernorm.weight")) d in
    let gw = dec_mat b (off (p ^ "mlp.gate_proj.weight")) ff d in
    let uw = dec_mat b (off (p ^ "mlp.up_proj.weight")) ff d in
    let dw = dec_mat b (off (p ^ "mlp.down_proj.weight")) d ff in
    let h2 = List.map (fun row -> f32_rmsnorm ln2 eps row) hidden2 in
    let gate = List.map (fun row -> f32_mat_vec_mul gw row) h2 in
    let up = List.map (fun row -> f32_mat_vec_mul uw row) h2 in
    let act = List.map2 (fun g u -> f32_vec_mult (f32_silu_vec g) u) gate up in
    let down = List.map (fun row -> f32_mat_vec_mul dw row) act in
    hidden := List.map2 f32_vec_add hidden2 down;
    Printf.eprintf "  layer %d/%d\n%!" (i + 1) nl
  done;
  let normw = dec_vec b (off "norm.weight") d in
  let final = List.map (fun row -> f32_rmsnorm normw eps row) !hidden in
  let last = List.nth final (List.length toks - 1) in
  Printf.eprintf "projecting %d logits (streaming the embedding)\n%!" topk;
  let a = Array.init topk (fun j ->
    if j mod 4096 = 0 then Printf.eprintf "  logit %d/%d\n%!" j topk;
    b2f (f32_dot last (emb_row j))) in
  let idx = Array.init topk (fun i -> i) in
  Array.sort (fun i j -> compare a.(j) a.(i)) idx;
  Printf.printf "top-10 next-token logits (inductive, no FPU trusted):\n";
  for r = 0 to min 9 (topk - 1) do
    let i = idx.(r) in Printf.printf "  %6d  %.9g\n" i a.(i)
  done
