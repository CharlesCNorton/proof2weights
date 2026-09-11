(* qwen_talk_inductive.ml - the Qwen3.5 forward against the INDUCTIVE
   extraction, where binary32 is Flocq's binary_float and Z is its inductive
   datatype. No floating-point hardware is trusted: every arithmetic operation
   is the computational content of its proof.

   The arithmetic is qwen_talk_native.ml's, operation for operation. Integers
   cross the boundary as inductive Z. A weight matrix is never held whole: a
   projection decodes one row at a time and takes its dot product with every
   position's input before moving to the next row, which performs exactly the
   products f32_mat_vec_mul performs. The embedding is decoded row by row for
   the token lookup and streamed for the logit projection.

   usage: qwen_talk_inductive <path> d nl nh nkv hd rd ff vocab lnh lhd ck <tok,...> *)

open Qwen_inductive

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
let last_n n l = let k = List.length l in if k <= n then l else fdrop (k - n) l

(* f32_mat_vec_mul w h for every h in hs, where w is stored as [rows] rows of
   [cols] entries starting at [start]. Row r of the result for position t is
   f32_dot row_r h_t, the product f32_mat_vec_mul forms. *)
let project b start rows cols hs =
  let n = List.length hs in
  let res = Array.make_matrix n rows f32_zero in
  for r = 0 to rows - 1 do
    let row = dec_vec b (start + 4 * r * cols) cols in
    List.iteri (fun t h -> res.(t).(r) <- f32_dot row h) hs
  done;
  List.init n (fun t -> Array.to_list res.(t))

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
  let toks  = List.map int_of_string (String.split_on_char ',' Sys.argv.(13)) in
  let group = nh / nkv in
  let kvd = nkv * hd in
  let ldim = lnh * lhd in
  let qkvd = 3 * ldim in
  let zerov n = List.init n (fun _ -> f32_zero) in
  let b = read_file path in
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let off name =
    match json_tensor_offsets header (coqstr name) with
    | s :: _ :: _ -> base + s
    | _ -> failwith ("offsets not found for " ^ name) in
  let eps = f32_div f32_one (f32_of_Z (z_of_int 1000000)) in
  let embed_s = off "embed_tokens.weight" in
  let emb_row t = dec_vec b (embed_s + 4 * t * d) d in
  let invf = dec_vec b (off "rope.inv_freq") (rd / 2) in
  let rope_cs pos =
    let pf = f32_of_Z (z_of_int pos) in
    let ang = List.map (fun fj -> f32_mult pf fj) invf in
    let cosv = List.map f32_cos ang and sinv = List.map f32_sin ang in
    (cosv @ cosv, sinv @ sinv) in
  let layer_is_full i = (i mod 4) = 3 in
  let lname i s = Printf.sprintf "layers.%d.%s" i s in
  let n = List.length toks in

  Printf.eprintf "prefill over %d tokens, %d layers, streaming weights\n%!" n nl;
  let hs = ref (List.map emb_row toks) in
  for i = 0 to nl - 1 do
    let ln1 = dec_vec b (off (lname i "input_layernorm.weight")) d in
    let hn = List.map (fun row -> f32_rmsnorm_zc ln1 eps row) !hs in
    let mixed =
      if layer_is_full i then begin
        let qgs = project b (off (lname i "self_attn.q_proj.weight")) (nh * hd * 2) d hn in
        let kraws = project b (off (lname i "self_attn.k_proj.weight")) kvd d hn in
        let vraws = project b (off (lname i "self_attn.v_proj.weight")) kvd d hn in
        let qnw = dec_vec b (off (lname i "self_attn.q_norm.weight")) hd in
        let knw = dec_vec b (off (lname i "self_attn.k_norm.weight")) hd in
        let kc = Array.make nkv [] and vc = Array.make nkv [] in
        let gated = List.mapi (fun pos (qg, (kraw, vraw)) ->
          let (cosv, sinv) = rope_cs pos in
          for c = 0 to nkv - 1 do
            let knew = f32_partial_rope rd cosv sinv
                         (f32_rmsnorm_zc knw eps (slice (c * hd) hd kraw)) in
            kc.(c) <- kc.(c) @ [knew];
            vc.(c) <- vc.(c) @ [slice (c * hd) hd vraw]
          done;
          let gate = List.concat
            (List.init nh (fun hh -> slice (hh * hd * 2 + hd) hd qg)) in
          let heads = List.init nh (fun hh ->
            let q = f32_partial_rope rd cosv sinv
                      (f32_rmsnorm_zc qnw eps (slice (hh * hd * 2) hd qg)) in
            f32_attend q kc.(hh / group) vc.(hh / group) hd) in
          f32_gate_sigmoid gate (List.concat heads))
          (List.combine qgs (List.combine kraws vraws)) in
        project b (off (lname i "self_attn.o_proj.weight")) d (nh * hd) gated
      end else begin
        let qkvs = project b (off (lname i "linear_attn.in_proj_qkv.weight")) qkvd d hn in
        let zs   = project b (off (lname i "linear_attn.in_proj_z.weight")) ldim d hn in
        let avs  = project b (off (lname i "linear_attn.in_proj_a.weight")) lnh d hn in
        let bvs  = project b (off (lname i "linear_attn.in_proj_b.weight")) lnh d hn in
        let cw   = dec_mat b (off (lname i "linear_attn.conv1d.weight")) qkvd ck in
        let alog = dec_vec b (off (lname i "linear_attn.A_log")) lnh in
        let dtb  = dec_vec b (off (lname i "linear_attn.dt_bias")) lnh in
        let nw   = dec_vec b (off (lname i "linear_attn.norm.weight")) lhd in
        let dstate = Array.init lnh (fun _ -> f32_delta_state0 lhd lhd) in
        let hist = ref [] in
        let outs = List.map (fun (qkv, (z, (av, bv))) ->
          let h = !hist @ [qkv] in
          let k = List.length h in
          let win = if k >= ck then last_n ck h
                    else List.init (ck - k) (fun _ -> zerov qkvd) @ h in
          hist := last_n (ck - 1) h;
          let c = f32_conv_step cw [] win in
          List.concat (List.init lnh (fun hh ->
            let q = f32_delta_prep_q eps lhd (slice (hh * lhd) lhd c) in
            let kk = f32_l2norm eps (slice (ldim + hh * lhd) lhd c) in
            let v = slice (2 * ldim + hh * lhd) lhd c in
            let beta = f32_sigmoid (List.nth bv hh) in
            let g = f32_delta_decay (List.nth alog hh) (List.nth dtb hh) (List.nth av hh) in
            let (st', o) = f32_delta_step beta g q kk v dstate.(hh) in
            dstate.(hh) <- st';
            f32_rmsnorm_gated nw eps (slice (hh * lhd) lhd z) o)))
          (List.combine qkvs (List.combine zs (List.combine avs bvs))) in
        project b (off (lname i "linear_attn.out_proj.weight")) d ldim outs
      end in
    let hidden2 = List.map2 f32_vec_add !hs mixed in
    let ln2 = dec_vec b (off (lname i "post_attention_layernorm.weight")) d in
    let h2 = List.map (fun row -> f32_rmsnorm_zc ln2 eps row) hidden2 in
    let gates = project b (off (lname i "mlp.gate_proj.weight")) ff d h2 in
    let ups = project b (off (lname i "mlp.up_proj.weight")) ff d h2 in
    let acts = List.map2 (fun g u -> f32_vec_mult (f32_silu_vec g) u) gates ups in
    let downs = project b (off (lname i "mlp.down_proj.weight")) d ff acts in
    hs := List.map2 f32_vec_add hidden2 downs;
    Printf.eprintf "  layer %d/%d (%s)\n%!" (i + 1) nl
      (if layer_is_full i then "attn" else "delta")
  done;
  let normw = dec_vec b (off "norm.weight") d in
  let final = List.map (fun row -> f32_rmsnorm_zc normw eps row) !hs in
  let last = List.nth final (n - 1) in
  Printf.eprintf "projecting %d logits (streaming the embedding)\n%!" vocab;
  let a = Array.init vocab (fun j ->
    if j mod 16384 = 0 then Printf.eprintf "  logit %d/%d\n%!" j vocab;
    b2f (f32_dot last (emb_row j))) in
  let idx = Array.init vocab (fun i -> i) in
  Array.sort (fun i j -> compare a.(j) a.(i)) idx;
  Printf.printf "top-10 next-token logits (inductive, no FPU trusted):\n";
  for r = 0 to 9 do
    let i = idx.(r) in Printf.printf "  %7d  %.9g\n" i a.(i)
  done
