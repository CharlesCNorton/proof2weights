(* qwen_talk_native.ml - run Qwen3.5 through the verified extracted IEEE-754
   operators (native build), with greedy generation.

   Qwen3.5 alternates three gated-DeltaNet layers with one gated full-attention
   layer. Both mixers are composed here from the verified primitives: the
   DeltaNet path uses f32_conv_step, f32_l2norm, f32_delta_decay and
   f32_delta_step, and the attention path uses f32_partial_rope,
   f32_rmsnorm_zc, f32_softmax and f32_gate_sigmoid. Every float value is
   produced by a verified operator; only byte addressing and the structural
   composition are native.

   Generation carries three caches: the keys and values of the full-attention
   layers, the recurrent state matrix of each DeltaNet head, and the trailing
   convolution window. A decode step therefore costs one token of arithmetic
   rather than a re-run of the prefix.

   The stack is walked layer by layer, so each layer's weights are decoded once
   per call and dropped, and memory stays well below the size of the model. The
   embedding is never held: rows are decoded on demand for the lookup and
   streamed for the logit projection.

   In serve mode the weights stay on disk and the process answers queries from
   stdin (one per line: "<comma ids> <max_new>"), streaming "TOK <id>" per token
   and "END <comma gen ids>" per query, after announcing the weight checksum a
   receipt binds to. Each query starts from an empty cache.

   usage:
     qwen_talk_native <path> d nl nh nkv hd rd ff vocab lnh lhd ck <tok,...>
       -> top-10 next-token logits
     qwen_talk_native <path> d nl nh nkv hd rd ff vocab lnh lhd ck <tok,...> <max_new> <eos>
       -> greedy continuation, one token id per line
     qwen_talk_native <path> d nl nh nkv hd rd ff vocab lnh lhd ck serve <eos>
       -> persistent server *)

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
let dec_mat b start rows cols = List.init rows (fun r -> dec_vec b (start + 4 * r * cols) cols)

let coqstr s = List.init (String.length s) (fun i -> s.[i])
let rec ftake n l = if n <= 0 then [] else match l with [] -> [] | x :: r -> x :: ftake (n - 1) r
let rec fdrop n l = if n <= 0 then l else match l with [] -> [] | _ :: r -> fdrop (n - 1) r
let slice off len row = ftake len (fdrop off row)
let last_n n l = let k = List.length l in if k <= n then l else fdrop (k - n) l

let () =
  let path  = Sys.argv.(1) in
  let d     = int_of_string Sys.argv.(2) in    (* hidden size            *)
  let nl    = int_of_string Sys.argv.(3) in    (* layers                 *)
  let nh    = int_of_string Sys.argv.(4) in    (* attention query heads  *)
  let nkv   = int_of_string Sys.argv.(5) in    (* attention kv heads     *)
  let hd    = int_of_string Sys.argv.(6) in    (* attention head dim     *)
  let rd    = int_of_string Sys.argv.(7) in    (* rotary dim (partial)   *)
  let ff    = int_of_string Sys.argv.(8) in    (* mlp intermediate       *)
  let vocab = int_of_string Sys.argv.(9) in
  let lnh   = int_of_string Sys.argv.(10) in   (* deltanet heads         *)
  let lhd   = int_of_string Sys.argv.(11) in   (* deltanet head dim      *)
  let ck    = int_of_string Sys.argv.(12) in   (* conv kernel            *)
  let toks_arg = Sys.argv.(13) in
  let serve = (toks_arg = "serve") in
  let prompt =
    if serve then [] else List.map int_of_string (String.split_on_char ',' toks_arg) in
  let max_new =
    if serve then 0
    else (if Array.length Sys.argv > 14 then int_of_string Sys.argv.(14) else 0) in
  let eos =
    if serve then (if Array.length Sys.argv > 14 then int_of_string Sys.argv.(14) else -1)
    else (if Array.length Sys.argv > 15 then int_of_string Sys.argv.(15) else -1) in

  let group = nh / nkv in
  let kvd = nkv * hd in
  let ldim = lnh * lhd in                      (* 2048: q, k and v each  *)
  let qkvd = 3 * ldim in                       (* 6144 conv channels     *)
  let zerov n = List.init n (fun _ -> f32_zero) in

  let b = read_file path in
  (* The weight checksum a receipt binds its answer to, over the whole file. *)
  let cksum =
    let r = ref 0 and n = Bytes.length b in
    for i = 0 to n - 1 do r := (!r * 31 + Char.code (Bytes.get b i)) mod 4294967296 done;
    !r in
  Printf.eprintf "checksum %d\n%!" cksum;
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let off name =
    match json_tensor_offsets header (coqstr name) with
    | s :: _ :: _ -> base + s
    | _ -> failwith ("offsets not found for " ^ name) in

  (* rms_norm_eps = 1e-6, and the DeltaNet kernel normalises with the same. *)
  let eps = f32_div f32_one (f32_of_Z 1000000) in
  let embed_s = off "embed_tokens.weight" in
  let emb_row t = dec_vec b (embed_s + 4 * t * d) d in
  let invf = dec_vec b (off "rope.inv_freq") (rd / 2) in

  (* cos/sin over the rotated prefix, in the half-split convention: entry i and
     entry i + rd/2 share an angle. The inverse frequencies come from the
     checkpoint, so the runner never raises theta to a fractional power. *)
  let rope_cs pos =
    let pf = f32_of_Z pos in
    let ang = List.map (fun fj -> f32_mult pf fj) invf in
    let cosv = List.map f32_cos ang and sinv = List.map f32_sin ang in
    (cosv @ cosv, sinv @ sinv) in

  let layer_is_full i = (i mod 4) = 3 in
  let lname i s = Printf.sprintf "layers.%d.%s" i s in

  (* Caches. Attention keeps keys and values per kv head; DeltaNet keeps the
     per-head recurrent state and the trailing convolution window. *)
  let kc = Array.make_matrix nl nkv [] in
  let vc = Array.make_matrix nl nkv [] in
  let dstate = Array.init nl (fun _ -> Array.make lnh []) in
  let dconv = Array.make nl [] in
  (* Every query starts from an empty cache and a zero recurrent state. *)
  let reset_caches () =
    for i = 0 to nl - 1 do
      for c = 0 to nkv - 1 do kc.(i).(c) <- []; vc.(i).(c) <- [] done;
      dstate.(i) <- Array.init lnh (fun _ -> f32_delta_state0 lhd lhd);
      dconv.(i) <- []
    done in
  reset_caches ();

  (* Run a batch of (position, token) through the whole stack, layer by layer,
     advancing every cache. Returns the final normalised hidden state of each
     input position. A prefill passes the whole prompt; a decode step passes a
     single token. *)
  let run batch =
    let hs = ref (List.map (fun (_, t) -> emb_row t) batch) in
    let poss = List.map fst batch in
    for i = 0 to nl - 1 do
      let ln1 = dec_vec b (off (lname i "input_layernorm.weight")) d in
      let hn = List.map (fun row -> f32_rmsnorm_zc ln1 eps row) !hs in
      let mixed =
        if layer_is_full i then begin
          let qw  = dec_mat b (off (lname i "self_attn.q_proj.weight")) (nh * hd * 2) d in
          let kw  = dec_mat b (off (lname i "self_attn.k_proj.weight")) kvd d in
          let vw  = dec_mat b (off (lname i "self_attn.v_proj.weight")) kvd d in
          let ow  = dec_mat b (off (lname i "self_attn.o_proj.weight")) d (nh * hd) in
          let qnw = dec_vec b (off (lname i "self_attn.q_norm.weight")) hd in
          let knw = dec_vec b (off (lname i "self_attn.k_norm.weight")) hd in
          let inv = f32_div f32_one (f32_sqrt (f32_of_Z hd)) in
          List.map2 (fun pos h ->
            let qg = f32_mat_vec_mul qw h in
            let kraw = f32_mat_vec_mul kw h in
            let vraw = f32_mat_vec_mul vw h in
            let (cosv, sinv) = rope_cs pos in
            for c = 0 to nkv - 1 do
              let knew = f32_partial_rope rd cosv sinv
                           (f32_rmsnorm_zc knw eps (slice (c * hd) hd kraw)) in
              kc.(i).(c) <- kc.(i).(c) @ [knew];
              vc.(i).(c) <- vc.(i).(c) @ [slice (c * hd) hd vraw]
            done;
            let gate = List.concat
              (List.init nh (fun hh -> slice (hh * hd * 2 + hd) hd qg)) in
            let heads = List.init nh (fun hh ->
              let q = f32_partial_rope rd cosv sinv
                        (f32_rmsnorm_zc qnw eps (slice (hh * hd * 2) hd qg)) in
              let kcache = kc.(i).(hh / group) and vcache = vc.(i).(hh / group) in
              let scores = List.map (fun kj -> f32_mult (f32_dot q kj) inv) kcache in
              let w = f32_softmax scores in
              List.init hd (fun j ->
                f32_dot w (List.map (fun vj -> List.nth vj j) vcache))) in
            f32_mat_vec_mul ow (f32_gate_sigmoid gate (List.concat heads)))
            poss hn
        end else begin
          let wqkv = dec_mat b (off (lname i "linear_attn.in_proj_qkv.weight")) qkvd d in
          let wz   = dec_mat b (off (lname i "linear_attn.in_proj_z.weight")) ldim d in
          let wa   = dec_mat b (off (lname i "linear_attn.in_proj_a.weight")) lnh d in
          let wb   = dec_mat b (off (lname i "linear_attn.in_proj_b.weight")) lnh d in
          let cw   = dec_mat b (off (lname i "linear_attn.conv1d.weight")) qkvd ck in
          let alog = dec_vec b (off (lname i "linear_attn.A_log")) lnh in
          let dtb  = dec_vec b (off (lname i "linear_attn.dt_bias")) lnh in
          let nw   = dec_vec b (off (lname i "linear_attn.norm.weight")) lhd in
          let dow  = dec_mat b (off (lname i "linear_attn.out_proj.weight")) d ldim in
          List.map (fun h ->
            let qkv = f32_mat_vec_mul wqkv h in
            let hist = dconv.(i) @ [qkv] in
            let k = List.length hist in
            let win = if k >= ck then last_n ck hist
                      else List.init (ck - k) (fun _ -> zerov qkvd) @ hist in
            dconv.(i) <- last_n (ck - 1) hist;
            let c = f32_conv_step cw [] win in
            let z  = f32_mat_vec_mul wz h in
            let av = f32_mat_vec_mul wa h in
            let bv = f32_mat_vec_mul wb h in
            let outs = List.init lnh (fun hh ->
              let q = f32_delta_prep_q eps lhd (slice (hh * lhd) lhd c) in
              let kk = f32_l2norm eps (slice (ldim + hh * lhd) lhd c) in
              let v = slice (2 * ldim + hh * lhd) lhd c in
              let beta = f32_sigmoid (List.nth bv hh) in
              let g = f32_delta_decay (List.nth alog hh) (List.nth dtb hh)
                        (List.nth av hh) in
              let (st', o) = f32_delta_step beta g q kk v dstate.(i).(hh) in
              dstate.(i).(hh) <- st';
              f32_rmsnorm_gated nw eps (slice (hh * lhd) lhd z) o) in
            f32_mat_vec_mul dow (List.concat outs)) hn
        end in
      let hidden2 = List.map2 f32_vec_add !hs mixed in
      let ln2 = dec_vec b (off (lname i "post_attention_layernorm.weight")) d in
      let gw = dec_mat b (off (lname i "mlp.gate_proj.weight")) ff d in
      let uw = dec_mat b (off (lname i "mlp.up_proj.weight")) ff d in
      let dw = dec_mat b (off (lname i "mlp.down_proj.weight")) d ff in
      let h2 = List.map (fun row -> f32_rmsnorm_zc ln2 eps row) hidden2 in
      hs := List.map2 f32_vec_add hidden2
              (List.map (fun row -> f32_swiglu gw uw dw row) h2);
      Printf.eprintf "  layer %d/%d (%s)\n%!" (i + 1) nl
        (if layer_is_full i then "attn" else "delta")
    done;
    let normw = dec_vec b (off "norm.weight") d in
    List.map (fun row -> f32_rmsnorm_zc normw eps row) !hs in

  let logits_of v = Array.init vocab (fun j -> f32_dot v (emb_row j)) in
  let argmax a =
    let bi = ref 0 in
    for j = 1 to Array.length a - 1 do if a.(j) > a.(!bi) then bi := j done; !bi in
  let rec last = function [] -> failwith "empty" | [x] -> x | _ :: r -> last r in

  (* One query against a fresh cache. Returns (first-position logits, generated
     ids); streams "TOK <id>" per token when asked. *)
  let run_query toks mx stream =
    reset_caches ();
    Printf.eprintf "prefill over %d tokens...\n%!" (List.length toks);
    let finals = run (List.mapi (fun p t -> (p, t)) toks) in
    Printf.eprintf "projecting %d logits...\n%!" vocab;
    let logits0 = logits_of (last finals) in
    let gen = ref [] in
    if mx > 0 then begin
      let cur = ref (argmax logits0) in
      let pos = ref (List.length toks) in
      (try
        for n = 1 to mx do
          if !cur = eos then raise Exit;
          gen := !cur :: !gen;
          if stream then Printf.printf "TOK %d\n%!" !cur
          else Printf.printf "%d\n%!" !cur;
          if n < mx then begin
            Printf.eprintf "decode %d/%d\n%!" n mx;
            let f = run [(!pos, !cur)] in
            incr pos;
            cur := argmax (logits_of (last f))
          end
        done
      with Exit -> ())
    end;
    (logits0, List.rev !gen) in

  if serve then begin
    Printf.printf "CKSUM %d\nREADY\n%!" cksum;
    (try
      while true do
        let line = String.trim (input_line stdin) in
        if line <> "" then begin
          match String.split_on_char ' ' line with
          | ids_csv :: mn :: _ ->
              let qtoks = List.map int_of_string (String.split_on_char ',' ids_csv) in
              let (_, g) = run_query qtoks (int_of_string mn) true in
              Printf.printf "END %s\n%!" (String.concat "," (List.map string_of_int g))
          | _ -> Printf.printf "END \n%!"
        end
      done
    with End_of_file -> ())
  end
  else if max_new > 0 then ignore (run_query prompt max_new false)
  else begin
    let (logits0, _) = run_query prompt 0 false in
    let idx = Array.init vocab (fun i -> i) in
    Array.sort (fun i j -> compare logits0.(j) logits0.(i)) idx;
    Printf.printf "top-10 next-token logits:\n";
    for r = 0 to 9 do
      let i = idx.(r) in Printf.printf "  %7d  %.4f\n" i logits0.(i)
    done
  end
