(* llama_talk_native.ml - run a real Llama-architecture safetensors model
   (SmolLM2) through the verified extracted IEEE-754 operators (native build),
   with a key/value cache and an optional persistent serve mode.

   Every float value is produced by the verified operators; only byte addressing
   and the structural composition are native. Weights are PyTorch Linear matrices
   stored [out, in]. All weights and the embedding are decoded once at startup.

   In serve mode the weights stay resident and the process answers queries from
   stdin (one per line: "<comma ids> <max_new>"), streaming "TOK <id>" per token
   and "END <comma gen ids>" per query. Otherwise it runs one query and exits.

   usage:
     llama_talk_native <path> d n_layer n_head n_kv ff vocab <tok,...>            (top-10)
     llama_talk_native <path> d n_layer n_head n_kv ff vocab <tok,...> <max_new> <eos>  (generate)
     llama_talk_native <path> d n_layer n_head n_kv ff vocab serve <eos>          (persistent server) *)

open Llama_native

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
let rec map3 f a b c =
  match a, b, c with x :: a', y :: b', z :: c' -> f x y z :: map3 f a' b' c' | _, _, _ -> []

type layer = {
  ln1 : binary32 list; ln2 : binary32 list;
  qw : binary32 list list; kw : binary32 list list;
  vw : binary32 list list; ow : binary32 list list;
  gw : binary32 list list; uw : binary32 list list; dw : binary32 list list;
}

let () =
  let path  = Sys.argv.(1) in
  let d     = int_of_string Sys.argv.(2) in
  let nl    = int_of_string Sys.argv.(3) in
  let nh    = int_of_string Sys.argv.(4) in
  let nkv   = int_of_string Sys.argv.(5) in
  let ff    = int_of_string Sys.argv.(6) in
  let vocab = int_of_string Sys.argv.(7) in
  let toks_arg = Sys.argv.(8) in
  let serve = (toks_arg = "serve") in
  let eos =
    if serve then (if Array.length Sys.argv > 9 then int_of_string Sys.argv.(9) else -1)
    else (if Array.length Sys.argv > 10 then int_of_string Sys.argv.(10) else -1) in
  let max_new =
    if serve then 0
    else (if Array.length Sys.argv > 9 then int_of_string Sys.argv.(9) else 0) in
  let prompt = if serve then [] else List.map int_of_string (String.split_on_char ',' toks_arg) in
  let hd = d / nh in
  let group = nh / nkv in
  let kvd = nkv * hd in
  let b = read_file path in
  let cksum =
    let r = ref 0 and n = Bytes.length b in
    for i = 0 to n - 1 do r := (!r * 31 + Char.code (Bytes.get b i)) mod 4294967296 done; !r in
  Printf.eprintf "checksum %d\n%!" cksum;
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  let header = coqstr (Bytes.sub_string b 8 hlen) in
  let off name =
    match json_tensor_offsets header (coqstr name) with
    | s :: _ :: _ -> base + s
    | _ -> failwith ("offsets not found for " ^ name) in
  let eps = f32_div (f32_of_Z 1) (f32_of_Z 100000) in
  let embed_s = off "embed_tokens.weight" in
  let invf = dec_vec b (off "rope.inv_freq") (hd / 2) in

  Printf.eprintf "decoding weights...\n%!";
  let emb = Array.init vocab (fun t -> dec_vec b (embed_s + 4 * t * d) d) in
  let lyr = Array.init nl (fun i ->
    let p = Printf.sprintf "layers.%d." i in
    { ln1 = dec_vec b (off (p ^ "input_layernorm.weight")) d;
      ln2 = dec_vec b (off (p ^ "post_attention_layernorm.weight")) d;
      qw = dec_mat b (off (p ^ "self_attn.q_proj.weight")) d d;
      kw = dec_mat b (off (p ^ "self_attn.k_proj.weight")) kvd d;
      vw = dec_mat b (off (p ^ "self_attn.v_proj.weight")) kvd d;
      ow = dec_mat b (off (p ^ "self_attn.o_proj.weight")) d d;
      gw = dec_mat b (off (p ^ "mlp.gate_proj.weight")) ff d;
      uw = dec_mat b (off (p ^ "mlp.up_proj.weight")) ff d;
      dw = dec_mat b (off (p ^ "mlp.down_proj.weight")) d ff }) in
  let normw = dec_vec b (off "norm.weight") d in
  Printf.eprintf "weights ready.\n%!";

  let rope pos x =
    let xa = ftake (hd / 2) x and xb = fdrop (hd / 2) x in
    let pf = f32_of_Z pos in
    let cs = List.map (fun fj -> let a = f32_mult pf fj in (f32_cos a, f32_sin a)) invf in
    let outa = map3 (fun xaj xbj (c, s) -> f32_minus (f32_mult xaj c) (f32_mult xbj s)) xa xb cs in
    let outb = map3 (fun xaj xbj (c, s) -> f32_plus (f32_mult xbj c) (f32_mult xaj s)) xa xb cs in
    outa @ outb in

  let attend qhead kcache vcache =
    let inv = f32_div f32_one (f32_sqrt (f32_of_Z hd)) in
    let scores = List.map (fun kj -> f32_mult (f32_dot qhead kj) inv) kcache in
    let w = f32_softmax scores in
    match f32_mat_mul [w] vcache with row :: _ -> row | [] -> [] in

  let logits_of v = Array.init vocab (fun j -> f32_dot v emb.(j)) in
  let argmax a =
    let bi = ref 0 in
    for j = 1 to Array.length a - 1 do if a.(j) > a.(!bi) then bi := j done; !bi in

  (* One query with a fresh key/value cache. Returns (first-position logits,
     generated token ids). Streams "TOK <id>" per generated token if stream. *)
  let run_query toks max_new stream =
    let kc = Array.make_matrix nl nkv [] in
    let vc = Array.make_matrix nl nkv [] in
    let prefill toks =
      let hidden = ref (List.map (fun t -> emb.(t)) toks) in
      for i = 0 to nl - 1 do
        let l = lyr.(i) in
        let h = List.map (fun row -> f32_rmsnorm l.ln1 eps row) !hidden in
        let q = List.map (fun row -> f32_mat_vec_mul l.qw row) h in
        let k = List.map (fun row -> f32_mat_vec_mul l.kw row) h in
        let v = List.map (fun row -> f32_mat_vec_mul l.vw row) h in
        let qheads = List.init nh  (fun hh -> List.mapi (fun pos row -> rope pos (slice (hh * hd) hd row)) q) in
        let kheads = List.init nkv (fun kk -> List.mapi (fun pos row -> rope pos (slice (kk * hd) hd row)) k) in
        let vheads = List.init nkv (fun kk -> List.map  (fun row -> slice (kk * hd) hd row) v) in
        List.iteri (fun kk kh -> kc.(i).(kk) <- kh) kheads;
        List.iteri (fun kk vh -> vc.(i).(kk) <- vh) vheads;
        let headouts = List.init nh (fun hh ->
          f32_causal_attention (List.nth qheads hh) (List.nth kheads (hh / group)) (List.nth vheads (hh / group)) hd) in
        let attn = List.map (fun row -> f32_mat_vec_mul l.ow row) (f32_concat_heads headouts) in
        let hidden2 = List.map2 f32_vec_add !hidden attn in
        let h2 = List.map (fun row -> f32_rmsnorm l.ln2 eps row) hidden2 in
        let gate = List.map (fun row -> f32_mat_vec_mul l.gw row) h2 in
        let up = List.map (fun row -> f32_mat_vec_mul l.uw row) h2 in
        let act = List.map2 (fun g u -> f32_vec_mult (f32_silu_vec g) u) gate up in
        let down = List.map (fun row -> f32_mat_vec_mul l.dw row) act in
        hidden := List.map2 f32_vec_add hidden2 down;
        if stream then Printf.printf "PFL %d\n%!" (i + 1)
      done;
      let final = List.map (fun row -> f32_rmsnorm normw eps row) !hidden in
      logits_of (List.nth final (List.length toks - 1)) in
    let decode_step pos token =
      let hv = ref emb.(token) in
      for i = 0 to nl - 1 do
        let l = lyr.(i) in
        let h = f32_rmsnorm l.ln1 eps !hv in
        let q = f32_mat_vec_mul l.qw h in
        let knew = f32_mat_vec_mul l.kw h in
        let vnew = f32_mat_vec_mul l.vw h in
        let qheads = List.init nh  (fun hh -> rope pos (slice (hh * hd) hd q)) in
        let knew_h = List.init nkv (fun kk -> rope pos (slice (kk * hd) hd knew)) in
        let vnew_h = List.init nkv (fun kk -> slice (kk * hd) hd vnew) in
        List.iteri (fun kk x -> kc.(i).(kk) <- kc.(i).(kk) @ [x]) knew_h;
        List.iteri (fun kk x -> vc.(i).(kk) <- vc.(i).(kk) @ [x]) vnew_h;
        let headouts = List.init nh (fun hh -> attend (List.nth qheads hh) kc.(i).(hh / group) vc.(i).(hh / group)) in
        let attn = f32_mat_vec_mul l.ow (List.concat headouts) in
        let hidden2 = f32_vec_add !hv attn in
        let h2 = f32_rmsnorm l.ln2 eps hidden2 in
        let gate = f32_mat_vec_mul l.gw h2 in
        let up = f32_mat_vec_mul l.uw h2 in
        let act = f32_vec_mult (f32_silu_vec gate) up in
        let down = f32_mat_vec_mul l.dw act in
        hv := f32_vec_add hidden2 down
      done;
      logits_of (f32_rmsnorm normw eps !hv) in
    let logits0 = prefill toks in
    let gen = ref [] in
    if max_new > 0 then begin
      let cur = ref (argmax logits0) in
      let pos = ref (List.length toks) in
      (try
        for _ = 1 to max_new do
          if !cur = eos then raise Exit;
          gen := !cur :: !gen;
          if stream then Printf.printf "TOK %d\n%!" !cur;
          let logits = decode_step !pos !cur in
          incr pos;
          cur := argmax logits
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
  else if max_new > 0 then begin
    let (_, g) = run_query prompt max_new false in
    print_string (String.concat " " (List.map string_of_int g)); print_newline ()
  end
  else begin
    let (logits0, _) = run_query prompt 0 false in
    let idx = Array.init vocab (fun i -> i) in
    Array.sort (fun i j -> compare logits0.(j) logits0.(i)) idx;
    Printf.printf "top-10 next-token logits:\n";
    for r = 0 to 9 do let i = idx.(r) in Printf.printf "  %6d  %.4f\n" i logits0.(i) done
  end
