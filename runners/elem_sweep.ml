(* elem_sweep.ml - every elementary function of the native extraction on every
   binary32 input of its domain, against the mathematical function in binary64.

   The functions evaluated are the extracted definitions themselves, from
   qwen_native and phases1_15_native. For each function the report gives the
   number of inputs, the largest error in units in the last place of the
   binary32 neighbourhood of the reference, the largest absolute and relative
   errors, the number of inputs whose result is not the correctly rounded
   binary32 value, and the number a binary64 reference cannot decide because
   the true value lies within a guard band of a binary32 midpoint.

   usage: elem_sweep <domains> [function ...]
     with no function named, every function is swept. *)

let fb u = Int32.float_of_bits (Int32.of_int u)
let bits f = Int32.to_int (Int32.bits_of_float f) land 0xFFFFFFFF
let tiny = Float.ldexp 1.0 (-149)

(* ulp of the binary32 neighbourhood of v *)
let ulp32 v =
  let f = fb (bits (Float.abs v)) in
  if not (f > 0.0) then tiny
  else
    let _, e = Float.frexp f in
    Float.max (Float.ldexp 1.0 (e - 24)) tiny

let clamp88 x = if x > 88.0 then 88.0 else if x < -88.0 then -88.0 else x
let r_sigmoid x = 1.0 /. (1.0 +. exp (clamp88 (-. x)))

let cases = [
  "exp", Qwen_native.f32_exp_approx, (fun x -> exp (clamp88 x)), -88.0, 88.0;
  "sigmoid", Qwen_native.f32_sigmoid, r_sigmoid, -88.0, 88.0;
  "tanh", Phases1_15_native.f32_tanh, (fun x -> r_sigmoid (2.0 *. x) *. 2.0 -. 1.0), -44.0, 44.0;
  "gelu", Phases1_15_native.f32_gelu,
    (fun x -> let t = 0.7978845608028654 *. (x +. 0.044715 *. x *. x *. x) in
              0.5 *. x *. (1.0 +. tanh t)), -8.0, 8.0;
  "log", Qwen_native.f32_log_unit, log, 1.0, 2.0;
  "softplus", Qwen_native.f32_softplus,
    (fun x -> Float.max x 0.0 +. Float.log1p (exp (-. Float.abs x))), -88.0, 88.0;
  "sin", Qwen_native.f32_sin, sin, -262144.0, 262144.0;
  "cos", Qwen_native.f32_cos, cos, -262144.0, 262144.0;
  "sqrt", Qwen_native.f32_sqrt, sqrt, 0.0, Float.infinity;
]

type acc = { mutable n : int; mutable bad : int; mutable undec : int;
             mutable worst : float; mutable worst_in : int;
             mutable wabs : float; mutable wrel : float }

let sweep f r lo hi (first, last) =
  let a = { n = 0; bad = 0; undec = 0; worst = 0.0; worst_in = 0; wabs = 0.0; wrel = 0.0 } in
  for u = first to last do
    let x = fb u in
    if Float.is_finite x && x >= lo && x <= hi then begin
      let got = f x in
      let rv = r x in
      if Float.is_finite rv then begin
        a.n <- a.n + 1;
        let uu = ulp32 rv in
        let aerr = Float.abs (got -. rv) in
        let err = aerr /. uu in
        if aerr > a.wabs then a.wabs <- aerr;
        if rv <> 0.0 then begin
          let rr = aerr /. Float.abs rv in
          if rr > a.wrel then a.wrel <- rr
        end;
        if bits got <> bits rv then begin
          let frac = Float.abs rv /. uu in
          if Float.abs (frac -. Float.floor frac -. 0.5) < 1e-6
          then a.undec <- a.undec + 1 else a.bad <- a.bad + 1
        end;
        if err > a.worst then begin a.worst <- err; a.worst_in <- u end
      end
    end
  done;
  a

(* the bit patterns whose values lie in [lo, hi]: the non-negative patterns
   increase with the value, the negative ones decrease *)
let ranges lo hi =
  let pos_lo = if lo > 0.0 then bits lo else 0 in
  let pos_hi = if Float.is_finite hi then bits hi else 0x7F7FFFFF in
  let neg = if lo < 0.0 then [0x80000000, bits lo]
            else if lo = 0.0 then [0x80000000, 0x80000000] else [] in
  (if hi >= 0.0 then [pos_lo, pos_hi] else []) @ neg

(* split the ranges into k pieces of nearly equal length *)
let split rs k =
  let total = List.fold_left (fun s (a, b) -> s + b - a + 1) 0 rs in
  let per = (total + k - 1) / k in
  let rec go rs acc = match rs with
    | [] -> List.rev acc
    | (a, b) :: rest ->
      if b - a + 1 <= per then go rest ((a, b) :: acc)
      else go ((a + per, b) :: rest) ((a, a + per - 1) :: acc) in
  go rs []

let () =
  let nd = int_of_string Sys.argv.(1) in
  let want = Array.to_list (Array.sub Sys.argv 2 (Array.length Sys.argv - 2)) in
  Printf.printf "%-9s %13s %12s %11s %11s %12s %10s\n%!"
    "function" "inputs" "max ulp" "max abs" "max rel" "not c.r." "undecided";
  List.iter (fun (name, f, r, lo, hi) ->
    if want = [] || List.mem name want then begin
      let parts = split (ranges lo hi) nd in
      let ds = List.map (fun rg -> Domain.spawn (fun () -> sweep f r lo hi rg)) parts in
      let t = { n = 0; bad = 0; undec = 0; worst = 0.0; worst_in = 0; wabs = 0.0; wrel = 0.0 } in
      List.iter (fun d ->
        let a = Domain.join d in
        t.n <- t.n + a.n; t.bad <- t.bad + a.bad; t.undec <- t.undec + a.undec;
        t.wabs <- Float.max t.wabs a.wabs; t.wrel <- Float.max t.wrel a.wrel;
        if a.worst > t.worst then begin t.worst <- a.worst; t.worst_in <- a.worst_in end)
        ds;
      Printf.printf "%-9s %13d %12.4f %11.3e %11.3e %12d %10d  ulp worst 0x%08x = %.9g\n%!"
        name t.n t.worst t.wabs t.wrel t.bad t.undec t.worst_in (fb t.worst_in)
    end) cases
