(* enc_common.ml - what the enclosure runners share: reading the checkpoint and
   the windows, comparing each binary32 logit with its enclosure, and sharing
   the windows among domains.

   For each logit the distance from the binary32 value to the far end of its
   interval bounds the logit's distance from the reference; it is computed in
   integers and printed rounded up. The reference's top-1 token is certified to
   be the binary32 top-1 when its interval lies above every other interval. *)

module ZZ = Z
open Enc_native

let read_file path =
  let ic = open_in_bin path in
  let n = in_channel_length ic in
  let b = Bytes.create n in really_input ic b 0 n; close_in ic; b

let u64_le b off =
  let r = ref 0 in
  for i = 7 downto 0 do r := (!r * 256) + Char.code (Bytes.get b (off + i)) done; !r

let coqstr s = List.init (String.length s) (fun i -> s.[i])

let read_lines path =
  let ic = open_in path in
  let rec go acc = match input_line ic with
    | l -> go (if String.trim l = "" then acc else String.trim l :: acc)
    | exception End_of_file -> close_in ic; List.rev acc in
  go []

(* The safetensors header as the loaders read it, and the byte reader over the
   tensor data they read through. *)
let checkpoint path =
  let b = read_file path in
  let hlen = u64_le b 0 in
  let base = 8 + hlen in
  (coqstr (Bytes.sub_string b 8 hlen), fun o -> ZZ.of_int (Char.code (Bytes.get b (base + o))))

(* The binary32 value of four little-endian bytes as m * 2^e, exactly. *)
let decomp bits =
  let s = bits lsr 31 and ex = (bits lsr 23) land 0xFF and fr = bits land 0x7FFFFF in
  if ex = 0 then ((if s = 1 then - fr else fr), -149)
  else let m = fr lor 0x800000 in ((if s = 1 then - m else m), ex - 150)

(* An integer at scale 2^-fb as a float, rounded up. *)
let up_float z fb = Float.succ (ldexp (ZZ.to_float z) (- fb))

(* One line per position, "p bound width certified", and the largest bound,
   the largest width, the certified positions and the failed logits. *)
let report oc fb vocab fb32 rows =
  let wmax = ref 0.0 and wwid = ref 0.0 and cert = ref 0 and failed = ref 0 in
  List.iteri (fun p row ->
    let pmax = ref ZZ.zero and pwid = ref ZZ.zero and pfail = ref 0 in
    let best = ref (-1) and bestx = ref neg_infinity in
    let los = Array.make vocab ZZ.zero and his = Array.make vocab ZZ.zero in
    List.iteri (fun j e ->
      let o = 4 * (p * vocab + j) in
      let bits = (Int32.to_int (Bytes.get_int32_le fb32 o)) land 0xFFFFFFFF in
      let x = Int32.float_of_bits (Int32.of_int bits) in
      if x > !bestx then (bestx := x; best := j);
      let (m, ex) = decomp bits in
      let xs = ZZ.shift_left (ZZ.of_int m) (ex + fb) in
      match e with
      | EIv (lo, hi) ->
          los.(j) <- lo; his.(j) <- hi;
          let dm = ZZ.max (ZZ.abs (ZZ.sub xs lo)) (ZZ.abs (ZZ.sub hi xs)) in
          if ZZ.gt dm !pmax then pmax := dm;
          let wd = ZZ.sub hi lo in if ZZ.gt wd !pwid then pwid := wd
      | EPt (mm, ee) ->
          let v = ZZ.shift_left mm (ZZ.to_int ee + fb) in
          los.(j) <- v; his.(j) <- v;
          let dm = ZZ.abs (ZZ.sub xs v) in if ZZ.gt dm !pmax then pmax := dm
      | EFail -> incr pfail) row;
    let others = ref None in
    Array.iteri (fun j h -> if j <> !best then
      match !others with
      | Some o when ZZ.leq h o -> ()
      | _ -> others := Some h) his;
    let certified = !pfail = 0 &&
      (match !others with Some o -> ZZ.gt los.(!best) o | None -> true) in
    if certified then incr cert;
    failed := !failed + !pfail;
    let pm = up_float !pmax fb and pw = up_float !pwid fb in
    if pm > !wmax then wmax := pm;
    if pw > !wwid then wwid := pw;
    Printf.fprintf oc "%d %.6e %.6e %d\n" p pm pw (if certified then 1 else 0)) rows;
  (!wmax, !wwid, !cert, !failed)

(* Window [w]: the rows [rows] computes on its tokens, against the dump
   <f32dir>/<w>.f32, written to <outdir>/<w>.txt through a rename, with a
   summary line on stderr. *)
let window fb vocab windows f32dir outdir rows w =
  let t0 = Unix.gettimeofday () in
  let toks = List.map int_of_string (String.split_on_char ',' windows.(w)) in
  let r = rows toks in
  let fb32 = read_file (Printf.sprintf "%s/%d.f32" f32dir w) in
  let tmp = Printf.sprintf "%s/%d.txt.tmp" outdir w in
  let oc = open_out tmp in
  let (wmax, wwid, cert, failed) = report oc fb vocab fb32 r in
  close_out oc;
  Sys.rename tmp (Printf.sprintf "%s/%d.txt" outdir w);
  Printf.eprintf "window %d: max bound %.3e, max width %.3e, certified top-1 %d/%d, failed %d, %.0f s\n%!"
    w wmax wwid cert (List.length r) failed (Unix.gettimeofday () -. t0)

(* The windows in [first, last) whose output does not exist, shared among
   P2W_DOMAINS domains (default one). *)
let over_windows outdir first last one =
  (try Unix.mkdir outdir 0o755 with _ -> ());
  let nd = try max 1 (int_of_string (Sys.getenv "P2W_DOMAINS")) with _ -> 1 in
  let work k =
    let w = ref (first + k) in
    while !w < last do
      if not (Sys.file_exists (Printf.sprintf "%s/%d.txt" outdir !w)) then one !w;
      w := !w + nd
    done in
  let ds = List.init (nd - 1) (fun k -> Domain.spawn (fun () -> work (k + 1))) in
  work 0;
  List.iter Domain.join ds
