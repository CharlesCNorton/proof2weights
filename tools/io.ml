(** io.ml - File I/O for proof2weights

    This module provides actual file I/O for the extracted Coq code.
    The extracted code produces data structures; this writes them to disk.

    The extraction keeps [Z] as its inductive binary representation, so
    tensor entries and serialized bytes arrive as [z] rather than [int] and
    are converted here at the file boundary.
*)

open Phases1_15_complete

(** Convert an extracted [positive] to an OCaml int. *)
let rec pos_to_int (p : positive) : int =
  match p with
  | XI q -> 1 + 2 * pos_to_int q
  | XO q -> 2 * pos_to_int q
  | XH -> 1

(** Convert an extracted [z] to an OCaml int. *)
let z_to_int (n : z) : int =
  match n with
  | Z0 -> 0
  | Zpos p -> pos_to_int p
  | Zneg p -> - (pos_to_int p)

(** Convert a Coq char list to an OCaml string. *)
let char_list_to_string chars =
  let buf = Buffer.create (List.length chars) in
  List.iter (Buffer.add_char buf) chars;
  Buffer.contents buf

(** Write 32-bit little-endian integer. *)
let write_int32_le oc n =
  let n = if n < 0 then n + 0x100000000 else n in
  output_byte oc (n land 0xFF);
  output_byte oc ((n lsr 8) land 0xFF);
  output_byte oc ((n lsr 16) land 0xFF);
  output_byte oc ((n lsr 24) land 0xFF)

(** Write 64-bit little-endian integer. *)
let write_int64_le oc n =
  let n64 = Int64.of_int n in
  write_int32_le oc (Int64.to_int (Int64.logand n64 0xFFFFFFFFL));
  write_int32_le oc (Int64.to_int (Int64.shift_right_logical n64 32))

(** Format shape as JSON array. *)
let shape_to_json shape =
  "[" ^ String.concat ", " (List.map string_of_int shape) ^ "]"

(** Compute tensor data size in bytes. *)
let tensor_size t = List.length (t_data t) * 4

(** Build safetensors JSON header for a network. *)
let build_header tensors =
  let rec aux offset = function
    | [] -> []
    | t :: rest ->
        let name = char_list_to_string (t_name t) in
        let size = tensor_size t in
        let entry = Printf.sprintf
          {|"%s": {"dtype": "I32", "shape": %s, "data_offsets": [%d, %d]}|}
          name (shape_to_json (t_shape t)) offset (offset + size) in
        entry :: aux (offset + size) rest
  in
  "{" ^ String.concat ", " (aux 0 tensors) ^ "}"

(** Write a network to a safetensors file. *)
let write_safetensors filename tensors =
  let header = build_header tensors in
  let oc = open_out_bin filename in
  write_int64_le oc (String.length header);
  output_string oc header;
  List.iter
    (fun t -> List.iter (fun v -> write_int32_le oc (z_to_int v)) (t_data t))
    tensors;
  close_out oc;
  header

(** Read 32-bit little-endian integer. *)
let read_int32_le ic =
  let b0 = input_byte ic in
  let b1 = input_byte ic in
  let b2 = input_byte ic in
  let b3 = input_byte ic in
  let unsigned = b0 + (b1 lsl 8) + (b2 lsl 16) + (b3 lsl 24) in
  if unsigned >= 0x80000000 then unsigned - 0x100000000 else unsigned

(** Read 64-bit little-endian integer. *)
let read_int64_le ic =
  let lo = read_int32_le ic in
  let hi = read_int32_le ic in
  Int64.(to_int (add (of_int (lo land 0xFFFFFFFF)) (shift_left (of_int hi) 32)))

(** Read raw bytes from file. *)
let read_bytes filename =
  let ic = open_in_bin filename in
  let len = in_channel_length ic in
  let buf = Bytes.create len in
  really_input ic buf 0 len;
  close_in ic;
  buf

(** Write a Coq-side byte list to a file. *)
let write_bytes filename (bs : byte list) =
  let oc = open_out_bin filename in
  List.iter (fun b -> output_byte oc (z_to_int b land 0xFF)) bs;
  close_out oc
