(** main.ml - Test harness for proof2weights

    Exports the example networks to safetensors files and verifies them.
*)

open Phases1_15_complete
open Io

(** Verify round-trip: Coq serialize -> file -> read back -> compare. *)
let verify_roundtrip filename tensors =
  (* Get expected bytes from Coq's serialize_list *)
  let all_data = List.concat (List.map t_data tensors) in
  let expected_bytes = List.map z_to_int (serialize_list all_data) in

  (* Read actual bytes from file (skip header) *)
  let file_bytes = read_bytes filename in
  let header_size =
    Char.code (Bytes.get file_bytes 0) +
    (Char.code (Bytes.get file_bytes 1) lsl 8) +
    (Char.code (Bytes.get file_bytes 2) lsl 16) +
    (Char.code (Bytes.get file_bytes 3) lsl 24) in
  let data_offset = 8 + header_size in

  (* Compare byte by byte *)
  let mismatch = ref None in
  List.iteri
    (fun i expected ->
      let actual = Char.code (Bytes.get file_bytes (data_offset + i)) in
      if expected <> actual && !mismatch = None then
        mismatch := Some (i, expected, actual))
    expected_bytes;

  match !mismatch with
  | None ->
      Printf.printf "  Roundtrip: VERIFIED (%d bytes match)\n"
        (List.length expected_bytes)
  | Some (i, exp, act) ->
      Printf.printf "  Roundtrip: FAILED at byte %d (expected %d, got %d)\n" i exp act

let export name filename network =
  Printf.printf "Exporting %s...\n" name;
  let header = write_safetensors filename network in
  Printf.printf "  Header: %s\n" header;
  let total =
    List.fold_left (fun acc t -> acc + List.length (t_data t)) 0 network in
  Printf.printf "  Tensors: %d, Values: %d, Data bytes: %d\n"
    (List.length network) total (total * 4);
  verify_roundtrip filename network;
  Printf.printf "\n"

let () =
  Printf.printf "proof2weights: Coq -> safetensors\n\n";
  export "majority_network" "majority.safetensors" majority_network;
  export "mod3_network" "mod3.safetensors" mod3_network;
  Printf.printf "Done. Verify with:\n";
  Printf.printf "  make verify\n"
