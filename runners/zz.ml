(* Zarith's Z under a name the extracted enc_native.ml, which defines its own
   module Z, does not shadow; ExtractEnc.v's directives refer to it. *)
include Z
