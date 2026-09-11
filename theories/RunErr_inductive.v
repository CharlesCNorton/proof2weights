(** * Inductive extraction of the annotated forward pass

    The annotated GPT-2 forward of RunErr.v, with binary32 and binary64 left as
    Flocq's inductive [binary_float] and [Z] as its inductive datatype, so the
    logits and their bounds are the computational content of the definitions
    and no floating-point hardware is trusted. runners/gpt2_bound_ref.ml drives
    it on models loaded through the verified list-based loader. *)

Require Import Phases1_15_complete Float_error Bound64 Annot AnnExp RunErr.

Set Extraction Output Directory ".".

Extraction "runerr_inductive.ml"
  binary32 f32_zero f32_ln_eps f32_gpt2_logits
  parse_header_size parse_header_string f32_load_model
  ann ann_const ann_ops ann_model g_gpt2_logits.
