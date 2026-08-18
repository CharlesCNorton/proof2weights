#!/usr/bin/env bash
# Build the oracle runner and produce canonical logits for every sample.
# Run from anywhere; paths resolve against the repository root.
cd "$(dirname "$0")/.."
opam exec --switch rocq9 -- ocamlopt -rectypes -w -a -I theories \
  theories/phases1_15_complete.mli theories/phases1_15_complete.ml \
  runners/ref_logits.ml -o ref_logits 2>&1 | tail -5
: > coq_out.txt
while read id nl toks d h ff vocab npos; do
  echo "ID $id" >> coq_out.txt
  ./ref_logits expbatch/"$id".safetensors "$nl" "$toks" "$d" "$h" "$ff" "$vocab" "$npos" >> coq_out.txt 2>>coq_out.txt || echo "FAIL $id" >> coq_out.txt
done < expbatch/manifest.txt
echo "samples: $(grep -c '^ID ' coq_out.txt), fails: $(grep -c '^FAIL ' coq_out.txt), nan-lines: $(grep -c nan coq_out.txt)"
