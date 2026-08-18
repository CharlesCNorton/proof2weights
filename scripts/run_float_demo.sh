#!/usr/bin/env bash
# Reproduce the verified float GPT-2 pipeline end to end:
#   compile the Coq development, extract OCaml, build, run.
#
# Requires the rocq9 opam switch with coq-flocq, and ocamlopt.
# For the numerical comparison, first generate the fixture:
#   python scripts/tiny_gpt2_ref.py   (writes tiny_gpt2.safetensors + reference logits)
set -e
cd "$(dirname "$0")/.."

RUN="opam exec --switch rocq9 --"
OCAMLOPT="$RUN ocamlopt -rectypes -w -a -I theories \
  theories/phases1_15_complete.mli theories/phases1_15_complete.ml"

echo "== compile + extract =="
(cd theories && $RUN rocq compile -R . "" Phases1_15_complete.v)

echo "== smoke test (tiny hand-built model) =="
$OCAMLOPT runners/float_smoke.ml -o float_smoke
./float_smoke

if [ -f tiny_gpt2.safetensors ]; then
  echo "== load real safetensors, run extracted float forward =="
  $OCAMLOPT runners/float_load_run.ml -o float_load_run
  echo "extracted logits:"
  ./float_load_run
  if [ -f tiny_gpt2_ref_logits.txt ]; then
    echo "reference logits:"
    cat tiny_gpt2_ref_logits.txt
  fi
else
  echo "(skipping comparison: run 'python scripts/tiny_gpt2_ref.py' to generate tiny_gpt2.safetensors)"
fi
