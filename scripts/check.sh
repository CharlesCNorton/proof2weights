#!/bin/bash
# check.sh - run everything that can be checked, and say what could not be.
#
# The proofs are checked by compiling them; the rest are programs that run in
# minutes. A step whose input is absent is reported as skipped and named, so
# the output says what was decided and what was not.
#
#   scripts/check.sh [--quick]
#
# --quick omits the assumption report.
#
# Compiling the theories rewrites the extraction output in theories/, which the
# runners link against, so build the runners after the theories and not before.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUICK=0
[ "${1-}" = "--quick" ] && QUICK=1

PY="${P2W_PYTHON:-python3}"
RUN_DIR="${P2W_RUN_DIR:-.}"

pass=0 fail=0 skip=0
t_start=$(date +%s)

step() {  # step <name> <command...>
  local name="$1"; shift
  local t0 t1 out rc
  t0=$(date +%s)
  out=$("$@" 2>&1); rc=$?
  t1=$(date +%s)
  if [ $rc -eq 0 ]; then
    printf '  %-46s %4ds  ok\n' "$name" "$((t1 - t0))"; pass=$((pass + 1))
  else
    printf '  %-46s %4ds  FAILED\n' "$name" "$((t1 - t0))"; fail=$((fail + 1))
    printf '%s\n' "$out" | tail -25 | sed 's/^/      /'
  fi
}

skipstep() {
  printf '  %-46s        skipped (%s)\n' "$1" "$2"; skip=$((skip + 1))
}

echo "proofs"
have_rocq=0
if [ -f theories/_CoqProject ] && command -v coqc > /dev/null 2>&1; then
  contrib="$(coqc -where 2>/dev/null)/user-contrib"
  if [ -d "$contrib/Flocq" ] && [ -d "$contrib/Interval" ]; then
    have_rocq=1
  else
    missing=""
    [ -d "$contrib/Flocq" ]    || missing="coq-flocq"
    [ -d "$contrib/Interval" ] || missing="${missing:+$missing and }coq-interval"
    why="$missing not installed"
  fi
else
  why="no coqc on the path"
fi
if [ $have_rocq -eq 1 ]; then
  step "theories compile" make -C theories
  if [ $QUICK -eq 1 ]; then
    skipstep "assumption report" "--quick"
  else
    step "assumption report" make -C theories Audit.vo
  fi
else
  skipstep "theories compile" "$why"
  skipstep "assumption report" "$why"
fi

echo "the host the native build runs on"
if [ -x "$RUN_DIR/fp_selftest" ]; then
  step "fp_selftest, 33 checks" "$RUN_DIR/fp_selftest"
else
  skipstep "fp_selftest, 33 checks" "not built"
fi

echo "provenance"
step "checkpoint revisions and tool versions" \
  "$PY" scripts/provenance.py --offline --out provenance.json

echo "the extracted elementary functions"
if [ -x "$RUN_DIR/prim_sweep" ]; then
  step "every primitive against the true function" \
    "$PY" scripts/prim_check.py --bin "$RUN_DIR" --n 40001
else
  skipstep "every primitive against the true function" "prim_sweep not built"
fi

echo "the checkpoints"
if [ -x "$RUN_DIR/llama_talk_native" ] || [ -x "$RUN_DIR/qwen_talk_native" ]; then
  step "each checkpoint against its PyTorch oracle" \
    "$PY" scripts/oracle_check.py --bin "$RUN_DIR"
else
  skipstep "each checkpoint against its PyTorch oracle" "runners not built"
fi

echo "the extracted forward pass on the checkpoints"
if [ -x "$RUN_DIR/gpt2_verified" ] || [ -x "$RUN_DIR/llama_verified" ]; then
  step "the extracted pass against each oracle" \
    "$PY" scripts/verified_check.py gpt2 smollm --bin "$RUN_DIR"
else
  skipstep "the extracted pass against each oracle" "runners not built"
fi

echo "the bound the annotated pass carries"
# GPT-2 small, on the prompt "The quick brown"
if [ -x "$RUN_DIR/gpt2_bound_native" ] && [ -f "$RUN_DIR/gpt2.safetensors" ]; then
  step "annotated GPT-2 pass" \
    "$RUN_DIR/gpt2_bound_native" "$RUN_DIR/gpt2.safetensors" \
    768 12 12 3072 50257 464,2068,7586
else
  skipstep "annotated GPT-2 pass" "runner or checkpoint absent"
fi

echo
printf 'passed %d, failed %d, skipped %d, in %d s\n' \
  "$pass" "$fail" "$skip" "$(( $(date +%s) - t_start ))"
[ "$fail" -eq 0 ] || exit 1
