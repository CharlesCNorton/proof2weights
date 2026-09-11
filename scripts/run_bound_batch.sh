#!/usr/bin/env bash
# The annotated GPT-2 forward pass over every sample of the sweep, through the
# inductive extraction (runners/gpt2_bound_ref). Each output row holds, per
# logit, the binary32 value and its bound as value:bound. Run from anywhere;
# paths resolve against the repository root. JOBS sets the parallelism.
cd "$(dirname "$0")/.."
mkdir -p bound_out
grep -v '^$' expbatch/manifest.txt | xargs -P "${JOBS:-8}" -L 1 sh -c \
  './gpt2_bound_ref expbatch/"$0".safetensors "$1" "$2" "$3" "$4" "$5" "$6" "$7" > bound_out/"$0".txt 2>&1 || echo FAIL >> bound_out/"$0".txt'
echo "samples: $(ls bound_out | wc -l), fails: $(grep -l FAIL bound_out/*.txt 2>/dev/null | wc -l)"
