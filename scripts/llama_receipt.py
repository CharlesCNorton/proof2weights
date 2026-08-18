"""Proof-carrying inference receipts for the verified SmolLM2 forward.

  python llama_receipt.py emit "prompt" [max_new]   # generate and write receipt.json
  python llama_receipt.py verify [receipt.json]     # recompute and re-run to check

A receipt records the weight checksum, the prompt, the full output token sequence,
and the semantics. Verification recomputes the checksum and re-runs the
deterministic verified generation, then checks both against the receipt; the
underlying check is the proven verify_receipt of Receipt.v. The forward runs in
the built runner, where every floating-point operation is the extracted
IEEE-754 arithmetic; tokenization runs in this script.
"""
import os, sys, json, subprocess, tempfile
from transformers import AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_LAYER = 30
CFG = ["576", "30", "9", "3", "1536", "49152"]
EOS = "2"
# Where the built llama_talk_native runner lives. The default runs it on this
# machine; set P2W_REMOTE to an ssh target to run it elsewhere.
REMOTE = os.environ.get("P2W_REMOTE", "local")
RUN_DIR = os.environ.get("P2W_RUN_DIR", "~/proof2weights")
WEIGHTS = os.environ.get("P2W_WEIGHTS", "smollm.safetensors")
RUN = f"cd {RUN_DIR} && ./llama_talk_native {WEIGHTS} " + " ".join(CFG)
SEMANTICS = "IEEE-754 binary32 round-nearest-even (Flocq), SmolLM2-135M-Instruct"
# Receipts land in the repository root unless P2W_WORK names another directory.
WORK = os.environ.get("P2W_WORK") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(WORK, exist_ok=True)
RECEIPT = os.path.join(WORK, "receipt.json")

def runner_argv(cmd):
    return ["sh", "-c", cmd] if REMOTE == "local" else ["ssh", REMOTE, cmd]

tok = AutoTokenizer.from_pretrained(MODEL)

def run_gen(ids, max_new):
    """Run the verified runner; return (generated_ids, weight_checksum)."""
    cmd = f"{RUN} {','.join(map(str, ids))} {max_new} {EOS}"
    fo, op = tempfile.mkstemp(); os.close(fo)
    fe, ep = tempfile.mkstemp(); os.close(fe)
    with open(op, "w") as out, open(ep, "w") as err:
        subprocess.run(runner_argv(cmd), stdout=out, stderr=err)
    gen = [int(x) for x in open(op).read().split()]
    cksum = None
    for line in open(ep):
        if line.startswith("checksum "):
            cksum = int(line.split()[1])
    os.remove(op); os.remove(ep)
    return gen, cksum

def tokenize(prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True, return_dict=False)
    return ids if isinstance(ids, list) else list(ids["input_ids"])

def emit(prompt, max_new):
    ids = tokenize(prompt)
    gen, cksum = run_gen(ids, max_new)
    receipt = {"model_checksum": cksum, "n_layer": N_LAYER, "prompt": ids,
               "output": ids + gen, "semantics": SEMANTICS}
    json.dump(receipt, open(RECEIPT, "w"), indent=2)
    print(f"prompt: {prompt!r}")
    print(f"answer: {tok.decode(gen)!r}")
    print(f"weight checksum: {cksum}")
    print(f"receipt written to {RECEIPT}")

def verify(path):
    r = json.load(open(path))
    gen2, cksum2 = run_gen(r["prompt"], len(r["output"]) - len(r["prompt"]))
    regenerated = r["prompt"] + gen2
    checksum_ok = (cksum2 == r["model_checksum"])
    output_ok = (regenerated == r["output"])
    prompt_ok = (r["output"][:len(r["prompt"])] == r["prompt"])
    print(f"weight checksum matches: {checksum_ok}  ({cksum2} vs {r['model_checksum']})")
    print(f"output reproduced:       {output_ok}")
    print(f"prompt preserved:        {prompt_ok}")
    print(f"semantics:               {r['semantics']}")
    ok = checksum_ok and output_ok and prompt_ok
    print("VERIFIED" if ok else "FAILED")
    if ok:
        gen = r["output"][len(r["prompt"]):]
        print(f"certified answer: {tok.decode(gen)!r}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("emit", "verify"):
        print(__doc__); return
    if sys.argv[1] == "emit":
        prompt = sys.argv[2]
        max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 64
        emit(prompt, max_new)
    else:
        verify(sys.argv[2] if len(sys.argv) > 2 else RECEIPT)

if __name__ == "__main__":
    main()
