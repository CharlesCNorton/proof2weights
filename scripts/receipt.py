"""Proof-carrying inference receipts for the verified forward passes.

  python receipt.py emit <model> "prompt" [max_new]   # generate, write receipt.json
  python receipt.py verify [receipt.json]             # recompute and re-run

A receipt records the weight checksum, the prompt, the full output token
sequence, and the semantics. Verification recomputes the checksum and re-runs
the deterministic verified generation, then checks both against the receipt;
the underlying check is the proven verify_receipt of Receipt.v. The forward
runs in the built runner, where every floating-point operation is the extracted
IEEE-754 arithmetic; tokenization runs in this script.

Both runners print the checksum of the weight file they loaded, so a receipt
can be emitted for either model.
"""
import os, sys, json, subprocess, tempfile
from transformers import AutoTokenizer

from models import MODELS, select, runner_command, runner_argv

# Receipts land in the repository root unless P2W_WORK names another directory.
WORK = os.environ.get("P2W_WORK") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(WORK, exist_ok=True)
RECEIPT = os.path.join(WORK, "receipt.json")


def run_gen(cfg, tok, ids, max_new):
    """Run the verified runner; return (generated ids, weight checksum)."""
    eos = tok.eos_token_id if tok.eos_token_id is not None else -1
    cmd = f"{runner_command(cfg)} {','.join(map(str, ids))} {max_new} {eos}"
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


def tokenize(tok, prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True,
                                  return_dict=False)
    return ids if isinstance(ids, list) else list(ids["input_ids"])


def emit(name, prompt, max_new):
    cfg = select(name)
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    ids = tokenize(tok, prompt)
    gen, cksum = run_gen(cfg, tok, ids, max_new)
    receipt = {"model": name, "model_checksum": cksum, "n_layer": cfg["n_layer"],
               "prompt": ids, "output": ids + gen, "semantics": cfg["semantics"]}
    json.dump(receipt, open(RECEIPT, "w"), indent=2)
    print(f"model:  {name}")
    print(f"prompt: {prompt!r}")
    print(f"answer: {tok.decode(gen)!r}")
    print(f"weight checksum: {cksum}")
    print(f"receipt written to {RECEIPT}")


def verify(path):
    r = json.load(open(path))
    cfg = select(r.get("model", "smollm"))
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    gen2, cksum2 = run_gen(cfg, tok, r["prompt"], len(r["output"]) - len(r["prompt"]))
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
        print(f"certified answer: {tok.decode(r['output'][len(r['prompt']):])!r}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("emit", "verify"):
        print(__doc__)
        print("models: " + ", ".join(MODELS))
        return
    if sys.argv[1] == "emit":
        name = sys.argv[2]
        prompt = sys.argv[3]
        max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 64
        emit(name, prompt, max_new)
    else:
        verify(sys.argv[2] if len(sys.argv) > 2 else RECEIPT)


if __name__ == "__main__":
    main()
