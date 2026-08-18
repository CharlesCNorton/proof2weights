"""Interactive chat against the verified SmolLM2 forward.

Tokenization and detokenization run in this script; the forward pass and greedy
generation run in the built llama_talk_native binary, where every
floating-point operation is the extracted IEEE-754 arithmetic. One turn:
apply the chat template to the conversation, send the token ids to the runner,
read back the generated ids, and decode.

  python llama_chat.py "your question" [max_new]   # one shot
  python llama_chat.py                             # interactive loop
"""
import os, sys, subprocess, tempfile
from transformers import AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
CFG = ["576", "30", "9", "3", "1536", "49152"]   # d n_layer n_head n_kv ff vocab
EOS = "2"                                          # <|im_end|>
# Where the built llama_talk_native runner lives. The default runs it on this
# machine; set P2W_REMOTE to an ssh target to run it elsewhere.
REMOTE = os.environ.get("P2W_REMOTE", "local")
RUN_DIR = os.environ.get("P2W_RUN_DIR", "~/proof2weights")
WEIGHTS = os.environ.get("P2W_WEIGHTS", "smollm.safetensors")
RUN = f"cd {RUN_DIR} && ./llama_talk_native {WEIGHTS} " + " ".join(CFG)

def runner_argv(cmd):
    return ["sh", "-c", cmd] if REMOTE == "local" else ["ssh", REMOTE, cmd]

tok = AutoTokenizer.from_pretrained(MODEL)

def generate(ids, max_new):
    idcsv = ",".join(map(str, ids))
    cmd = f"{RUN} {idcsv} {max_new} {EOS}"
    fd, outpath = tempfile.mkstemp()
    os.close(fd)
    # Redirect remote stdout to a file rather than capturing, to avoid the
    # Windows OpenSSH pipe-close hang.
    with open(outpath, "w") as out:
        subprocess.run(runner_argv(cmd), stdout=out, stderr=subprocess.DEVNULL)
    with open(outpath) as f:
        text = f.read()
    os.remove(outpath)
    return [int(x) for x in text.split()]

def turn(history, max_new):
    ids = tok.apply_chat_template(history, add_generation_prompt=True, tokenize=True, return_dict=False)
    if not isinstance(ids, list):
        ids = list(ids["input_ids"])
    gen = generate(ids, max_new)
    return tok.decode(gen)

def main():
    args = sys.argv[1:]
    max_new = 64
    one_shot = None
    if args and args[-1].isdigit():
        max_new = int(args[-1]); args = args[:-1]
    if args:
        one_shot = " ".join(args)
    if one_shot is not None:
        reply = turn([{"role": "user", "content": one_shot}], max_new)
        print(reply)
        return
    print("verified SmolLM2 chat. Every floating-point operation in each reply is the")
    print("extracted IEEE-754 arithmetic. Generation uses a key/value cache. Ctrl-C to exit.")
    history = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        reply = turn(history, max_new)
        history.append({"role": "assistant", "content": reply})
        print("bot>", reply)

if __name__ == "__main__":
    main()
