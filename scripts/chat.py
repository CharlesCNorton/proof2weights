"""Interactive chat against a verified forward pass.

Tokenization and detokenization run in this script; the forward pass and greedy
generation run in the built runner, where every floating-point operation is the
extracted IEEE-754 arithmetic. One turn: apply the chat template to the
conversation, send the token ids to the runner, read back the generated ids,
and decode.

  python chat.py smollm "your question" [max_new]   # one shot
  python chat.py qwen                               # interactive loop
"""
import os, sys, subprocess, tempfile
from transformers import AutoTokenizer

from models import select, runner_command, runner_argv


def generate(cfg, tok, ids, max_new):
    eos = tok.eos_token_id if tok.eos_token_id is not None else -1
    cmd = f"{runner_command(cfg)} {','.join(map(str, ids))} {max_new} {eos}"
    fd, outpath = tempfile.mkstemp()
    os.close(fd)
    # Redirect stdout to a file rather than capturing, to avoid the Windows
    # OpenSSH pipe-close hang when the runner is remote.
    with open(outpath, "w") as out:
        subprocess.run(runner_argv(cmd), stdout=out, stderr=subprocess.DEVNULL)
    with open(outpath) as f:
        text = f.read()
    os.remove(outpath)
    return [int(x) for x in text.split()]


def turn(cfg, tok, history, max_new):
    ids = tok.apply_chat_template(history, add_generation_prompt=True,
                                  tokenize=True, return_dict=False)
    if not isinstance(ids, list):
        ids = list(ids["input_ids"])
    return tok.decode(generate(cfg, tok, ids, max_new))


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    cfg = select(args[0])
    args = args[1:]
    max_new = 64
    if args and args[-1].isdigit():
        max_new = int(args[-1]); args = args[:-1]
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    if args:
        print(turn(cfg, tok, [{"role": "user", "content": " ".join(args)}], max_new))
        return
    print("Every floating-point operation in each reply is the extracted")
    print("IEEE-754 arithmetic. Generation uses a cache. Ctrl-C to exit.")
    history = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        reply = turn(cfg, tok, history, max_new)
        history.append({"role": "assistant", "content": reply})
        print("bot>", reply)


if __name__ == "__main__":
    main()
