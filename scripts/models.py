"""The models the verified runners can be driven against, and how to reach one.

Each entry names the Hugging Face repository the tokenizer comes from, the
built runner, the weight file, and the configuration arguments that runner
takes ahead of the token ids. Tokenization runs in this script; the forward
pass and greedy generation run in the runner, where every floating-point
operation is the extracted IEEE-754 arithmetic.
"""
import os

MODELS = {
    "smollm": {
        "hf": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "runner": "llama_talk_native",
        "weights": "smollm.safetensors",
        # d n_layer n_head n_kv ff vocab
        "args": ["576", "30", "9", "3", "1536", "49152"],
        "n_layer": 30,
        "semantics": "IEEE-754 binary32 round-nearest-even (Flocq), "
                     "SmolLM2-135M-Instruct",
    },
    "qwen": {
        "hf": "Qwen/Qwen3.5-0.8B",
        "runner": "qwen_talk_native",
        "weights": "qwen.safetensors",
        # d n_layer n_head n_kv head_dim rotary_dim ff vocab lnh lhd conv_k
        "args": ["1024", "24", "8", "2", "256", "64", "3584", "248320",
                 "16", "128", "4"],
        "n_layer": 24,
        "semantics": "IEEE-754 binary32 round-nearest-even (Flocq), "
                     "Qwen3.5-0.8B",
    },
}

# Host holding the built runner. The default runs it on this machine; set
# P2W_REMOTE to an ssh target to run it elsewhere.
REMOTE = os.environ.get("P2W_REMOTE", "local")
RUN_DIR = os.environ.get("P2W_RUN_DIR", ".")


def select(name):
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; choose from {', '.join(MODELS)}")
    cfg = dict(MODELS[name])
    cfg["weights"] = os.environ.get("P2W_WEIGHTS", cfg["weights"])
    return cfg


def runner_command(cfg):
    return (f"cd {RUN_DIR} && ./{cfg['runner']} {cfg['weights']} "
            + " ".join(cfg["args"]))


def runner_argv(cmd):
    return ["sh", "-c", cmd] if REMOTE == "local" else ["ssh", REMOTE, cmd]
