"""Pin the inputs every reported number was produced from.

Writes a JSON record naming, for each model, the Hugging Face repository and
the commit its weights were read at, the digest of every file the conversion
consumed, the digest of the converted safetensors the runners load, and the
version of every tool that touched a reported number. With --gguf it also
compares a float32 GGUF conversion against the safetensors tensor by tensor,
so the llama.cpp columns are known to start from the same bytes.

  python provenance.py [--out provenance.json] [--work DIR]
                       [--gguf name=path ...] [--offline]

The record is the answer to "which bytes": a result that cites it names a
checkpoint revision, not a checkpoint name.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# repository, the revision the conversion read, and the converted file the
# runners load. The revision is filled in from the hub unless --offline.
MODELS = {
    "gpt2": {"repo": "gpt2", "converted": "gpt2.safetensors",
             "setup": "scripts/gpt2_setup.py", "n_head": 12, "n_kv_head": 12},
    "smollm": {"repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
               "converted": "smollm.safetensors",
               "setup": "scripts/smollm_setup.py", "n_head": 9, "n_kv_head": 3},
    "qwen": {"repo": "Qwen/Qwen3.5-0.8B", "converted": "qwen.safetensors",
             "setup": "scripts/qwen_setup.py", "n_head": 8, "n_kv_head": 2,
             # the multi-token-prediction block, which inference does not run
             "gguf_skip": "blk.24."},
}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def digest_of(path):
    if not os.path.exists(path):
        return None
    return {"bytes": os.path.getsize(path), "sha256": sha256(path)}


def run(cmd, cwd=None):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                             timeout=120)
        return out.stdout.strip() or out.stderr.strip() or None
    except Exception:
        return None


def hub_revision(repo):
    """The commit the repository's main branch points at, and the digest the
    hub records for every weight file on it."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    files = {}
    for s in info.siblings:
        if not s.rfilename.endswith((".safetensors", ".bin", ".json")):
            continue
        lfs = getattr(s, "lfs", None)
        files[s.rfilename] = {
            "bytes": getattr(s, "size", None),
            "sha256": (lfs or {}).get("sha256") if isinstance(lfs, dict)
                      else getattr(lfs, "sha256", None),
        }
    return {"sha": info.sha, "last_modified": str(info.last_modified),
            "files": files}


def versions():
    v = {"python": sys.version.split()[0],
         "platform": sys.platform}
    for mod in ("torch", "transformers", "numpy", "safetensors", "huggingface_hub"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = None
    try:
        import torch
        v["torch_cuda"] = torch.version.cuda
        v["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            v["gpu"] = torch.cuda.get_device_name(0)
            v["driver"] = run(["nvidia-smi", "--query-gpu=driver_version",
                               "--format=csv,noheader"])
    except Exception:
        pass
    v["rocq"] = run(["coqc", "--version"])
    for lib in ("Flocq", "Interval"):
        where = run(["coqc", "-where"])
        if where:
            p = os.path.join(where, "user-contrib", lib)
            v[lib.lower()] = "present" if os.path.isdir(p) else None
    v["repo_commit"] = run(["git", "rev-parse", "HEAD"], cwd=ROOT)
    v["repo_dirty"] = bool(run(["git", "status", "--porcelain"], cwd=ROOT))
    lc = os.environ.get("P2W_LLAMACPP")
    if lc and os.path.isdir(lc):
        v["llamacpp_commit"] = run(["git", "rev-parse", "HEAD"], cwd=lc)
        v["llamacpp_describe"] = run(["git", "describe", "--tags", "--always"], cwd=lc)
    return v


def _unpermute(w, n_head):
    """Undo the row permutation llama.cpp's converter applies to the query and
    key projections so that its rotary convention matches the checkpoint's.

    The converter reshapes a projection of [n_head * head_dim, in] to
    (n_head, 2, head_dim/2, in) and swaps the middle two axes. The inverse
    reshapes to (n_head, head_dim/2, 2, in) and swaps them back. It is a
    permutation of rows: it moves values, it does not change them.
    """
    import numpy as np
    if w.ndim != 2 or w.shape[0] % (n_head * 2):
        return None
    return (w.reshape(n_head, w.shape[0] // n_head // 2, 2, w.shape[1])
             .swapaxes(1, 2)
             .reshape(w.shape))


def _ulps(a, b):
    """Largest distance between two float32 arrays in units in the last place."""
    import numpy as np
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-2**31) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2**31) - ib, ib)
    return int(np.abs(ia - ib).max())


def gguf_vs_safetensors(gguf_path, st_path, n_head=None, n_kv_head=None,
                        skip=None):
    """Compare a float32 GGUF against the safetensors it was converted from,
    tensor by tensor, on the values both hold.

    A tensor counts as identical when its bytes match, when it matches after a
    transpose, or, for a query or key projection, when it matches after undoing
    the converter's rotary row permutation. The two folds the Qwen3.5 converter
    makes are counted apart: a zero-centred normalization weight stored with one
    added in binary32, and a decay parameter stored as the negated exponential
    of itself, within one unit in the last place of the correctly rounded value.
    Tensors whose names start with `skip` are left out. Anything else is
    reported by name.
    """
    import numpy as np
    from gguf import GGUFReader
    from safetensors import safe_open

    reader = GGUFReader(gguf_path)
    gg = {t.name: np.array(t.data) for t in reader.tensors
          if not (skip and t.name.startswith(skip))}
    with safe_open(st_path, framework="np") as f:
        st = {k: f.get_tensor(k) for k in f.keys()}
    report = {"gguf_tensors": len(gg), "safetensors_tensors": len(st),
              "compared": 0, "identical": 0, "identical_after_permute": 0,
              "plus_one": 0, "neg_exp": 0, "unmatched": []}
    st_by_shape = {}
    for k, v in st.items():
        st_by_shape.setdefault((v.shape, v.dtype.str), []).append((k, v))

    for name, g in gg.items():
        if g.dtype != np.float32:
            continue
        report["compared"] += 1
        cands = st_by_shape.get((g.shape, "<f4"), [])
        if any(np.array_equal(v, g) for _, v in cands):
            report["identical"] += 1
            continue
        tcands = st_by_shape.get((g.shape[::-1], "<f4"), [])
        if any(np.array_equal(v.T, g) for _, v in tcands):
            report["identical"] += 1
            continue
        heads = n_head if name.endswith("attn_q.weight") else \
                n_kv_head if name.endswith("attn_k.weight") else None
        if heads and any(np.array_equal(_unpermute(g, heads), v)
                         for _, v in cands):
            report["identical_after_permute"] += 1
            continue
        if any(np.array_equal(g, v + np.float32(1)) for _, v in cands):
            report["plus_one"] += 1
            continue
        if any(_ulps(g, (-np.exp(v.astype(np.float64))).astype(np.float32)) <= 1
               for _, v in cands):
            report["neg_exp"] += 1
            continue
        report["unmatched"].append(name)

    report["all_accounted"] = not report["unmatched"]
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "provenance.json"))
    ap.add_argument("--work", default=ROOT)
    ap.add_argument("--gguf", action="append", default=[],
                    help="name=path of a float32 GGUF to compare")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    record = {"generated": datetime.datetime.now(datetime.timezone.utc)
                           .replace(microsecond=0).isoformat(),
              "tools": versions(), "models": {}}

    for name, cfg in MODELS.items():
        entry = {"repo": cfg["repo"], "setup": cfg["setup"]}
        conv = os.path.join(args.work, cfg["converted"])
        entry["converted"] = {"file": cfg["converted"], **(digest_of(conv) or {})}
        if not args.offline:
            try:
                entry["hub"] = hub_revision(cfg["repo"])
            except Exception as exc:
                entry["hub_error"] = str(exc)
        record["models"][name] = entry

    for spec in args.gguf:
        name, path = spec.split("=", 1)
        st = os.path.join(args.work, MODELS[name]["converted"])
        entry = record["models"].setdefault(name, {})
        entry["gguf"] = {"file": os.path.basename(path), **(digest_of(path) or {})}
        try:
            entry["gguf"]["identity"] = gguf_vs_safetensors(
                path, st, MODELS[name].get("n_head"),
                MODELS[name].get("n_kv_head"), MODELS[name].get("gguf_skip"))
        except Exception as exc:
            entry["gguf"]["identity_error"] = str(exc)

    with open(args.out, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(json.dumps(record, indent=1, sort_keys=True))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
