// llamacpp_logits.cpp - next-token logits from llama.cpp for the agreement evaluation.
//
// Loads a GGUF model through libllama, runs each window of a windows file
// (scripts/agree_setup.py) as one batch with an output at every position, and
// writes <outdir>/<index>.f32: the logits of every position as little-endian
// binary32, positions in order, the layout the runners' dump mode writes. Flash
// attention is disabled and the memory is cleared before each window. The
// key/value cache keeps llama.cpp's default type unless the last argument names
// f32 or f16.
//
//   llamacpp_logits <model.gguf> <windows.txt> <outdir> <first> <last> [n_gpu_layers] [threads] [kv_type]
//
// The CMakeLists.txt beside this file builds it against a llama.cpp checkout.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.gguf> <windows.txt> <outdir> <first> <last> [n_gpu_layers] [threads] [kv_type]\n", argv[0]);
        return 1;
    }
    const std::string outdir = argv[3];
    const int first = atoi(argv[4]);
    const int last = atoi(argv[5]);
    const int ngl = argc > 6 ? atoi(argv[6]) : 0;
    const int threads = argc > 7 ? atoi(argv[7]) : 8;
    const std::string kv_type = argc > 8 ? argv[8] : "";

    std::vector<std::vector<llama_token>> windows;
    std::ifstream in(argv[2]);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \r\n\t") == std::string::npos) {
            continue;
        }
        std::vector<llama_token> toks;
        std::stringstream ss(line);
        std::string t;
        while (std::getline(ss, t, ',')) {
            toks.push_back((llama_token) std::stol(t));
        }
        windows.push_back(toks);
    }
    if (windows.empty()) {
        fprintf(stderr, "no windows\n");
        return 1;
    }
    const uint32_t T = (uint32_t) windows[0].size();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = T;
    cp.n_batch = T;
    cp.n_ubatch = T;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = true;
    if (kv_type == "f32" || kv_type == "f16") {
        cp.type_k = kv_type == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
        cp.type_v = cp.type_k;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create a context\n");
        return 1;
    }
    fprintf(stderr, "vocab %d, window %u tokens, kv cache %s\n", n_vocab, T, ggml_type_name(cp.type_k));

    llama_batch batch = llama_batch_init((int32_t) T, 0, 1);
    for (int w = first; w < last && w < (int) windows.size(); w++) {
        const std::vector<llama_token> & toks = windows[w];
        llama_memory_clear(llama_get_memory(ctx), true);
        batch.n_tokens = (int32_t) toks.size();
        for (size_t i = 0; i < toks.size(); i++) {
            batch.token[i] = toks[i];
            batch.pos[i] = (llama_pos) i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at window %d\n", w);
            return 1;
        }
        const std::string dst = outdir + "/" + std::to_string(w) + ".f32";
        const std::string tmp = dst + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "cannot write %s\n", tmp.c_str());
            return 1;
        }
        for (size_t i = 0; i < toks.size(); i++) {
            fwrite(llama_get_logits_ith(ctx, (int32_t) i), sizeof(float), (size_t) n_vocab, f);
        }
        fclose(f);
        std::remove(dst.c_str());
        std::rename(tmp.c_str(), dst.c_str());
        fprintf(stderr, "window %d done\n", w);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
