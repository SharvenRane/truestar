// Truestar's driver for `truestar conform llamacpp` (docs/CONFORM_LLAMACPP.md).
//
// Compiled by Truestar against the llama.h of the llama.cpp build under test and linked with that
// build's libllama, so the struct layouts are the ones of that exact version. It only loads a GGUF,
// runs token ids through llama_decode with the requested settings and writes the float32 logits the
// engine produced. It never samples (greedy is argmax, lowest index on ties) and never changes a
// weight.
//
// usage: truestar_llamacpp_driver MODEL.gguf REQUEST.txt OUT.bin
// REQUEST.txt, one item per line:
//   threads N | batch N | ubatch N | flash 0|1 | ctx N | kv f16|f32 | rope_base X (0 = the file's)
//   prompt  T id1 .. idT            teacher forced: logits of every position
//   greedy  NEW T id1 .. idT        prompt with logits of its last position, then NEW greedy steps
// OUT.bin: "TSLC" u32 version(1) u32 n_vocab u32 n_items, then per item:
//   u32 kind (0 prompt, 1 greedy) u32 rows, i32 tokens[rows], f32 logits[rows * n_vocab]
//   prompt rows = T (tokens = input ids); greedy rows = NEW (tokens = the token chosen from each row)
// stdout: one line "info key=value ..." with what the context actually used.

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Item {
    int kind = 0;
    int n_new = 0;
    std::vector<llama_token> ids;
};

static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static int fail(const char * msg) {
    fprintf(stderr, "truestar driver: %s\n", msg);
    return 3;
}

static bool decode_ids(llama_context * ctx, const std::vector<llama_token> & ids, int pos0, int n_batch,
                       bool all_logits, int n_vocab, std::vector<float> * rows) {
    llama_batch b = llama_batch_init(n_batch, 0, 1);
    const int n = (int) ids.size();
    for (int s = 0; s < n; s += n_batch) {
        const int e = s + n_batch < n ? s + n_batch : n;
        b.n_tokens = 0;
        for (int i = s; i < e; ++i) {
            const int j = b.n_tokens++;
            b.token[j] = ids[i];
            b.pos[j] = pos0 + i;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = all_logits || i == n - 1;
        }
        if (llama_decode(ctx, b) != 0) {
            llama_batch_free(b);
            return false;
        }
        // logits of a call are only readable until the next call, so each chunk is copied here
        for (int j = 0; rows && j < b.n_tokens; ++j) {
            if (b.logits[j]) {
                const float * lg = llama_get_logits_ith(ctx, j);
                rows->insert(rows->end(), lg, lg + n_vocab);
            }
        }
    }
    llama_batch_free(b);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        return fail("usage: truestar_llamacpp_driver MODEL.gguf REQUEST.txt OUT.bin");
    }
    int threads = 4, n_batch = 512, n_ubatch = 512, flash = 0, n_ctx = 512;
    std::string kv = "f16";
    float rope_base = 0.0f;
    std::vector<Item> items;
    std::ifstream req(argv[2]);
    if (!req) {
        return fail("cannot read the request file");
    }
    std::string line;
    while (std::getline(req, line)) {
        std::istringstream ss(line);
        std::string key;
        if (!(ss >> key)) {
            continue;
        }
        if (key == "threads") ss >> threads;
        else if (key == "batch") ss >> n_batch;
        else if (key == "ubatch") ss >> n_ubatch;
        else if (key == "flash") ss >> flash;
        else if (key == "ctx") ss >> n_ctx;
        else if (key == "kv") ss >> kv;
        else if (key == "rope_base") ss >> rope_base;
        else if (key == "prompt" || key == "greedy") {
            Item it;
            it.kind = key == "greedy";
            if (it.kind) ss >> it.n_new;
            int t = 0;
            ss >> t;
            for (int i = 0; i < t; ++i) {
                int id = 0;
                ss >> id;
                it.ids.push_back(id);
            }
            if ((int) it.ids.size() != t || t == 0) {
                return fail("a prompt line has fewer ids than it says");
            }
            items.push_back(it);
        } else {
            return fail(("unknown request key " + key).c_str());
        }
    }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return fail("llama_model_load_from_file failed");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_ubatch;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = kv == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
    cp.type_v = cp.type_k;
    cp.no_perf = true;
    cp.rope_freq_base = rope_base;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        return fail("llama_init_from_model failed");
    }

    FILE * out = fopen(argv[3], "wb");
    if (!out) {
        return fail("cannot write the output file");
    }
    const uint32_t head[3] = {1u, (uint32_t) n_vocab, (uint32_t) items.size()};
    fwrite("TSLC", 1, 4, out);
    fwrite(head, 4, 3, out);

    for (const Item & it : items) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int T = (int) it.ids.size();
        std::vector<llama_token> toks;
        std::vector<float> rows;
        if (it.kind == 0) {
            if (!decode_ids(ctx, it.ids, 0, (int) llama_n_batch(ctx), true, n_vocab, &rows)) {
                return fail("llama_decode failed on a prompt");
            }
            toks = it.ids;
        } else {
            if (!decode_ids(ctx, it.ids, 0, (int) llama_n_batch(ctx), false, n_vocab, nullptr)) {
                return fail("llama_decode failed on a greedy prompt");
            }
            int pos = T;
            for (int s = 0; s < it.n_new; ++s) {
                const float * lg = llama_get_logits_ith(ctx, -1);
                rows.insert(rows.end(), lg, lg + n_vocab);
                int best = 0;
                for (int v = 1; v < n_vocab; ++v) {
                    if (lg[v] > lg[best]) best = v;
                }
                toks.push_back(best);
                if (s + 1 < it.n_new) {
                    std::vector<llama_token> one(1, best);
                    if (!decode_ids(ctx, one, pos, 1, true, n_vocab, nullptr)) {
                        return fail("llama_decode failed on a greedy step");
                    }
                    ++pos;
                }
            }
        }
        const uint32_t kind = (uint32_t) it.kind, n_rows = (uint32_t) toks.size();
        fwrite(&kind, 4, 1, out);
        fwrite(&n_rows, 4, 1, out);
        fwrite(toks.data(), 4, toks.size(), out);
        fwrite(rows.data(), 4, rows.size(), out);
    }
    fclose(out);
    printf("info n_vocab=%d threads=%d threads_batch=%d n_batch=%u n_ubatch=%u n_ctx=%u flash=%d kv=%s\n",
           n_vocab, llama_n_threads(ctx), llama_n_threads_batch(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx),
           llama_n_ctx(ctx), flash, kv.c_str());
    printf("ggml version=%s commit=%s\n", ggml_version(), ggml_commit());
    printf("system %s\n", llama_print_system_info());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
