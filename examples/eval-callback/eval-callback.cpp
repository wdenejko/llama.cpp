#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstring>
#include <string>
#include <vector>

// [Q4X] Q4X_TOPK_DUMP=<file>: instead of the verbose debug printer, binary-dump
// every prefill-width I32 tensor whose name contains Q4X_TOPK_DUMP_FILTER
// (default "indexer_top_k") - the per-token selected KV cells of the QSA
// indexer. Record: 64-byte name, 4x int64 ne, then the i32 data. Capped by
// Q4X_TOPK_DUMP_MAX records (default 24). Feeds the top-k overlap analysis
// for the gathered sparse-prefill design.
struct q4x_topk_dump_state {
    FILE * f = nullptr;
    const char * filter = nullptr;
    int remaining = 0;
    int skip = 0;  // Q4X_TOPK_DUMP_SKIP: drop this many matches first (early
                   // ubatches select everything while n_kv <= k - only the
                   // deep ubatches carry a real sparsity pattern)
};

static bool q4x_topk_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (q4x_topk_dump_state *) user_data;
    const bool match = st->remaining > 0 &&
                       t->type == GGML_TYPE_I32 &&
                       t->ne[1] >= 512 &&
                       strstr(t->name, st->filter) != nullptr &&
                       strstr(t->name, "(view)") == nullptr;
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }
    if (st->skip > 0) {
        st->skip--;
        return true;
    }
    const size_t n = ggml_nelements(t);
    std::vector<int32_t> host(n);
    ggml_backend_tensor_get(t, host.data(), 0, n * sizeof(int32_t));
    char name[64] = {0};
    snprintf(name, sizeof(name) - 1, "%s", t->name);
    int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    fwrite(name, 1, sizeof(name), st->f);
    fwrite(ne, sizeof(int64_t), 4, st->f);
    fwrite(host.data(), sizeof(int32_t), n, st->f);
    fflush(st->f);
    st->remaining--;
    fprintf(stderr, "[topk-dump] %s ne=[%lld,%lld,%lld,%lld] (%d records left)\n",
            t->name, (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3], st->remaining);
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_debug_cb_user_data cb_data;

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    static q4x_topk_dump_state dump_state;
    const char * dump_path = getenv("Q4X_TOPK_DUMP");
    if (dump_path != nullptr) {
        dump_state.f = fopen(dump_path, "wb");
        if (dump_state.f == nullptr) {
            LOG_ERR("Q4X_TOPK_DUMP: cannot open %s\n", dump_path);
            return 1;
        }
        const char * filt = getenv("Q4X_TOPK_DUMP_FILTER");
        const char * maxs = getenv("Q4X_TOPK_DUMP_MAX");
        const char * skps = getenv("Q4X_TOPK_DUMP_SKIP");
        dump_state.filter    = filt ? filt : "indexer_top_k";
        dump_state.remaining = maxs ? atoi(maxs) : 24;
        dump_state.skip      = skps ? atoi(skps) : 0;
        params.cb_eval = q4x_topk_dump_cb;
        params.cb_eval_user_data = &dump_state;
    } else {
        params.cb_eval = common_debug_cb_eval;
        params.cb_eval_user_data = &cb_data;
    }
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
