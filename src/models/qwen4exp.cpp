#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include <cstdlib>
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <thread>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <cinttypes>

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // [HC_COLLAPSE] set by the offline w_up row-permute: hc_*_up rows are channel-major, which
    // lets the Vulkan mul_mm epilog fuse the mix collapse into the up-projection's store
    ml.get_key(LLM_KV_HYPER_CONNECTION_UP_PERM,  hparams.hc_up_perm, false);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;


    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { hc_dim }, 0);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, 0);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    // An MTP-only file carries just the draft block. Keep walking the trunk so the
    // per-layer bookkeeping still runs, but let its tensors be absent.
    const bool mtp_only = hparams.n_layer_nextn > 0 && ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr;
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, trunk_flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, trunk_flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, trunk_flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, trunk_flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, trunk_flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, trunk_flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, trunk_flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, trunk_flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, trunk_flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, trunk_flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, trunk_flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, trunk_flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, trunk_flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, trunk_flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, trunk_flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, trunk_flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, trunk_flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, trunk_flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, trunk_flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, trunk_flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, trunk_flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, trunk_flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, trunk_flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, trunk_flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, trunk_flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, trunk_flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, trunk_flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { hc_dim }, trunk_flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { hc_dim }, trunk_flags);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { hc_dim }, trunk_flags);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, trunk_flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, trunk_flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, trunk_flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, trunk_flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, trunk_flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, trunk_flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, trunk_flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, trunk_flags);
    }
    // The MTP draft block sits one past the trunk. It is a full qwen4exp layer plus the
    // three nextn tensors, and is skipped unless the file is opened as a draft.
    for (int il = n_layer; il < n_layer + (int) hparams.n_layer_nextn; ++il) {
        auto & layer = layers[il];

        const int flags = ml.load_mtp ? 0 : TENSOR_SKIP;

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;
        const int64_t idx_dim    = hparams.indexer_head_size;

        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

        // the draft attends dense, so the indexer weights are present but never read
        const int idx_flags = flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP;
        layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, idx_flags);
        layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, idx_flags);
        layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, idx_flags);
        layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, idx_flags);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        // hnorm spans the whole hyper-connection row, enorm just one stream
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);
    }

}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    // A draft-only MTP file carries just the nextn block, so the trunk hc weights are
    // absent (mtp_only at load time). Building the trunk graph would dereference them and
    // crash; fail clearly instead. The draft path (LLM_GRAPH_TYPE_DECODER_MTP) above is fine.
    if (hparams.n_layer_nextn > 0 && !layers.empty() && layers[0].hc_attn_norm == nullptr) {
        throw std::runtime_error(
            "qwen4exp: draft-only MTP head has no trunk layers; load it as a draft "
            "(-md / --model-draft) alongside a base model, not as the main model (-m)");
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [hc_dim] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    // A2: multiply by the per-(channel,stream) gamma BEFORE flattening, so RMS_NORM and MUL stay
    // adjacent and the Vulkan RMS_NORM_MUL fusion fires (the reshape between them blocked it).
    // w_norm is [hc_dim] laid out n_embd-fastest == [n_embd, hc]; the fused rms_norm shader indexes
    // src1 as src1_idx(0,row,channel,samp)+col, so [n_embd, hc, 1] broadcasts over the token axis.
    xn = ggml_mul(ctx0, xn, ggml_reshape_3d(ctx0, w_norm, n_embd, hc, 1));
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * up_out = build_lora_mm(w_up, lo);

    // [Q4X] fused mix tail: sigmoid + gate-mul + mean-collapse in one op
    // (replaces ~7 tiny dispatches per call, x96 per decode token). VERDICT
    // 2026-08-31: +0.5-0.9% tg / +0.3-0.6% pp, needle-validated (32k/64k HIT,
    // harness 10/10, accept unchanged; text diff = benign FMA contraction).
    // Q4X_HC_MIX_NOFUSE=1 forces the primitive chain (A/B and correctness checks).
    static const bool q4x_mix_nofuse = getenv("Q4X_HC_MIX_NOFUSE") != nullptr;

    // [HC_COLLAPSE] with the offline-permuted w_up (hparams.hc_up_perm) up_out's rows are
    // channel-major, so only the perm-aware collapse op reads them correctly: force the fused
    // op (the primitive chain below assumes stream-major rows). On Vulkan the backend then
    // folds this op into the up-projection's mul_mm store and up_out is never materialized.
    ggml_tensor * mixed;
    if (!q4x_mix_nofuse || hparams.hc_up_perm != 0) {
        // lo rides along as the keep-alive src so the allocator cannot hand its (dead) slot to
        // mixed: the fused epilog writes mixed while the up GEMM's other workgroups still read lo,
        // and the backend refuses that overlap by dropping the fusion for the layer.
        mixed = ggml_q4x_hc_mix_collapse(ctx0, xn, up_out, (int32_t) hc, (int32_t) hparams.hc_up_perm, lo);
        cb(mixed, "hc_mixed", il);
    } else {
        ggml_tensor * gate = ggml_sigmoid(ctx0, up_out);
        cb(gate, "hc_gate", il);

        ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
        cb(mixed, "hc_mixed", il);
    }

    if (inject) {
        // The inject projection is w_inject^T @ xn, i.e. M = hc (= 4), N = n_tokens: a
        // pathological small-M GEMM. On Vulkan/gfx1151 the tiled matmul wastes all but
        // hc rows of its M-tile, so this single op costs ~16% of prefill GPU time.
        // Compute it transposed instead -- mul_mat(xn, w_inject) has N = hc, which is
        // <= mul_mat_vec_max_cols, so the backend dispatches the efficient GEMV
        // (mul_mat_vec) path -- then transpose the tiny [n_tokens, hc] result back.
        // The swap puts w_inject in the GEMV `b` slot, which Vulkan only accepts as
        // F32/F16/Q8_1, so a quantized inject weight (e.g. the Q8_0 MTP draft head) or a
        // LoRA target takes the plain path instead. The trunk's inject is F16/F32, so the
        // prefill win is unaffected; the draft runs this only on small batches anyway.
        bool inject_swap_ok = w_inject->type == GGML_TYPE_F32 || w_inject->type == GGML_TYPE_F16;
        for (const auto & lora : *loras) {
            if (lora.first->get_weight(w_inject)) { inject_swap_ok = false; break; }
        }
        if (!inject_swap_ok) {
            *inject = build_lora_mm(w_inject, xn);
        } else {
            ggml_tensor * inj_t = ggml_mul_mat(ctx0, xn, w_inject);   // [n_tokens, hc]
            *inject = ggml_cont(ctx0, ggml_transpose(ctx0, inj_t));   // [hc, n_tokens]
        }
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so an untrained injection matrix
    // reproduces the plain residual add
    // escape hatch: Q4X_HC_NOFUSE=1 forces the primitive path (A/B correctness checks)
    static const bool q4x_hc_nofuse = getenv("Q4X_HC_NOFUSE") != nullptr;
    if (cparams.fused_q4x_hc_comb && il >= 0 && !q4x_hc_nofuse) {
        ggml_tensor * fused = ggml_q4x_hc_combine(ctx0, residual, block_out, inject);
        res->add_fused_node({LLM_FUSED_OP_Q4X_HC_COMBINE, fused, il});
        cb(fused, "hc_combine", il);
        return fused;
    }

    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

// members only: graph_mtp builds a single block instead of the trunk
llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params, bool) :
    llm_build_delta_net_base(params), model(model) {}

// ref: PR 27739 (JJJYmmm), reconciled against this tree's hyper-connection helpers.
// The draft is one block: embedding and handed-over hidden state are each normed, joined
// and projected, then run through the same attention and FFN the trunk uses.
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : graph(model, params, true) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "qwen4exp MTP supports a single block");
    GGML_ASSERT(ubatch.token && "qwen4exp MTP needs token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t n_embd = hparams.n_embd;

    const int    il    = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block is missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block is missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block is missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h = inp->h;
    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    // The draft block is the only layer in an MTP context and attends dense, so it takes a
    // plain attention input. create_memory gives that context an attention-only filter.
    auto        * inp_attn    = build_attn_inp_kv();

    // the reference norms the whole hyper-connection row and splits into streams after,
    // so hnorm spans hc*n_embd rather than one stream
    ggml_tensor * h_norm = build_norm(h, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    // every stream sees the same embedding term
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // fc_embedding(e) + fc_hidden(h) is one projection of the concatenation; the converter
    // merges the two checkpoint tensors into this single eh_proj.
    // Fold the hc streams into the column dimension so the projection is ONE GEMM with
    // n = hc*n_tokens. As a [2*n_embd, hc, n_tokens] operand the backends treat n_tokens as a
    // batch of mat-vecs with n = hc and re-read the weight once per token: measured 44 ms per
    // 512-token draft ubatch on gfx1151, 58% of the draft's prefill GPU time and ~4.5% of the
    // served prefill. Every output column is the same dot product either way; eh_proj_s is a
    // scalar, so it broadcasts over the 2D result just as it did over the 3D one.
    ggml_tensor * eh_in = ggml_concat(ctx0, e_norm, h_norm, 0);   // [2*n_embd, hc, n_tokens], contiguous
    eh_in = ggml_reshape_2d(ctx0, eh_in, 2*n_embd, hc*n_tokens);
    ggml_tensor * inpL = build_lora_mm(layer.nextn.eh_proj, eh_in, layer.nextn.eh_proj_s);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, hc, n_tokens);
    cb(inpL, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur    = build_hc_mix(inpL,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    // dense attention for the draft: a QSA indexer would need a cache of its own, and one
    // block spends its time reading weights rather than attending
    cur = build_layer_attn(inp_attn, nullptr, cur, inp_pos, sections, il);
    inpL = build_hc_combine(inpL, cur, inject, il);
    cb(inpL, "mtp_hc_attn_post", il);

    cur = build_hc_mix(inpL,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_hc_combine(inpL, cur, inject, il);
    cb(inpL, "mtp_l_out", il);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, hc*n_embd, n_tokens);

    // a chained head reads the streams back the way the trunk hands them over
    res->t_h_nextn = flat;

    if (inp_out_ids) {
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_mix(inpL, model.hc_head_norm, model.hc_head_down, model.hc_head_up, nullptr, nullptr, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // The output-row gather is deferred to after the loop (see t_h_nextn below):
        // the MTP head reads every token's wide hidden state, which the nextn extraction
        // copies unmasked over n_tokens, so res_hc must stay full through the last layer.
        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // Hand the wide residual to an MTP head, before the final mix collapses it. The draft
    // reads all hc streams, which is what its hnorm is sized for. res_hc is passed as is
    // rather than reshaped: a bare view is not a node the scheduler places, and the
    // extraction is a flat copy, for which [n_embd, hc, T] and [hc*n_embd, T] are identical.
    // hand the full-width residual (every token) to the MTP head; the nextn extraction
    // reads all n_tokens rows unmasked, so this must not be collapsed to output rows
    res->t_h_nextn = res_hc;

    // now drop the rows that produce no output: the head and logits are per-output
    if (inp_out_ids) {
        res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias, bool blk_topk, bool pooled, bool pooled_score) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias), blk_topk(blk_topk), pooled(pooled), pooled_score(pooled_score) {}
    virtual ~llm_graph_input_qsa() = default;

    // [TAG_QSA_POOLED_CACHE] the store keeps one sequence's summaries of one stream, so the
    // pooled graph is only sound for an ubatch that is wholly one sequence in one stream.
    // The reserve pass's mock ubatch (null seq pointers) counts: it stands in for that shape
    static bool pooled_usable(const llama_memory_hybrid_idx_context * mctx, const llama_ubatch & ubatch) {
        if (mctx->get_pooled_rows() == 0) {
            return false;
        }

        if (ubatch.seq_id == nullptr || ubatch.seq_id[0] == nullptr) {
            return true;
        }

        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.seq_id[i] == nullptr || ubatch.seq_id[i][0] != ubatch.seq_id[0][0]) {
                return false;
            }
        }

        return true;
    }

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_cells_dup, blk_pos, bias, ubatch, ratio, blk_bias, blk_topk,
                dirty_cells, dirty_pos, dirty_rows, blk_slot_bias);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= bias->ne[0]      == (blk_bias ? n_blocks : n_kv);
        res &= bias->ne[1]      == params.ubatch.n_tokens/n_stream;

        // [TAG_QSA_POOLED_CACHE] the graph committed to its write/score paths when it was
        // built; an ubatch on the other side of either line needs the other graph
        res &= pooled == (pooled_usable(mctx, params.ubatch) && n_stream == 1);
        res &= pooled_score == (pooled && params.ubatch.n_tokens <= 16);

        if (pooled_score) {
            // the summary view and the k_all gather view are both sized by the build-time n_kv
            res &= n_kv_build == n_kv;

            // a watermark clamp (rollback, sequence switch) can ask for more rows than the
            // built tensors hold; the fill pads any excess capacity with dustbin writes
            const int64_t need = mctx->qsa_pooled_n_dirty_max(&params.ubatch, ratio);
            res &= need <= dirty_rows->ne[0];

            // and a catch-up-sized graph must not persist into steady decode, where its
            // full-width dustbin-padded write chain would run every token
            res &= dirty_rows->ne[0] <= 4*need + 8;
        }

        if (!pooled_score) {
            res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
            res &= blk_cells->ne[1] == n_stream;
            res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        }

        if (blk_topk) {
            // no cell_blk to carry the exact n_kv (blk_cells only pins ceil(n_kv/r)), and the
            // k_all view is sized by the build-time n_kv, so require an exact match
            res &= n_kv_build == n_kv;
            res &= blk_cells_dup != nullptr && blk_cells_dup->ne[0] == (int64_t) ratio*n_blocks;

            // [TAG_QSA_GATHER] gather-activity is pinned by checks above (exact n_kv, exact
            // n_tokens via k_idxs, static threshold), so presence only needs a size check
            res &= blk_slot_bias == nullptr || blk_slot_bias->ne[0] == (int64_t) ratio*n_blocks;
        } else {
            res &= cell_blk->ne[0] == n_kv;
            res &= cell_blk->ne[1] == n_stream;
        }

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs        = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk      = nullptr;   // I32 [n_kv, n_stream]; null under blk_topk (unused there)
    ggml_tensor * blk_cells     = nullptr;   // I32 [ratio*n_blocks, n_stream]; null under pooled (unused there)
    ggml_tensor * blk_cells_dup = nullptr;   // I32 [ratio*n_blocks, n_stream], unfilled slots repeat a filled cell
    ggml_tensor * blk_pos       = nullptr;   // I32 [4*n_blocks*n_stream]; null under pooled
    ggml_tensor * bias          = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    // [TAG_QSA_POOLED_CACHE] under pooled only: the store rows this ubatch writes
    ggml_tensor * dirty_cells   = nullptr;   // I32 [ratio*n_dirty, 1]
    ggml_tensor * dirty_pos     = nullptr;   // I32 [4*n_dirty]
    ggml_tensor * dirty_rows    = nullptr;   // I64 [n_dirty]

    // [TAG_QSA_GATHER] under blk_topk when the gather can activate: 0 where a block slot holds
    // its own cell, -inf where blk_cells_dup repeated one -- a gathered softmax must not count
    // a duplicated row twice (the dense path's set_rows unmask is idempotent and never cared)
    ggml_tensor * blk_slot_bias = nullptr;   // F32 [ratio*n_blocks, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    // select whole blocks from the per-block scores (never expanding to per-cell scores); needs blk_bias
    const bool blk_topk;

    // the pooled summary store is usable for this ubatch (single sequence, single stream)
    const bool pooled;

    // write the completed-since-watermark blocks into the store and score from it, instead of
    // re-pooling every block. Decode-sized batches only: the write chain measured ~3% pp at
    // prefill widths, while scoring from the store is +6-8% tg at decode -- so prefill keeps
    // pure recompute and the first decode graph catches up the store in one write
    const bool pooled_score;

    // n_kv the graph was built for; can_reuse needs it when cell_blk is absent (blk_topk)
    int64_t n_kv_build = 0;
};

llama_model_qwen4exp::graph::qsa_selection llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // block-level top-k: select whole blocks from the [n_blocks] scores instead of expanding to the
    // [n_kv, n_tps] per-cell scores -- that expansion (plus the f32 mask cast) is the ubatch-scaled
    // prefill memory hog at deep context. Needs the per-block bias to carry visibility on its own.
    static const bool blk_topk_env = getenv("Q4X_QSA_BLK_TOPK") != nullptr;
    const bool blk_topk = blk_topk_env && blk_bias;

    // [TAG_QSA_POOLED_CACHE] a complete block's summary never changes, so decode-sized batches
    // keep a store of them and score O(new) instead of O(n_kv) per token (+6-8% tg at depth).
    // Prefill stays pure recompute -- the store write chain alone measured ~3% pp -- and the
    // first decode graph catches the store up over the whole prefilled span in one write.
    // Allocation already gated on the unified layout; the ubatch must also be a single
    // sequence in a single stream, or the graph falls back to recompute-only
    const bool pooled_use       = llm_graph_input_qsa::pooled_usable(mctx_hyb, ubatch) && n_stream == 1;
    const bool pooled_score_use = pooled_use && n_tokens <= 16;

    // [TAG_QSA_GATHER] when the attention will gather the selected rows, the selection widens
    // to the gather's padded row count here, so both ends agree on one width
    const int64_t width_blk_nom = std::min<int64_t>(n_blocks,
            ((int64_t) hparams.indexer_top_k + r - 1 + r - 1)/r);
    const int64_t n_sel_blk     = blk_topk ? qsa_gather_n_sel(n_kv, r*width_blk_nom, r) : 0;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias, blk_topk, pooled_use, pooled_score_use);

        qsa->k_idxs = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->bias   = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        ggml_set_input(qsa->bias);

        // an input the graph never consumes is never allocated, and set_input would then write
        // through a null buffer -- so create exactly the tensors this graph's paths read:
        // the dirty list feeds the store write, the gather tables feed recompute scoring.
        // [TAG_QSA_POOLED_CACHE] writes are lazy: prefill graphs skip the store entirely
        // (the write chain measured ~3% pp) and the first decode graph catches up the whole
        // span in one write, sized by the same qsa_pooled_n_dirty_max
        if (pooled_score_use) {
            const int64_t n_dirty = mctx_hyb->qsa_pooled_n_dirty_max(&ubatch, (uint32_t) r);

            qsa->dirty_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_dirty, 1);
            qsa->dirty_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_dirty);
            qsa->dirty_rows  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_dirty);

            ggml_set_input(qsa->dirty_cells);
            ggml_set_input(qsa->dirty_pos);
            ggml_set_input(qsa->dirty_rows);
        }
        if (!pooled_score_use) {
            qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
            qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);

            ggml_set_input(qsa->blk_cells);
            ggml_set_input(qsa->blk_pos);
        }

        // cell_blk only feeds the per-cell score expansion; the blk_topk path returns before it
        if (blk_topk) {
            qsa->blk_cells_dup = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
            ggml_set_input(qsa->blk_cells_dup);

            // [TAG_QSA_GATHER] only the gathered softmax consumes the slot validity
            if (n_sel_blk > 0) {
                qsa->blk_slot_bias = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, r*n_blocks, n_stream);
                ggml_set_input(qsa->blk_slot_bias);
            }
        } else {
            qsa->cell_blk = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
            ggml_set_input(qsa->cell_blk);
        }

        qsa->n_kv_build = n_kv;

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    ggml_tensor * pooled = nullptr;

    if (inp->pooled_score) {
        // [TAG_QSA_POOLED_CACHE] pool the blocks completed since the watermark (after a prefill
        // this is one catch-up covering the whole new span), write their summary rows into the
        // layer's store, and score against the store. Rows past the watermark hold
        // stale-but-finite values whose scores the bias masks, like the garbage partial pools
        ggml_tensor * store = mctx_hyb->get_pooled_k(il);
        GGML_ASSERT(store != nullptr);
        GGML_ASSERT(n_blocks <= (int64_t) mctx_hyb->get_pooled_rows() - 1 && "qsa: pooled store view would cover the dustbin");

        const int64_t n_dirty = inp->dirty_rows->ne[0];

        // gather the member cells of the fresh blocks: same mean/norm/rope as the recompute
        // path, over n_dirty blocks instead of n_blocks
        ggml_tensor * fresh = ggml_get_rows(ctx0, k_all, inp->dirty_cells);
        fresh = ggml_reshape_4d(ctx0, fresh, idx_dim, r, n_dirty, 1);

        ggml_tensor * sum = nullptr;
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_cont(ctx0,
                    ggml_view_3d(ctx0, fresh, idx_dim, n_dirty, 1,
                            fresh->nb[2], fresh->nb[3], i*fresh->nb[1]));
            sum = sum ? ggml_add(ctx0, sum, slice) : slice;
        }
        sum = ggml_scale(ctx0, sum, 1.0f/(float) r);
        cb(sum, "indexer_k_pooled", il);

        sum = ggml_reshape_3d(ctx0, sum, idx_dim, 1, n_dirty);
        sum = build_norm(sum, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
        sum = ggml_rope_multi(ctx0, sum, inp->dirty_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        sum = ggml_reshape_2d(ctx0, sum, idx_dim, n_dirty);

        // expanded before the score below is built, so the write orders ahead of the read
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, store, sum, inp->dirty_rows));

        pooled = ggml_view_3d(ctx0, store, idx_dim, n_blocks, 1,
                store->nb[1], store->nb[1]*n_blocks, 0);
        cb(pooled, "indexer_k", il);
    }

    if (pooled == nullptr) {
        // gathers per stream: blk_cells row s indexes stream s's own cells
        ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
        members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_blocks, n_stream);

        // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_cont(ctx0,
                    ggml_view_3d(ctx0, members, idx_dim, n_blocks, n_stream,
                            members->nb[2], members->nb[3], i*members->nb[1]));
            pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
        }
        pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
        cb(pooled, "indexer_k_pooled", il);

        // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);
        pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
        pooled = ggml_rope_multi(ctx0, pooled, inp->blk_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
        cb(pooled, "indexer_k", il);
    }

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
    // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
    // (q is the rope output, already contiguous, so it reshapes directly)
    ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
            ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream));
    score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
    score = ggml_relu(ctx0, score);

    // the heads sit side by side on ne[1] and there are only a few of them
    ggml_tensor * summed = nullptr;
    for (int64_t h = 0; h < n_idx_h; ++h) {
        ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, n_tps, n_stream,
                score->nb[2], score->nb[3], h*score->nb[1]);
        summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
    }

    score = summed;
    cb(score, "indexer_score", il);

    // one value per block, so it is cheaper to bias here than after the cells are expanded
    if (blk_bias) {
        score = ggml_add(ctx0, score, inp->bias);
    }

    if (inp->blk_topk) {
        // select whole blocks straight from the per-block scores. The blk_topk bias variant
        // carries visibility on its own (tail forced in; strictly-future and incomplete blocks
        // -inf), and build_attn_qsa re-applies the causal KQ mask after unmasking, so the tail
        // overshoot and any -inf block picked in the underfull regime are re-masked. This skips
        // the [n_kv, n_tps] expanded scores, the f32 mask cast and the n_kv-wide sort.
        // [TAG_QSA_GATHER] under the gather the selection widens to n_sel_blk blocks; the extra
        // entries are real score-ranked blocks the gathered mask re-masks
        const int64_t sel_blk = n_sel_blk > 0 ? n_sel_blk/r : width_blk_nom;

        ggml_tensor * top_blk = ggml_cont(ctx0, ggml_top_k(ctx0, score, sel_blk));
        cb(top_blk, "indexer_top_blk", il);

        // expand block ids to their member cells; blk_cells_dup repeats a filled cell into a
        // block's unfilled slots so the extra set_rows writes are idempotent
        ggml_tensor * blk2cell = ggml_reshape_3d(ctx0, inp->blk_cells_dup, r, n_blocks, n_stream);
        ggml_tensor * blk_idx  = ggml_reshape_2d(ctx0, top_blk, sel_blk*n_tps, n_stream);

        ggml_tensor * top_k = ggml_get_rows(ctx0, blk2cell, blk_idx);

        // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
        top_k = ggml_reshape_4d(ctx0, top_k, r*sel_blk, n_tps, 1, n_stream);
        cb(top_k, "indexer_top_k", il);

        // [TAG_QSA_GATHER] expand per-slot validity through the same block gather, so a row
        // blk_cells_dup duplicated is counted once by the gathered softmax
        ggml_tensor * sel_bias = nullptr;
        if (n_sel_blk > 0) {
            GGML_ASSERT(inp->blk_slot_bias != nullptr);

            ggml_tensor * slot2bias = ggml_reshape_3d(ctx0, inp->blk_slot_bias, r, n_blocks, n_stream);

            sel_bias = ggml_get_rows(ctx0, slot2bias, blk_idx);
            sel_bias = ggml_reshape_4d(ctx0, sel_bias, 1, r*sel_blk, n_tps, n_stream);
            cb(sel_bias, "indexer_sel_bias", il);
        }

        return { top_k, sel_bias, r*width_blk_nom };
    }

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    if (blk_bias) {
        // flash attention keeps the mask in f16; the scores are f32
        ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx0, expanded, inp->bias);
    }
    cb(expanded, "indexer_score_tokens", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    // [TAG_QSA_GATHER] widen to the padded gather width when active; the per-cell sort never
    // duplicates an index, so no slot bias is needed on this path
    const int64_t n_sel_cell = qsa_gather_n_sel(n_kv, width, r);
    const int64_t n_sel      = n_sel_cell > 0 ? n_sel_cell : width;

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, n_sel));

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, n_sel, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return { top_k, nullptr, width };
}

// [TAG_QSA_GATHER] (ported from the apepojken fork, 28c9d7302)
// At depth, masked-dense QSA attention still streams the whole KV cache to discard all but
// ~2k selected rows. Past a threshold it is cheaper to gather the selected rows into small
// contiguous tensors and run dense flash attention over those. This gate returns the padded
// gathered-row count, or 0 for the masked-dense path; build_qsa_top_k widens its selection by
// the same call, so the attention gathers exactly what was selected.
int64_t llama_model_qwen4exp::graph::qsa_gather_n_sel(int64_t n_kv, int64_t width, int64_t r) const {
    // Q4X_QSA_GATHER: unset or <= 0 -> OFF; N -> gather from N cells (1 = always).
    // Default flipped to off 2026-08-30: the first honest on/off A/B (the auto-on-at-32768
    // default had silently been active in every deep measurement) has masked-dense ahead at
    // every depth tried - 47.0 vs 46.1 tg @32k, 45.3 vs 37.2 @48k
    static const int64_t min_kv = [] {
        const char * env = getenv("Q4X_QSA_GATHER");
        if (env == nullptr) {
            return (int64_t) INT64_MAX;
        }
        const long long v = atoll(env);
        return v <= 0 ? (int64_t) INT64_MAX : (int64_t) v;
    }();

    // the gather wants FA's non-transposed V cache and a mask that only carries visibility
    if (!cparams.flash_attn || hparams.use_alibi) {
        return 0;
    }

    // each token gathers its own rows, so the cost scales with the batch: decode and
    // speculative verify batches only, never prefill
    if (n_tokens > 16 || n_kv < min_kv) {
        return 0;
    }

    // FA's fastest kernels want the KV length padded to 256; also keep it a whole number of
    // blocks so the blk-topk route can select block-wise (any power-of-two r <= 256 divides it)
    int64_t n_sel = GGML_PAD(width, 256);
    n_sel = ((n_sel + r - 1)/r)*r;

    if (n_sel >= n_kv) {
        return 0;
    }

    return n_sel;
}

ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa_gather(
        ggml_tensor *         k,
        ggml_tensor *         v,
        ggml_tensor *         kq_mask,
        ggml_tensor *         q_cur,
        const qsa_selection & sel,
        float                 kq_scale,
        int                   il) {
    ggml_tensor * top_k = sel.top_k;

    const int64_t n_sel = top_k->ne[0];
    const int64_t n_tps = top_k->ne[1];
    const int64_t ns    = top_k->ne[3];
    const int64_t nt    = n_tps*ns;

    const int64_t d_k  = k->ne[0];
    const int64_t hkv  = k->ne[1];
    const int64_t n_kv = k->ne[2];
    const int64_t d_v  = v->ne[0];

    const int64_t n_head = q_cur->ne[1];

    GGML_ASSERT(ggml_is_contiguous(top_k));
    GGML_ASSERT(kq_mask->type == GGML_TYPE_F16);
    GGML_ASSERT(sel.width <= n_sel);

    // one flat row-list per stream
    ggml_tensor * idx = ggml_view_2d(ctx0, top_k, n_sel*n_tps, ns, top_k->nb[3], 0);

    // a cache cell is one contiguous d*hkv row, so fold the heads away and gather per cell
    ggml_tensor * k_rows = ggml_view_3d(ctx0, k, d_k*hkv, n_kv, ns, k->nb[2], k->nb[3], 0);
    ggml_tensor * v_rows = ggml_view_3d(ctx0, v, d_v*hkv, n_kv, ns, v->nb[2], v->nb[3], 0);

    // get_rows emits f32; cast back for FA (a fused src-type->f16 gather would skip this)
    ggml_tensor * k_sel = ggml_cast(ctx0, ggml_get_rows(ctx0, k_rows, idx), GGML_TYPE_F16);
    ggml_tensor * v_sel = ggml_cast(ctx0, ggml_get_rows(ctx0, v_rows, idx), GGML_TYPE_F16);

    // token t of stream s is ne3 slice s*n_tps + t (the ubatch order); heads stay interleaved
    // inside a row (nb2 < nb1), the same stride pattern decode FA reads from the cache
    const size_t row_k = ggml_row_size(GGML_TYPE_F16, d_k);
    const size_t row_v = ggml_row_size(GGML_TYPE_F16, d_v);

    ggml_tensor * k_g = ggml_view_4d(ctx0, k_sel, d_k, n_sel, hkv, nt,
            row_k*hkv, row_k, row_k*hkv*n_sel, 0);
    ggml_tensor * v_g = ggml_view_4d(ctx0, v_sel, d_v, n_sel, hkv, nt,
            row_v*hkv, row_v, row_v*hkv*n_sel, 0);

    // each selected cell keeps its original mask value, so selection never unmasks anything:
    // causally-invisible, foreign and empty cells that got selected stay -inf
    ggml_tensor * idx_w = ggml_view_3d(ctx0, top_k, sel.width, n_tps, ns, top_k->nb[1], top_k->nb[3], 0);

    ggml_tensor * m_sel = ggml_get_rows(ctx0,
            ggml_reshape_4d(ctx0, kq_mask, 1, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[3]), idx_w);

    // entries past the reference width are kernel padding
    if (sel.width < n_sel) {
        ggml_tensor * pad = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_sel - sel.width, n_tps, ns);
        pad = ggml_fill(ctx0, pad, -INFINITY);

        m_sel = ggml_concat(ctx0, m_sel, pad, 1);
    }

    // a row blk_cells_dup duplicated must not be counted twice by the softmax
    if (sel.sel_bias != nullptr) {
        m_sel = ggml_add(ctx0, m_sel, sel.sel_bias);
    }

    m_sel = ggml_cast(ctx0, m_sel, GGML_TYPE_F16);
    m_sel = ggml_reshape_4d(ctx0, m_sel, n_sel, 1, 1, nt);

    // one query row per ne3 slice, matching the gathered KV order
    ggml_tensor * q = ggml_reshape_4d(ctx0, q_cur, d_k, 1, n_head, nt);

    ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q, k_g, v_g, m_sel, kq_scale, hparams.f_max_alibi_bias,
                              hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
    res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

    ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

    cur = ggml_reshape_2d(ctx0, cur, d_v*n_head, n_tokens);

    ggml_build_forward_expand(gf, cur);
    cb(cur, "kqv_out_gather", il);

    return cur;
}

// [TAG_QSA_GP] gathered sparse prefill. At depth, masked-dense QSA prefill streams the
// whole KV cache through flash attention to discard all but the selected cells. Tokens of
// one tile share most of their selection (union of W=128 neighbours ~ 45-50% of n_kv at
// 32k, measured), so per W-token tile: bitmap-union the tile's top-k lists into one
// ascending cell list, gather those K/V rows f16->f16, gather each token's row of the
// sparse mask (already -inf outside its own selection - per-token strictness survives the
// union widening), and run standard FA over the C gathered cells.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa_gather_prefill(
        ggml_tensor *         k,
        ggml_tensor *         v,
        ggml_tensor *         kq_mask_sparse,
        ggml_tensor *         q_cur,
        const qsa_selection & sel,
        int64_t               W,
        int64_t               C,
        float                 kq_scale,
        int                   il) {
    ggml_tensor * top_k = sel.top_k;

    const int64_t n_sel = top_k->ne[0];
    const int64_t n_tps = top_k->ne[1];

    const int64_t d_k  = k->ne[0];
    const int64_t hkv  = k->ne[1];
    const int64_t n_kv = k->ne[2];
    const int64_t d_v  = v->ne[0];

    const int64_t n_head  = q_cur->ne[1];
    const int64_t n_tiles = n_tps / W;

    GGML_ASSERT(top_k->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous(top_k));
    GGML_ASSERT(kq_mask_sparse->type == GGML_TYPE_F16);
    GGML_ASSERT(kq_mask_sparse->ne[1] >= n_tps);
    GGML_ASSERT(n_tps % W == 0);

    // a cache cell is one contiguous d*hkv row, so fold the heads away and gather per cell
    ggml_tensor * k_rows = ggml_view_2d(ctx0, k, d_k*hkv, n_kv, k->nb[2], 0);
    ggml_tensor * v_rows = ggml_view_2d(ctx0, v, d_v*hkv, n_kv, v->nb[2], 0);

    const size_t row_k = ggml_row_size(GGML_TYPE_F16, d_k);
    const size_t row_v = ggml_row_size(GGML_TYPE_F16, d_v);

    ggml_tensor * out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_v*n_head, n_tps);

    for (int64_t ti = 0; ti < n_tiles; ++ti) {
        // union of the tile's selections; pads in top_k name real (lowest-scoring) cells
        // and the bitmap dedups them, so the full n_sel width goes in
        ggml_tensor * tk = ggml_view_2d(ctx0, top_k, n_sel, W, top_k->nb[1], ti*W*top_k->nb[1]);
        ggml_tensor * list = ggml_q4x_qsa_union(ctx0, tk, (int32_t) n_kv, (int32_t) C);

        // each token keeps its own sparse-mask row: union cells outside its selection
        // stay -inf, list pads (cell 0) get -inf from the count guard
        ggml_tensor * m_t = ggml_view_2d(ctx0, kq_mask_sparse, n_kv, W,
                kq_mask_sparse->nb[1], ti*W*kq_mask_sparse->nb[1]);
        ggml_tensor * m_g = ggml_q4x_qsa_mask_gather(ctx0, m_t, list);

        ggml_tensor * k_sel = ggml_q4x_qsa_kv_gather(ctx0, k_rows, list);
        ggml_tensor * v_sel = ggml_q4x_qsa_kv_gather(ctx0, v_rows, list);

        // heads stay interleaved inside a row (nb2 < nb1), the same stride pattern the
        // decode gather feeds FA
        ggml_tensor * k_g = ggml_view_4d(ctx0, k_sel, d_k, C, hkv, 1,
                row_k*hkv, row_k, row_k*hkv*C, 0);
        ggml_tensor * v_g = ggml_view_4d(ctx0, v_sel, d_v, C, hkv, 1,
                row_v*hkv, row_v, row_v*hkv*C, 0);

        ggml_tensor * q_t = ggml_view_3d(ctx0, q_cur, d_k, n_head, W,
                q_cur->nb[1], q_cur->nb[2], ti*W*q_cur->nb[2]);
        ggml_tensor * q = ggml_permute(ctx0, q_t, 0, 2, 1, 3);   // [d_k, W, n_head, 1]

        ggml_tensor * cur = ggml_flash_attn_ext(ctx0, q, k_g, v_g, m_g, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});

        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, d_v*n_head, W);

        // the tiles land in slices of one output tensor; expansion order keeps the
        // copies ahead of the consumer on the single-backend graph
        ggml_tensor * slice = ggml_view_2d(ctx0, out, d_v*n_head, W, out->nb[1], ti*W*out->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, cur, slice));
    }

    ggml_build_forward_expand(gf, out);
    cb(out, "kqv_out_gp", il);

    return out;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        const qsa_selection &     sel,
        float                     kq_scale,
        int                       il) {
    ggml_tensor * top_k = sel.top_k;
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    // [TAG_QSA_GATHER] past the threshold, gather the selected rows and run dense FA over
    // them instead of masking the full cache; the current ubatch's rows are already stored
    // above, so they are gatherable. Same gate call as the selection widening, so the widths
    // agree; the assert catches any drift between the two call sites (the attention and idx
    // caches share slot infos and padding, so their n_kv are always equal)
    {
        const int64_t n_sel = qsa_gather_n_sel(mctx_cur->get_n_kv(), sel.width, hparams.dsv4_compress_ratios[il]);

        if (n_sel > 0) {
            GGML_ASSERT(top_k->ne[0] == n_sel);

            ggml_tensor * k = mctx_cur->get_k(ctx0, il);
            ggml_tensor * v = mctx_cur->get_v(ctx0, il);

            ggml_tensor * cur = build_attn_qsa_gather(k, v, kq_mask, q_cur, sel, kq_scale, il);

            // the rotation is its own inverse, so undo it on the value side of the output
            if (inp->self_v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
            }

            return cur;
        }
    }

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    // [TAG_QSA_GP] gathered sparse prefill (see build_attn_qsa_gather_prefill).
    // Q4X_QSA_GP: unset or <= 0 -> OFF; N -> engage from n_kv >= N (1 = always).
    // Q4X_QSA_GP_W: tile width (default 128, multiple of 64). Q4X_QSA_GP_FRAC:
    // gathered-list capacity as a fraction of n_kv (default 0.62; the tile union
    // must fit or the highest cells get dropped - check depth with the topk dump).
    {
        static const int64_t gp_min_kv = [] {
            const char * env = getenv("Q4X_QSA_GP");
            if (env == nullptr) {
                return (int64_t) 0;
            }
            const long long val = atoll(env);
            return val <= 0 ? (int64_t) 0 : (int64_t) val;
        }();
        static const int64_t gp_w = [] {
            const char * env = getenv("Q4X_QSA_GP_W");
            const long long val = env ? atoll(env) : 128;
            return (val >= 64 && val % 64 == 0) ? (int64_t) val : (int64_t) 128;
        }();
        static const double gp_frac = [] {
            const char * env = getenv("Q4X_QSA_GP_FRAC");
            const double val = env ? atof(env) : 0.62;
            return (val > 0.0 && val <= 1.0) ? val : 0.62;
        }();

        const int64_t n_tps = top_k->ne[1];
        const int64_t ns    = top_k->ne[3];
        const int64_t n_kv  = mctx_cur->get_n_kv();

        // prefill only (the decode gather above owns small batches); the union shader's
        // bitmap covers 262144 cells
        if (gp_min_kv > 0 && cparams.flash_attn && !hparams.use_alibi &&
                n_tokens > 16 && ns == 1 && n_tps % gp_w == 0 &&
                n_kv >= gp_min_kv && n_kv <= 262144) {
            int64_t c_pad = (int64_t) (gp_frac*(double) n_kv);
            c_pad = std::max<int64_t>(c_pad, 3*(int64_t) sel.width);
            c_pad = std::min<int64_t>(c_pad, n_kv);
            c_pad = GGML_PAD(c_pad, 256);

            ggml_tensor * k = mctx_cur->get_k(ctx0, il);
            ggml_tensor * v = mctx_cur->get_v(ctx0, il);

            if (c_pad < n_kv && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) {
                static bool logged = false;
                if (!logged) {
                    fprintf(stderr, "qwen4exp: QSA gathered prefill engaged (n_kv=%" PRId64 " C=%" PRId64 " W=%" PRId64 ")\n",
                            n_kv, c_pad, gp_w);
                    logged = true;
                }

                ggml_tensor * cur = build_attn_qsa_gather_prefill(k, v, kq_mask_top_k, q_cur, sel, gp_w, c_pad, kq_scale, il);

                // the rotation is its own inverse, so undo it on the value side of the output
                if (inp->self_v_rot) {
                    cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
                }

                return cur;
            }
        }
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // Deep-context FA query-chunking lives in build_attn_mha (shared by this QSA dense path and
    // the non-QSA build_attn path), so the single deep FLASH_ATTN_EXT dispatch cannot trip the
    // amdgpu compute-ring watchdog. See llm_graph_context::build_attn_mha.
    // n_kv_max = 0: sparse FA (upstream #27970) is CUDA/Metal-only and upstream keeps it off for
    // qwen4exp as well ("TODO: enable sparse attention when we are ready")
    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb != nullptr && mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    const qsa_selection sel = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il) : qsa_selection{};

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (sel.top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, sel, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// [Q4X_PLE_PREFETCH] The PLE table is a 27G mapping-resident (TENSOR_READ_LAZY) host tensor gathered on
// the CPU at the start of every ubatch. On a box that cannot keep the whole table resident, that gather
// page-faults its rows in one by one, serially with the GPU work (measured: a 2048-token real-code
// prefill loses 15-25% to those faults). The rows are a pure function of the tokens, so they can be
// asked for ahead of time: this computes the same n-gram rows set_input() will gather and issues an
// asynchronous MADV_WILLNEED for their pages from a background thread. Predecessors before the supplied
// context read as EOS (the first few rows of a prompt may be missed; the real gather still faults them).
// Q4X_PLE_PREFETCH=0 disables. A no-op unless the table is host-resident.
void llama_model_qwen4exp::prefetch_prompt(const llama_token * tokens, int32_t n_tokens, int32_t n_ctx_prev) const {
#ifdef __linux__
    static const bool enabled = [] {
        const char * e = getenv("Q4X_PLE_PREFETCH");
        return !(e && e[0] == '0');
    }();
    const ggml_tensor * t = per_layer_tok_embd;
    if (!enabled || !t || !t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) || !tokens) {
        return;
    }
    if (n_ctx_prev < 0 || n_tokens <= n_ctx_prev) {
        return;
    }

    const auto & hp = hparams;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;

    const int64_t t_start = ggml_time_us();

    std::vector<int32_t> idx;
    idx.reserve((size_t) (n_tokens - n_ctx_prev) * hp.ple_n_heads);

    for (int64_t i = n_ctx_prev; i < n_tokens; ++i) {
        // same window and EOS-cut rules as llm_graph_input_ple::set_input()
        int64_t ctx[LLAMA_MAX_PLE_NGRAM];
        ctx[0] = tokens[i];
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            const int64_t     k = i - s;
            const llama_token p = (cut || k < 0) ? LLAMA_TOKEN_NULL : tokens[k];
            cut = cut || p < 0 || p == eos;
            ctx[s] = cut ? eos : p;
        }
        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx.push_back((int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]));
            }
        }
    }

    // rows -> page-aligned byte ranges of the mapping, coalesced
    const size_t    pg        = (size_t) sysconf(_SC_PAGESIZE);
    const size_t    row_bytes = ggml_row_size(t->type, t->ne[0]);
    const uintptr_t base      = (uintptr_t) t->data;

    std::vector<uintptr_t> pages;
    pages.reserve(idx.size() * 2);
    for (const int32_t r : idx) {
        const uintptr_t a  = base + (uintptr_t) r * (uintptr_t) t->nb[1];
        const uintptr_t p0 = a & ~(uintptr_t) (pg - 1);
        const uintptr_t p1 = (a + row_bytes - 1) & ~(uintptr_t) (pg - 1);
        for (uintptr_t p = p0; p <= p1; p += pg) {
            pages.push_back(p);
        }
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());

    std::vector<std::pair<uintptr_t, size_t>> ranges;
    for (const uintptr_t p : pages) {
        if (!ranges.empty() && ranges.back().first + ranges.back().second == p) {
            ranges.back().second += pg;
        } else {
            ranges.emplace_back(p, pg);
        }
    }

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "qwen4exp: ple prefetch engaged (first: %d tokens -> %zu rows, %zu pages in %zu ranges, %.1f ms to compute)\n",
                (int) (n_tokens - n_ctx_prev), idx.size(), pages.size(), ranges.size(), (ggml_time_us() - t_start) / 1000.0);
    }

    // the hint is issued off the calling thread: the caller is about to submit GPU work and must not
    // wait on 10^4 madvise calls. The mapping outlives every request (model lifetime).
    std::thread([ranges = std::move(ranges)]() {
        for (const auto & r : ranges) {
            madvise((void *) r.first, r.second, MADV_WILLNEED);
        }
    }).detach();
#else
    GGML_UNUSED(tokens); GGML_UNUSED(n_tokens); GGML_UNUSED(n_ctx_prev);
#endif
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    // cont the transposed activations before the concat: concat's non-contiguous path costs
    // ~13ms per GDN layer on 2048-token prefill chunks (~463ms per chunk over 36 layers,
    // measured on gfx1151 by the apepojken fork - +8% prefill at every depth from this line)
    // A3 (measured +4.2% pp d0, byte-exact): the transposed-concat kernel subsumes the explicit
    // cont, which the stale comment above credited with +8% before that kernel existed. Default now
    // DROPS the cont; Q4X_A3_KEEP_CONT=1 restores it (regression fallback).
    ggml_tensor * xt = ggml_transpose(ctx0, x);
    { static const bool a3_keep_cont = getenv("Q4X_A3_KEEP_CONT") != nullptr; if (a3_keep_cont) xt = ggml_cont(ctx0, xt); }
    ggml_tensor * conv_input = ggml_concat(ctx0, state, xt, 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    // with rollback enabled, also fill the per-token snapshot planes: the conv state
    // "s tokens back" is just the window ending s tokens earlier, so each plane is a
    // shifted view of conv_input (mirrors the shared build_conv_state K-loop). Slot 0 is
    // the current state, slots 1..n_rs_seq the snapshots a rollback of that many tokens
    // reads; BOTH conv rows (delta-net r_l and PLE p_l) get every plane, or a rollback
    // read them stale and generation degraded a little on every speculative rejection.
    // n_rs_seq == 0 (rollback off) degenerates to the single slot-0 write.
    // [TAG_RECURRENT_ROLLBACK_SPLITS] the last K tokens of a seq share one ubatch.
    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);
        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a weight over the whole hc*n_embd layout
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
