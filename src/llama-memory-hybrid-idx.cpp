#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // [TAG_QSA_POOLED_CACHE] one summary row per position block, kept beside the indexer keys
    // per-stream cell arrays would need a row set per stream, so only the unified layout gets the cache
    if (mem_idx && mem_idx->get_n_stream() == 1 && getenv("LLAMA_QSA_NO_POOLED_CACHE") == nullptr) {
        uint32_t ratio = 0;

        std::vector<int32_t> layers;
        for (const int32_t il : mem_idx->get_layer_ids()) {
            const uint32_t r = model.hparams.dsv4_compress_ratios[il];
            if (r == 0) {
                continue;
            }

            // a single pooled_rows serves every layer, so the ratio must not vary between them
            GGML_ASSERT(ratio == 0 || ratio == r);
            ratio = r;

            layers.push_back(il);
        }

        const uint32_t idx_dim = model.hparams.indexer_head_size;

        if (ratio > 0 && idx_dim > 0 && !layers.empty()) {
            // one row per complete block, one the tail can complete into, one dustbin for padded writes
            pooled_rows  = kv_size/ratio + 2;
            pooled_ratio = ratio;

            ggml_init_params params = {
                /*.mem_size   =*/ layers.size()*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            pooled_ctx.reset(ggml_init(params));

            for (const int32_t il : layers) {
                ggml_tensor * t = ggml_new_tensor_2d(pooled_ctx.get(), GGML_TYPE_F32, idx_dim, pooled_rows);
                ggml_format_name(t, "pooled_k_l%d", il);
                pooled_k[il] = t;
            }

            // same placement as the indexer keys the summaries are pooled from
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(mem_idx->get_k_storage(layers[0])->buffer);

            pooled_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(pooled_ctx.get(), buft));
            if (pooled_buf) {
                // unwritten rows must stay finite: their scores are masked, but only against a finite value
                ggml_backend_buffer_clear(pooled_buf.get(), 0);

                LLAMA_LOG_INFO("%s: QSA pooled key cache, %u rows x %zu layers, %.2f MiB\n", __func__,
                        pooled_rows, layers.size(), ggml_backend_buffer_get_size(pooled_buf.get())/1024.0/1024.0);
            } else {
                LLAMA_LOG_WARN("%s: failed to allocate the QSA pooled key cache, scores will be recomputed\n", __func__);
                pooled_k.clear();
                pooled_ctx.reset();
                pooled_rows = 0;
            }
        }
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    pooled_reset(-1);
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    pooled_rm(seq_id, p0, p1);

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    // the copy shares cells, but any rows the destination goes on to score must be its own
    pooled_reset(seq_id_dst);
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }

    if (pooled_owner == seq_id) {
        // the kept sequence's rows stay valid; forget everyone else's watermark
        const int64_t w = pooled_w[seq_id];
        pooled_w.clear();
        pooled_w[seq_id] = w;
    } else {
        pooled_reset(-1);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    // a shift moves tokens between blocks, which no clamp can express
    pooled_reset(seq_id);
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    pooled_reset(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }

            // restored cells land wherever the caches found room, so every summary is suspect
            // a PARTIAL_ONLY restore leaves the cells alone; the matching rollback arrives via seq_rm
            pooled_reset(seq_id);
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }

    pooled_reset(seq_id);
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

// [TAG_QSA_POOLED_CACHE]

ggml_tensor * llama_memory_hybrid_idx::get_pooled_k(int32_t il) const {
    const auto it = pooled_k.find(il);

    return it != pooled_k.end() ? it->second : nullptr;
}

int64_t & llama_memory_hybrid_idx::pooled_valid(llama_seq_id seq_id) const {
    if (pooled_owner != seq_id) {
        // another sequence's summaries occupy the rows; this one recomputes from the start
        pooled_owner = seq_id;
        pooled_w[seq_id] = 0;
    }

    return pooled_w[seq_id];
}

void llama_memory_hybrid_idx::pooled_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (pooled_k.empty()) {
        return;
    }

    if (seq_id < 0) {
        pooled_reset(-1);
        return;
    }

    // every block from p0 up is suspect; a middle removal also leaves holes above p1, which the
    // completeness bias masks on its own (their cells are gone, so the blocks can never refill)
    const int64_t blk = std::max<llama_pos>(p0, 0)/pooled_ratio;

    const auto it = pooled_w.find(seq_id);
    if (it != pooled_w.end()) {
        it->second = std::min(it->second, blk);
    }

    GGML_UNUSED(p1);
}

void llama_memory_hybrid_idx::pooled_reset(llama_seq_id seq_id) {
    if (seq_id < 0) {
        pooled_w.clear();
        pooled_owner = -1;
        return;
    }

    pooled_w[seq_id] = 0;
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

// [TAG_QSA_POOLED_CACHE]

ggml_tensor * llama_memory_hybrid_idx_context::get_pooled_k(int32_t il) const {
    return mem ? mem->get_pooled_k(il) : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::get_pooled_rows() const {
    return mem ? mem->get_pooled_rows() : 0;
}

int64_t llama_memory_hybrid_idx_context::qsa_pooled_n_dirty_max(const llama_ubatch * ubatch, uint32_t ratio) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr);

    const int64_t r = ratio;

    // the reserve pass probes with a mock ubatch that carries no sequence data;
    // size for a fill that completes a block with every r tokens
    if (ubatch->seq_id == nullptr || ubatch->seq_id[0] == nullptr) {
        return std::max<int64_t>(2, ((int64_t) ubatch->n_tokens + r - 1)/r + 1);
    }

    const llama_seq_id seq = ubatch->seq_id[0][0];

    llama_pos p_max = -1;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        p_max = std::max(p_max, ubatch->pos[i]);
    }

    const int64_t n_complete = (int64_t) (p_max + 1)/r;
    const int64_t w          = mem->pooled_valid(seq);

    // floor of 2: steady decode alternates between 0 and 1 fresh blocks, and a size that
    // flipped with it would break graph reuse on every ratio-th token
    return std::max<int64_t>(2, n_complete - w + 1);
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_cells_dup,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        bool blk_topk,
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_rows,
        ggml_tensor * blk_slot_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(!blk_topk || (blk_bias && blk_cells_dup != nullptr));
    GGML_ASSERT(blk_slot_bias == nullptr || blk_topk);

    // [TAG_QSA_POOLED_CACHE] all three dirty tensors travel together
    GGML_ASSERT((dirty_cells != nullptr) == (dirty_rows != nullptr));
    GGML_ASSERT((dirty_pos   != nullptr) == (dirty_rows != nullptr));

    // the blk_topk graph never consumes cell_blk, so it is not created there (an unconsumed
    // input is never allocated and cannot be written)
    GGML_ASSERT(blk_topk == (cell_blk == nullptr));

    // [TAG_QSA_POOLED_CACHE] the pooled graph consumes neither gather table, so neither exists
    // there; the cell mapping below is still built, into a local buffer, for the other fills
    GGML_ASSERT(blk_cells != nullptr || dirty_rows != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(blk_cells ? blk_cells->buffer : dirty_rows->buffer));

    const int64_t n_kv     = cell_blk ? cell_blk->ne[0] : get_idx()->get_n_kv();
    const int64_t n_ns     = blk_cells ? blk_cells->ne[1] : (cell_blk ? cell_blk->ne[1] : 1);   // streams in this ubatch
    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(blk_pos == nullptr || blk_pos->ne[0] == 4*n_blocks*n_ns);

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = cell_blk  ? (int32_t *) cell_blk->data  : nullptr;
    int32_t * dst_blk_cells = blk_cells ? (int32_t *) blk_cells->data : nullptr;
    int32_t * dst_blk_pos   = blk_pos   ? (int32_t *) blk_pos->data   : nullptr;
    float   * dst_bias      = (float   *) bias->data;

    // block b covers [b*ratio, (b+1)*ratio), so its first token is at b*ratio
    // all mrope sections carry it: exact for text, approximate for images
    if (dst_blk_pos != nullptr) {
        for (int64_t sec = 0; sec < 4; ++sec) {
            for (int64_t s = 0; s < n_ns; ++s) {
                for (int64_t b = 0; b < n_blocks; ++b) {
                    dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = (int32_t) (b*r);
                }
            }
        }
    }

    // one pass per stream: cell j is a different token in each, so no mapping is shared
    std::vector<int32_t> blk_of(n_kv);
    std::vector<int32_t> filled(n_blocks);
    std::vector<uint8_t> slot_set;
    std::vector<int32_t> first_cell;
    std::vector<int32_t> loc_blk_cells;
    if (blk_topk) {
        slot_set.resize(r*n_blocks);
        first_cell.resize(n_blocks);
    }
    if (dst_blk_cells == nullptr) {
        loc_blk_cells.resize(r*n_blocks);
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  ? dst_cell_blk  + s*n_kv          : nullptr;
        int32_t * cur_blk_cells = dst_blk_cells ? dst_blk_cells + s*(r*n_blocks)  : loc_blk_cells.data();
        int32_t * cur_dup       = blk_topk ? (int32_t *) blk_cells_dup->data + s*(r*n_blocks) : nullptr;
        float   * cur_slot_bias = blk_slot_bias ? (float *) blk_slot_bias->data + s*(r*n_blocks) : nullptr;

        // an incomplete block cannot be pooled; the bias below forces those tail cells in
        // -1 means no usable block, and block 0 only keeps the gather in range
        std::fill(blk_of.begin(),  blk_of.end(),  -1);
        std::fill(filled.begin(),  filled.end(),   0);
        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);
        if (blk_topk) {
            std::fill(slot_set.begin(),   slot_set.end(),    0);
            std::fill(first_cell.begin(), first_cell.end(), -1);
        }

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const llama_pos p = cells.pos_get(j);
            const int64_t   b = p/r;

            if (b >= n_blocks) {
                oor = true;
                continue;
            }

            blk_of[j] = (int32_t) b;
            cur_blk_cells[b*r + (p%r)] = (int32_t) j;
            filled[b]++;

            if (blk_topk) {
                slot_set[b*r + (p%r)] = 1;
                if (first_cell[b] < 0) {
                    first_cell[b] = (int32_t) j;
                }
            }
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        if (blk_topk) {
            // unfilled slots repeat a filled cell of the same block, so a block->cells gather stays
            // in range and its extra set_rows writes land on an already-selected row (idempotent).
            // a fully empty block only gets selected in the underfull regime, where every visible
            // cell is attended anyway, so falling back to cell 0 there is harmless
            for (int64_t b = 0; b < n_blocks; ++b) {
                for (int64_t t = 0; t < r; ++t) {
                    cur_dup[b*r + t] = slot_set[b*r + t] ? cur_blk_cells[b*r + t]
                                     : (first_cell[b] >= 0 ? first_cell[b] : 0);
                    if (cur_slot_bias != nullptr) {
                        cur_slot_bias[b*r + t] = slot_set[b*r + t] ? 0.0f : -INFINITY;
                    }
                }
            }
        }

        // [TAG_QSA_POOLED_CACHE] list the blocks this ubatch completes, so the graph writes their
        // summaries into the pooled store; every earlier row then survives to the next graph
        if (dirty_rows != nullptr) {
            // the store holds one sequence's rows of one stream; the graph only asks under that shape
            GGML_ASSERT(n_ns == 1);

            const int64_t cap     = dirty_rows->ne[0];
            const int64_t dustbin = (int64_t) mem->get_pooled_rows() - 1;

            int64_t * dst_dirty_rows  = (int64_t *) dirty_rows->data;
            int32_t * dst_dirty_cells = (int32_t *) dirty_cells->data;
            int32_t * dst_dirty_pos   = (int32_t *) dirty_pos->data;

            // same last position the sizing pass used, so the fill provably fits
            llama_pos p_max = -1;
            for (int64_t i = 0; i < n_tokens; ++i) {
                p_max = std::max(p_max, ubatch->pos[i]);
            }

            const int64_t n_complete = (int64_t) (p_max + 1)/r;

            int64_t & w = mem->pooled_valid(seq_of_stream);

            int64_t d = 0;
            for (int64_t b = w; b < n_complete; ++b) {
                if (b >= n_blocks || filled[b] < r) {
                    // a hole from a middle removal; its cells are gone, the bias masks it forever
                    continue;
                }

                GGML_ASSERT(d < cap && "qsa: pooled dirty list outgrew its graph tensors");

                dst_dirty_rows[d] = b;
                for (int64_t t = 0; t < r; ++t) {
                    dst_dirty_cells[d*r + t] = cur_blk_cells[b*r + t];
                }
                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_dirty_pos[sec*cap + d] = (int32_t) (b*r);
                }

                ++d;
            }

            // spare slots gather cell 0 and write it to the dustbin row, which nothing reads
            for (; d < cap; ++d) {
                dst_dirty_rows[d] = dustbin;
                for (int64_t t = 0; t < r; ++t) {
                    dst_dirty_cells[d*r + t] = 0;
                }
                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_dirty_pos[sec*cap + d] = 0;
                }
            }

            w = std::max(w, n_complete);
        }

        // per-block mode keeps an unpooled cell's real block, so the block's own -inf reaches it
        // per-cell mode carries that -inf itself and only needs the gather in range
        if (cur_cell_blk != nullptr) {
            for (int64_t j = 0; j < n_kv; ++j) {
                if (blk_of[j] >= 0 && filled[blk_of[j]] < r && !blk_bias) {
                    blk_of[j] = -1;
                }
                cur_cell_blk[j] = blk_of[j] < 0 ? 0 : blk_of[j];
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            const llama_pos    q      = ubatch->pos[i];

            // the tail is an incomplete block and is always visible, as in the reference
            const llama_pos tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                if (blk_topk) {
                    // a block-level top-k selects on this bias alone -- no per-cell mask add cleans
                    // up afterwards -- so the bias must carry visibility itself: force the one tail
                    // block (it contains q), and drop strictly-future and incomplete blocks
                    for (int64_t b = 0; b < n_blocks; ++b) {
                        // finite, so it can never meet a -inf and produce a nan
                        cur_blk_bias[b] = b*r >= tail_start ? (b*r <= q ? 1e9f : -INFINITY)
                                        : (filled[b] < r ? -INFINITY : 0.0f);
                    }
                } else {
                    for (int64_t b = 0; b < n_blocks; ++b) {
                        // finite, so it can never meet a -inf and produce a nan
                        cur_blk_bias[b] = b*r >= tail_start ? 1e9f : (filled[b] < r ? -INFINITY : 0.0f);
                    }
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id) && cells.pos_get(j) <= q) {
                    // finite, so it can never meet a -inf and produce a nan
                    v = cells.pos_get(j) >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                }

                cur_bias[j] = v;
            }
        }
    }
}
