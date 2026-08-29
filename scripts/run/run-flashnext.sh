#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on the REBASED flashnext-v2 build (build-v2). MTP self-speculative decode
# is OPT-IN because it ONLY helps at TEMP 0 (greedy). MEASURED on this box (build-v2, single stream):
#   temp 1.0 (this script's default creative sampling): MTP = WASH (25.7 vs 25.8) -> leave it OFF.
#   temp 0 + code:            base 25.7 -> MTP 35.6  (+39%), and the gain GROWS with depth:
#   temp 0 + code @ ~32k:     16.1 -> 28.9 (+79%);   @ ~64k: 12.0 -> 25.8 (+115%).
# Rule: enable MTP only for DETERMINISTIC work (code / tools / extraction) run at TEMP=0.
# Q4_K_XL is the only quant we keep. Creative sampling = unsloth thinking-mode rec.
# Usage: [QUANT=Q4_K_XL] [CTX=131072] [NGL=99] [PORT=8080] [FA=on] [NGRAM_OFFLOAD=1] [REASONING=medium] [MTP=0] [TEMP=1.0] [PARALLEL=1] [UB=2048]
#   Creative/chat (default):     ./run-flashnext.sh
#   Fast deterministic code:     MTP=1 TEMP=0 REASONING=none ./run-flashnext.sh
export PATH="$PATH:/usr/sbin:/sbin"
TOOLBOX=llama-nudge-vulkan
BIN=/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin   # rebased build WITH MTP (old build/ has NO MTP)
MODELS=/home/wdenejko/models/Qwen3.8-Flash-Next-GGUF
MD=/home/wdenejko/models/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF/Qwen3.8-Flash-Next-MTP-Q8_0.gguf  # MTP draft head

QUANT="${QUANT:-Q4_K_XL}"
PORT="${PORT:-8080}"
NGL="${NGL:-99}"
CTX="${CTX:-131072}"                 # model supports up to 262144; MTP adds ~5GB, lower CTX if it OOMs at 128k
FA="${FA:-on}"
NGRAM_OFFLOAD="${NGRAM_OFFLOAD:-1}"  # 1 = keep the 28.8GB PLE table off GTT as an NVMe mmap (needed for Q4 + big ctx)
REASONING="${REASONING:-medium}"     # xhigh|medium|low = think ON at that effort; none/off = think OFF.
                                     # NB: this template rejects effort 'none' — none maps to --reasoning-budget 0
                                     # (enable_thinking=false -> empty <think></think>), NOT --reasoning-effort none.
TEMP="${TEMP:-1.0}"                  # 1.0 = creative default; set 0 for deterministic work (REQUIRED for MTP to help)
MTP="${MTP:-0}"                      # 0 = plain base decode (right for temp 1.0); 1 = MTP (only useful at TEMP=0)
PARALLEL="${PARALLEL:-1}"            # 1 = single stream = fastest PER REQUEST; higher = concurrency but dilutes each
UB="${UB:-2048}"                     # physical batch. MEASURED pp4096 on this box: 512->2048 is +25% coopmat-OFF
                                     # (386->484) and +14% coopmat-ON (494->564); plateaus by 3072. ub2048 verified
                                     # NON-OOM at full 131072 ctx (peak 87GB / 121GB free). Was 512.

case "$QUANT" in
  Q4_K_XL) MODEL="$MODELS/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" ;;
  *)       MODEL="$MODELS/$QUANT" ;;
esac
if [ ! -f "$MODEL" ]; then
  echo "[run-flashnext] quant/model not found: $MODEL" >&2
  echo "[run-flashnext] only UD-Q4_K_XL is installed; see run-flashnext-q4.sh" >&2
  exit 1
fi

OT_ARGS=()
[ "$NGRAM_OFFLOAD" = "1" ] && OT_ARGS=(--override-tensor "per_layer_token_embd.weight=CPU")
# reasoning control. This template has NO 'none' effort level — effort is only xhigh|medium|low and it
# raise_exception()s on anything else. Thinking is turned OFF by enable_thinking=false, which the server
# flag `--reasoning off` sets (verified: yields an empty <think></think> and pure output, no error).
# So 'none'/'off' -> `--reasoning off`; any other value -> a real effort level.
REASON_ARGS=()
case "$REASONING" in
  none|off|0) REASON_ARGS=(--reasoning off) ;;
  *)          REASON_ARGS=(--reasoning on --reasoning-effort "$REASONING") ;;
esac

MTP_ARGS=()
VK_ENV=()
if [ "$MTP" = "1" ]; then
  # draft-mtp ONLY. Do NOT add ngram-mod (its 48-64 tok drafts only make the bug below worse).
  # p-min 0.75 gates low-confidence drafts (keeps prose from regressing).
  MTP_ARGS=(-md "$MD" --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75)
  # REQUIRED with MTP: without this the GPU DEADLOCKS mid-run. Root-caused 2026-08-28: a KHR_coopmat
  # matmul shader hangs (Vulkan fence never signals; llama_decode spins forever in ggml_vk_wait_for_fence;
  # no device-lost) on the small speculative verify/draft batch shapes. By elimination f16b-off,
  # GGML_VK_SERIALIZE_SUBMISSIONS=1 and --flash-attn off ALL still hang; only disabling coopmat fixes it
  # (repro 30/30 clean). Decode stays full-speed (33-36 t/s, accept 0.90); only cold-prefill matmul is slower.
  VK_ENV=(GGML_VK_DISABLE_COOPMAT=1)
  if [ "$TEMP" != "0" ] && [ "$TEMP" != "0.0" ]; then
    echo "[run-flashnext] WARNING: MTP=1 with TEMP=$TEMP is pointless — MTP is a WASH above temp 0 (measured 25.7 vs 25.8)." >&2
    echo "[run-flashnext]          It only speeds up greedy decode. Set TEMP=0, or use MTP=0 to save ~4GB on the draft model." >&2
  fi
fi

echo "[run-flashnext] build=build-v2 quant=$QUANT ctx=$CTX ngl=$NGL fa=$FA port=$PORT ngram_offload=$NGRAM_OFFLOAD reasoning=$REASONING temp=$TEMP mtp=$MTP parallel=$PARALLEL" >&2
podman start "$TOOLBOX" >/dev/null 2>&1 || true
exec toolbox run --container "$TOOLBOX" env "${VK_ENV[@]}" LD_LIBRARY_PATH="$BIN" \
  "$BIN/llama-server" -m "$MODEL" \
    --alias flashnext --host 0.0.0.0 --port "$PORT" \
    -ngl "$NGL" -c "$CTX" -ub "$UB" --flash-attn "$FA" "${OT_ARGS[@]}" --metrics \
    --parallel "$PARALLEL" \
    --temp "$TEMP" --top-p 0.95 --top-k 20 --min-p 0.0 \
    --presence-penalty 0.0 --repeat-penalty 1.0 \
    "${REASON_ARGS[@]}" \
    "${MTP_ARGS[@]}"
