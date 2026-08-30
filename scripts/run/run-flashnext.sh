#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on the REBASED flashnext-v2 build (build-v2). MTP self-speculative decode
# is OPT-IN because it ONLY helps at TEMP 0 (greedy). MEASURED on this box (build-v2, single stream):
#   temp 1.0 (this script's default creative sampling): MTP = WASH (25.7 vs 25.8) -> leave it OFF.
#   temp 0 + code:            base 25.7 -> MTP 35.6  (+39%), and the gain GROWS with depth:
#   temp 0 + code @ ~32k:     16.1 -> 28.9 (+79%);   @ ~64k: 12.0 -> 25.8 (+115%).
# Rule: enable MTP only for DETERMINISTIC work (code / tools / extraction) run at TEMP=0.
# Q4_K_XL is the only quant we keep. Creative sampling = unsloth thinking-mode rec.
# Usage: [QUANT=Q4_K_XL] [CTX=131072] [NGL=99] [PORT=8080] [FA=on] [NGRAM_OFFLOAD=1] [REASONING=medium] [MTP=0] [TEMP=1.0] [PARALLEL=1] [UB=2048] [COOPMAT=1] [FREE_GPU=1] [UBD=512] [KVD=q8_0] [MD=<draft.gguf>]
#   Creative/chat (default):     ./run-flashnext.sh
#   Fast deterministic code:     MTP=1 TEMP=0 REASONING=none ./run-flashnext.sh
export PATH="$PATH:/usr/sbin:/sbin"
TOOLBOX=llama-nudge-vulkan
BIN=/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin   # rebased build WITH MTP (old build/ has NO MTP)
MODELS=/home/wdenejko/models/Qwen3.8-Flash-Next-GGUF
MD="${MD:-/home/wdenejko/models/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF/Qwen3.8-Flash-Next-MTP-Q8_0.gguf}"  # MTP draft head (env-overridable)

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
COOPMAT="${COOPMAT:-1}"              # KHR_coopmat matmul. 1 = ON (default): recovers prefill (measured ub2048:
                                     # +16% at d0, +11% @32k — shrinks with depth), safe under MTP since 81aa39c17.
                                     # 0 = force OFF (GGML_VK_DISABLE_COOPMAT=1) as a regression fallback.
FREE_GPU="${FREE_GPU:-1}"            # 1 = stop comfyui/OCR + wait for GTT/RAM headroom before the ~103GB
                                     # load, and restart them when the server exits. 0 = leave services be.

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
# COOPMAT=0 is a global kill switch forcing KHR_coopmat OFF (regression fallback). Default is ON.
[ "$COOPMAT" = "0" ] && VK_ENV=(GGML_VK_DISABLE_COOPMAT=1)
# pass experiment env vars into the toolbox (toolbox run does not forward the host env)
for _v in GGML_TOPK_LOG GGML_VK_EVENT_DEVICE_WAIT GGML_VK_EVENT_TL_OFF; do
  if [ -n "${!_v:-}" ]; then VK_ENV+=("$_v=${!_v}"); fi
done
# Q4X_RS_ROLLBACK: enable qwen4exp recurrent partial rollback (n_rs_seq=4). Without it every
# speculative rejection restores a ~112 MiB checkpoint — and when coopmat's restore+redecode is
# not bit-reproducible that restore loops forever (the historical "fence deadlock"/decode stall).
# Validated 2026-08-29: 10/10 harness tasks, 0 restores, stall task 62s->7s. RSROLL=0 disables
# (the server-side non-convergence guard then covers the livelock, at checkpoint-churn cost).
[ "${RSROLL:-1}" != "0" ] && VK_ENV+=(Q4X_RS_ROLLBACK=1)
# Q4X_QSA_BLK_TOPK: block-level QSA indexer top-k. A/B 2026-08-30 (symmetric, 98k ctx): pp +25%
# and tg +11% at 32-64k depth, pp +5% shallow, ~2GB less GTT at full ctx, needle retrieval
# intact at 32k/64k, harness 10/10 — and the k=2051 CPU top_k fallback disappears (~14k
# dispatches/run -> 0). Costs ~3pt draft acceptance (net tg still ahead). BLKTOPK=0 disables.
[ "${BLKTOPK:-1}" != "0" ] && VK_ENV+=(Q4X_QSA_BLK_TOPK=1)
if [ "$MTP" = "1" ]; then
  # draft-mtp ONLY. Do NOT add ngram-mod (its 48-64 tok drafts regress acceptance here).
  # p-min 0.75 gates low-confidence drafts (keeps prose from regressing).
  UBD="${UBD:-512}"   # draft-context physical batch. The draft otherwise inherits UB and reserves a
                      # SECOND ub x n_ctx mask/compute set; 512 shrinks it 4x so the TARGET keeps
                      # UB=2048 prefill (+11-13% pp vs ub512 at 0-64k). Decode-side cost: none.
  KVD="${KVD:-q8_0}"  # draft KV cache type (1 attn layer; q8_0 halves it, ~120MB at 128k)
  # VALIDATED ENVELOPE (2026-08-29): with the default Q8 draft, ub2048+MTP prefill is proven to ~96k
  # (dies ~129k on a HOST oom-kill: 103GB model + drafts + deep transients fill the 128GB box).
  # For genuinely deeper jobs pick ONE: MD=<...MTP-Q4_K_M.gguf> (frees 1.35GB, full 128k proven, but
  # shallow decode drops 44->32 t/s at 74% accept, and the box peaks at free=0) or UB=512 (pp at 128k
  # is within 6.5% of ub2048 anyway: 141.5 vs 150.7).
  MTP_ARGS=(-md "$MD" --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75
            --spec-draft-ubatch "$UBD" --spec-draft-type-k "$KVD" --spec-draft-type-v "$KVD")
  # HISTORY: coopmat under MTP used to DEADLOCK the GPU — a KHR_coopmat matmul shader's Vulkan fence never
  # signalled on the small speculative batch shapes (llama_decode spun in ggml_vk_wait_for_fence, no
  # device-lost). FIXED 2026-08-29 by ordering the cross-backend event wait through its timeline semaphore
  # (commit 81aa39c17). VALIDATED: coopmat-ON MTP ran 20 min / 84 code gens with 0 deadlocks, 0 livelocks,
  # 0 stalls at full decode speed (32-37 t/s). The old OFF workaround is now opt-in via COOPMAT=0. The
  # coopmat-ON graph init is GTT-sensitive under memory pressure -> see the launch preamble below.
  if [ "$TEMP" != "0" ] && [ "$TEMP" != "0.0" ]; then
    echo "[run-flashnext] WARNING: MTP=1 with TEMP=$TEMP is pointless — MTP is a WASH above temp 0 (measured 25.7 vs 25.8)." >&2
    echo "[run-flashnext]          It only speeds up greedy decode. Set TEMP=0, or use MTP=0 to save ~4GB on the draft model." >&2
  fi
fi

echo "[run-flashnext] build=build-v2 quant=$QUANT ctx=$CTX ngl=$NGL fa=$FA port=$PORT ngram_offload=$NGRAM_OFFLOAD reasoning=$REASONING temp=$TEMP mtp=$MTP coopmat=$COOPMAT free_gpu=$FREE_GPU parallel=$PARALLEL" >&2

# Don't stack a second server on the coopmat-ON MTP path: its graph init can wedge graph_reserve into an
# unkillable amdgpu D-state (needs a GPU reset/reboot) when a prior server's GTT is still held.
if [ "$MTP" = "1" ] && [ "$COOPMAT" != "0" ] && pgrep -f "build-v2/bin/[l]lama-server" >/dev/null 2>&1; then
  echo "[run-flashnext] a build-v2 llama-server is already running — stop it before launching coopmat-ON MTP." >&2
  exit 1
fi

# Clean shutdown: kill the server and (if we stopped them) restart the user's services. The server runs
# inside the toolbox via `toolbox run`; a signal to this script kills the toolbox-run client but can ORPHAN
# the containerized llama-server (podman only forwards the signal with a TTY), leaving it holding ~85GB GTT
# while the services come back = contention. So pkill it explicitly (the toolbox shares the host PID ns).
# Fires on normal exit and on Ctrl-C/TERM; defined AFTER the refuse-check above so it never touches a
# pre-existing server.
_FREED=0
_shutdown() {
  pkill -f "build-v2/bin/[l]lama-server" 2>/dev/null || true
  if [ "$_FREED" = "1" ]; then
    sleep 2
    echo "[run-flashnext] restarting comfyui + OCR" >&2
    systemctl --user start dashi-unlimited-ocr.service comfyui.service 2>/dev/null || true
  fi
}
trap _shutdown EXIT INT TERM

# Free GPU/host memory for the ~103GB model: stop comfyui/OCR and wait for headroom (restored on exit above).
if [ "$FREE_GPU" = "1" ]; then
  echo "[run-flashnext] stopping comfyui + OCR to free memory (FREE_GPU=0 to skip)" >&2
  systemctl --user stop dashi-unlimited-ocr.service comfyui.service 2>/dev/null || true
  _FREED=1
  _gttf=$(ls /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)
  g=0; a=0
  for _ in $(seq 1 40); do            # up to ~80s for GTT to drain and RAM headroom for the load
    g=$(( $(cat "$_gttf" 2>/dev/null)/1073741824 )); a=$(free -g | awk '/^Mem:/{print $7}')
    [ "$g" -lt 6 ] && [ "$a" -ge 110 ] && break
    sleep 2
  done
  echo "[run-flashnext] memory ready: gtt=${g}GB free_ram=${a}GB" >&2
elif [ "$MTP" = "1" ] && [ "$COOPMAT" != "0" ]; then
  # FREE_GPU off but coopmat-ON: still let a just-exited server's GTT finish draining before graph init.
  _gtt() { echo $(( $(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1) / 1073741824 )); }
  _prev=$(_gtt); _stable=0
  for _ in $(seq 1 15); do
    sleep 2; _cur=$(_gtt)
    if [ "$_cur" -le "$_prev" ]; then _stable=$((_stable+1)); else _stable=0; fi
    [ "$_stable" -ge 2 ] && break
    _prev=$_cur
  done
  echo "[run-flashnext] gtt=$(_gtt)GB stable" >&2
fi

# Start the toolbox — restart it fresh on the coopmat-ON MTP path so graph init sees a clean container.
if [ "$MTP" = "1" ] && [ "$COOPMAT" != "0" ]; then
  podman restart "$TOOLBOX" >/dev/null 2>&1 || podman start "$TOOLBOX" >/dev/null 2>&1 || true
  sleep 2
else
  podman start "$TOOLBOX" >/dev/null 2>&1 || true
fi

# NOTE: intentionally NOT `exec` — the shell must stay alive so the FREE_GPU trap restarts comfyui/OCR
# when the server exits or is interrupted.
VERBOSE_ARGS=()
[ "${VERBOSE:-0}" = "1" ] && VERBOSE_ARGS=(--verbose)
toolbox run --container "$TOOLBOX" env "${VK_ENV[@]}" LD_LIBRARY_PATH="$BIN" \
  "$BIN/llama-server" -m "$MODEL" "${VERBOSE_ARGS[@]}" \
    --alias flashnext --host 0.0.0.0 --port "$PORT" \
    -ngl "$NGL" -c "$CTX" -ub "$UB" --flash-attn "$FA" "${OT_ARGS[@]}" --metrics \
    --parallel "$PARALLEL" \
    --temp "$TEMP" --top-p 0.95 --top-k 20 --min-p 0.0 \
    --presence-penalty 0.0 --repeat-penalty 1.0 \
    "${REASON_ARGS[@]}" \
    "${MTP_ARGS[@]}"
