#!/usr/bin/env bash
# run-flashnext.sh — serve Qwen3.8-Flash-Next on Strix Halo (Vulkan/RADV).
#
#   chat:          MTP=1 ./run-flashnext.sh              (full 262k context by default;
#   code (fast):   MTP=1 TEMP=0 REASONING=none ./run-flashnext.sh    the guard auto-tiers)
#   smaller ctx:   CTX=131072 ./run-flashnext.sh         (skips the deep-ctx knobs)
#   image build:   BIN=/usr/local/bin ./run-flashnext.sh (serve the GHCR image's binaries)
#
# Every default below is measured; the rationale lives in the published release notes
# (https://wdenejko.github.io/llama.cpp/) and the git history — not here.

export PATH="$PATH:/usr/sbin:/sbin"

# --- environment ----------------------------------------------------------------------
TOOLBOX="${TOOLBOX:-llama-vulkan-wdenejko}"   # serving container (from the GHCR image)
BIN="${BIN:-/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin}"
SRV_PAT="${BIN}/[l]lama-server"               # kill/refuse patterns track BIN
MODELS=/home/wdenejko/models/Qwen3.8-Flash-Next-GGUF

# the deep-ctx guard only auto-picks knobs the user left alone
MD_USER="${MD:+1}"
UB_USER="${UB:+1}"
KV_USER="${KV:+1}"
MD="${MD:-/home/wdenejko/models/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF/Qwen3.8-Flash-Next-MTP-Q8_0.gguf}"
MDQ4=/home/wdenejko/models/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF/Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf

# --- knobs ----------------------------------------------------------------------------
QUANT="${QUANT:-Q4_K_XL}"            # Q4_K_XL (base) | IQ4_XS (orcarouter Uncensored finetune)
PORT="${PORT:-8080}"
NGL="${NGL:-99}"
CTX="${CTX:-262144}"                 # full model context; the deep-ctx guard below keeps it safe
FA="${FA:-on}"
NGRAM_OFFLOAD="${NGRAM_OFFLOAD:-1}"  # 1 = keep the 28.8GB PLE table on CPU (needed at big ctx)
REASONING="${REASONING:-medium}"     # xhigh|medium|low = think at that effort; none/off = think OFF
TEMP="${TEMP:-1.0}"                  # 0 = deterministic (fastest for code)
MTP="${MTP:-0}"                      # 1 = self-speculative decode (worth it at any temperature)

# Deep-context guard (DEEPMTP=0 disables all three). Thresholds are measured envelopes:
# Q8 draft host-OOMs past ~129k; KV=q8_0 frees ~3.9GB (a <1% wash); ub2048 is OOM-killed
# at ~192k of ACTUAL fill, ub1024 completes a full 252k fill.
if [ "${DEEPMTP:-1}" != "0" ]; then
  if [ "$MTP" = "1" ] && [ -z "$MD_USER" ] && [ "$CTX" -gt 98304 ] && [ -f "$MDQ4" ]; then
    MD="$MDQ4"
    echo "[run-flashnext] deep-ctx: CTX=$CTX > 98304 with MTP — switching to the Q4 draft (MD=<path> or DEEPMTP=0 overrides)" >&2
  fi
  if [ -z "$KV_USER" ] && [ "$CTX" -gt 131072 ]; then
    KV=q8_0
    echo "[run-flashnext] deep-ctx: CTX=$CTX > 131072 — KV=q8_0 frees ~3.9GB of host RAM (KV=f16 or DEEPMTP=0 overrides)" >&2
  fi
  if [ -z "$UB_USER" ] && [ "$CTX" -gt 196608 ]; then
    UB=1024
    echo "[run-flashnext] deep-ctx: CTX=$CTX > 196608 — UB=1024 so a full-depth fill survives (ub2048 host-OOMs at ~192k fill; UB=2048 to force, DEEPMTP=0 disables)" >&2
  fi
fi
PARALLEL="${PARALLEL:-1}"            # 1 = fastest per request
UB="${UB:-2048}"                     # physical batch (512->2048 measured +14-25% pp)
COOPMAT="${COOPMAT:-1}"              # 0 = force KHR_coopmat OFF (regression fallback)
FREE_GPU="${FREE_GPU:-1}"            # 1 = stop comfyui/OCR for the load, restart on exit

# --- model ----------------------------------------------------------------------------
case "$QUANT" in
  Q4_K_XL) MODEL="$MODELS/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" ;;
  IQ4_XS)  MODEL="$HOME/models/Qwen3.8-Flash-Next-Uncensored-GGUF/IQ4_XS/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf" ;;
  *)       MODEL="$MODELS/$QUANT" ;;
esac
if [ ! -f "$MODEL" ]; then
  echo "[run-flashnext] quant/model not found: $MODEL" >&2
  echo "[run-flashnext] only UD-Q4_K_XL is installed; see run-flashnext-q4.sh" >&2
  exit 1
fi

# --- server args ----------------------------------------------------------------------
OT_ARGS=()
[ "$NGRAM_OFFLOAD" = "1" ] && OT_ARGS=(--override-tensor "per_layer_token_embd.weight=CPU")

# the Uncensored template only accepts xhigh|medium|low and rejects 'high'
if [ "$QUANT" = "IQ4_XS" ] && [ "$REASONING" = "high" ]; then
  REASONING=xhigh
  echo "[run-flashnext] IQ4_XS template: reasoning high -> xhigh (this template's levels: xhigh|medium|low)" >&2
fi
REASON_ARGS=()
case "$REASONING" in
  none|off|0) REASON_ARGS=(--reasoning off) ;;
  *)          REASON_ARGS=(--reasoning on --reasoning-effort "$REASONING") ;;
esac

# the Uncensored template rejects mid-conversation system messages (agent harnesses like
# opencode send them); serve a relaxed copy that renders them in place. TPLFIX=0 reverts.
TPL="${TPL:-/home/wdenejko/src/llama-qwen4exp-src/scripts/run/uncensored-chat-template.jinja}"
TPL_ARGS=()
[ "$QUANT" = "IQ4_XS" ] && [ "${TPLFIX:-1}" != "0" ] && [ -f "$TPL" ] && TPL_ARGS=(--chat-template-file "$TPL")

MTP_ARGS=()
VK_ENV=()
[ "$COOPMAT" = "0" ] && VK_ENV=(GGML_VK_DISABLE_COOPMAT=1)
# forward experiment env into the toolbox (toolbox run does not pass the host env)
for _v in GGML_TOPK_LOG GGML_VK_EVENT_DEVICE_WAIT GGML_VK_EVENT_TL_OFF GGML_VK_DISABLE_FUSION GGML_VK_PERF_LOGGER GGML_VK_PERF_LOGGER_FREQUENCY GGML_VK_DENSE_F16B GGML_VK_DENSE_WAVE32 RADV_PERFTEST \
             GGML_VK_MMID_INT GGML_VK_MMID_F16B GGML_VK_MMID_SCALE_EPILOGUE GGML_VK_MMID_PROBE GGML_VK_MMID_COMPACT GGML_VK_DENSE_THIN_BM GGML_VK_MMID_BN48 GGML_VK_MMID_BK64 GGML_VK_MM_TILE_EPILOG GGML_VK_DENSE_BM256 GGML_VK_MMV_RM_STDQ GGML_VK_MMV_RM_KQ Q4X_SPEC_STOCH_TOPK Q4X_SPEC_STOCH_TEMP Q4X_HC_MIX_NOFUSE GGML_VK_MMID_SMALLN GGML_VK_MMID_BM64 GGML_VK_MEMORY_LOGGER GGML_VK_JOB_BUDGET_MS GGML_VK_JOB_LOG GGML_VK_JOB_THROTTLE GGML_VK_SERIALIZE_SUBMISSIONS GGML_VK_SUBMIT_TRACE GGML_VK_MAX_NODES_PER_SUBMIT GGML_VK_MAX_MB_PER_SUBMIT; do
  if [ -n "${!_v:-}" ]; then VK_ENV+=("$_v=${!_v}"); fi
done
[ "${RSROLL:-1}" != "0" ]  && VK_ENV+=(Q4X_RS_ROLLBACK=1)     # partial recurrent rollback (+19% tg @temp1)
[ "${BLKTOPK:-1}" != "0" ] && VK_ENV+=(Q4X_QSA_BLK_TOPK=1)    # block-level indexer top-k (pp +25% at depth)
[ -n "${GATHER:-}" ]       && VK_ENV+=(Q4X_QSA_GATHER="$GATHER")  # gathered decode attn (off: loses A/B)
GP="${GP:-28672}"                                             # gathered sparse prefill engage depth; 0 = off
[ "$GP" != "0" ]      && VK_ENV+=(Q4X_QSA_GP="$GP")
[ -n "${GP_W:-}" ]    && VK_ENV+=(Q4X_QSA_GP_W="$GP_W")
[ -n "${GP_FRAC:-}" ] && VK_ENV+=(Q4X_QSA_GP_FRAC="$GP_FRAC")
[ "${POOLED:-1}" = "0" ] && VK_ENV+=(LLAMA_QSA_NO_POOLED_CACHE=1)  # A/B fallback: recompute pooled keys

if [ "$MTP" = "1" ]; then
  UBD="${UBD:-512}"     # draft ubatch — decoupled so the TARGET keeps UB=2048
  KVD="${KVD:-q8_0}"    # draft KV type
  PMIN="${PMIN:-0.75}"  # draft confidence gate
  # draft depth: 6 wins greedy, 4 wins sampled; NMAX+1 must stay <= 8 (mat-vec batch cliff)
  case "$TEMP" in
    0|0.0) NMAX="${NMAX:-6}" ;;
    *)     NMAX="${NMAX:-4}" ;;
  esac
  MTP_ARGS=(-md "$MD" --spec-type draft-mtp --spec-draft-n-max "$NMAX" --spec-draft-p-min "$PMIN"
            --spec-draft-ubatch "$UBD" --spec-draft-type-k "$KVD" --spec-draft-type-v "$KVD")
  # stochastic speculative sampling: draft samples with target params, server verifies by
  # rejection sampling (lossless). Auto-on at temp>0; without it MTP at temp>0 is a wash.
  [ "${SPEC_STOCH:-1}" != "0" ] && [ "${TEMP:-0}" != "0" ] && VK_ENV+=(Q4X_SPEC_STOCH=1)
  if [ "${SPEC_STOCH:-1}" = "0" ] && [ "$TEMP" != "0" ] && [ "$TEMP" != "0.0" ]; then
    echo "[run-flashnext] WARNING: MTP=1 + SPEC_STOCH=0 at TEMP=$TEMP is a wash (exact-match verify only accepts at ~p)." >&2
    echo "[run-flashnext]          Drop SPEC_STOCH=0, or set TEMP=0, or use MTP=0 to save ~4GB on the draft model." >&2
  fi
fi

echo "[run-flashnext] build=build-v2 quant=$QUANT ctx=$CTX ngl=$NGL fa=$FA port=$PORT ngram_offload=$NGRAM_OFFLOAD reasoning=$REASONING temp=$TEMP mtp=$MTP coopmat=$COOPMAT free_gpu=$FREE_GPU parallel=$PARALLEL" >&2

# --- lifecycle ------------------------------------------------------------------------
# never stack a second server: booting into another server's GTT can wedge amdgpu (reboot-only)
if [ "$MTP" = "1" ] && [ "$COOPMAT" != "0" ] && pgrep -f "$SRV_PAT" >/dev/null 2>&1; then
  echo "[run-flashnext] a build-v2 llama-server is already running — stop it before launching coopmat-ON MTP." >&2
  exit 1
fi

# on exit: kill the containerized server explicitly (a signal to this script would orphan it
# holding ~85GB GTT — podman only forwards signals with a TTY), then restore services.
_FREED=0
_shutdown() {
  pkill -f "$SRV_PAT" 2>/dev/null || true
  if [ "$_FREED" = "1" ]; then
    sleep 2
    echo "[run-flashnext] restarting comfyui + OCR" >&2
    systemctl --user start dashi-unlimited-ocr.service comfyui.service 2>/dev/null || true
  fi
}
trap _shutdown EXIT INT TERM

if [ "$FREE_GPU" = "1" ]; then
  echo "[run-flashnext] stopping comfyui + OCR to free memory (FREE_GPU=0 to skip)" >&2
  systemctl --user stop dashi-unlimited-ocr.service comfyui.service 2>/dev/null || true
  _FREED=1
  _gttf=$(ls /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)
  g=0; a=0
  for _ in $(seq 1 40); do
    g=$(( $(cat "$_gttf" 2>/dev/null)/1073741824 )); a=$(free -g | awk '/^Mem:/{print $7}')
    [ "$g" -lt 6 ] && [ "$a" -ge 110 ] && break
    sleep 2
  done
  echo "[run-flashnext] memory ready: gtt=${g}GB free_ram=${a}GB" >&2
elif [ "$MTP" = "1" ] && [ "$COOPMAT" != "0" ]; then
  # let a just-exited server's GTT finish draining before graph init
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

# fresh container on the coopmat-ON MTP path so graph init sees a clean state
if [ "$MTP" = "1" ] && [ "$COOPMAT" != "0" ]; then
  podman restart "$TOOLBOX" >/dev/null 2>&1 || podman start "$TOOLBOX" >/dev/null 2>&1 || true
  sleep 2
else
  podman start "$TOOLBOX" >/dev/null 2>&1 || true
fi

VERBOSE_ARGS=()
[ "${VERBOSE:-0}" = "1" ] && VERBOSE_ARGS=(--verbose)
KV_ARGS=()
[ -n "${KV:-}" ] && KV_ARGS=(-ctk "$KV" -ctv "$KV")   # target KV quant (deep-ctx fit lever)

# not `exec` — the shell must survive to run the trap
toolbox run --container "$TOOLBOX" env "${VK_ENV[@]}" LD_LIBRARY_PATH="$BIN" \
  "$BIN/llama-server" -m "$MODEL" "${VERBOSE_ARGS[@]}" \
    --alias flashnext --host 0.0.0.0 --port "$PORT" \
    -ngl "$NGL" -c "$CTX" -ub "$UB" --flash-attn "$FA" "${KV_ARGS[@]}" "${OT_ARGS[@]}" --metrics \
    --parallel "$PARALLEL" \
    --temp "$TEMP" --top-p 0.95 --top-k 20 --min-p 0.0 \
    --presence-penalty 0.0 --repeat-penalty 1.0 \
    "${REASON_ARGS[@]}" "${TPL_ARGS[@]}" \
    "${MTP_ARGS[@]}"
