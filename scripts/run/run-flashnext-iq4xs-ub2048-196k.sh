#!/usr/bin/env bash
# run-flashnext-iq4xs-ub2048-196k.sh
# Qwen3.8-Flash-Next Uncensored IQ4_XS (orcarouter finetune) · ubatch 2048 · ctx 196608 (max context without OOM at ub2048)
# ONE fixed configuration — to change anything, edit this file. Source of truth: this file in the repo (scripts/run/).
set -u
log() { echo "[$(basename "$0")] $*" >&2; }
TOOLBOX=llama-vulkan-wdenejko
BIN=/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin
MODEL=/home/wdenejko/models/Qwen3.8-Flash-Next-Uncensored-GGUF/IQ4_XS-a1perm/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf   # K1: the A1 model with the 97 hc_*_up rows permuted channel-major + KV hyper_connection.up_perm=1 so the fused HC collapse epilog engages (+3.6% pp d0 on top of A1, lossless). Fallbacks: IQ4_XS-a1inject/ (A1 only), IQ4_XS/ (pre-A1).
DRAFT=/home/wdenejko/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M-hcfix.gguf   # Q4 MTP head: with the PLE in RAM the Q8 head host-OOMs past ~129k
CTX=196608
UB=2048
PORT=8080
NEED_FREE_GB=100   # clean-box gate (a drained box shows ~118G free); a cleanliness check, not a fit predictor
TPL="$(cd "$(dirname "$0")" && pwd)/uncensored-chat-template.jinja"   # Uncensored template relaxed to accept mid-conversation system messages (agent harnesses)
[ -f "$TPL" ] || { log "template missing: $TPL"; exit 1; }
# the row-permuted model is only correct on a binary that reads hyper_connection.up_perm; an older build would silently misread it
grep -qa "hyper_connection.up_perm" "$BIN/libllama.so" 2>/dev/null || { log "ABORT: $BIN lacks hc up_perm support — it would misread the row-permuted a1perm model (serve IQ4_XS-a1inject/ with that binary instead)"; exit 1; }
# --- box safety (every measured wedge trigger): one server at a time; boot only into a drained, clean box ---
if pgrep -f "alias flashnext" >/dev/null; then log "a flashnext llama-server is already running — stop it first"; exit 1; fi
log "stopping comfyui + OCR for the duration"
systemctl --user stop dashi-unlimited-ocr.service comfyui.service 2>/dev/null || true
_shutdown() {   # podman only forwards signals with a TTY: kill the containerized server ourselves, then restore services
  pkill -f "$BIN/[l]lama-server" 2>/dev/null || true
  sleep 2
  systemctl --user start dashi-unlimited-ocr.service comfyui.service 2>/dev/null || true
}
trap _shutdown EXIT INT TERM
python3 - <<'PY'   # evict stale model pages from the page cache so 'free' is real (no root needed)
import os, glob
for f in glob.glob('/home/wdenejko/models/Qwen3.8-Flash-Next*GGUF/*/*.gguf'):
    fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
PY
_gtt()  { echo $(( $(cat /sys/class/drm/card*/device/mem_info_gtt_used | head -1) / 1073741824 )); }
_free() { free -g | awk '/^Mem:/{print $4}'; }
for _ in $(seq 1 45); do
  [ "$(_gtt)" -le 2 ] && [ "$(_free)" -ge "$NEED_FREE_GB" ] && break
  sleep 2
done
if [ "$(_gtt)" -gt 2 ] || [ "$(_free)" -lt "$NEED_FREE_GB" ]; then
  log "ABORT: gtt=$(_gtt)G free=$(_free)G (need gtt<=2 and free>=${NEED_FREE_GB}G) — not booting into a dirty box"
  exit 1
fi
log "memory ready: gtt=$(_gtt)G free=$(_free)G"
podman restart "$TOOLBOX" >/dev/null 2>&1 || podman start "$TOOLBOX" >/dev/null 2>&1 || true   # fresh container: clean state for coopmat MTP graph init
sleep 2

log "Uncensored IQ4_XS (orcarouter finetune) ub=$UB ctx=$CTX load-mode=none kv=q8_0 mtp=Q4-draft(n4,pmin0.75) temp=1.0 reasoning=medium port=$PORT"
# not exec — the shell must survive to run the trap
toolbox run --container "$TOOLBOX" env Q4X_RS_ROLLBACK=1 Q4X_QSA_BLK_TOPK=1 Q4X_QSA_GP=28672 Q4X_SPEC_STOCH=1 LD_LIBRARY_PATH="$BIN" \
  "$BIN/llama-server" -m "$MODEL" --load-mode none \
    --alias flashnext --host 0.0.0.0 --port "$PORT" \
    -ngl 99 -c "$CTX" -ub "$UB" --flash-attn on -ctk q8_0 -ctv q8_0 \
    --override-tensor per_layer_token_embd.weight=CPU --metrics --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
    --reasoning on --reasoning-effort medium --chat-template-file "$TPL" \
    -md "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75 \
    --spec-draft-ubatch 512 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
