#!/usr/bin/env bash
# run-flashnext-q4kxl-ub2048-262k.sh
# Qwen3.8-Flash-Next UD-Q4_K_XL (Unsloth base) · ubatch 2048 · ctx 262144 (the model's full context)
# ONE fixed configuration — to change anything, edit this file. Source of truth: this file in the repo (scripts/run/).
#
# --load-mode mmap (NOT none): the 26.8G per-layer-token-embedding table (PLE, offloaded to the CPU) is served as a
# mapping-resident, pre-populated, EVICTABLE file mapping instead of a 27G anonymous copy. That is the ~27G of host
# headroom that lets ub2048 fill the full 262k window on the 123G box (validated 2026-09-05: fill to 253952 with
# min 21.1G available, tg 29.6 t/s at 254k, greedy output byte-identical to --load-mode none). Under none the same
# config OOM-kills at ~192k of fill and sits at ~3G available with an EMPTY context at 196k.
# Cost: under memory pressure the kernel evicts cold PLE rows, so a d0 prefill of a prompt with an unusually wide
# vocabulary re-faults them (random-token probe: -25..-29%); deep prefill is unaffected, real prose ~0%, real code -5% median (-2% once warm)
# (see the 2026-09-05 d0 diag numbers in the commit message).
set -u
log() { echo "[$(basename "$0")] $*" >&2; }
TOOLBOX=llama-vulkan-wdenejko
BIN=/home/wdenejko/src/llama-qwen4exp-src/build-v2/bin
MODEL=/home/wdenejko/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL-a1perm/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf   # K1: base UD-Q4_K_XL with the 97 hc_*_up rows permuted channel-major + KV hyper_connection.up_perm=1 so the fused HC collapse epilog engages on the Q8_0 up-weights (+4.0% pp d0, lossless/greedy-identical). Fallback: UD-Q4_K_XL/ (base; already carries A1 F32-inject + correct D1 compress_ratios, no K1).
DRAFT=/home/wdenejko/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M-hcfix.gguf   # Q4 MTP head (the Q8 head trips a radv per-submission ceiling at ub2048/262k)
CTX=262144
UB=2048
PORT=8080
NEED_FREE_GB=100   # clean-box gate (a drained box shows ~118G free); a cleanliness check, not a fit predictor
# the row-permuted a1perm model is only correct on a binary that reads hyper_connection.up_perm; an older build would silently misread the permuted rows
grep -qa "hyper_connection.up_perm" "$BIN/libllama.so" 2>/dev/null || { log "ABORT: $BIN lacks hc up_perm support — it would misread the row-permuted a1perm model (serve UD-Q4_K_XL/ with that binary instead)"; exit 1; }
# --load-mode mmap is only safe on a binary with the fixed mapping-resident loader (model-wide no-prefetch + bulk-readahead
# populate): an older build MAP_POPULATEs the non-PLE shards under mmap, the 77G device pin then crawls at ~110 MB/s and a
# kill mid-pin wedges TTM (reboot-only). The fixed loader carries the POSIX_MADV_SEQUENTIAL populate path.
grep -qa "POSIX_MADV_SEQUENTIAL" "$BIN/libllama.so" 2>/dev/null || { log "ABORT: $BIN lacks the fixed mapping-resident loader — never serve --load-mode mmap on it (use the ub1024-262k runner or --load-mode none at 196k)"; exit 1; }
# --- box safety (every measured wedge trigger): one server at a time; boot only into a drained, clean box ---
if pgrep -f "alias flashnext" >/dev/null; then log "a flashnext llama-server is already running — stop it first"; exit 1; fi
log "stopping comfyui + OCR for the duration"
systemctl --user stop dashi-unlimited-ocr.service comfyui.service 2>/dev/null || true
_shutdown() {   # podman only forwards signals with a TTY: kill the containerized server ourselves, then restore services
  pkill -INT -f "$BIN/[l]lama-server" 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f "$BIN/[l]lama-server" >/dev/null || break; sleep 1; done
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

log "UD-Q4_K_XL-a1perm (Unsloth base + K1 row-permute) ub=$UB ctx=$CTX load-mode=mmap (mapping-resident PLE) kv=q8_0 mtp=Q4-draft(n4,pmin0.75) temp=1.0 reasoning=medium port=$PORT"
# not exec — the shell must survive to run the trap
toolbox run --container "$TOOLBOX" env Q4X_RS_ROLLBACK=1 Q4X_QSA_BLK_TOPK=1 Q4X_QSA_GP=28672 Q4X_SPEC_STOCH=1 LD_LIBRARY_PATH="$BIN" \
  "$BIN/llama-server" -m "$MODEL" --load-mode mmap \
    --alias flashnext --host 0.0.0.0 --port "$PORT" \
    -ngl 99 -c "$CTX" -ub "$UB" --flash-attn on -ctk q8_0 -ctv q8_0 \
    --override-tensor per_layer_token_embd.weight=CPU --metrics --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
    --reasoning on --reasoning-effort medium \
    -md "$DRAFT" --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75 \
    --spec-draft-ubatch 512 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0
