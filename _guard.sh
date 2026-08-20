# Refuse to start if another local inference process is resident, or if memory
# is genuinely low. Uses memory_pressure's system-wide free percentage --
# vm_stat free+inactive badly understates availability on macOS, because most
# memory is held as reclaimable file cache / compressed pages.
# `bin/dflash` catches the speculative-decoding runtime in dflash/.venv, which holds
# ~6GB of its own. It is a separate venv, so nothing else here would notice it.
_others=$(pgrep -f "mlx_vlm|mistralrs|bin/dflash" || true)
if [ -n "$_others" ]; then
  echo "REFUSING TO START: an inference process is already running:" >&2
  ps -o pid=,rss=,command= -p $(echo "$_others" | tr '\n' ' ') 2>/dev/null \
    | awk '{printf "  pid %s  rss=%.1fGB  %s\n", $1, $2/1048576, $3}' >&2
  echo "Stop it first:  pkill -f mlx_vlm   (or pkill -f bin/dflash)" >&2
  exit 1
fi
_total_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
_freepct=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
_free_gb=$(( _total_gb * ${_freepct:-100} / 100 ))
if [ "${_free_gb}" -lt 8 ]; then
  echo "WARNING: ~${_free_gb}GB free of ${_total_gb}GB (${_freepct}%); this model needs ~7GB." >&2
  echo "Close some apps, or override with FORCE=1." >&2
  [ "${FORCE:-0}" = "1" ] || exit 1
fi
