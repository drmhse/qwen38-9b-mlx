#!/usr/bin/env bash
# OpenAI-compatible server on 127.0.0.1:8080.
#   curl 127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
#     -d '{"model":"local","messages":[{"role":"user","content":"hi"}]}'
set -euo pipefail
cd "$(dirname "$0")"
source ./_guard.sh
# Automatic Prefix Cache: off by default upstream. Without it every request
# re-prefills the whole prompt (a 16k-token prompt cost ~80s, twice). The disk
# tier makes the cache survive restarts.
export APC_ENABLED="${APC_ENABLED:-1}"
export APC_BLOCK_SIZE="${APC_BLOCK_SIZE:-16}"
export APC_NUM_BLOCKS="${APC_NUM_BLOCKS:-2048}"      # x block_size = 32768 tokens
export APC_DISK_PATH="${APC_DISK_PATH:-$PWD/.apc-cache}"
export APC_DISK_MAX_GB="${APC_DISK_MAX_GB:-8}"
# Default 2.0 compares against psutil's "available", which reads ~1.7GB on macOS
# even at 74% free -- that silently disables all warm restores here.
export APC_DISK_MIN_FREE_RAM_GB="${APC_DISK_MIN_FREE_RAM_GB:-0.5}"
# In-RAM snapshots are the ONLY place prefix matching happens (disk is exact-key
# only). Default 2 is enough for one conversation; raise it for concurrent ones.
export APC_EXACT_CACHE_ENTRIES="${APC_EXACT_CACHE_ENTRIES:-8}"

exec .venv/bin/python -m mlx_vlm.server \
  --model models/Qwen3.8-9B-Abliterated-MLX/4bit \
  --host 127.0.0.1 --port "${PORT:-8080}" "$@"
