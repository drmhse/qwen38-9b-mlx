#!/usr/bin/env bash
# One-shot prompt. Usage: ./ask.sh "why is the sky blue?"  [--image pic.jpg]
set -euo pipefail
cd "$(dirname "$0")"
source ./_guard.sh
exec .venv/bin/python -m mlx_vlm.generate \
  --model models/Qwen3.8-9B-Abliterated-MLX/4bit \
  --max-tokens "${MAX_TOKENS:-512}" --temperature "${TEMP:-0.7}" \
  --prompt "$@"
