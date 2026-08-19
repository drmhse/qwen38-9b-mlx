#!/usr/bin/env bash
# Interactive chat. Usage: ./chat.sh [--image path] ["prompt"]
set -euo pipefail
cd "$(dirname "$0")"
source ./_guard.sh
exec .venv/bin/python -m mlx_vlm.chat --model models/Qwen3.8-9B-Abliterated-MLX/4bit "$@"
