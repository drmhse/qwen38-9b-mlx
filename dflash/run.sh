#!/usr/bin/env bash
# Speculative decoding, guarded. Usage: dflash/run.sh "your prompt" [--max-tokens N]
#
# The dflash CLI is normally invoked by hand, which skips ../_guard.sh and lets you
# start a second ~6GB runtime beside a running serve.sh. Both fit in 16GB right up
# until they don't, and the failure mode is a GPU OOM that takes the machine with it.
#
# Defaults are the bit-exact configuration: verify_linear off, 1.70x. Set
# DFLASH_VERIFY_LINEAR=1 for 2.01x and read the README section on what it costs.
set -euo pipefail
cd "$(dirname "$0")"
source ../_guard.sh

PROMPT="${1:?usage: dflash/run.sh \"prompt\" [extra dflash args]}"
shift
[ -x ./.venv/bin/dflash ] || { echo "dflash venv missing; run dflash/setup.sh first" >&2; exit 1; }
exec ./.venv/bin/dflash generate \
  --model ../models/Qwen3.8-9B-Abliterated-MLX/4bit \
  --draft ./draft-9b \
  --verify-mode adaptive --draft-quant w4 \
  --max-tokens "${MAX_TOKENS:-100}" \
  --prompt "$PROMPT" "$@"
