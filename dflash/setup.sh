#!/bin/bash
# Stand up the dflash-mlx speculative-decoding runtime alongside (not inside) the
# mlx-vlm .venv, so serve.sh is untouched. See the DFlash section of ../README.md.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv"
DRAFT="$HERE/draft-9b"

# 3.13; the repo .venv is 3.12 and stays separate. dflash-mlx needs >=3.10.
python3.13 -m venv "$VENV"
"$VENV/bin/pip" -q install --upgrade pip
"$VENV/bin/pip" -q install dflash-mlx        # pulls mlx + mlx-lm (NOT mlx-vlm)

# Drafter. huggingface_hub died silently mid-transfer here twice; curl resumes.
mkdir -p "$DRAFT"
curl -sL -o "$DRAFT/config.json" \
  https://huggingface.co/z-lab/Qwen3.5-9B-DFlash/resolve/main/config.json
curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
  -o "$DRAFT/model.safetensors" \
  https://huggingface.co/z-lab/Qwen3.5-9B-DFlash/resolve/main/model.safetensors
test "$(stat -f%z "$DRAFT/model.safetensors")" = 2583816465

python3 "$HERE/shim_draft_config.py" "$DRAFT/config.json"
python3 "$HERE/patch_dflash_verify_linear.py" \
  "$VENV/lib/python3.13/site-packages"

"$VENV/bin/dflash" doctor
echo
echo "run:  $HERE/run.sh 'your prompt'"
echo
echo "run.sh wraps the CLI above with the same memory guard serve.sh uses; the"
echo "raw command is in its header if you need to vary the flags."
