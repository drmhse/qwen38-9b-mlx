# Qwen3.8-9B-Abliterated on a 16 GB M4

Local inference for `PocketAiHub/Qwen3.8-9B-Abliterated-MLX` (4-bit), via mlx-vlm.
Serving scripts, two prompt-cache fixes that make multi-turn 57x faster, and the
measured ceilings — so nobody re-runs the experiments that found them.

## Files

    serve.sh          OpenAI/Anthropic-compatible server on 127.0.0.1:8080 (APC env baked in)
    ask.sh            one-shot prompt
    chat.sh           interactive
    _guard.sh         sourced by all three: blocks a 2nd instance and low-memory starts
    patch_mlx_vlm.py  re-applies the two caching fixes; RUN AFTER any mlx-vlm reinstall
                      or model re-download, or prefill silently regresses ~57x
    download.py       re-fetch the 4-bit variant at the pinned revision
    bench_fly.py      re-measure the throughput levers (regression check)

Untracked, rebuilt locally: `.venv/` (mlx 0.32.0 + mlx-vlm 0.6.8), `models/`
(5.6 GB, via `download.py`), `.apc-cache/` (self-regenerating). Only the patched
`chat_template.jinja` is tracked out of `models/`.

## Run it

    ./ask.sh "your prompt"          # one-shot
    ./chat.sh                       # interactive
    ./serve.sh                      # server on 127.0.0.1:8080

All three `source ./_guard.sh`, which refuses a second instance or a start below
~8 GB reclaimable memory. **Respect it** — see Memory.

Requests must name the model exactly, not an alias:

    curl 127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"models/Qwen3.8-9B-Abliterated-MLX/4bit",
           "messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

Three dialects on one server: `/v1/chat/completions` (OpenAI), `/v1/responses`
(stateful, with cancel), `/v1/messages` + `/v1/messages/count_tokens` (Anthropic).
Plus `/v1/cache/{stats,reset}`, `/health`, `/metrics`, `/unload`, audio and image
endpoints. Supports `tools`, `tool_choice`, `response_format`/`json_schema`,
`stream`, `logprobs`, `seed`, `stop`, `image_url`.

## Naming: 3.8 vs 3.5

`Qwen3.8-9B` names the **weights**; `qwen3_5` names the **architecture**.
`config.json` declares `model_type: "qwen3_5"`, and the manifests record a
third-party full-parameter *distillation* of `Qwen/Qwen3.5-9B` — explicitly "not an
official Qwen3.8 release". So runtime paths named `qwen3_5` are the right ones, but
stock-Qwen3.5 assumptions are not safe (see the norm ambiguity in the appendix).

Hybrid attention: 24 gated-DeltaNet linear-attention layers plus full attention
every 4th (`full_attention_interval: 4`), `attn_output_gate`, interleaved mrope,
and a 27-block SigLIP-style vision tower kept at BF16. Hidden 4096 /
intermediate 12288 / 32 layers / vocab 248320.

## Memory: use 4-bit, not 8-bit

16 GB unified memory, **11.5 GB Metal budget**. The 8-bit variant is 9.74 GiB of
weights with a ~11.4 GB measured peak, so it does not fit — confirmed three ways:
mistral.rs's device mapper, `mistralrs tune`, and MLX raising
`kIOGPUCommandBufferCallbackErrorOutOfMemory`. That GPU OOM hard-crashed the machine
once. 4-bit peaks at 6.7–8.7 GB and is the only sane choice here.

## Performance: both ends are at the hardware roof

Measured, 4-bit, base M4 (4P+6E, 10-core GPU):

| | prefill | decode | peak |
|---|---|---|---|
| mlx-vlm 0.6.8 | 217–229 tok/s @ 2.5–7k ctx | **20.8 tok/s** | 6.7–8.2 GB |
| mistral.rs 0.9.1 (patched) | 109 tok/s | 15.2 tok/s | — |

**Decode is memory-bandwidth-bound** — every token streams all weights. Achievable
bandwidth is 102.5 GB/s stream / 81 GB/s matvec (of 120 theoretical); at 20.8 tok/s
that is ~4.3–4.9 GB read per token, i.e. 85–100% of achievable. No runtime beats
this. Confirmed by mlx 0.32.0 vs 0.32.1 (20.84 vs 20.69) and by quantized KV cache
changing nothing (20.8 / 20.0 / 21.1 for bf16 / 8-bit / 4-bit at 7k) — KV is not
the bottleneck. The only way past it is fewer weight-bytes per token: speculative
decoding, a smaller model, or lower precision.

**Prefill is compute-bound and equally maxed.** afq4 GEMM measures 3.29 TFLOP/s
(88% of the 3.73 dense bf16 peak). The model costs 13.8–15.9 GFLOP per prefill
token (mlp 4.832B + linear_attn 1.617B + self_attn 0.470B), so 224 tok/s @ 2.5k is
3.10–3.56 TFLOP/s — 94–108% of that ceiling. Consequences:

- `--prefill-step-size` tuning is pointless; GEMM is flat from M=1024 up, so the
  2048 default is already optimal.
- Dequantize-then-dense-GEMM caps at 1.13x before dequant cost. Not worth it.
- FlashAttention/sparse-attention/sinks have nothing to win: 2.5k -> 16k costs only
  224 -> 199 tok/s, because just 8 of 32 layers are full attention.
- Batching does not help prefill (it already saturates compute); it helps decode.

**Prefill dominates agent latency, not decode.** A 4k prompt costs ~19s and 16k
~80s, while 40 output tokens cost ~2s. Prompt size is the lever.

## What actually moves the needle

| lever | effect | verdict |
|---|---|---|
| concurrency 4 vs 1 | 18.7 -> **38.2 tok/s** aggregate (2.04x) | **best free win** |
| prompt caching, warm | prefill 19.4s -> **0.35s** (57x) | **biggest lever** |
| keep server warm | avoids 2.5s model load per call | free |
| thinking off | mlx-vlm default (mistral.rs had it ON, burning 2631–4556 tokens) | free |
| shorter prompts | ~4.5s per 1k tokens | biggest latency lever |
| quantized KV (8/4-bit) | 20.8 / 20.0 / 21.1 tok/s @ 7k | **no effect** |
| newer mlx (0.32.1) | 20.69 vs 20.84 | **no effect** |

Decode reads all weights per step regardless of batch size, so N concurrent
requests amortise one weight pass — hit `/v1/chat/completions` concurrently rather
than serially. (8x untested.)

Beyond caching, the only real prefill wins are fewer tokens (context curation, or
LLMLingua-2-style learned pruning for 2–5x lossy reduction), response-level
semantic caching (GPTCache-style — architecture-independent, so it sidesteps the
exact-mode limit below), and quantizing the BF16 vision tower if you send images.
The publisher's 3197 tok/s prefill was an M5 Max (~40 GPU cores); that 14x is
hardware.

## Prompt caching

Off upstream; `serve.sh` enables and tunes it, since the defaults do not work on a
16 GB Mac:

    APC_ENABLED=1                     # off by default
    APC_BLOCK_SIZE=16 APC_NUM_BLOCKS=2048   # 32768 tokens
    APC_DISK_PATH=./.apc-cache APC_DISK_MAX_GB=8   # survives restarts, ~160 MB per 4.3k snapshot
    APC_DISK_MIN_FREE_RAM_GB=0.5      # default 2.0 vs psutil "available" ~1.7 GB on macOS,
                                      # which silently skips every warm restore
    APC_EXACT_CACHE_ENTRIES=8         # in-RAM snapshots are the ONLY place prefix matching
                                      # happens; disk is exact-key only. Default 2.
    APC_EXACT_PREFIX_GUARD_TOKENS=16  # (default) why a hit reports len-16 cached tokens
    APC_TRACE=1                       # logs store/lookup decisions

Measured on a ~4.3k-token system prompt: turn 1 cold 19.42s prefill; turns 2–4
0.35s / 0.34s / 0.31s at 99–100% cached. **57x faster prefill, ~30x lower latency
on every turn after the first.**

**Exact mode only.** `apc.model_apc_mode()` returns `"exact"`, not `"block"`,
because the gated-DeltaNet layers hold recurrent state — quoting the source,
*"recurrent state cannot be reconstructed by concatenating K/V blocks alone."*
(Cache layout confirms it: `ArraysCache` x3 then `KVCache`, repeating, matching
`full_attention_interval: 4`.) So there is no arbitrary-prefix reuse — only
whole-prompt snapshots reused as prefixes of longer prompts.

Practical consequence: *independent* queries sharing a system prompt never hit,
because no snapshot exists at the shared boundary, and warming with a bare prefix
does not help (the template terminates it with `<|im_end|><|im_start|>assistant`).
Run that pattern as one growing conversation instead.

### The two fixes in `patch_mlx_vlm.py`

Both were needed; fixing either alone still yields `cached=0`. Idempotent, and
**must be re-run after any mlx-vlm reinstall or model re-download.**

**1. `inputs_embeds` in the APC salt** (mlx-vlm bug,
`generate/ar.py:_apc_extra_hash`). The salt folded in `inputs_embeds`, which the
server always supplies, and which is a function of the whole prompt — so every
distinct prompt got a distinct cache key and **prefix reuse was impossible by
construction**. Instrumenting `lookup_exact_cache` proved it: candidates were
rejected with `reason=extra_hash`, never on token mismatch. The salt exists to
separate media payloads sharing token ids, so the fix omits it for text-only
requests and keeps it whenever media is present.

**2. `<think>` asymmetry in the chat template** (model-packaging issue). Under
`enable_thinking=false` the generation prompt ends with `<think>\n\n</think>\n\n`,
but history assistant turns rendered as plain `content` — the `loop.index0 >
ns.last_query_index` gate keeps the scaffold only on the most recent turn. So turn
N's prompt was **not** a prefix of turn N+1's; they diverged exactly at `<think>`.
Fixed by giving history turns the same scaffold, making prompts strictly
append-only.

## Driving it from Pi (`@earendil-works/pi-coding-agent`)

Pi reads custom OpenAI-compatible providers from `~/.pi/agent/models.json`; no
extension needed (`pi.registerProvider()` is only for custom auth or streaming).

    {
      "providers": {
        "qwen38-local": {
          "name": "Qwen3.8-9B (local MLX)",
          "baseUrl": "http://127.0.0.1:8080/v1",
          "api": "openai-completions",
          "apiKey": "local",
          "compat": {
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false,
            "supportsStore": false,
            "maxTokensField": "max_tokens"
          },
          "models": [{
            "id": "models/Qwen3.8-9B-Abliterated-MLX/4bit",
            "name": "Qwen3.8-9B Abliterated 4bit",
            "reasoning": false,
            "input": ["text", "image"],
            "contextWindow": 32768,
            "maxTokens": 4096,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
          }]
        }
      }
    }

Leave `./serve.sh` running, then `pi` -> `/login qwen38-local` (any value) ->
`/model`. Or skip the login prompt:

    pi --api-key local --provider qwen38-local \
       --model models/Qwen3.8-9B-Abliterated-MLX/4bit

- `id` is the literal served path; aliases are rejected.
- `supportsDeveloperRole: false` — mlx-vlm's server does not accept the `developer`
  role Pi sends for reasoning-capable models.
- `reasoning: false` matches the server's `enable_thinking=False`. To enable
  thinking, set `reasoning: true` plus `compat.thinkingFormat: "qwen-chat-template"`
  — but that changes the rendered prompt and so invalidates existing APC snapshots.
- `contextWindow: 32768` matches the APC budget. Larger just evicts cache.
- `/login` is required even though the key is ignored; Pi hides models with no
  stored credential.

Expect it to feel slow for coding: 20.8 tok/s decode, ~4.5s per 1k prompt tokens,
and Pi is single-stream so the 2.04x concurrency win does not apply. The
compensation is that Pi grows one conversation — exactly the append-only shape
exact mode needs, so turn 2 onward prefills in ~0.3s.

## Speculative decoding: the one way past the decode roof

It reduces weight-bytes per accepted token, and is **lossless in output quality**
regardless of drafter fit — drafts are verified by the target, so a mismatched
drafter costs speed, never correctness. It is also **decode-only**: MTP, DFlash and
EAGLE alike do nothing for prefill, which already processes all prompt tokens in
parallel.

**Do not try to port a 27B drafter.** Both the 27B MTP head and
`incoai/Qwen3.8-27B-DFlash2` are shaped to their host's residual stream — the
DFlash2 checkpoint is 5120-wide throughout with `fc.weight [5120, 25600]`, fusing
hidden states tapped from 5 of the 27B's 64 layers (`target_layer_ids
[5,19,33,47,61]`), and it ships no `embed_tokens`/`lm_head` of its own. Nothing
lines up with 4096 / 12288 / 32 except the vocab.

**The 9B has native options.** `z-lab/Qwen3.5-9B-DFlash` matches exactly — hidden
4096, intermediate 12288, `num_target_layers 32`, vocab 248320, `fc.weight
[4096, 32768]` (8 taps at `[1,5,9,13,17,21,25,29]`), `block_size 16`, 6 layers
≈ 1.29B params, ~0.7 GB at w4. `bstnxbt/dflash-mlx` supports Qwen3.5 hybrid
GatedDeltaNet targets and quantizes the draft to w4 by default, reporting 4.37x at
1k ctx falling to 2.22x at 8k on an M5 Max. `guglxni/Qwen3.5-9B-abliterated-DFlash`
is an abliterated-tuned variant. The 9B also has its own MTP head in
`empero-ai/Qwen3.8-9B` (15 tensors, 0.49 GB bf16, fetchable by HTTP range read
instead of pulling 19.3 GB), which this publisher strips.

Two costs before switching: abliteration re-projected `out_proj`/`down_proj` in
layers 12–31, so a drafter trained on the un-abliterated stream will accept below
the published rates; and dflash-mlx is single-request with no continuous batching,
so it trades the measured 2.04x concurrency win for single-stream latency. Leaving
mlx-vlm also leaves the caching fixes above. Untested here.

## Provenance

All 14 files of the 4-bit variant verified against the publisher's SHA-256
`artifact-manifest.json`. Pinned revision `1836724...`. The download is kept
unmodified except for `chat_template.jinja` (fix 2 above).

## Appendix: the mistral.rs / MTP path (abandoned)

Dropped because mlx-vlm is both faster (20.8 vs 15.2 tok/s decode) and correct, and
because MTP is decode-only and so cannot touch the prefill cost that dominates. The
source tree, the patched model copy, `prepare_mistralrs.py`, the MTP fetch/quantize
scripts and the MTP shards were all deleted. What was learned:

- **Three weight transforms** are needed to run an MLX-converted checkpoint under
  mistral.rs, which is written against the source HF layout. (1) Conv weights are
  channels-last: MLX stores `(out, *kernel, in)`, candle wants `(out, in, *kernel)`
  — 25 tensors, hard shape error. (2) **RMSNorm weights are absolute, not offset**
  (81 tensors) — *this is the one that silently produces garbage.* mlx-vlm bakes
  `+1.0` into its `NORM_WEIGHT_SUFFIXES` families; mistral.rs builds the same five
  with `GemmaRmsNorm`, whose constructor adds 1.0 again (`layers.rs:489`), so the
  model computes `2+w` and emits multilingual noise while loading cleanly. Subtract
  1.0. `linear_attn.norm` and the vision LayerNorms are *not* offset by mlx-vlm and
  must be left alone. (3) `lm_head` is nested at `language_model.lm_head`; mistral.rs
  reads it from the root varbuilder (`text.rs:458,581`). Rename only. Transforms 1
  and 3 are arguably mistral.rs bugs, since it already detects mlx-vlm naming
  (`mod.rs:61`, `text.rs:436`); 2 is genuine ambiguity — the checkpoint does not
  record which form its norms are in.
- Also needed `-n "0:32"` to bypass the auto device mapper, which sizes
  AFQ-prequantized layers as if BF16 (claims 19169 MB) and refuses to load.
- `mtp.rs` / `speculative.rs` for qwen3_5 exist **only on master**, not at tag
  v0.9.1. Build: `MISTRALRS_METAL_PRECOMPILE=0 cargo build --release -p
  mistralrs-cli --features metal -j 3` (3m12s). Use `-j 3` and `nice`; a wide
  parallel link can exhaust 16 GB.
- Xcode 26.3 ships `metal` as a **stub**, so ahead-of-time kernel compilation fails
  ("cannot execute tool 'metal' due to missing Metal Toolchain"). `PRECOMPILE=0`
  writes 0-byte metallibs and compiles at runtime, which works for mistralrs-quant
  but **not** paged-attention: those sources hit `error: 'function_constant' has a
  duplicate index '10'`. Getting further needs
  `xcodebuild -downloadComponent MetalToolchain`.
- MTP requires PagedAttention, which on Metal hit `Failed to create metal resource:
  Buffer`. Root cause: hybrid models request **0 KV blocks** for linear/recurrent
  layers (24 of 32 here), and `cache_engine.rs` then calls
  `dev.new_private_buffer(0, ...)` — Metal refuses a zero-length buffer where CUDA
  tolerates it. Fixed with five one-word edits, `elem_count` ->
  `elem_count.max(1)`. Worth upstreaming.
- mistral.rs only loads weight files matching `model-\d+-of-\d+\.safetensors`
  (`pipeline/paths.rs:27`); an extra shard under any other name is **silently
  ignored**. (Its *local-dir* path globs `*.safetensors` instead —
  `pipeline/paths.rs:196`.)
- MTP preconditions all held: `mtp_num_hidden_layers == 1`,
  `mtp_use_dedicated_embeddings == false`, `mtp.fc` shaped `2*hidden -> hidden`
  (`[4096, 8192]`). Enable with `--mtp` (+ `--mtp-n-predict N`); `--mtp-model` is
  for a separate assistant model instead.
