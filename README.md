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
    dflash/           speculative-decoding runtime (separate venv; see DFlash section)

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

## Speculative decoding: measured, working, 1.70x

`dflash-mlx` runs the 9B-native DFlash drafter against this checkpoint and clears
the 20.8 tok/s decode roof. Measured here, same prompt throughout, 100 tokens out,
86% draft acceptance:

| configuration | tok/s | vs baseline | bit-exact vs target |
|---|---|---|---|
| target only (mlx_lm, greedy) | 21.4 | 1.00x | reference |
| DFlash, verify_linear off | **36.4** | **1.70x** | **yes** |
| DFlash, verify_linear on + qmm on | 42.9 | 2.01x | **no** (rel 4.3e-3) |
| DFlash, verify_linear on + qmm off | 36.4 | 1.70x | yes |

Peak memory 5.99 GB, against 5.20 GB for the target alone — the drafter adds only
0.79 GB, because mlx-lm loads lazily and quantizes per tensor, so the 2.6 GB bf16
copy never exists. Speedup does **not** decay with context here (1.51x at 1865
prompt tokens vs 1.48x at 23); the published M5 Max curve falls 4.37x -> 2.22x
because it starts far higher. Target-only measured 20.1-21.4 tok/s and 226.9 tok/s
prefill under mlx-lm, independently reproducing the roofs above on a second runtime.

**Losslessness was verified, not assumed.** With verify_linear off, DFlash output
is byte-identical to `mlx_lm generate --temp 0` on the same prompt.

Why it fits: `z-lab/Qwen3.5-9B-DFlash` is hidden 4096 / intermediate 12288 /
`num_target_layers 32` / vocab 248320, and `fc.weight [4096, 32768]` fuses hidden
states from 8 target layers `[1,5,9,13,17,21,25,29]`. `bind_target_model` asserts
no dimensions at all -- it only reads `embed_scale` -- so that concatenation width
is the entire contract. The drafter ships no `embed_tokens`/`lm_head`, borrowing
the target's, which means draft logits come out of *this* abliterated head.

### Setup

`dflash/setup.sh` builds a separate 3.13 venv (dflash-mlx depends on **mlx-lm**,
not mlx-vlm), fetches the drafter, and applies both patches below. It does not
touch `.venv/` or `serve.sh`. Two fixes are required:

1. **`dflash/shim_draft_config.py`** -- dflash-mlx 0.1.8 reads `rope_theta` and
   `block_size` from the config root; z-lab publishes them nested
   (`rope_parameters.rope_theta` per transformers 5.x, `block_size` inside
   `dflash_config`), so `from_dict` raises `TypeError: missing 2 required
   positional arguments`. Pure schema drift; the shim lifts both.
2. **`dflash/patch_dflash_verify_linear.py`** -- optional, see below.

Model loading needed no porting: mlx-lm's `qwen3_5.py` already accepts the VLM
wrapper (`ModelArgs.from_dict` takes a nested `text_config`, and `sanitize()`
drops `vision_tower`/`model.visual` and rewrites the `language_model.` prefix),
and dflash's target adapter is duck-typed -- `"qwen" in model_type` plus a shape
check -- so it resolves to `qwen_gdn` / `hybrid_gdn` with recurrent rollback.

### verify_linear: an 18% gain that costs bit-exactness

dflash-mlx gates its hand-written Metal verify kernels on
`_supports_verify_linear`, which for dense models is `num_layers >= 40`. This
model has 32, so it is excluded on layer count alone -- not on capability: the
per-linear gate `is_verify_eligible()` accepts **248 of 249** QuantizedLinears
here (the one rejection is `lm_head`, excluded on purpose by `N < 100_000`), and
`verify_linear._PROJ_TAGS` carries explicit `gdn_qkv`/`gdn_z`/`gdn_o` tags for
gated-DeltaNet projections. The threshold is validation policy.

The patch does **not** invent a new threshold. The package already ships
`DFLASH_VERIFY_LINEAR`, but `loading.py` computes `supports_verify_linear AND
_verify_enabled_for(...)` and the env var only feeds the right operand, so it
could disable the feature and never enable it. The patch consults the override at
the top of `_supports_verify_linear`, and makes `DFLASH_VERIFY_QMM` a tri-state
(`VerifyConfig.enable_qmm` defaults True with no CLI flag, so the qmm path was
otherwise unavoidable). Unset, upstream behaviour is unchanged. Both edits are
idempotent and write `.orig` backups.

**The result is worth knowing before enabling it.** Swapping the 248 linears buys
*nothing* on its own -- 36.4 tok/s either way. The entire +18% comes from one
kernel, the M=16 `qmm` path, and that kernel is exactly the one that is not
bit-exact. Isolated per layer on `mlp.gate_proj`, stock vs `VerifyQuantizedLinear`
is identical at n=1/8/16 with qmm off, and at n=1/8 with qmm on, diverging **only
at n=16** -- which is the DFlash block size, so it perturbs the verifier on every
block (`_build_kernel_m16_super_tree_fp16_ktmpl`: fp16 accumulation).

So it is a speed-for-numerics trade, not free performance. Default is off:
1.70x while remaining byte-identical to the reference target. Enabling it means
the verifier is no longer numerically the same model, and "verified by the
target" stops meaning what it means everywhere else in this file.

    DFLASH_VERIFY_LINEAR=1                    # 2.01x, NOT bit-exact
    DFLASH_VERIFY_LINEAR=1 DFLASH_VERIFY_QMM=0  # bit-exact, but no gain (36.4)

### Costs of switching runtimes

Not adopted as the default server here, because leaving mlx-vlm costs:

- **No vision tower** -- dflash-mlx builds on mlx-lm; no image input.
- **No continuous batching**, so the 2.04x concurrency win above disappears.
  mlx-vlm at 38.2 tok/s aggregate still beats DFlash for concurrent agent traffic;
  DFlash wins single-stream interactive use.
- **The two APC fixes do not come along**, so prefill returns to cold cost.
- **Thinking is on by default** in this runtime, unlike mlx-vlm, so real tasks
  burn reasoning tokens before answering.

Note the drafter already uses the verify kernels unconditionally at w4
(`load_draft_bundle` installs them with `enable_qmm=True` regardless of gating).
That is harmless -- draft errors are corrected by verification. Applying them to
the *target* is the part that changes the guarantee.

Also still true: **do not try to port a 27B drafter.** Both the 27B MTP head and
`incoai/Qwen3.8-27B-DFlash2` are shaped to their host's residual stream -- the
DFlash2 checkpoint is 5120-wide throughout with `fc.weight [5120, 25600]`, fusing
5 of the 27B's 64 layers, and ships no embeddings of its own. Nothing lines up
with 4096 / 12288 / 32 except the vocab. The 9B also has its own MTP head in
`empero-ai/Qwen3.8-9B` (15 tensors, 0.49 GB bf16, fetchable by HTTP range read).

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
