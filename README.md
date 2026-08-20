# Qwen3.8-9B on a 16 GB Mac

This repository runs a 9-billion-parameter model on a laptop that has 16 GB of
memory for the model, the GPU, your browser, and everything else. It works. It is
also slower than you want, and I am going to tell you exactly how slow and
exactly why, because the thing that wastes the most time on hardware like this is
not the model. It is the week you spend tuning knobs that cannot move.

Everything below is measured on one machine: an Apple M4, 4 performance cores and
6 efficiency cores, 10 GPU cores, 16 GB unified memory, macOS 15.7.2. If you have
a bigger Mac, your numbers are better and some of my conclusions do not apply to
you. I will say so where it matters.

## Start here

```
git clone git@github.com:drmhse/qwen38-9b-mlx.git && cd qwen38-9b-mlx
python3 -m venv .venv
.venv/bin/pip install mlx==0.32.0 mlx-vlm==0.6.8 huggingface_hub
.venv/bin/python download.py        # 5.6 GB at a pinned revision
.venv/bin/python patch_mlx_vlm.py   # do not skip this, see Prompt caching
./ask.sh "explain unified memory in two sentences"
```

That is the whole thing. `./chat.sh` gives you a REPL, `./serve.sh` gives you an
HTTP server on `127.0.0.1:8080`.

All three scripts source `_guard.sh`, which refuses to start a second instance or
to start at all below roughly 8 GB of reclaimable memory. Respect the guard. I
added it after a GPU out-of-memory took the entire machine down, and the section
on memory explains why that is not an exaggeration.

Requests must name the model by its literal served path. Aliases are rejected:

```
curl 127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"models/Qwen3.8-9B-Abliterated-MLX/4bit",
       "messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

You get three API dialects on one server: `/v1/chat/completions` (OpenAI),
`/v1/responses` (stateful, cancellable), and `/v1/messages` plus
`/v1/messages/count_tokens` (Anthropic). Also `/v1/cache/{stats,reset}`,
`/health`, `/metrics`, `/unload`, and audio and image endpoints. `tools`,
`tool_choice`, `response_format`, `json_schema`, `stream`, `logprobs`, `seed`,
`stop` and `image_url` all work.

### What is in here

```
serve.sh          the server, with the cache environment already tuned
ask.sh            one-shot prompt
chat.sh           interactive
_guard.sh         sourced by all three; blocks a second instance or a low-memory start
patch_mlx_vlm.py  the two caching fixes; RE-RUN AFTER ANY REINSTALL
download.py       fetch the 4-bit weights at the pinned revision
bench_fly.py      re-measure the three server-side levers (thinking, cache, batching)
dflash/           speculative decoding, separate venv, separate section below
                  setup.sh builds it, run.sh runs it under the same memory guard
```

`.venv/`, `models/` and `.apc-cache/` are rebuilt locally and not tracked. The one
exception is `models/.../chat_template.jinja`, which is tracked because it is
patched.

## A naming trap worth thirty seconds

`Qwen3.8-9B` is the name of the weights. `qwen3_5` is the name of the
architecture. `config.json` says `model_type: "qwen3_5"`, and the manifests
describe a third-party full-parameter distillation of `Qwen/Qwen3.5-9B` that is
explicitly "not an official Qwen3.8 release."

This matters because every runtime dispatches on architecture, not on marketing.
Code paths named `qwen3_5` are the correct ones to read. Assumptions about stock
Qwen3.5 are not safe, and the appendix has a story about a norm convention that
loaded cleanly and produced fluent nonsense.

The shape: 32 layers, hidden 4096, intermediate 12288, vocab 248320. Attention is
hybrid, 24 gated-DeltaNet linear-attention layers with full attention every fourth
(`full_attention_interval: 4`), plus `attn_output_gate`, interleaved mrope, and a
27-block SigLIP-style vision tower kept at BF16. That hybrid is not trivia. It
decides what the prompt cache can and cannot do, and it comes back twice below.

## Use 4-bit. The 8-bit build does not fit.

16 GB of unified memory gives you an 11.5 GB Metal budget. The 8-bit variant is
9.74 GiB of weights with a measured peak around 11.4 GB, so it does not fit, and I
confirmed that three separate ways: mistral.rs's device mapper refused it,
`mistralrs tune` refused it, and MLX raised
`kIOGPUCommandBufferCallbackErrorOutOfMemory` partway through generation. That
last one hard-crashed the machine.

4-bit peaks at 6.7 to 8.7 GB depending on prompt length. It is the only sane
choice here, and it is not a compromise you should feel bad about.

One warning about capacity tools. The device mapper that refused the 8-bit load
was right for the wrong reason: it priced already-quantized weights as if they
were BF16 and claimed 19,169 MB. A published peak memory figure describes the
model on the machine that measured it, and the 8-bit number here came from an M5
Max with 128 GB. Trust your own allocator failure over anyone's estimate.

## Know your ceilings before you tune anything

Two limits explain nearly every result in this repository. Learn them and you can
predict which optimizations are worth your weekend.

| | prefill | decode | peak |
|---|---|---|---|
| mlx-vlm 0.6.8 | 217-229 tok/s at 2.5-7k ctx | **20.8 tok/s** | 6.7-8.2 GB |
| mistral.rs 0.9.1 (patched) | 109 tok/s | 15.2 tok/s | - |

**Decode is memory-bandwidth-bound.** Every token streams every weight. This GPU
reaches 102.5 GB/s on a streaming read and 81 GB/s on a matrix-vector product,
against 120 GB/s theoretical. At 20.8 tok/s the model is moving 4.3 to 4.9 GB per
token, which is 85 to 100 percent of what the hardware can actually deliver. That
is why upgrading mlx from 0.32.0 to 0.32.1 changed nothing (20.84 versus 20.69),
and why quantizing the KV cache changed nothing (20.8 / 20.0 / 21.1 tok/s for
bf16 / 8-bit / 4-bit at 7k context). The KV cache was never the bottleneck.

The only way past a bandwidth wall is to move fewer bytes per token. A smaller
model, lower precision, or speculative decoding. Two of those cost you quality.
The third one does not, and it has its own section.

**Prefill is compute-bound and equally maxed.** The afq4 GEMM measures 3.29
TFLOP/s, which is 88 percent of the 3.73 TFLOP/s dense BF16 peak. The model costs
13.8 to 15.9 GFLOP per prefill token (4.832B parameters in MLPs, 1.617B in linear
attention, 0.470B in full attention), so 224 tok/s at 2.5k context is 3.10 to 3.56
TFLOP/s. That is 94 to 108 percent of the ceiling. You are done.

Here is what that buys you, which is the right to skip four things:

- `--prefill-step-size` tuning is pointless. GEMM is flat from M=1024 up, so the
  2048 default is already optimal.
- Dequantize-then-dense-GEMM caps at 1.13x before you pay for the dequantization.
- FlashAttention, sparse attention and attention sinks have nothing to win here.
  Going from 2.5k to 16k context costs 224 to 199 tok/s, because only 8 of 32
  layers use full attention.
- Batching does not help prefill. It already saturates compute. It helps decode.

**Prefill dominates agent latency, not decode.** A 4k prompt costs about 19
seconds. A 16k prompt costs about 80. Forty output tokens cost about 2. If you are
optimizing the wrong end of that, you are optimizing the part that is 10 percent
of your wall clock.

## What actually moves the needle

| lever | effect | verdict |
|---|---|---|
| prompt caching, warm | prefill 19.4s to **0.35s** (57x) | **biggest lever** |
| speculative decoding (DFlash) | decode 21.4 to **36.4 tok/s** (1.70x) | best single-stream win |
| concurrency 4 vs 1 | 18.7 to **38.2 tok/s** aggregate (2.04x) | best free win |
| shorter prompts | about 4.5s per 1k tokens | biggest latency lever |
| keep the server warm | avoids a 2.5s model load per call | free |
| thinking off | mlx-vlm's default; mistral.rs had it on, burning 2631-4556 tokens | free |
| quantized KV (8/4-bit) | 20.8 / 20.0 / 21.1 tok/s at 7k | **no effect** |
| newer mlx (0.32.1) | 20.69 versus 20.84 | **no effect** |

Concurrency deserves a sentence of explanation, because it looks like magic and is
not. Decode reads all the weights per step no matter the batch size, so four
concurrent requests amortize one pass over the weights across four streams. Total
throughput goes up 2.04x. Each individual request gets slower, around 9.5 tok/s.
Hit the server concurrently, not serially. I have not tested 8.

Beyond caching, the only real prefill wins left are sending fewer tokens (context
curation, or LLMLingua-2-style learned pruning for a lossy 2-5x reduction),
response-level semantic caching in the GPTCache style, and quantizing the BF16
vision tower if you send images. The publisher's 3197 tok/s prefill figure is an
M5 Max with roughly 40 GPU cores. That 14x is hardware. You cannot code your way
to it.

## Prompt caching, or how to make turn two 57x faster

This is the highest-value section in this file.

mlx-vlm ships an automatic prefix cache. It is off by default, and its defaults do
not work on a 16 GB Mac. `serve.sh` turns it on and fixes the defaults:

```
APC_ENABLED=1                     # off upstream
APC_BLOCK_SIZE=16 APC_NUM_BLOCKS=2048        # 32768 tokens
APC_DISK_PATH=./.apc-cache APC_DISK_MAX_GB=8 # survives restarts, ~160 MiB per 4.3k snapshot
APC_DISK_MIN_FREE_RAM_GB=0.5      # default 2.0, compared against psutil "available",
                                  # which reads ~1.7 GB on macOS even at 74% free, so
                                  # every warm restore was silently skipped
APC_EXACT_CACHE_ENTRIES=8         # default 2; in-RAM snapshots are the ONLY place
                                  # prefix matching happens, disk is exact-key only
APC_EXACT_PREFIX_GUARD_TOKENS=16  # default; why a hit reports len-16 cached tokens
APC_TRACE=1                       # logs every store and lookup decision
```

Measured on a 4,312-token system prompt: turn 1 costs 19.42s of prefill cold,
turns 2 through 4 cost 0.35s, 0.34s and 0.31s at 99 to 100 percent cached. That is
57x on prefill and roughly 30x on end-to-end latency for every turn after the
first.

**Now the limitation, because it changes how you structure your application.** The
cache runs in exact mode, not block mode. `apc.model_apc_mode()` returns `"exact"`
because of those gated-DeltaNet layers. Quoting the source: "recurrent state
cannot be reconstructed by concatenating K/V blocks alone." The cache layout
confirms it, `ArraysCache` three times then `KVCache`, repeating, exactly matching
`full_attention_interval: 4`.

So there is no arbitrary-prefix reuse. There are only whole-prompt snapshots,
reused as prefixes of longer prompts. Which means: independent queries that share a
big system prompt will never hit the cache, and warming with a bare prefix does not
help, because the template terminates it with `<|im_end|><|im_start|>assistant`.

Run that workload as one growing conversation instead. Append-only is the shape
this cache rewards.

### The two bugs in `patch_mlx_vlm.py`

Both had to be fixed. Fixing either one alone still gives you `cached=0`, which is
what made this take a day instead of an hour. The script is idempotent, and **you
must re-run it after any mlx-vlm reinstall or model re-download** or your prefill
silently regresses by 57x.

**Bug one: `inputs_embeds` in the cache key.** In `generate/ar.py:_apc_extra_hash`,
the salt folded in `inputs_embeds`. The server always supplies those, and they are
a function of the entire prompt, so every distinct prompt produced a distinct cache
key. Prefix reuse was impossible by construction, not by accident. Instrumenting
`lookup_exact_cache` proved it: candidates were rejected with `reason=extra_hash`
and never on a token mismatch. The salt exists to separate media payloads that
share token ids, so the fix omits it for text-only requests and keeps the
conservative key whenever pixels or audio are present.

**Bug two: `<think>` asymmetry in the chat template.** Under
`enable_thinking=false` the generation prompt ends with `<think>\n\n</think>\n\n`,
but assistant turns already in the history render as plain content, because the
`loop.index0 > ns.last_query_index` gate keeps that scaffold only for the most
recent turn. Turn N's prompt was therefore not a prefix of turn N+1's. They
diverged at exactly that token. The fix gives history turns the same scaffold, so
prompts are strictly append-only.

One is an upstream bug and one is a packaging choice. Neither announced itself.
The cache reported zero hits and no errors, which is the worst failure mode a
cache has.

## Speculative decoding: 1.70x, and it is lossless

I wrote in an earlier version of this file that speculative decoding had nothing
to offer here. I was wrong, and the way I was wrong is instructive: I reasoned
about prefill, where drafting genuinely does not help because the prompt is
already processed in parallel, and then applied that conclusion to decode, where
the bound is bytes moved rather than work done. A drafter proposes a block, the
target verifies the whole block in one pass, and every accepted token in that
block shares one read of the weights. The bandwidth ceiling does not move. The
tokens you get per byte does.

Same prompt throughout, 100 tokens out, 86 percent draft acceptance:

| configuration | tok/s | vs baseline | bit-exact vs target |
|---|---|---|---|
| target only (mlx_lm, greedy) | 21.4 | 1.00x | reference |
| DFlash, verify_linear off | **36.4** | **1.70x** | **yes** |
| DFlash, verify_linear on + qmm on | 42.9 | 2.01x | **no** (rel 4.3e-3) |
| DFlash, verify_linear on + qmm off | 36.4 | 1.70x | yes |

Losslessness was verified, not assumed. With `verify_linear` off, the output is
byte-identical to `mlx_lm generate --temp 0` on the same prompt.

Peak memory is 5.99 GB against 5.20 GB for the target alone, so a 2.4 GiB BF16
drafter costs 0.79 GB. mlx-lm loads lazily and quantizes per tensor, so the BF16
copy never exists whole. On an 11.5 GB budget, that is the difference between
running and not running.

The speedup does not decay with context here, 1.51x at 1865 prompt tokens versus
1.48x at 23. The published M5 Max curve falls from 4.37x to 2.22x, but that is
because it starts far higher. Target-only under mlx-lm measured 20.1-21.4 tok/s
decode and 226.9 tok/s prefill, which independently reproduces both ceilings above
on a second runtime.

Why this particular drafter fits: `z-lab/Qwen3.5-9B-DFlash` is hidden 4096,
intermediate 12288, `num_target_layers 32`, vocab 248320, and its `fc.weight` is
`[4096, 32768]`, fusing hidden states from 8 target layers `[1,5,9,13,17,21,25,29]`.
`bind_target_model` asserts no dimensions at all, it only reads `embed_scale`, so
that concatenation width is the entire contract between the two models. The drafter
ships no `embed_tokens` and no `lm_head` and borrows the target's, which means
draft logits come out of this abliterated head rather than a stock one.

### Setup

```
dflash/setup.sh
```

That builds a separate Python 3.13 venv (dflash-mlx depends on **mlx-lm**, not
mlx-vlm), fetches the drafter with curl because huggingface_hub died silently
mid-transfer twice, and applies the two patches below. It does not touch `.venv/`
or `serve.sh`.

Then run it:

```
dflash/run.sh "explain unified memory in two sentences"
```

`run.sh` sources the same `_guard.sh` the server does, which matters more here
than it looks: this is a second runtime holding roughly 6 GB in its own venv, and
without the guard nothing stops you starting it beside a live `serve.sh`. The raw
CLI invocation is in the script header if you need to vary the flags.

Model loading needed no porting at all, which was a pleasant surprise. mlx-lm's
`qwen3_5.py` already accepts the VLM wrapper: `ModelArgs.from_dict` takes a nested
`text_config`, and `sanitize()` drops `vision_tower` and `model.visual` and
rewrites the `language_model.` prefix. dflash's target adapter is duck-typed,
`"qwen" in model_type` plus a shape check, so it resolves to `qwen_gdn` /
`hybrid_gdn` with recurrent rollback.

What did need a patch:

1. **`dflash/shim_draft_config.py`** — dflash-mlx 0.1.8 reads `rope_theta` and
   `block_size` from the config root, and z-lab publishes them nested
   (`rope_parameters.rope_theta` per transformers 5.x, `block_size` inside
   `dflash_config`), so `from_dict` raises `TypeError: missing 2 required
   positional arguments`. Pure schema drift. The shim lifts both.
2. **`dflash/patch_dflash_verify_linear.py`** — optional. Read the next section
   before you use it.

### verify_linear buys 18 percent and costs you bit-exactness

dflash-mlx gates its hand-written Metal verify kernels on
`_supports_verify_linear`, which for dense models means `num_layers >= 40`. This
model has 32, so it is excluded on layer count alone, not on capability. The
per-linear gate `is_verify_eligible()` accepts **248 of 249** QuantizedLinears
here (the single rejection is `lm_head`, excluded on purpose by `N < 100_000`),
and `verify_linear._PROJ_TAGS` already carries explicit `gdn_qkv` / `gdn_z` /
`gdn_o` tags for gated-DeltaNet projections. The threshold is validation policy,
not a statement about this model.

The patch does not invent a new threshold. The package already ships
`DFLASH_VERIFY_LINEAR`, but `loading.py` computes `supports_verify_linear AND
_verify_enabled_for(...)` and the environment variable only feeds the right
operand, so it could disable the feature and never enable it. The patch consults
the override at the top of `_supports_verify_linear`, and makes `DFLASH_VERIFY_QMM`
a tri-state, because `VerifyConfig.enable_qmm` defaults to True with no CLI flag
and the qmm path was otherwise unavoidable. Unset, upstream behavior is unchanged.
Both edits are idempotent and write `.orig` backups.

**Now the part you actually need to know.** Swapping all 248 linears buys nothing
by itself: 36.4 tok/s either way. The entire 18 percent comes from one kernel, the
M=16 `qmm` path, and that kernel is exactly the one that is not bit-exact.
Isolated per layer on `mlp.gate_proj`, stock versus `VerifyQuantizedLinear` is
identical at n=1, 8 and 16 with qmm off, and at n=1 and 8 with qmm on, diverging
**only at n=16** — which is the DFlash block size, so it perturbs the verifier on
every single block (`_build_kernel_m16_super_tree_fp16_ktmpl`, fp16 accumulation).

So this is a speed-for-numerics trade, not free performance. Default is off. You
get 1.70x while staying byte-identical to the reference target. Turn it on and
your verifier is no longer numerically the same model as your target, and
"verified by the target" stops meaning what it means everywhere else in this file.

```
DFLASH_VERIFY_LINEAR=1                      # 2.01x, NOT bit-exact
DFLASH_VERIFY_LINEAR=1 DFLASH_VERIFY_QMM=0  # bit-exact, but no gain (36.4)
```

For what it is worth, the drafter already uses the verify kernels unconditionally
at w4 (`load_draft_bundle` installs them with `enable_qmm=True` regardless of
gating). That is harmless, because draft errors get corrected by verification.
Applying them to the *target* is the part that changes the guarantee.

### Why this is not the default server

Because leaving mlx-vlm costs four things at once, and for most real traffic they
add up to more than 1.70x:

- **No vision tower.** dflash-mlx builds on mlx-lm. No image input.
- **No continuous batching.** The 2.04x concurrency win disappears. mlx-vlm at
  38.2 tok/s aggregate still beats DFlash at 36.4 single-stream for concurrent
  agent traffic. DFlash wins interactive single-stream use.
- **The two APC fixes do not come along**, so prefill goes back to cold cost. You
  are trading a 57x for a 1.70x.
- **Thinking is on by default** in this runtime, unlike mlx-vlm, so a real task
  burns reasoning tokens before it answers.

Use DFlash when you are one human at a terminal. Use `serve.sh` when you are
serving anything else.

### Do not try to port a 27B drafter

Both the 27B MTP head and `incoai/Qwen3.8-27B-DFlash2` are shaped to their host's
residual stream. The DFlash2 checkpoint is 5120-wide throughout with `fc.weight`
`[5120, 25600]`, fusing 5 of the 27B's 64 layers, and ships no embeddings of its
own. Nothing lines up with 4096 / 12288 / 32 except the vocabulary.

The 9B does have its own MTP head in `empero-ai/Qwen3.8-9B` (15 tensors, 0.49 GB
bf16, fetchable by HTTP range read). It remains unmeasured, and now that DFlash
carries decode past the roof losslessly, it is an increment on a solved axis
rather than the open question it used to be.

## Driving it from Pi

Pi (`@earendil-works/pi-coding-agent`) reads custom OpenAI-compatible providers
from `~/.pi/agent/models.json`. No extension needed; `pi.registerProvider()` is
only for custom auth or streaming.

```json
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
```

Leave `./serve.sh` running, then `pi`, `/login qwen38-local` (any value), `/model`.
Or skip the login prompt:

```
pi --api-key local --provider qwen38-local \
   --model models/Qwen3.8-9B-Abliterated-MLX/4bit
```

Four fields there are load-bearing:

- `id` is the literal served path. Aliases are rejected.
- `supportsDeveloperRole: false`, because mlx-vlm's server does not accept the
  `developer` role Pi sends for reasoning-capable models.
- `reasoning: false` matches the server's `enable_thinking=False`. You can set
  `reasoning: true` plus `compat.thinkingFormat: "qwen-chat-template"`, but that
  changes the rendered prompt and invalidates every APC snapshot you have.
- `contextWindow: 32768` matches the APC budget. Larger just evicts cache.

`/login` is required even though the key is ignored, because Pi hides models with
no stored credential.

Set your expectations honestly: this will feel slow for coding. 20.8 tok/s decode,
about 4.5s per 1k prompt tokens, and Pi is single-stream so the concurrency win
does not apply. The compensation is that Pi grows one conversation, which is
exactly the append-only shape exact mode rewards, so turn 2 onward prefills in
about 0.3s.

## Provenance

All 14 files of the 4-bit variant are verified against the publisher's SHA-256
`artifact-manifest.json`, at pinned revision `1836724...`. The download is
unmodified except for `chat_template.jinja`, which carries fix two above.

## Appendix: the mistral.rs path, and why I abandoned it

I assumed a compiled Rust serving stack would beat Python. It did not: 15.2 tok/s
decode against 20.8, and 109 tok/s prefill against 217. Both stacks dispatch the
same Metal kernels, and at 9B the interpreter is not on the critical path. That
assumption cost me the most time of anything in this repository, so it goes first.

The source tree, the patched model copy, `prepare_mistralrs.py`, the MTP scripts
and the MTP shards are all deleted. Here is what was worth keeping:

- **Three weight transforms** are needed to run an MLX-converted checkpoint under
  mistral.rs, which is written against the source HF layout. (1) Conv weights are
  channels-last: MLX stores `(out, *kernel, in)`, candle wants `(out, in, *kernel)`,
  25 tensors, and it fails loudly with a shape error. (2) **RMSNorm weights are
  absolute, not offset**, 81 tensors, and *this is the one that silently produces
  garbage.* mlx-vlm bakes `+1.0` into its `NORM_WEIGHT_SUFFIXES` families;
  mistral.rs builds the same five with `GemmaRmsNorm`, whose constructor adds 1.0
  again (`layers.rs:489`), so the model computes `2+w`, loads cleanly, and emits
  fluent multilingual noise. Subtract 1.0. Note that `linear_attn.norm` and the
  vision LayerNorms are *not* offset by mlx-vlm and must be left alone. (3)
  `lm_head` is nested at `language_model.lm_head` and mistral.rs reads it from the
  root varbuilder (`text.rs:458,581`), so that one is a rename. Transforms 1 and 3
  are arguably mistral.rs bugs, since it already detects mlx-vlm naming
  (`mod.rs:61`, `text.rs:436`). Transform 2 is genuine ambiguity: the checkpoint
  does not record which convention its norms use.
- `-n "0:32"` is needed to bypass the auto device mapper, which sizes
  AFQ-prequantized layers as if they were BF16 and refuses to load.
- `mtp.rs` and `speculative.rs` for qwen3_5 exist **only on master**, not at tag
  v0.9.1. Build with `MISTRALRS_METAL_PRECOMPILE=0 cargo build --release -p
  mistralrs-cli --features metal -j 3` (3m12s). Use `-j 3` and `nice`; a wide
  parallel link will exhaust 16 GB.
- Xcode 26.3 ships `metal` as a **stub**, so ahead-of-time kernel compilation
  fails with "cannot execute tool 'metal' due to missing Metal Toolchain".
  `PRECOMPILE=0` writes zero-byte metallibs and compiles at runtime, which works
  for mistralrs-quant but **not** for paged attention, where those sources hit
  `error: 'function_constant' has a duplicate index '10'`. Getting further needs
  `xcodebuild -downloadComponent MetalToolchain`.
- MTP requires PagedAttention, which on Metal hit `Failed to create metal resource:
  Buffer`. Root cause: hybrid models request **0 KV blocks** for their recurrent
  layers, 24 of 32 here, and `cache_engine.rs` then calls
  `dev.new_private_buffer(0, ...)`. Metal refuses a zero-length buffer where CUDA
  tolerates it. Five one-word edits fix it, `elem_count` to `elem_count.max(1)`.
  Worth upstreaming.
- mistral.rs only loads weight files matching `model-\d+-of-\d+\.safetensors`
  (`pipeline/paths.rs:27`). An extra shard under any other name is **silently
  ignored**. Its local-dir path globs `*.safetensors` instead
  (`pipeline/paths.rs:196`), which is how you get two different behaviors from
  what looks like one loader.
- Every MTP precondition held: `mtp_num_hidden_layers == 1`,
  `mtp_use_dedicated_embeddings == false`, and `mtp.fc` shaped `2*hidden -> hidden`
  (`[4096, 8192]`). Enable with `--mtp` plus `--mtp-n-predict N`; `--mtp-model` is
  for a separate assistant model instead.

If you take one thing from this appendix, take this: the norm bug loaded without
an error and produced confident, grammatical output in several languages. Silent
correctness failures are the expensive ones. Measure output against a known-good
runtime before you measure its speed.
