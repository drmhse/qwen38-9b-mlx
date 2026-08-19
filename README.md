# Qwen3.8-9B-Abliterated on a 16 GB M4

Local inference for `PocketAiHub/Qwen3.8-9B-Abliterated-MLX` (4-bit variant).

## Naming: 3.8 vs 3.5

`Qwen3.8-9B` names the **weights**; `qwen3_5` names the **architecture**.
`config.json` declares `model_type: "qwen3_5"` / `Qwen3_5ForConditionalGeneration`,
and the manifests record it as a third-party full-parameter *distillation* of
`Qwen/Qwen3.5-9B` — explicitly "not an official Qwen3.8 release". So runtime code
paths named `qwen3_5` are the correct ones, but stock-Qwen3.5 assumptions are not
safe (see the norm issue below).

Architecture: hybrid attention — 24 gated-DeltaNet linear-attention layers plus
full attention every 4th layer (`full_attention_interval: 4`), `attn_output_gate`,
interleaved mrope, and a 27-block SigLIP-style vision tower kept at BF16.

## Files

    serve.sh          OpenAI/Anthropic-compatible server on 127.0.0.1:8080 (APC env baked in)
    ask.sh            one-shot prompt
    chat.sh           interactive
    _guard.sh         sourced by all three: blocks a 2nd instance and low-memory starts
    patch_mlx_vlm.py  re-applies the two caching fixes; RUN AFTER any mlx-vlm reinstall
                      or model re-download, or prefill silently regresses ~57x
    download.py       re-fetch the 4-bit variant at the pinned revision
    bench_fly.py      re-measure the throughput levers (regression check)
    .venv/            mlx 0.32.0 + mlx-vlm 0.6.8 (+ the ar.py fix)
    models/           the 4-bit weights (+ the patched chat_template.jinja)
    .apc-cache/       prompt-cache disk tier; regenerates itself, capped at
                      APC_DISK_MAX_GB (8), ~160 MB per 4.3k-token snapshot. Safe to delete.

The mistral.rs/Rust path, its patched model copy and the MTP shards were removed --
mlx-vlm is faster and correct. See the appendix for what was learned.

## Run it

    ./ask.sh "your prompt"          # one-shot
    ./chat.sh                       # interactive
    ./serve.sh                      # OpenAI-compatible on 127.0.0.1:8080

All three `source ./_guard.sh`, which refuses to start a second instance or to
start at all below ~8 GB reclaimable memory. **Respect it** — see Memory.

Server needs the exact model id, not an alias:

    curl 127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"models/Qwen3.8-9B-Abliterated-MLX/4bit",
           "messages":[{"role":"user","content":"hi"}],"max_tokens":64}'

## Memory: use 4-bit, not 8-bit

This machine has 16 GB unified memory and an **11.5 GB Metal budget**
(`mistralrs doctor`). The 8-bit variant is 9.74 GiB of weights with a ~11.4 GB
measured peak, so it does not fit — confirmed three independent ways: mistral.rs's
device mapper, `mistralrs tune`, and MLX itself raising
`kIOGPUCommandBufferCallbackErrorOutOfMemory`. That GPU OOM hard-crashed the
machine once. 4-bit peaks at 6.7–8.7 GB and is the only sane choice here.

## Speed: you are at the hardware roof

Measured (4-bit, base M4, 4P+6E, 10-core GPU):

| | prefill | decode | peak |
|---|---|---|---|
| mlx-vlm 0.6.8 | 217–229 tok/s @ 2.5–7k ctx | **20.8 tok/s** | 6.7–8.2 GB |
| mistral.rs 0.9.1 (patched) | 109 tok/s | 15.2 tok/s | — |

Decode is memory-bandwidth-bound: every token streams all weights. Measured
achievable bandwidth is **102.5 GB/s** stream / 81 GB/s matvec (of 120 theoretical).
At 20.8 tok/s that is ~4.3–4.9 GB read per token — the language-model weights at
85–100% of achievable bandwidth. **No runtime can beat this**; it is not language
overhead. Confirmed by mlx 0.32.0 vs 0.32.1 being identical (20.84 vs 20.69), and
by quantized KV cache changing nothing (20.8 / 20.0 / 21.1 for bf16 / 8-bit / 4-bit
at 7k context) — KV is not the bottleneck.

The only way past the roof is to read fewer weight-bytes per token:
speculative decoding, a smaller model, or lower precision. See MTP below.

## MTP / speculative decoding

The 27B's MTP head **cannot** be ported: MTP tensors are shaped to their host
model's residual stream (27B is hidden 5120 / intermediate 17408 / 64 layers vs
the 9B's 4096 / 12288 / 32), so `mtp.fc.weight` alone is `[5120,10240]` vs
`[4096,8192]`. It is also trained to predict that specific model's distribution
from its hidden states — not a portable adapter.

But the 9B has its own: `empero-ai/Qwen3.8-9B` contains all 15 `mtp.*` tensors
(0.49 GB bf16); the publisher strips them ("Native source MTP tensors are
intentionally excluded"). safetensors is byte-addressable, so only those ranges
need fetching (~0.5 GB, not 19.3 GB).

Blockers, in order:
- **mlx-vlm** discards them: `models/qwen3_5/qwen3_5.py:148` drops any `mtp.` key,
  and `--draft-kind mtp` is documented as Gemma 4 only. No qwen3_5 MTP module exists.
- **mistral.rs** ships `vision_models/qwen3_5/{mtp,speculative}.rs` — but **only on
  master**, not in the v0.9.1 release binary (there those two files do not exist).
  So this route needs a source build of master. Its loader now works (below), at
  15.2 tok/s base; MTP typically yields 1.5–2x, which would put it ~23–30 tok/s
  and past MLX.
- Caveat: abliteration re-projected `out_proj`/`down_proj` in layers 12–31, so an
  MTP head trained on the un-abliterated residual stream will have a lower
  acceptance rate. Speculative decoding stays **lossless in output quality**
  regardless (drafts are verified by the main model) — a mismatched drafter only
  reduces the speedup.

## mlx-vlm -> mistral.rs conversion (`prepare_mistralrs.py`)

mistral.rs implements qwen3_5 but is written against the **source HF checkpoint**,
which differs from mlx-vlm's converted output in three ways. All three verified
against `empero-ai/Qwen3.8-9B` by HTTP range-reading its safetensors header.

1. **conv weights are channels-last** (25 tensors). MLX stores `(out, *kernel, in)`;
   candle wants `(out, in, *kernel)`. `conv1d [8192,4,1] -> [8192,1,4]`,
   `patch_embed [1152,2,16,16,3] -> [1152,3,2,16,16]`. Symptom: hard shape error.
2. **RMSNorm weights are absolute, not offset** (81 tensors) — *the cause of the
   garbage output*. mlx-vlm bakes `+1.0` into the five families in its
   `NORM_WEIGHT_SUFFIXES`; mistral.rs builds those same five with `GemmaRmsNorm`,
   whose constructor does `weight = original_weight + 1.0` (`layers.rs:489`).
   Uncorrected the model computes `2+w` and emits multilingual noise — it loads
   cleanly and silently produces nonsense. Subtract 1.0 to restore offset form.
   `linear_attn.norm` (24) and the vision LayerNorms are **not** offset by mlx-vlm
   and are byte-identical to source — left alone.
3. **`lm_head` is nested** (3 tensors). mistral.rs detects the body prefix
   (`text.rs:436-450`) but always reads `lm_head` from the root varbuilder
   (`text.rs:458,581`); mlx-vlm puts it at `language_model.lm_head`. Rename only.

Also needed: `-n "0:32"` to pin layers and bypass the auto device mapper, which
sizes AFQ-prequantized layers as if they were BF16 (claims 19169 MB) and refuses
to load. Not a real memory limit — an estimator that ignores on-disk quantization.

Rebuild:

    .venv/bin/python prepare_mistralrs.py \
      models/Qwen3.8-9B-Abliterated-MLX/4bit \
      models/Qwen3.8-9B-Abliterated-mistralrs/4bit

    ./bin/mistralrs run --max-seqs 1 --prefix-cache-n 0 -i "prompt" \
      multimodal -m models/Qwen3.8-9B-Abliterated-mistralrs/4bit -n "0:32" --max-seq-len 4096

Byte lengths are unchanged by all three transforms, so data offsets are preserved
and shards are rewritten by streaming (never 5 GB in RAM). The norm subtraction is
done in float32 and rounded back to bf16 round-to-nearest-even; the effective
scale `(patched + 1.0)` round-trips **bit-exactly** to what mlx-vlm uses.

Upstream: 1 and 3 are arguably mistral.rs bugs for MLX-converted qwen3_5 repos
(it already detects mlx-vlm naming at `mod.rs:61` and `text.rs:436`, so these
repos are meant to work). 2 is a genuine ambiguity — the checkpoint does not record
which form its norms are in.

## Provenance

All 14 files of the 4-bit variant verified against the publisher's SHA-256
`artifact-manifest.json`. Pinned revision `1836724...`. The pristine download is
kept unmodified; `prepare_mistralrs.py` writes a separate tree and hardlinks
what it does not change.


## MTP build-out (in progress)

Everything needed is confirmed present; see `bench_mtp.sh` for the A/B.

**How mistral.rs consumes it.** `qwen3_5/mtp.rs` reads the source tensor names
verbatim under an `mtp.` prefix (`mtp.fc.weight`, `mtp.layers.0.*`, `mtp.norm.weight`,
`mtp.pre_fc_norm_{embedding,hidden}.weight`) and is gated on `mtp.fc.weight` existing,
so adding the tensors switches it on. Its hard preconditions, all satisfied here:
`mtp_num_hidden_layers == 1` (ours: 1), `mtp_use_dedicated_embeddings == false`
(ours: False), `mtp.fc` shaped `2*hidden -> hidden` (ours: `[4096, 8192]`).

Enabled at runtime with `--mtp` (plus optional `--mtp-n-predict N`), which builds
`SpeculativeConfig::Mtp(MtpConfig::builtin(..))` and injects the `_mistralrs_mtp`
top-level config key. `--mtp-model` is for a *separate* assistant model instead.

**Steps done.**
1. `fetch_mtp.py` — pulls the 15 `mtp.*` tensors out of `empero-ai/Qwen3.8-9B` using
   HTTP range reads: 0.49 GB fetched instead of the 19.3 GB file.
2. `quantize_mtp.py` — quantizes the 8 linear tensors to AFQ4/group-64 via
   `mx.quantize` (the same quantizer that produced the checkpoint), emitting MLX's
   weight/scales/biases triplets; leaves the 7 norms as bf16. Per-weight relative
   error ~0.10, normal for 4-bit affine. Output `mtp-afq4.safetensors`, 136.9 MB,
   31 tensors. Packing matches the checkpoint's own convention (in=4096 -> 512 u32).
   The MTP norms are NOT offset by -1.0: mlx-vlm never touched them (it drops
   `mtp.` keys), so the source's offset form is already what `GemmaRmsNorm` wants.
3. Dropped into the model dir as an extra shard — mistral.rs globs `*.safetensors`
   for a local model dir (`pipeline/paths.rs:196`) rather than reading the index.

**Remaining.** Build master with Metal, then A/B. Two build gotchas hit:
- `mtp.rs`/`speculative.rs` are absent at tag v0.9.1 — must build master.
- Xcode 26.3 ships `metal` as a stub; ahead-of-time kernel compilation fails with
  "cannot execute tool 'metal' due to missing Metal Toolchain". Either run
  `xcodebuild -downloadComponent MetalToolchain`, or build and run with
  `MISTRALRS_METAL_PRECOMPILE=0` to compile kernels at runtime instead (no
  offline toolchain needed). Using the latter.
- Build with `-j 3` and `nice`; a wide parallel link can exhaust 16 GB.

**Expected ceiling.** MTP reduces weight-passes per accepted token, which is the
only way past the 20.8 tok/s bandwidth roof. Abliteration re-projected
`out_proj`/`down_proj` in layers 12-31 while this MTP head was trained on the
un-abliterated residual stream, so acceptance will be below ideal. Speculative
decoding remains **lossless in output quality** either way — drafts are verified
by the target model, so a mismatched drafter costs speed, never correctness.


## Making mlx-vlm fly: what actually moves the needle

All measured on this machine, 4-bit, base M4.

| lever | effect | verdict |
|---|---|---|
| concurrency 4 vs 1 | 18.7 -> **38.2 tok/s** aggregate (2.04x) | **best free win** |
| APC on identical prompt | prefill 18.9s -> **0.22s** (86x) | huge, but narrow (see below) |
| keep server warm | avoids 2.5s model load per call | free |
| thinking off | mlx-vlm default; mistral.rs had it ON and burned 2631-4556 tokens answering a one-paragraph question | free |
| quantized KV (8/4-bit) | 20.8 / 20.0 / 21.1 tok/s at 7k ctx | **no effect** |
| newer mlx (0.32.1) | 20.69 vs 20.84 tok/s | **no effect** |
| shorter prompts | prefill is ~200-225 tok/s, so ~4.5s per 1k tokens | biggest latency lever |

**Decode cannot be improved.** 20.8 tok/s is ~85-100% of measured achievable bandwidth
(102.5 GB/s stream, 81 GB/s matvec). Nothing in software gets past it.

**Prefill dominates agent latency, not decode.** A 4k prompt costs ~19s; 16k costs ~80s.
Generating 40 tokens costs ~2s. So prompt size is the thing to optimise.

**Concurrency is the real multiplier.** Decode reads all weights per step regardless of
batch size, so N concurrent requests amortise one weight pass. 4x concurrency gave 2.04x
aggregate. Use `/v1/chat/completions` concurrently rather than serially. (8x untested.)

### Prompt caching: big but narrow on this architecture

Off by default. `serve.sh` now enables it:

    APC_ENABLED=1 APC_BLOCK_SIZE=16 APC_NUM_BLOCKS=2048   # 32768 tokens
    APC_DISK_PATH=./.apc-cache APC_DISK_MAX_GB=8          # survives restarts

Verified: an identical 4181-token prompt went 18.9s -> **0.22s** prefill (4165/4181 tokens
served from cache, restored from the disk tier).

Two hard limits found:
- **Exact mode only.** `apc.model_apc_mode()` returns `"exact"` rather than `"block"` for
  this model, because the gated-DeltaNet layers hold recurrent state and, quoting the
  source, *"recurrent state cannot be reconstructed by concatenating K/V blocks alone."*
  The self-check confirms the layout: `ArraysCache`(checkpoint) x3 then `KVCache`(pageable),
  repeating - matching `full_attention_interval: 4`. So there is **no arbitrary-prefix
  reuse**; only whole-prompt snapshots reused as prefixes of longer prompts.
- **RAM gate.** Restores are skipped when psutil's `available` is under
  `APC_DISK_MIN_FREE_RAM_GB` (default 2.0). On macOS that reads ~1.7 GB even at 74% free,
  so the default silently disables warm restores here: `APC: skipping exact disk restore
  (free RAM 1.7 GB < 2.0 GB)`. Lower it (0.5 works) or the cache never restores. Note a
  16k-token snapshot is ~573 MB, so this gate is not pure paranoia.

**FIXED — multi-turn caching now works.** See "Two caching bugs" below. Measured on a
~4.3k-token system prompt:

| turn | prompt | cached | prefill | wall |
|---|---|---|---|---|
| 1 (cold) | 4312 | 0 | 19.42s | 20.10s |
| 2 | 4335 | 4312 (99%) | **0.35s** | 0.69s |
| 3 | 4359 | 4335 (99%) | **0.34s** | 0.68s |
| 4 | 4380 | 4359 (100%) | **0.31s** | 2.77s |

**57x faster prefill and ~30x lower latency on every turn after the first.** Run
`patch_mlx_vlm.py` to (re-)apply the fixes; it is idempotent and must be re-run after
any `mlx-vlm` reinstall or model re-download.

Still not reusable: *independent* queries that merely share a prefix (same system prompt,
different single question, no conversation history). Exact mode needs the new prompt to
extend a previously-seen prompt, and no snapshot exists at the shared boundary. Warming
with a bare-prefix request does not help either, because the template terminates it with
`<|im_end|><|im_start|>assistant`. For that pattern, run it as one growing conversation.

## Two caching bugs (both fixed locally by `patch_mlx_vlm.py`)

**1. `inputs_embeds` in the APC salt (mlx-vlm bug, `generate/ar.py:_apc_extra_hash`).**
The salt folded in `inputs_embeds` and `attention_mask`. The server *always* supplies
inputs_embeds (`server/generation.py`: "BatchGenerator requires inputs_embeds"), and
embeddings are a function of the whole prompt -- so every distinct prompt produced a
distinct cache key and **prefix reuse was impossible by construction**. Only
byte-identical prompts could ever hit, which is exactly what the symptoms showed.
Instrumenting `lookup_exact_cache` proved it: candidates were rejected with
`reason=extra_hash`, never on token mismatch. The salt exists to separate media payloads
that share token ids; for text-only requests the embeddings are derived deterministically
from those ids, so they add nothing. Fix omits them for text-only and keeps the
conservative salt whenever media is present.

**2. `<think>` asymmetry in the chat template (model-packaging issue).**
Under `enable_thinking=false` the generation prompt ends with `<think>\n\n</think>\n\n`,
but history assistant turns rendered as plain `content` -- the `loop.index0 >
ns.last_query_index` gate keeps the scaffold only on the most recent turn. So turn N's
prompt was **not** a prefix of turn N+1's; they diverged exactly at the `<think>` token.
Verified at the token level, then fixed by giving history turns the same scaffold, which
makes prompts strictly append-only. (Confirmed the server does use
`enable_thinking=False`: its reported 4312 prompt tokens match that branch exactly, vs
4310 for thinking-on.)

Both were needed -- fixing either alone still yields `cached=0`.

### Cache tuning that matters

`serve.sh` sets these; the upstream defaults do not work on a 16 GB Mac:

    APC_ENABLED=1                     # off by default
    APC_DISK_MIN_FREE_RAM_GB=0.5      # default 2.0 vs psutil "available" ~1.7GB -> restores
                                      # silently skipped ("APC: skipping exact disk restore")
    APC_EXACT_CACHE_ENTRIES=8         # in-RAM snapshots are the ONLY place prefix matching
                                      # happens; disk is exact-key only. Default 2.
    APC_EXACT_PREFIX_GUARD_TOKENS=16  # (default) why a hit reports len-16 cached tokens
    APC_DISK_PATH=./.apc-cache        # persists across restarts; ~160 MB per 4.3k snapshot
    APC_TRACE=1                       # logs store/lookup decisions -- how this was debugged

### API surface (good for agent use)

Three dialects on one server: `/v1/chat/completions` (OpenAI), `/v1/responses`
(OpenAI Responses, stateful, with cancel + input_items), and `/v1/messages` +
`/v1/messages/count_tokens` (**Anthropic Messages**). Plus `/v1/cache/{stats,reset}`,
`/health`, `/metrics`, `/unload`, audio and image endpoints. Request support includes
`tools`, `tool_choice`, `response_format`/`json_schema`, `stream`, `logprobs`,
`top_logprobs`, `seed`, `stop`, and `image_url`. Backend reports
`continuous_batching`. Requests must name the model exactly
(`models/Qwen3.8-9B-Abliterated-MLX/4bit`), not an alias.


### Driving it from Pi (`@earendil-works/pi-coding-agent`)

Pi reads custom OpenAI-compatible providers from `~/.pi/agent/models.json` (no
extension needed -- `pi.registerProvider()` is only for custom auth or streaming):

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

Then leave `./serve.sh` running and start `pi`; `/login qwen38-local` (any value),
then `/model`. Or skip the login prompt entirely:

    pi --api-key local --provider qwen38-local \
       --model models/Qwen3.8-9B-Abliterated-MLX/4bit

Why each field is what it is:

- **`id` is the literal served path.** The server rejects aliases (see above).
- **`supportsDeveloperRole: false`** -- mlx-vlm's server does not accept the
  `developer` role that Pi sends for reasoning-capable models; the system prompt
  has to go as `system`.
- **`reasoning: false`** matches the server, which runs `enable_thinking=False`.
  To turn thinking on, set `reasoning: true` plus
  `compat.thinkingFormat: "qwen-chat-template"` -- that is Pi's variant for local
  Qwen servers that read `chat_template_kwargs.enable_thinking`. Note this changes
  the rendered prompt, so it invalidates existing APC snapshots.
- **`contextWindow: 32768`** matches the APC budget
  (`APC_BLOCK_SIZE 16 x APC_NUM_BLOCKS 2048`). Larger just evicts cache.
- **`/login` is required even though the key is ignored** -- Pi hides models with
  no stored credential for their provider.

Expect it to feel slow for coding work: 20.8 tok/s decode and ~4.5s per 1k prompt
tokens, and Pi is single-stream, so the 2.04x concurrency win does not apply. The
compensation is that Pi grows one conversation, which is exactly the append-only
shape APC exact mode needs -- turn 2 onward prefills in ~0.3s.


## Prefill: there is no software headroom left

Prefill is compute-bound, so the question is what fraction of the GPU's arithmetic
throughput it already uses. Measured on this 10-core M4 GPU:

    bf16 dense GEMM      3.73 TFLOP/s   (peak, 4096^3 square: 3.62)
    afq4 quantized GEMM  3.29 TFLOP/s   = 88% of dense
    (flat for M = 1024, 2048, 4096 -- no shape sensitivity above M~1024)

Model's prefill arithmetic, counted from real tensor shapes (language model only;
vision tower idle for text, embed_tokens is a lookup):

    mlp 4.832B + linear_attn 1.617B + self_attn 0.470B  = 6.919B
    (+ lm_head 1.017B, but that runs on the last position only)
    => 13.8-15.9 GFLOP per prefill token

Against measured prefill rates:

    224 tok/s @ 2.5k ctx  ->  3.10-3.56 TFLOP/s   = 94-108% of the afq4 GEMM ceiling
    217 tok/s @ 7k        ->  3.00-3.44
    199 tok/s @ 16k       ->  2.75-3.15

**Prefill is running at the GPU's arithmetic roof.** No kernel flag, compile pass, or
step-size tweak can help, because the multiply-accumulates themselves are the limit.

Three consequences worth knowing, all following from the numbers above:

- **`--prefill-step-size` tuning is pointless.** GEMM throughput is flat from M=1024
  upward, so the 2048 default is already optimal.
- **Dequantize-then-dense-GEMM won't help.** afq4 matmul is already at 88% of dense bf16,
  so the theoretical best case is 1.13x before paying dequant cost and memory.
- **Attention tricks are irrelevant here.** Going 2.5k -> 16k only cost 224 -> 199 tok/s,
  because just 8 of 32 layers are full attention (`full_attention_interval: 4`). The
  hybrid design already made attention cheap, so FlashAttention variants, sparse
  attention, and attention sinks have almost nothing to win.
- **Batching does not help prefill.** It gives ~2x on *decode* because decode is
  bandwidth-bound and batching amortises one weight pass. Prefill already saturates
  compute, so concurrent prefills split the same FLOPs rather than adding throughput.

### What genuinely speeds up prefill

Only two categories: skip the work, or do less of it.

1. **Skip it — snapshot caching.** 18.9s -> 0.22s (86x) measured. The single biggest lever
   by an order of magnitude. Constrained to exact mode on this architecture (see above).
2. **Do less — fewer tokens.** Prefill is linear in token count at ~4.5s per 1k tokens, so
   token count *is* latency. Prompt compression (LLMLingua-2-style learned token pruning,
   2-5x reduction, lossy) is the main modern technique that actually reduces prefill FLOPs.
   Plain context curation gets much of the same win for free.
3. **Response-level / semantic caching** (GPTCache-style): skip inference entirely for
   near-duplicate queries. Architecture-independent, so it sidesteps the exact-mode limit
   that constrains APC.
4. **Quantize the vision tower** (currently BF16) if you send images -- images also inflate
   token counts sharply, so this is where multimodal prefill cost actually lives.
5. **Smaller model / faster GPU.** The publisher's 3197 tok/s prefill was an M5 Max
   (~40 GPU cores); that 14x is hardware, not software.

### Explicitly NOT prefill wins (common misconceptions)

- **Speculative decoding in every form** -- MTP, dflash, eagle3 -- is **decode-only**.
  It proposes future tokens; prefill already processes all prompt tokens in parallel.
  Zero prefill benefit.
- **Lower-bit weights (3-bit, 2-bit)** help *decode* (bandwidth) but not prefill (compute),
  since the afq4 GEMM is already near dense throughput.
- **`mx.compile`** fuses elementwise/small ops; GEMM-bound work is unaffected.
- **Quantized KV cache** -- measured no effect at all (20.8 / 20.0 / 21.1 tok/s).


## Appendix: the mistral.rs / MTP path (removed, recorded here)

Abandoned because mlx-vlm is both faster (20.8 vs 17.4 tok/s decode) and correct, and
because MTP is decode-only and so cannot touch the prefill cost that dominates. The
source tree, patched model copy and MTP shards were deleted; this is what was learned,
should anyone revisit it.

- `mtp.rs` / `speculative.rs` for qwen3_5 exist **only on master**, not at tag v0.9.1.
  Build: `MISTRALRS_METAL_PRECOMPILE=0 cargo build --release -p mistralrs-cli
  --features metal -j 3` (3m12s). Use `-j 3` and `nice`; a wide parallel link can
  exhaust 16 GB.
- Xcode 26.3 ships `metal` as a **stub**; ahead-of-time kernel compilation fails with
  "cannot execute tool 'metal' due to missing Metal Toolchain". `PRECOMPILE=0` writes
  0-byte metallibs and compiles kernels at runtime instead, which works for
  mistralrs-quant (the model ran) but **not** for paged-attention: runtime compilation
  of those sources collides with `error: 'function_constant' has a duplicate index '10'`.
  Getting further needs `xcodebuild -downloadComponent MetalToolchain`.
- MTP requires PagedAttention, which on Metal hit `Failed to create metal resource:
  Buffer`. Root cause: hybrid models request **0 KV blocks** for linear/recurrent layers
  (24 of 32 here), and `cache_engine.rs` then calls `dev.new_private_buffer(0, ...)` --
  Metal refuses a zero-length buffer where CUDA tolerates it. Fixed with five one-word
  edits, `elem_count` -> `elem_count.max(1)`, keeping shape and declared element count at
  zero since those buffers are never read. That got PagedAttention past model load.
  Worth upstreaming.
- mistral.rs only loads weight files matching `model-\d+-of-\d+\.safetensors`
  (`pipeline/paths.rs:27`); an extra shard under any other name is **silently ignored**.
- Enable with `--mtp` (+ `--mtp-n-predict N`), which builds
  `SpeculativeConfig::Mtp(MtpConfig::builtin(..))` and injects `_mistralrs_mtp`.
  Preconditions all held: `mtp_num_hidden_layers == 1`, `mtp_use_dedicated_embeddings ==
  false`, `mtp.fc` shaped `2*hidden -> hidden` (`[4096, 8192]`).
- The 9B's own MTP head lives in `empero-ai/Qwen3.8-9B` (15 tensors, 0.49 GB bf16) and is
  fetchable by HTTP range read rather than pulling the 19.3 GB file. The 27B's head is
  **not** portable: hidden 5120 / intermediate 17408 / 64 layers vs 4096 / 12288 / 32.
- Running the MLX checkpoint under mistral.rs also needed three weight transforms
  (channels-last convs, `-1.0` on the pre-offset norms, root `lm_head`) -- documented
  above; `prepare_mistralrs.py` implemented them and was deleted with the rest.
