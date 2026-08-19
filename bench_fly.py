#!/usr/bin/env python3
"""Measure the three levers that actually matter for mlx-vlm throughput.

Single-stream decode is pinned at the memory-bandwidth roof (~21 tok/s), so the
wins are elsewhere: not emitting tokens you don't need (thinking), not
recomputing prefill (prompt cache), and amortizing weight reads across
concurrent requests (continuous batching).
"""
import json, sys, threading, time, urllib.request

BASE = "http://127.0.0.1:8080/v1"
MODEL = "models/Qwen3.8-9B-Abliterated-MLX/4bit"

def post(path, payload, timeout=1200):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.load(r)
    return body, time.perf_counter() - t0

def chat(msg, max_tokens=200, **kw):
    p = {"model": MODEL, "messages": [{"role": "user", "content": msg}],
         "max_tokens": max_tokens, "temperature": 0}
    p.update(kw)
    return post("/chat/completions", p)

def stats(b):
    u = b.get("usage", {}) or {}
    t = b.get("timings", {}) or {}
    return dict(
        prompt=u.get("prompt_tokens"),
        cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        out=u.get("completion_tokens"),
        prefill_tps=t.get("prompt_per_second"),
        decode_tps=t.get("predicted_per_second"),
        prompt_ms=t.get("prompt_ms"),
    )

Q = "What is 17 * 23? Answer with just the number."
print("=" * 72)
print("LEVER 1 — thinking mode: same question, reasoning on vs off")
print("=" * 72)
for label, kw in [("thinking OFF", {"chat_template_kwargs": {"enable_thinking": False}}),
                  ("thinking ON",  {"chat_template_kwargs": {"enable_thinking": True}})]:
    try:
        b, wall = chat(Q, max_tokens=800, **kw)
        s = stats(b)
        txt = b["choices"][0]["message"]["content"] or ""
        print(f"  {label:13s} wall={wall:6.2f}s  out_tokens={s['out']:4d}  "
              f"decode={s['decode_tps'] or 0:5.1f} t/s   answer={txt.strip()[:40]!r}")
    except Exception as e:
        print(f"  {label:13s} FAILED: {e}")

print()
print("=" * 72)
print("LEVER 2 — prompt cache: identical long prefix sent twice")
print("=" * 72)
long_ctx = ("You are a careful assistant. Reference material follows.\n"
            + "Fact %d: the value of item %d is %d.\n" * 1 * 0)
long_ctx = "You are a careful assistant. Reference material follows.\n" + \
           "".join(f"Fact {i}: item {i} has value {i*7 % 97}.\n" for i in range(900))
for attempt in (1, 2):
    try:
        b, wall = chat(long_ctx + "\nWhat is the value of item 42? Answer briefly.",
                       max_tokens=30, chat_template_kwargs={"enable_thinking": False})
        s = stats(b)
        print(f"  pass {attempt}: prompt={s['prompt']:6d} cached={s['cached']:6d}  "
              f"prefill={s['prefill_tps'] or 0:7.1f} t/s  prefill_time={(s['prompt_ms'] or 0)/1000:5.2f}s  wall={wall:5.2f}s")
    except Exception as e:
        print(f"  pass {attempt}: FAILED: {e}")

print()
print("=" * 72)
print("LEVER 3 — continuous batching: N concurrent requests")
print("=" * 72)
def worker(i, out):
    try:
        b, wall = chat(f"Write exactly two sentences about the number {i}.",
                       max_tokens=120, chat_template_kwargs={"enable_thinking": False})
        s = stats(b)
        out[i] = (s["out"], wall, s["decode_tps"])
    except Exception as e:
        out[i] = (None, None, None)
        print(f"    worker {i} failed: {e}")

for n in (1, 4):
    res = {}
    ts = [threading.Thread(target=worker, args=(i, res)) for i in range(n)]
    t0 = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    toks = sum(v[0] or 0 for v in res.values())
    print(f"  concurrency {n}: {toks:4d} tokens in {wall:6.2f}s  "
          f"=> aggregate {toks/wall:6.2f} tok/s")
