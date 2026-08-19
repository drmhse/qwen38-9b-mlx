#!/usr/bin/env python3
"""Re-apply the two local fixes that make prompt caching work. Idempotent.

Both live outside our code (site-packages / the model dir), so a reinstall or a
re-download wipes them -- run this again after either.

FIX 1 (mlx_vlm/generate/ar.py, _apc_extra_hash): `inputs_embeds` and
`attention_mask` were folded into the APC cache salt. The server always supplies
inputs_embeds ("BatchGenerator requires inputs_embeds"), and embeddings are a
function of the whole prompt -- so every distinct prompt got a distinct salt and
prefix reuse was impossible by construction. Only byte-identical prompts could
hit. For text-only requests embeddings are derived deterministically from the
token ids, so they add no information; media requests keep the conservative salt.

FIX 2 (chat_template.jinja): under enable_thinking=false the generation prompt
ends with `<think>\\n\\n</think>\\n\\n`, but history assistant turns rendered
without it, so turn N's prompt was not a prefix of turn N+1's -- they diverged at
the `<think>` token. History now carries the same scaffold, making prompts
strictly append-only.
"""
import sys
from pathlib import Path

VENV = Path(__file__).parent / ".venv/lib/python3.12/site-packages/mlx_vlm"
TPL = Path(__file__).parent / "models/Qwen3.8-9B-Abliterated-MLX/4bit/chat_template.jinja"

OLD1 = """        tenant = prompt_kwargs.get("_apc_tenant")
        return _apc.semantic_extra_hash(
            tenant=tenant,
            image_hash=img,
            media={
                "audio": prompt_kwargs.get("input_features"),
                "video": prompt_kwargs.get("pixel_values_videos"),
                "embeddings": prompt_kwargs.get("inputs_embeds"),
                "masks": prompt_kwargs.get("attention_mask"),
            },
            model=getattr(self, "model", None),
            processor=getattr(self, "processor", None),
        )"""
NEW1 = """        tenant = prompt_kwargs.get("_apc_tenant")
        media = {
            "audio": prompt_kwargs.get("input_features"),
            "video": prompt_kwargs.get("pixel_values_videos"),
        }
        _media_present = (
            prompt_kwargs.get("_apc_image_hash") is not None
            or prompt_kwargs.get("pixel_values") is not None
            or prompt_kwargs.get("pixel_values_videos") is not None
            or prompt_kwargs.get("input_features") is not None
        )
        if _media_present:
            media["embeddings"] = prompt_kwargs.get("inputs_embeds")
            media["masks"] = prompt_kwargs.get("attention_mask")
        return _apc.semantic_extra_hash(
            tenant=tenant,
            image_hash=img,
            media=media,
            model=getattr(self, "model", None),
            processor=getattr(self, "processor", None),
        )"""

OLD2 = """        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- else %}"""
NEW2 = """        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
        {%- elif enable_thinking is defined and enable_thinking is false %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n\\n</think>\\n\\n' + content }}
        {%- else %}"""

def apply(path, old, new, name, marker=None):
    s = path.read_text()
    if (marker in s) if marker else (new in s):
        print(f"  {name}: already applied")
        return True
    if old not in s:
        print(f"  {name}: TARGET NOT FOUND -- upstream changed, re-check manually")
        return False
    path.write_text(s.replace(old, new, 1))
    print(f"  {name}: applied")
    return True

ok = True
print("applying local caching fixes:")
ok &= apply(VENV / "generate/ar.py", OLD1, NEW1, "ar.py _apc_extra_hash",
            marker="_media_present")
ok &= apply(TPL, OLD2, NEW2, "chat_template.jinja",
            marker="elif enable_thinking is defined and enable_thinking is false")
sys.exit(0 if ok else 1)
