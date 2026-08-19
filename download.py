from huggingface_hub import snapshot_download
p = snapshot_download(
    "PocketAiHub/Qwen3.8-9B-Abliterated-MLX",
    revision="183672470c9c603080a376da4b6858798a4903f1",
    allow_patterns=["4bit/*", "LICENSE", "README.md", "release-manifest.json"],
    local_dir="models/Qwen3.8-9B-Abliterated-MLX",
    max_workers=8,
)
print("DONE", p)
