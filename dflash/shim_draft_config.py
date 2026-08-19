#!/usr/bin/env python3
"""dflash-mlx 0.1.8 reads rope_theta and block_size from the config root.
z-lab/Qwen3.5-9B-DFlash publishes them nested (transformers 5.x rope_parameters,
and block_size inside dflash_config). Lift them; leave the originals in place."""
import json, sys, shutil

p = sys.argv[1]
cfg = json.load(open(p))
before = dict(cfg)

if "rope_theta" not in cfg:
    cfg["rope_theta"] = cfg["rope_parameters"]["rope_theta"]
if "block_size" not in cfg:
    cfg["block_size"] = cfg["dflash_config"]["block_size"]

if cfg != before:
    shutil.copy(p, p + ".orig")
    json.dump(cfg, open(p, "w"), indent=2)
    print(f"lifted: rope_theta={cfg['rope_theta']} block_size={cfg['block_size']}")
else:
    print("no change needed")
