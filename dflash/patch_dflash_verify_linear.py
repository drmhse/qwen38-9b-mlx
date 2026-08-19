#!/usr/bin/env python3
"""Make DFLASH_VERIFY_LINEAR able to ENABLE target verify_linear, not only disable it.

Why this is needed
------------------
dflash-mlx gates the hand-written Metal verify kernels on the target model with
QwenGdnTargetOps._supports_verify_linear(), which for dense models is:

    return num_layers >= 40

Qwen3.5-9B has 32 layers, so it is excluded on layer count alone. Nothing about
the architecture is unsupported: verify_linear's own per-linear gate,
is_verify_eligible(), accepts 248 of this model's 249 QuantizedLinears (the one
rejection is lm_head, excluded on purpose by the N < 100_000 rule), and
verify_linear._PROJ_TAGS carries explicit gdn_qkv / gdn_z / gdn_o tags for
gated-DeltaNet projections. The 40 is validation policy, not capability.

The package already ships an override, internal_debug.verify_linear_override()
reading DFLASH_VERIFY_LINEAR, but it cannot enable this path, because
loading.py:194 computes

    supports_verify_linear AND _verify_enabled_for(...)

and the override only feeds the right-hand side. So env=1 ANDed with a False
capability stays False: the flag can turn the feature off, never on.

This patch consults the same override at the start of _supports_verify_linear.
Default behaviour is unchanged (override returns None when the var is unset),
so with no env var the upstream >=40 policy still applies verbatim.

Idempotent. Writes a .orig backup on first run.
"""
import re, shutil, sys
from pathlib import Path

MARKER = "# dflash-mlx patch: allow DFLASH_VERIFY_LINEAR to enable"

OLD_IMPORT = "from dflash_mlx.engine.target_ops import TargetCapabilities"
NEW_IMPORT = (
    "from dflash_mlx.engine.target_ops import TargetCapabilities\n"
    "from dflash_mlx.internal_debug import verify_linear_override as _verify_linear_override"
)

OLD_FN = """    def _supports_verify_linear(self, target_model: Any) -> bool:
        wrapper = self.text_wrapper(target_model)"""
NEW_FN = f"""    def _supports_verify_linear(self, target_model: Any) -> bool:
        {MARKER}
        _override = _verify_linear_override()
        if _override is not None:
            return _override
        wrapper = self.text_wrapper(target_model)"""


def main(site_packages: str) -> int:
    p = Path(site_packages) / "dflash_mlx" / "engine" / "target_qwen_gdn.py"
    if not p.exists():
        print(f"not found: {p}")
        return 1
    src = p.read_text()

    if MARKER in src:
        print("already patched (idempotent no-op)")
        return 0

    if OLD_FN not in src:
        print("FAILED: _supports_verify_linear signature not found; "
              "dflash-mlx version differs from 0.1.8 - inspect manually")
        return 1
    if src.count(OLD_IMPORT) != 1:
        print("FAILED: import anchor not unique")
        return 1

    shutil.copy(p, str(p) + ".orig")
    src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    src = src.replace(OLD_FN, NEW_FN, 1)
    p.write_text(src)
    print(f"patched {p}")
    print("  DFLASH_VERIFY_LINEAR=1 now enables target verify_linear")
    print("  DFLASH_VERIFY_LINEAR=0 disables it")
    print("  unset  -> upstream num_layers >= 40 policy, unchanged")
    return 0


MARKER2 = "# dflash-mlx patch: tri-state DFLASH_VERIFY_QMM"

OLD_QMM = """def _verify_qmm_enabled(verify_config: VerifyConfig | None) -> bool:
    if verify_config is not None:
        return bool(verify_config.enable_qmm)
    return _debug_verify_qmm_enabled()"""

NEW_QMM = f"""def _verify_qmm_enabled(verify_config: VerifyConfig | None) -> bool:
    {MARKER2}
    # VerifyConfig.enable_qmm defaults True and the CLI exposes no flag, so the
    # M=16 fp16-accumulating kernel is unavoidable on the CLI path. That kernel is
    # NOT bit-exact at n=16 (rel ~4e-3), which is exactly the DFlash block size,
    # so it perturbs the verifier every block. Allow an explicit opt-out.
    import os as _os
    _raw = _os.environ.get("DFLASH_VERIFY_QMM", "").strip()
    if _raw == "0":
        return False
    if _raw == "1":
        return True
    if verify_config is not None:
        return bool(verify_config.enable_qmm)
    return _debug_verify_qmm_enabled()"""


def patch_loading(site_packages: str) -> int:
    p = Path(site_packages) / "dflash_mlx" / "runtime" / "loading.py"
    src = p.read_text()
    if MARKER2 in src:
        print("loading.py already patched (idempotent no-op)")
        return 0
    if OLD_QMM not in src:
        print("FAILED: _verify_qmm_enabled body not found - inspect manually")
        return 1
    shutil.copy(p, str(p) + ".orig")
    p.write_text(src.replace(OLD_QMM, NEW_QMM, 1))
    print(f"patched {p}")
    print("  DFLASH_VERIFY_QMM=0 forces the bit-exact path (no M=16 fp16 kernel)")
    return 0


if __name__ == "__main__":
    rc = main(sys.argv[1]) or patch_loading(sys.argv[1])
    sys.exit(rc)
