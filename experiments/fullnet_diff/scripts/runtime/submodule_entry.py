#!/usr/bin/env python3
"""Wrapper entry for full-network graph verification with runtime shared-weight patch."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _resolve_target() -> Path:
    target = os.getenv(
        "LMSV_SUBMODULE_TARGET_SCRIPT",
        "mutate_and_forward/load_and_forward_graph.py",
    ).strip()
    target_path = Path(target)
    if target_path.is_absolute():
        return target_path
    # Prefer current working directory first so ad-hoc local scripts still work.
    cwd_candidate = Path.cwd() / target_path
    if cwd_candidate.exists():
        return cwd_candidate
    # Fallback to migrated runtime package under utils/runtime.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "utils" / "runtime" / target_path


def _apply_patch() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    replace_dir = repo_root / "utils" / "replace"
    replace_dir_str = str(replace_dir)
    if replace_dir_str not in sys.path:
        sys.path.insert(0, replace_dir_str)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    # Force enable here to avoid being disabled by inherited shell env.
    os.environ["LMSV_ENABLE_SUBMODULE_SHARED_WEIGHT_PATCH"] = "1"
    try:
        from shared_weight_patch import apply_shared_weight_patch

        applied = apply_shared_weight_patch()
        if not applied:
            raise RuntimeError("apply_shared_weight_patch returned False")
        sys.stderr.write("[LMSV_PATCH] shared weight patch active\n")
        sys.stderr.flush()
    except Exception as exc:
        sys.stderr.write(f"[LMSV_PATCH] failed to apply shared weight patch: {exc}\n")
        sys.stderr.flush()
        raise


def main() -> None:
    _apply_patch()
    target_path = _resolve_target()
    if not target_path.exists():
        raise FileNotFoundError(f"full-network target script not found: {target_path}")

    sys.argv = [str(target_path)] + sys.argv[1:]
    runpy.run_path(str(target_path), run_name="__main__")


if __name__ == "__main__":
    main()
