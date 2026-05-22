#!/usr/bin/env python3
"""Shared paths for migrated runtime code and assets."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
ASSET_ROOT = REPO_ROOT / "assets" / "runtime"
CONFIG_DIR = ASSET_ROOT / "configs"
MODEL_CONFIG_DIR = WORKSPACE_ROOT / "frame_diff_common" / "model_configs"
TOKENIZER_DIR = ASSET_ROOT / "tokenizers"
SCRIPT_TEMPLATE_DIR = REPO_ROOT / "scripts" / "templates" / "pretrain_example"
MUTATION_SCRIPT_DIR = REPO_ROOT / "scripts" / "mutation"
RUNTIME_SCRIPT_DIR = REPO_ROOT / "scripts" / "runtime"


def repo_rel(path: Path) -> str:
    """Return a POSIX path relative to the repo root when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
