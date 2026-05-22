#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import time


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _emit(message: str) -> None:
    try:
        if os.getenv("LMSV_PATCH_LOG", "1").strip().lower() in {"1", "true", "yes", "on"}:
            rank = os.getenv("RANK", "?")
            sys.stderr.write(f"[{_timestamp()}] [LMSV_PATCH] [RANK={rank}] {message}\n")
            sys.stderr.flush()
    except Exception:
        pass

try:
    if os.getenv("LMSV_ENABLE_TRAINING_LOG_PATCH", "0").strip().lower() in {"1", "true", "yes", "on"}:
        _emit("sitecustomize loaded, trying to apply training log patch")
        from training_log_patch import apply_training_log_patch

        applied = apply_training_log_patch()
        _emit(f"sitecustomize apply result: {'applied' if applied else 'not-applied'}")
except Exception:
    # Do not block training startup on hook failures.
    _emit("sitecustomize failed to apply patch; continue without blocking training")
