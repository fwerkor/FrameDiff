#!/usr/bin/env python3
"""Helpers for persisting per-step runtime metrics."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


def resolve_train_iters(args: Any, default: int = 1) -> int:
    env_value = os.getenv("LMSV_TRAIN_ITERS", "").strip()
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass

    value = getattr(args, "train_iters", None)
    if value is None:
        value = getattr(args, "train_steps", None)
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return max(1, resolved)


def resolve_step_log_csv() -> str:
    for env_name in ("LMSV_TRAINING_LOG_CSV", "LMSV_STEP_LOG_CSV"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def append_step_metrics(csv_path: str | Path, step: int, elapsed_s: float, memory_mb: float, loss: float) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size <= 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Step", "Execution Time (s)", "NPU Memory (MB)", "loss"])

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([step, round(elapsed_s, 8), round(memory_mb, 8), float(loss)])
