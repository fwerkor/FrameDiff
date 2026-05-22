#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime training-log patch for CSV metric export."""

from __future__ import annotations

import csv
import inspect
import os
import sys
import threading
import time
from typing import Any, Optional

_PATCH_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_PATCH_APPLIED = False
_IMPORT_HOOK_INSTALLED = False
_ORIGINAL_IMPORT = None

_TARGET_MODULES = (
    "megatron.training.training",
    "mindspeed_llm.training.training",
    "mindspeed.training.training",
    "msm_replace.new_training",
)
_PATCHED_MODULES = set()
_SEEN_MISSING_TARGET = set()
_SEEN_SKIP_REASON = set()
_WRITE_COUNT = 0


_STATE = {
    "csv_path": "",
    "row_index": 0,
    "last_wall_time": None,
}

_TRUE_VALUES = {"1", "true", "yes", "on"}
_LOG_TRUE_VALUES = {"1", "true", "yes", "on"}


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _is_log_enabled() -> bool:
    return os.getenv("LMSV_PATCH_LOG", "1").strip().lower() in _LOG_TRUE_VALUES


def _rank_hint() -> str:
    for key in ("RANK", "LOCAL_RANK"):
        value = os.getenv(key)
        if value is not None and value != "":
            return f"{key}={value}"
    return "RANK=?"


def _emit(message: str) -> None:
    if not _is_log_enabled():
        return
    try:
        sys.stderr.write(f"[{_timestamp()}] [LMSV_PATCH] [{_rank_hint()}] {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _is_enabled() -> bool:
    return os.getenv("LMSV_ENABLE_TRAINING_LOG_PATCH", "0").strip().lower() in _TRUE_VALUES



def _csv_path() -> str:
    return os.getenv("LMSV_TRAINING_LOG_CSV", "").strip()


def _writer_mode() -> str:
    return os.getenv("LMSV_PATCH_WRITE_RANK", "last").strip().lower()


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "float"):
            value = value.float()
        if hasattr(value, "mean"):
            value = value.mean()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return None


def _fmt_float(value: Any, precision: int = 6, scientific: bool = False) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "na"
    try:
        if scientific:
            return f"{numeric:.{precision}e}"
        return f"{numeric:.{precision}f}"
    except Exception:
        return "na"


def _extract_loss(loss_dict: Any) -> Optional[float]:
    if not isinstance(loss_dict, dict):
        return None
    preferred = []
    fallback = []
    for key, value in loss_dict.items():
        numeric = _as_float(value)
        if numeric is None:
            continue
        if isinstance(key, str) and "loss" in key.lower():
            preferred.append(numeric)
        else:
            fallback.append(numeric)
    values = preferred if preferred else fallback
    if not values:
        return None
    return sum(values) / len(values)


def _peak_memory_mb() -> float:
    try:
        import torch

        if hasattr(torch, "npu") and hasattr(torch.npu, "max_memory_allocated"):
            return float(torch.npu.max_memory_allocated()) / 1024.0 / 1024.0
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 1024.0 / 1024.0
    except Exception:
        pass
    return 0.0


def _should_write(args: Any, iteration: Any, skipped_iter: Any, csv_path: str) -> tuple[bool, str]:
    if not csv_path:
        return False, "csv path unset"
    try:
        rank = int(getattr(args, "rank"))
        world_size = int(getattr(args, "world_size"))
        log_interval = int(getattr(args, "log_interval", 1) or 1)
    except Exception:
        return False, "args missing rank/world_size/log_interval"

    mode = _writer_mode()
    target_rank = world_size - 1
    if mode in ("rank0", "0", "first"):
        target_rank = 0
    elif mode == "all":
        target_rank = None
    elif mode.startswith("rank:"):
        try:
            target_rank = int(mode.split(":", 1)[1])
        except Exception:
            target_rank = world_size - 1

    if target_rank is not None and rank != target_rank:
        return False, f"rank mismatch (rank={rank}, target={target_rank}, mode={mode})"

    try:
        iteration = int(iteration)
    except Exception:
        return False, "iteration missing/unreadable"

    if log_interval <= 0 or iteration % log_interval != 0:
        return False, f"iteration not on log_interval (iter={iteration}, log_interval={log_interval})"
    try:
        if int(skipped_iter) != 0:
            return False, f"skipped_iter={skipped_iter}"
    except Exception:
        pass
    return True, "ok"


def _ensure_state_locked(csv_path: str) -> None:
    if _STATE["csv_path"] == csv_path:
        return
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    row_index = 0
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as handle:
                line_count = sum(1 for _ in handle)
            row_index = max(0, line_count - 1)
        except Exception:
            row_index = 0
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Iteration", "Execution Time (s)", "NPU Memory (MB)", "loss"])

    _STATE["csv_path"] = csv_path
    _STATE["row_index"] = row_index
    _STATE["last_wall_time"] = time.time()
    _emit(f"csv target set: {csv_path} (existing rows={row_index})")


def _append_row_locked(csv_path: str, elapsed_s: float, memory_mb: float, loss: float) -> None:
    global _WRITE_COUNT
    _STATE["row_index"] = int(_STATE["row_index"]) + 1
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([_STATE["row_index"], elapsed_s, memory_mb, loss])
    _WRITE_COUNT += 1
    if _WRITE_COUNT <= 3:
        _emit(
            "csv row written: "
            f"row={_STATE['row_index']} elapsed={_fmt_float(elapsed_s, 6)}s "
            f"mem={_fmt_float(memory_mb, 2)}MB loss={_fmt_float(loss, 6, scientific=True)}"
        )


def _wrap_training_log(original):
    try:
        signature = inspect.signature(original)
        param_index = {name: idx for idx, name in enumerate(signature.parameters.keys())}
    except Exception:
        signature = None
        param_index = {}

    def _get_arg(call_args, call_kwargs, name: str, fallback_index: Optional[int], default=None):
        if name in call_kwargs:
            return call_kwargs[name]
        idx = param_index.get(name)
        if idx is not None and idx < len(call_args):
            return call_args[idx]
        if fallback_index is not None and fallback_index < len(call_args):
            return call_args[fallback_index]
        return default

    def wrapped(*args, **kwargs):
        start_memory = _peak_memory_mb()
        result = original(*args, **kwargs)

        try:
            from megatron.training import get_args

            runtime_args = get_args()
            csv_path = _csv_path()
            loss_dict = _get_arg(args, kwargs, "loss_dict", 0, default=None)
            iteration = _get_arg(args, kwargs, "iteration", 4, default=None)
            skipped_iter = _get_arg(args, kwargs, "skipped_iter", 7, default=0)

            should_write, reason = _should_write(runtime_args, iteration, skipped_iter, csv_path)
            if not should_write:
                if reason not in _SEEN_SKIP_REASON:
                    _SEEN_SKIP_REASON.add(reason)
                    _emit(f"skip csv write: {reason}")
                return result

            loss_value = _extract_loss(loss_dict)
            if loss_value is None:
                reason = "loss_dict unreadable or empty"
                if reason not in _SEEN_SKIP_REASON:
                    _SEEN_SKIP_REASON.add(reason)
                    _emit(f"skip csv write: {reason}")
                return result
            if abs(loss_value) <= 1e-12:
                reason = "loss value too small"
                if reason not in _SEEN_SKIP_REASON:
                    _SEEN_SKIP_REASON.add(reason)
                    _emit(f"skip csv write: {reason}")
                return result

            end_memory = _peak_memory_mb()
            now = time.time()
            log_interval = int(getattr(runtime_args, "log_interval", 1) or 1)

            with _STATE_LOCK:
                _ensure_state_locked(csv_path)
                previous = _STATE["last_wall_time"]
                if previous is None:
                    elapsed = 0.0
                else:
                    elapsed = max(0.0, now - previous) / max(1, log_interval)
                _STATE["last_wall_time"] = now
                _append_row_locked(csv_path, elapsed, max(start_memory, end_memory), loss_value)
        except Exception:
            return result

        return result

    wrapped._lmsv_csv_wrapped = True
    wrapped._lmsv_csv_original = original
    return wrapped


def _patch_module_obj(module_name: str, module: Any) -> bool:
    if module_name in _PATCHED_MODULES:
        return False

    if module is None:
        return False

    try:
        target = getattr(module, "training_log", None)
    except Exception:
        return False

    if target is None:
        if module_name not in _SEEN_MISSING_TARGET:
            _SEEN_MISSING_TARGET.add(module_name)
            _emit(f"module loaded but training_log missing: {module_name}")
        return False
    if getattr(target, "_lmsv_csv_wrapped", False):
        _emit(f"module already patched: {module_name}")
        _PATCHED_MODULES.add(module_name)
        return True
    module.training_log = _wrap_training_log(target)
    _emit(f"patched training_log in module: {module_name}")
    _PATCHED_MODULES.add(module_name)
    return True


def _try_patch_loaded_modules(trigger: str) -> bool:
    patched_any = False
    for module_name in _TARGET_MODULES:
        module = sys.modules.get(module_name)
        if _patch_module_obj(module_name, module):
            patched_any = True
    if patched_any:
        _emit(f"patch applied on trigger: {trigger}")
    return patched_any


def _install_import_hook() -> bool:
    global _IMPORT_HOOK_INSTALLED, _ORIGINAL_IMPORT
    if _IMPORT_HOOK_INSTALLED:
        return True

    try:
        import builtins
    except Exception:
        return False

    _ORIGINAL_IMPORT = builtins.__import__

    def _lmsv_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
        try:
            if isinstance(name, str) and (
                name.startswith("megatron.")
                or name.startswith("msm_replace.")
                or name.startswith("mindspeed.")
                or name.startswith("mindspeed_llm.")
            ):
                _try_patch_loaded_modules(trigger=name)
        except Exception:
            pass
        return module

    builtins.__import__ = _lmsv_import
    _IMPORT_HOOK_INSTALLED = True
    _emit("lazy import hook installed")
    return True


def apply_training_log_patch() -> bool:
    """Install lazy patch hook and patch target modules once they are imported."""
    global _PATCH_APPLIED
    if not _is_enabled():
        return False
    with _PATCH_LOCK:
        if _PATCH_APPLIED:
            _try_patch_loaded_modules(trigger="re-apply")
            _emit("apply_training_log_patch called again; hook already active")
            return True

        if not _install_import_hook():
            _emit("failed to install lazy import hook")
            return False

        _PATCH_APPLIED = True
        patched_now = _try_patch_loaded_modules(trigger="initial")
        if patched_now:
            _emit("training log patch enabled (immediate)")
        else:
            _emit("training log patch armed (waiting for module import)")
        return True
