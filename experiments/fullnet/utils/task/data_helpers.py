#!/usr/bin/env python3
"""Shared data helpers for task runners."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def normalize_int_list(raw_values) -> list[int]:
    if isinstance(raw_values, str):
        items = [part.strip() for part in raw_values.split(",") if part.strip()]
    elif isinstance(raw_values, (list, tuple)):
        items = list(raw_values)
    else:
        return []

    values = []
    for item in items:
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            return []
    return values


def cleanup_shared_weight_file(weight_path: str | Path) -> None:
    path = Path(weight_path)
    if path.exists():
        path.unlink(missing_ok=True)


def remove_iteration_rows(csv_path: str | Path, iteration: int, log_warn, log_info) -> bool:
    """Delete all rows matching the target iteration from a CSV file."""
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size <= 0:
        return True

    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except Exception as exc:
        log_warn(f"读取CSV失败，无法清理迭代行: {csv_path} | {exc}")
        return False

    if not rows:
        return True

    header = rows[0]
    iter_idx = header.index("Iteration") if "Iteration" in header else 0
    kept_rows = [header]
    removed = 0
    for row in rows[1:]:
        iter_value = row[iter_idx].strip() if len(row) > iter_idx else ""
        hit = False
        if iter_value:
            try:
                hit = int(float(iter_value)) == int(iteration)
            except Exception:
                hit = False
        if hit:
            removed += 1
        else:
            kept_rows.append(row)

    if removed == 0:
        return True

    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(kept_rows)
    log_info(f"已清理CSV中迭代{iteration}的旧记录: {csv_path} | 删除{removed}行")
    return True


def csv_has_iteration(csv_path: str | Path, iteration: int) -> bool:
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size <= 0:
        return False

    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                iter_str = str(row.get("Iteration", "")).strip()
                if iter_str and int(float(iter_str)) == int(iteration):
                    return True
    except Exception:
        return False
    return False


def csv_iteration_is_valid(csv_path: str | Path, iteration: int, metric_keys: list[str] | None = None) -> bool:
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size <= 0:
        return False

    keys = metric_keys or ["Execution Time (s)", "NPU Memory (MB)", "loss"]
    try:
        last_row = None
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                iter_str = str(row.get("Iteration", "")).strip()
                if not iter_str:
                    continue
                try:
                    matched = int(float(iter_str)) == int(iteration)
                except Exception:
                    matched = False
                if not matched:
                    continue
                last_row = row

        if last_row is None:
            return False
        values = [str(last_row.get(key, "")).strip() for key in keys if key in last_row]
        if not values:
            return True
        return not all(value == "-" for value in values)
    except Exception:
        return False
    return False


def csv_has_valid_metrics_row(csv_path: str | Path, metric_keys: list[str] | None = None) -> bool:
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size <= 0:
        return False

    keys = metric_keys or ["Execution Time (s)", "NPU Memory (MB)", "loss"]
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                values = [str(row.get(key, "")).strip() for key in keys if key in row]
                if not values:
                    return True
                if not all(value == "-" for value in values):
                    return True
    except Exception:
        return False
    return False


def csv_iteration_metric_value(csv_path: str | Path, iteration: int, metric_key: str):
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size <= 0:
        return None

    try:
        last_value = None
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                iter_str = str(row.get("Iteration", "")).strip()
                if not iter_str:
                    continue
                try:
                    matched = int(float(iter_str)) == int(iteration)
                except Exception:
                    matched = False
                if not matched:
                    continue
                last_value = row.get(metric_key)
        return last_value
    except Exception:
        return None
    return None


def csv_iteration_loss(csv_path: str | Path, iteration: int):
    raw = csv_iteration_metric_value(csv_path, iteration, "loss")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except Exception:
        return None


def find_mutation_artifacts(res_root: str | Path, iteration: int):
    """Locate mutation JSON/YAML artifacts for a task2 iteration."""
    root = Path(res_root)
    if not root.exists():
        return [], [], []

    succ_json = sorted(root.glob(f"submodule_*/mutating-{iteration}.json"))
    err_json = sorted(root.glob(f"submodule_*/mutating-{iteration}-err.json"))
    yaml_cfg = sorted(root.glob(f"submodule_*/mutated_config_iter_{iteration:03d}.yaml"))
    return succ_json, err_json, yaml_cfg


def recover_err_mutation_json(err_path: str | Path, succ_path: str | Path, log_warn) -> bool:
    """Clean an err mutation record into a loadable success record."""
    try:
        with Path(err_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        log_warn(f"读取err变异记录失败: {err_path} | {exc}")
        return False

    if not isinstance(data, dict):
        log_warn(f"err变异记录格式非法（非dict）: {err_path}")
        return False

    cleaned = {}
    for key, value in data.items():
        key_str = str(key)
        if key_str.isdigit() or key_str in {"block_num_list", "success"}:
            cleaned[key_str] = value

    if not any(str(key).isdigit() for key in cleaned):
        log_warn(f"err变异记录缺少节点配置，无法恢复: {err_path}")
        return False

    cleaned["success"] = True
    success_path = Path(succ_path)
    success_path.parent.mkdir(parents=True, exist_ok=True)
    with success_path.open("w", encoding="utf-8") as handle:
        json.dump(cleaned, handle, ensure_ascii=False, indent=2)

    log_warn(f"检测到err变异记录，已清洗并恢复为可加载记录: {success_path}")
    return True
