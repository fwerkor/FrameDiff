from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .models import validate_models
from .paths import CONFIG_EXAMPLE_PATH, CONFIG_PATH


DEFAULT_CONFIG: dict[str, Any] = {
    "entry": "fullnet",
    "PTA_PATH": "<YOUR_PTA_PATH>",
    "MSA_PATH": "<YOUR_MSA_PATH>",
    "OUTPUT_ROOT": "output",
    "fullnet": {
        "MODELS": ["qwen2"],
        "PERTURB_EPS": "1e-5",
        "BASELINE_LOSS_TOLERANCE": 0.0,
    },
}

_REMOVED_TOP_LEVEL_KEYS = {
    "PTA_NAME",
    "MSA_NAME",
    "SAVE_ABNORMAL_WEIGHTS",
    "TRACE",
    "PRECISION",
    "task_type",
    "tasks",
    "MF_NAME",
}

_REMOVED_FULLNET_KEYS = {
    "COMPARE_MODE",
    "ENABLE_MF_WEIGHT_LOAD",
    "MF_ARGS_PATH",
    "PTA_MAX_RUNTIME",
    "MSA_MAX_RUNTIME",
    "MAX_VALIDATE_TIME",
    "TOTAL_ITER",
    "TEST_ITERATIONS",
    "LOG_INIT_WAIT",
    "LOG_STABLE_THRESHOLD",
    "MAX_MUTATION_WAIT",
    "BASE_SEED",
    "MUTNM",
    "NODE_NUM",
    "FULLNET_ASSEMBLY_MODE",
    "SAVE_STEPS",
    "LOAD_STEPS",
    "MUTATION_ROUNDS",
    "PERTURB_SIGMA",
}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _sanitize_paper_config(config: dict[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(config)
    for key in _REMOVED_TOP_LEVEL_KEYS:
        sanitized.pop(key, None)
    fullnet = sanitized.get("fullnet")
    if isinstance(fullnet, dict):
        for key in _REMOVED_FULLNET_KEYS:
            fullnet.pop(key, None)
    return sanitized


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    source = path if path.exists() else CONFIG_EXAMPLE_PATH
    if not source.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _sanitize_paper_config(_deep_merge(DEFAULT_CONFIG, data))


def write_config(config: dict[str, Any], path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_run_config(
    base: dict[str, Any] | None = None,
    *,
    models: list[str] | None = None,
    pta_path: str | None = None,
    msa_path: str | None = None,
    perturb_eps: str | None = None,
    baseline_loss_tolerance: float | None = None,
) -> dict[str, Any]:
    config = _sanitize_paper_config(_deep_merge(DEFAULT_CONFIG, base or {}))
    config["entry"] = "fullnet"
    fullnet = config.setdefault("fullnet", {})

    if models is not None:
        fullnet["MODELS"] = validate_models(models)
    else:
        fullnet["MODELS"] = validate_models(list(fullnet.get("MODELS") or []))

    if perturb_eps is not None:
        fullnet["PERTURB_EPS"] = str(perturb_eps)
    if baseline_loss_tolerance is not None:
        fullnet["BASELINE_LOSS_TOLERANCE"] = float(baseline_loss_tolerance)

    if pta_path is not None:
        config["PTA_PATH"] = pta_path
    if msa_path is not None:
        config["MSA_PATH"] = msa_path

    return config
