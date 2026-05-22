from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .models import validate_models
from .paths import CONFIG_EXAMPLE_PATH, CONFIG_PATH


DEFAULT_CONFIG: dict[str, Any] = {
    "entry": "fullnet",
    "PTA_NAME": "mindspeed",
    "PTA_PATH": "<YOUR_PTA_PATH>",
    "MSA_NAME": "msadapter",
    "MSA_PATH": "<YOUR_MSA_PATH>",
    "SAVE_ABNORMAL_WEIGHTS": True,
    "TRACE": {
        "ENABLED": False,
        "DEBUG_COMPARE": False,
        "LAYER_SUMMARY": False,
        "EXPORT_FULL_WEIGHTS": True,
        "PERTURBATION_RUNS": True,
        "PERTURB_SIGMA": "1e-6",
    },
    "PRECISION": {
        "BASELINE_ALIGNMENT_REQUIRED": True,
        "BASELINE_LOSS_TOLERANCE": 0.0,
    },
    "fullnet": {
        "MODELS": ["qwen2"],
        "TOTAL_ITER": 10,
        "PTA_MAX_RUNTIME": 3000,
        "MSA_MAX_RUNTIME": 3000,
        "LOG_INIT_WAIT": 240,
        "LOG_STABLE_THRESHOLD": 150,
        "MAX_MUTATION_WAIT": 600,
        "BASE_SEED": 43,
        "MUTNM": 2,
        "NODE_NUM": 0,
        "FULLNET_ASSEMBLY_MODE": "single_model_fullnet",
        "SAVE_STEPS": 1,
        "LOAD_STEPS": 15,
    },
}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    source = path if path.exists() else CONFIG_EXAMPLE_PATH
    if not source.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _deep_merge(DEFAULT_CONFIG, data)


def write_config(config: dict[str, Any], path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_run_config(
    base: dict[str, Any] | None = None,
    *,
    models: list[str] | None = None,
    total_iter: int | None = None,
    pta_path: str | None = None,
    msa_path: str | None = None,
    pta_env: str | None = None,
    msa_env: str | None = None,
    save_steps: int | None = None,
    load_steps: int | None = None,
    mutnm: int | None = None,
    base_seed: int | None = None,
    trace: bool | None = None,
    debug_compare: bool | None = None,
) -> dict[str, Any]:
    config = _deep_merge(DEFAULT_CONFIG, base or {})
    config["entry"] = "fullnet"
    fullnet = config.setdefault("fullnet", {})
    config.pop("task" + "_type", None)
    config.pop("tasks", None)
    config.pop("M" + "F_NAME", None)
    for key in ("COMPARE_" + "MODE", "M" + "F_ARGS_PATH", "ENABLE_" + "M" + "F_WEIGHT_LOAD"):
        fullnet.pop(key, None)

    if models is not None:
        fullnet["MODELS"] = validate_models(models)
    else:
        fullnet["MODELS"] = validate_models(list(fullnet.get("MODELS") or []))

    if total_iter is not None:
        fullnet["TOTAL_ITER"] = int(total_iter)
    if save_steps is not None:
        fullnet["SAVE_STEPS"] = int(save_steps)
    if load_steps is not None:
        fullnet["LOAD_STEPS"] = int(load_steps)
    if mutnm is not None:
        fullnet["MUTNM"] = int(mutnm)
    if base_seed is not None:
        fullnet["BASE_SEED"] = int(base_seed)

    if pta_path is not None:
        config["PTA_PATH"] = pta_path
    if msa_path is not None:
        config["MSA_PATH"] = msa_path
    if pta_env is not None:
        config["PTA_NAME"] = pta_env
    if msa_env is not None:
        config["MSA_NAME"] = msa_env

    trace_cfg = config.setdefault("TRACE", {})
    if trace is not None:
        trace_cfg["ENABLED"] = bool(trace)
    if debug_compare is not None:
        trace_cfg["DEBUG_COMPARE"] = bool(debug_compare)

    return config
