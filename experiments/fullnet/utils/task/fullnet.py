#!/usr/bin/env python3
"""
语言模型整网组装与差分验证链路。
重构自旧版 internal_exe_script.sh
"""

import json
import os
import re
import subprocess
import shutil
import shlex
import time
from datetime import datetime
from pathlib import Path
import yaml

import utils
from utils.analyze.precision import find_preferred_loss_mismatch
from utils.runtime.paths import MODEL_CONFIG_DIR, MUTATED_CONFIG_DIR, RUNTIME_SCRIPT_DIR, TOKENIZER_DIR, repo_rel
from utils.runtime.profiler_tools import generate_profile_report
from utils.task import data_helpers, runtime_helpers

LMSV_ROOT = Path(__file__).resolve().parents[2]
PROJECT_TMP_ROOT = LMSV_ROOT / "tmp"
FULLNET_TMP_ROOT = PROJECT_TMP_ROOT / "fullnet"
MODEL_CONFIG_REL = repo_rel(MODEL_CONFIG_DIR)
RUNTIME_SCRIPT_REL = repo_rel(RUNTIME_SCRIPT_DIR)
TOKENIZER_BAICHUAN_REL = repo_rel(TOKENIZER_DIR / "baichuan2")


class Config:
    # 任务参数
    MODE = "DEVELOP"
    TOTAL_ITER = 1
    TEST_ITERATIONS = 1
    BASE_SEED = 43
    MUTNM = 0
    MODELS = sorted(path.stem for path in MODEL_CONFIG_DIR.glob("*.yaml")) or ["qwen2"]
    NODE_NUM = 0
    RUNTIME_ROUNDS = TOTAL_ITER
    SAVE_STEPS = 1
    LOAD_STEPS = 1
    FULLNET_ASSEMBLY_MODE = "single_model_fullnet"

    # 运行配置
    PTA_ENV = "mindspeed"
    MSA_ENV = "msadapter"
    PTA_MAX_RUNTIME = 6000
    MSA_MAX_RUNTIME = 6000
    LOG_INIT_WAIT = 240
    LOG_STABLE_THRESHOLD = 300
    SAVE_ABNORMAL_WEIGHTS = True
    TARGET_TENSOR_PARALLEL_SIZE = 0
    TARGET_PIPELINE_PARALLEL_SIZE = 0
    TARGET_EXPERT_PARALLEL_SIZE = 0
    TARGET_CONTEXT_PARALLEL_SIZE = 0
    TARGET_NPUS_PER_NODE = 0
    TARGET_WORLD_SIZE = 0
    ENABLE_DATA_PARALLEL = False
    TARGET_MASTER_ADDR = "localhost"
    TARGET_MASTER_PORT = 6000

    # 路径配置
    LOG_PATH = "res/internal_execution.log"
    MSA_MONITOR_LOG = "msrun_log/worker_0.log"
    PTA_CSV_PATH = "res/execution_pta.csv"
    MSA_CSV_PATH = "res/execution_msa.csv"
    PERSIST_ROOT = ""
    RECORDS_ROOT = ""
    SHARED_WEIGHT_TMP_ROOT = str(FULLNET_TMP_ROOT / "shared_weight")
    TRACE_ENABLED = True
    TRACE_FULL_WEIGHTS = True
    TRACE_PERTURBATION_RUNS = True
    TRACE_PERTURB_EPS = "1e-5"
    TRACE_PERTURB_SEED = ""
    BASELINE_ALIGNMENT_REQUIRED = True
    BASELINE_LOSS_TOLERANCE = 0.0


LOG_SCOPE = "FullNet"

LAUNCHER_OVERRIDE_KEYS = {
    "TARGET_TENSOR_PARALLEL_SIZE",
    "TARGET_PIPELINE_PARALLEL_SIZE",
    "TARGET_EXPERT_PARALLEL_SIZE",
    "TARGET_CONTEXT_PARALLEL_SIZE",
    "TARGET_NPUS_PER_NODE",
    "TARGET_WORLD_SIZE",
    "ENABLE_DATA_PARALLEL",
    "TARGET_MASTER_ADDR",
    "TARGET_MASTER_PORT",
}

RUNTIME_CONTROL_KEYS = {
    "LOAD_STEPS",
    "SAVE_STEPS",
}

ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _format_log(tag, msg):
    text = str(msg)
    if tag:
        return f"[{LOG_SCOPE}][{tag}] {text}"
    return f"[{LOG_SCOPE}] {text}"


def log_info(msg):
    utils.log.write.info(_format_log(None, msg))


def log_warn(msg):
    utils.log.write.warn(_format_log(None, msg))


def log_error(msg):
    utils.log.write.error(_format_log(None, msg))


def log_step(msg):
    utils.log.write.info(_format_log("阶段", msg))


def log_kv(group, key, value):
    utils.log.write.info(_format_log(group, f"{key}: {value}"))


def configure_project_tmp_env():
    return runtime_helpers.configure_project_tmp_env(PROJECT_TMP_ROOT)


def _parse_positive_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return parsed if parsed > 0 else int(default)


def _parse_optional_positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _infer_visible_device_count():
    for env_name in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        raw = str(os.environ.get(env_name, "")).strip()
        if not raw:
            continue
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        if parts:
            return len(parts)

    try:
        import torch

        if hasattr(torch, "npu") and callable(getattr(torch.npu, "device_count", None)):
            count = int(torch.npu.device_count())
            if count > 0:
                return count
        if callable(getattr(torch.cuda, "device_count", None)):
            count = int(torch.cuda.device_count())
            if count > 0:
                return count
    except Exception:
        pass

    for cmd in (
        ["bash", "-lc", "npu-smi info -l | grep -c '^\\s*NPU ID'"],
        ["bash", "-lc", "npu-smi info | grep -c '| NPU '"],
    ):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError:
            continue
        if result.returncode != 0:
            continue
        count = _parse_optional_positive_int((result.stdout or "").strip())
        if count > 0:
            return count

    return 1


def _load_transformer_config(model_path):
    config_path = Path(model_path)
    if not config_path.is_absolute():
        config_path = LMSV_ROOT / config_path
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = (
        data.get("TransformerConfig")
        or data.get("MLATransformerConfig")
        or data.get("config")
        or data
    )
    return config if isinstance(config, dict) else {}


def _infer_fullnet_decoder_count(model_paths):
    if not model_paths:
        return 1
    cfg = _load_transformer_config(model_paths[0])
    for key in ("num_layers", "n_layer", "num_hidden_layers"):
        value = _parse_optional_positive_int(cfg.get(key))
        if value > 0:
            return value
    return max(1, len(model_paths))


def _assembly_model_paths(model_paths):
    if Config.FULLNET_ASSEMBLY_MODE == "single_model_fullnet" and model_paths:
        return [model_paths[0]]
    return model_paths


def _largest_dividing_tensor_parallel_size(constraints, max_tp):
    max_tp = max(1, int(max_tp or 1))
    values = [int(value) for value in (constraints or []) if _parse_optional_positive_int(value) > 0]
    if not values:
        return max_tp
    for candidate in range(max_tp, 0, -1):
        if all(value % candidate == 0 for value in values):
            return candidate
    return 1


def _infer_model_aware_tensor_parallel_size(model_paths, visible_cards):
    max_tp = max(1, int(visible_cards))
    if not model_paths:
        return max_tp

    constraints = []
    for model_path in model_paths:
        cfg = _load_transformer_config(model_path)
        for key in ("num_attention_heads", "num_query_groups", "hidden_size", "ffn_hidden_size"):
            value = _parse_optional_positive_int(cfg.get(key))
            if value > 0:
                constraints.append(value)

    return _largest_dividing_tensor_parallel_size(constraints, max_tp)


def _load_variant_transformer_config(yaml_path):
    try:
        with Path(yaml_path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:
        log_info(f"读取变体yaml失败，无法做TP自适应: {yaml_path} | {exc}")
        return {}
    if not isinstance(data, dict):
        return {}
    base_config = data.get("base_config") if isinstance(data.get("base_config"), dict) else data
    cfg = (base_config or {}).get("config", {})
    return cfg if isinstance(cfg, dict) else {}


def _load_variant_transformer_constraints(yaml_path):
    cfg = _load_variant_transformer_config(yaml_path)
    constraints = []
    for key in ("num_attention_heads", "num_query_groups", "hidden_size", "ffn_hidden_size"):
        value = _parse_optional_positive_int(cfg.get(key))
        if value > 0:
            constraints.append(value)
    return constraints


def apply_variant_tensor_parallel_constraints(launcher_overrides, yaml_path):
    """Lower TP for config variants whose mutated config cannot be built at current TP."""
    overrides = dict(launcher_overrides or {})
    cfg = _load_variant_transformer_config(yaml_path)
    num_query_groups = _parse_optional_positive_int(cfg.get("num_query_groups"))
    if num_query_groups > 0 and "TARGET_NUM_QUERY_GROUPS" not in overrides:
        overrides["TARGET_NUM_QUERY_GROUPS"] = num_query_groups
    if "TARGET_TENSOR_PARALLEL_SIZE" in overrides:
        return overrides
    constraints = []
    for key in ("num_attention_heads", "num_query_groups", "hidden_size", "ffn_hidden_size"):
        value = _parse_optional_positive_int(cfg.get(key))
        if value > 0:
            constraints.append(value)
    if not constraints:
        return overrides
    current_tp = resolve_distributed_config(overrides)["tp"]
    adjusted_tp = _largest_dividing_tensor_parallel_size(constraints, current_tp)
    if adjusted_tp < current_tp:
        overrides["TARGET_TENSOR_PARALLEL_SIZE"] = adjusted_tp
        log_info(
            f"变体配置与当前 TP={current_tp} 不匹配，使用 TP={adjusted_tp}: "
            f"yaml={yaml_path}, constraints={constraints}"
        )
    return overrides


def _largest_usable_worker_count(visible_cards, parallel_cards):
    visible_cards = max(1, int(visible_cards or 1))
    parallel_cards = max(1, int(parallel_cards or 1))
    for candidate in range(visible_cards, 0, -1):
        if candidate % parallel_cards == 0:
            return candidate
    return parallel_cards


def _override_value(overrides, key, default):
    if isinstance(overrides, dict) and key in overrides:
        return overrides.get(key)
    return default


def resolve_distributed_config(launcher_overrides=None):
    inferred_cards = _infer_visible_device_count()
    launcher_overrides = launcher_overrides if isinstance(launcher_overrides, dict) else {}
    has_topology_override = any(
        key in launcher_overrides
        for key in (
            "TARGET_TENSOR_PARALLEL_SIZE",
            "TARGET_PIPELINE_PARALLEL_SIZE",
            "TARGET_EXPERT_PARALLEL_SIZE",
            "TARGET_CONTEXT_PARALLEL_SIZE",
        )
    )
    if has_topology_override:
        tp_default = pp_default = ep_default = cp_default = 1
    else:
        tp_default = Config.TARGET_TENSOR_PARALLEL_SIZE
        pp_default = Config.TARGET_PIPELINE_PARALLEL_SIZE
        ep_default = Config.TARGET_EXPERT_PARALLEL_SIZE
        cp_default = Config.TARGET_CONTEXT_PARALLEL_SIZE

    tp = _parse_optional_positive_int(_override_value(launcher_overrides, "TARGET_TENSOR_PARALLEL_SIZE", tp_default))
    pp = _parse_optional_positive_int(_override_value(launcher_overrides, "TARGET_PIPELINE_PARALLEL_SIZE", pp_default))
    ep = _parse_optional_positive_int(_override_value(launcher_overrides, "TARGET_EXPERT_PARALLEL_SIZE", ep_default))
    cp = _parse_optional_positive_int(_override_value(launcher_overrides, "TARGET_CONTEXT_PARALLEL_SIZE", cp_default))

    if tp <= 0 and pp <= 0 and ep <= 0 and cp <= 0:
        tp = inferred_cards
        pp = 1
        ep = 1
        cp = 1
    else:
        tp = max(1, tp)
        pp = max(1, pp)
        ep = max(1, ep)
        cp = max(1, cp)

    parallel_cards = max(1, tp * pp * ep * cp)
    configured_world = int(_override_value(launcher_overrides, "TARGET_WORLD_SIZE", Config.TARGET_WORLD_SIZE) or 0)

    configured_npus = int(_override_value(launcher_overrides, "TARGET_NPUS_PER_NODE", Config.TARGET_NPUS_PER_NODE) or 0)
    enable_data_parallel = _to_bool(_override_value(launcher_overrides, "ENABLE_DATA_PARALLEL", Config.ENABLE_DATA_PARALLEL))
    if configured_npus <= 0:
        if configured_world > 0:
            configured_npus = configured_world
        elif enable_data_parallel:
            configured_npus = _largest_usable_worker_count(inferred_cards, parallel_cards)
        else:
            configured_npus = parallel_cards
    npus_per_node = max(1, configured_npus)
    if configured_world > 0:
        world_size = max(parallel_cards, configured_world)
    else:
        world_size = max(parallel_cards, npus_per_node)

    return {
        "tp": tp,
        "pp": pp,
        "ep": ep,
        "cp": cp,
        "inferred_cards": inferred_cards,
        "nnodes": 1,
        "node_rank": 0,
        "master_addr": str(_override_value(launcher_overrides, "TARGET_MASTER_ADDR", Config.TARGET_MASTER_ADDR)),
        "master_port": int(_override_value(launcher_overrides, "TARGET_MASTER_PORT", Config.TARGET_MASTER_PORT)),
        "npus_per_node": npus_per_node,
        "world_size": world_size,
    }


def configure_auto_parallel_from_models(model_paths):
    if any(
        _parse_optional_positive_int(value) > 0
        for value in (
            Config.TARGET_TENSOR_PARALLEL_SIZE,
            Config.TARGET_PIPELINE_PARALLEL_SIZE,
            Config.TARGET_EXPERT_PARALLEL_SIZE,
            Config.TARGET_CONTEXT_PARALLEL_SIZE,
        )
    ):
        return

    visible_cards = _infer_visible_device_count()
    Config.TARGET_TENSOR_PARALLEL_SIZE = _infer_model_aware_tensor_parallel_size(model_paths, visible_cards)
    Config.TARGET_PIPELINE_PARALLEL_SIZE = 1
    Config.TARGET_EXPERT_PARALLEL_SIZE = 1
    Config.TARGET_CONTEXT_PARALLEL_SIZE = 1


def resolve_msa_monitor_log(launcher_overrides=None):
    worker_index = max(0, resolve_distributed_config(launcher_overrides)["npus_per_node"] - 1)
    return f"msrun_log/worker_{worker_index}.log"


def _to_bool(value):
    return data_helpers.parse_bool(value)


def load_variant_runtime_context(json_path):
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception as exc:
        log_warn(f"读取变体运行元数据失败，按无override处理: {json_path} | {exc}")
        return {
            "metadata": {},
            "runtime_args": [],
            "launcher": {},
            "env": {},
            "optimizer_env": {},
            "runtime_control": {},
        }
    if not isinstance(data, dict):
        data = {}
    overrides = data.get("runtime_overrides", {})
    if not isinstance(overrides, dict):
        overrides = {}
    runtime_args = overrides.get("runtime_args", [])
    if isinstance(runtime_args, str):
        runtime_args = shlex.split(runtime_args)
    elif isinstance(runtime_args, (tuple, list)):
        runtime_args = [str(item) for item in runtime_args if str(item).strip()]
    else:
        runtime_args = []
    runtime_args = _normalize_runtime_args(runtime_args)

    def _dict_value(name):
        value = overrides.get(name, {})
        return value if isinstance(value, dict) else {}

    launcher = {
        key: value
        for key, value in _dict_value("launcher").items()
        if key in LAUNCHER_OVERRIDE_KEYS
    }
    runtime_control = {
        key: value
        for key, value in _dict_value("runtime_control").items()
        if key in RUNTIME_CONTROL_KEYS
    }
    return {
        "metadata": data,
        "runtime_args": runtime_args,
        "launcher": launcher,
        "env": _dict_value("env"),
        "optimizer_env": _dict_value("optimizer_env"),
        "runtime_control": runtime_control,
    }


def _runtime_args_have_option(args, option):
    return any(str(item) == option for item in (args or []))


def _normalize_runtime_args(runtime_args):
    args = [str(item) for item in (runtime_args or []) if str(item).strip()]

    def add_option(option, *values):
        if not _runtime_args_have_option(args, option):
            args.append(option)
            args.extend(str(value) for value in values)

    has_low_precision = _runtime_args_have_option(args, "--bf16") or _runtime_args_have_option(args, "--fp16")
    if _runtime_args_have_option(args, "--reuse-fp32-param") and not has_low_precision:
        args.append("--bf16")
        has_low_precision = True
    if _runtime_args_have_option(args, "--fp32-residual-connection") and not has_low_precision:
        args.append("--bf16")
        has_low_precision = True

    if _runtime_args_have_option(args, "--recompute-num-layers"):
        add_option("--recompute-granularity", "full")

    if (
        _runtime_args_have_option(args, "--recompute-method")
        and not _runtime_args_have_option(args, "--recompute-granularity")
    ):
        add_option("--recompute-granularity", "full")

    for index, item in enumerate(args):
        if item == "--recompute-granularity" and index + 1 < len(args) and args[index + 1] == "full":
            add_option("--recompute-method", "uniform")
            add_option("--recompute-num-layers", "1")
            break
    return args


def _runtime_control_int(runtime_context, key, default):
    value = (runtime_context or {}).get("runtime_control", {}).get(key, default)
    return _parse_positive_int(value, default)


def _build_runtime_args_block(runtime_args_list):
    return " ".join(shlex.quote(str(arg)) for arg in (runtime_args_list or []) if str(arg).strip())


def _load_model_config_candidates(model_path):
    if model_path is None:
        return []
    model_text = str(model_path).split(",", 1)[0].strip()
    if not model_text:
        return []
    model_path = Path(model_text)
    candidates = [model_path]
    if not model_path.is_absolute():
        candidates.append((LMSV_ROOT / model_path).resolve())
        candidates.append((LMSV_ROOT.parent.parent / model_path).resolve())
        candidates.append((MODEL_CONFIG_DIR / model_path.name).resolve())
    return candidates


def _walk_config_dicts(data):
    if not isinstance(data, dict):
        return []
    candidates = [data]
    for key in (
        "base_config",
        "config",
        "TransformerConfig",
        "MLATransformerConfig",
        "get_gpt_layer_local_spec",
        "production_config",
    ):
        value = data.get(key)
        if isinstance(value, dict):
            candidates.extend(_walk_config_dicts(value))
    return candidates


def _extract_model_config_path_from_mutate_args(mutate_args):
    try:
        tokens = shlex.split(str(mutate_args or ""))
    except ValueError:
        return None
    for index, token in enumerate(tokens):
        if token in {"-m", "--model-config", "--model_config"} and index + 1 < len(tokens):
            return Path(tokens[index + 1].split(",", 1)[0].strip())
        if token.startswith("--model-config="):
            return Path(token.split("=", 1)[1].split(",", 1)[0].strip())
        if token.startswith("--model_config="):
            return Path(token.split("=", 1)[1].split(",", 1)[0].strip())
    return None


def _read_model_config_dicts_from_mutate_args(mutate_args):
    model_path = _extract_model_config_path_from_mutate_args(mutate_args)
    for candidate in _load_model_config_candidates(model_path):
        try:
            if not candidate.exists():
                continue
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if isinstance(data, dict):
            return _walk_config_dicts(data)
    return []


def _infer_num_layers_from_mutate_args(mutate_args):
    for cfg in _read_model_config_dicts_from_mutate_args(mutate_args):
        value = _parse_optional_positive_int(cfg.get("num_layers"))
        if value > 0:
            return value
    return None


def _infer_num_query_groups_from_mutate_args(mutate_args):
    for cfg in _read_model_config_dicts_from_mutate_args(mutate_args):
        value = _parse_optional_positive_int(cfg.get("num_query_groups"))
        if value > 0:
            return value
    return None


def _build_num_query_groups_args_block(launcher_overrides=None, mutate_args=None, runtime_args_list=None):
    if _runtime_args_have_option(runtime_args_list, "--num-query-groups"):
        return ""
    num_query_groups = _parse_optional_positive_int((launcher_overrides or {}).get("TARGET_NUM_QUERY_GROUPS"))
    if num_query_groups <= 0:
        num_query_groups = _infer_num_query_groups_from_mutate_args(mutate_args)
    if not num_query_groups:
        return ""
    return f"--num-query-groups {num_query_groups}"


def _build_precision_dependency_args_block(runtime_args_list=None):
    args = [str(arg) for arg in (runtime_args_list or [])]
    has_low_precision = _runtime_args_have_option(args, "--bf16") or _runtime_args_have_option(args, "--fp16")
    needs_low_precision = (
        _runtime_args_have_option(args, "--reuse-fp32-param")
        or _runtime_args_have_option(args, "--fp32-residual-connection")
    )
    return "--bf16" if needs_low_precision and not has_low_precision else ""


def _infer_num_experts_from_mutate_args(mutate_args):
    for cfg in _read_model_config_dicts_from_mutate_args(mutate_args):
        for key in ("num_experts", "num_moe_experts"):
            value = _parse_optional_positive_int(cfg.get(key))
            if value > 0:
                return value
    return None


def _build_context_parallel_args_block(dist_cfg, mutate_args, runtime_args_list=None):
    cp = int((dist_cfg or {}).get("cp", 1) or 1)
    if cp <= 1:
        return ""
    args = ["--context-parallel-size", str(cp)]
    if not _runtime_args_have_option(runtime_args_list, "--cp-comm-type"):
        num_layers = _infer_num_layers_from_mutate_args(mutate_args) or 16
        args.append("--cp-comm-type")
        args.extend(["p2p"] * max(1, int(num_layers)))
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _build_expert_parallel_args_block(dist_cfg, mutate_args, runtime_args_list=None):
    if int((dist_cfg or {}).get("ep", 1) or 1) <= 1:
        return ""
    if _runtime_args_have_option(runtime_args_list, "--num-experts"):
        return ""
    num_experts = _infer_num_experts_from_mutate_args(mutate_args)
    if not num_experts:
        return ""
    return f"--num-experts {num_experts}"


def _format_env_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_variant_env_override_block(env_overrides=None, optimizer_env=None):
    lines = []
    combined = {}
    for source in (env_overrides or {}, optimizer_env or {}):
        if isinstance(source, dict):
            combined.update(source)
    for name in sorted(combined):
        if not ENV_NAME_RE.match(str(name)):
            log_warn(f"忽略非法环境变量名: {name}")
            continue
        value = combined[name]
        if value is None or str(value).strip().lower() in {"unset", "__unset__", "default"}:
            lines.append(f"unset {name}")
        else:
            lines.append(f"export {name}={shlex.quote(_format_env_value(value))}")
    if not lines:
        return "# no variant env overrides"
    return "\n    ".join(lines)


def _build_distributed_deterministic_env_block(dist_cfg) -> str:
    return """
    export HCCL_DETERMINISTIC=true
    export ASCEND_LAUNCH_BLOCKING=1
    export NCCL_DETERMINISTIC=1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    """.strip()


def _build_msa_profile_dir_env_block(profile_dir=None):
    if profile_dir:
        return f"export LMSV_MSA_PROFILE_DIR={shlex.quote(str(Path(profile_dir).resolve()))}"
    return "unset LMSV_MSA_PROFILE_DIR"


def _build_trace_env_block(
    *,
    iter_num,
    trace_dir,
    backend,
    run_name,
    trace_record_dir=None,
    perturb=False,
    perturb_eps=None,
    trace_enabled=None,
):
    should_trace = Config.TRACE_ENABLED or os.environ.get("LMSV_FULLNET_TRACE") == "1"
    if trace_enabled is not None:
        should_trace = bool(trace_enabled)
    enabled = "1" if should_trace else "0"
    perturb_value = "1" if perturb else "0"
    eps = str(perturb_eps or Config.TRACE_PERTURB_EPS)
    lines = [
        f"export LMSV_FULLNET_TRACE={enabled}",
        f"export LMSV_FULLNET_TRACE_BACKEND={shlex.quote(str(backend))}",
        f"export LMSV_FULLNET_TRACE_RUN={shlex.quote(str(run_name))}",
        f"export LMSV_FULLNET_TRACE_ITER={int(iter_num)}",
        f"export LMSV_FULLNET_TRACE_DIR={shlex.quote(str(Path(trace_dir).resolve()))}",
        f"export LMSV_FULLNET_TRACE_RECORD_DIR={shlex.quote(str(Path(trace_record_dir or trace_dir).resolve()))}",
        f"export LMSV_FULLNET_TRACE_FULL_WEIGHTS={'1' if Config.TRACE_FULL_WEIGHTS else '0'}",
        f"export LMSV_FULLNET_PERTURB={perturb_value}",
        f"export LMSV_FULLNET_PERTURB_EPS={shlex.quote(eps)}",
    ]
    if Config.TRACE_PERTURB_SEED:
        lines.append(f"export LMSV_FULLNET_PERTURB_SEED={shlex.quote(str(Config.TRACE_PERTURB_SEED))}")
    if should_trace:
        lines.append("export LMSV_DEBUG_COMPARE=${LMSV_DEBUG_COMPARE:-1}")
    else:
        lines.append("export LMSV_DEBUG_COMPARE=0")
    return "\n    ".join(lines)


def run_shell_to_file(cmd, log_file, check=False, timeout=None, timeout_label=None):
    return runtime_helpers.run_shell_to_file(
        cmd,
        log_file,
        LMSV_ROOT,
        log_error,
        check=check,
        timeout=timeout,
        timeout_label=timeout_label,
    )


def write_runtime_script(script_path, cmd):
    return runtime_helpers.write_runtime_script(script_path, cmd)


def build_conda_activate_block(env_name, load_ascend=False):
    return runtime_helpers.build_conda_activate_block(env_name, load_ascend=load_ascend)


def normalize_models(raw_models):
    return runtime_helpers.normalize_models(raw_models, MODEL_CONFIG_DIR)


def build_mutate_args(model_paths, node_num, mutnm, rounds):
    model_arg = ",".join(model_paths)
    return (
        f"-c {MODEL_CONFIG_REL} -r {rounds} --mutnm {mutnm} "
        f"-n {node_num} -m {model_arg}"
    )


def cleanup_shared_weight_file(weight_path):
    data_helpers.cleanup_shared_weight_file(weight_path)


def csv_has_iteration(csv_path, iteration):
    return data_helpers.csv_has_iteration(csv_path, iteration)


def csv_iteration_is_valid(csv_path, iteration):
    return data_helpers.csv_iteration_is_valid(csv_path, iteration)


def wait_msa_finish_csv(iter_num, csv_path, label, monitor_log=None):
    log_step(f"等待MSA验证完成 | {label} | 迭代{iter_num}")
    log_path = LMSV_ROOT / (monitor_log or Config.MSA_MONITOR_LOG)
    csv_path = Path(csv_path)
    return runtime_helpers.wait_msa_finish(
        iter_num=iter_num,
        log_path=log_path,
        total_timeout=Config.MSA_MAX_RUNTIME,
        init_wait=Config.LOG_INIT_WAIT,
        stable_threshold=Config.LOG_STABLE_THRESHOLD,
        poll_interval=60,
        log_info=log_info,
        log_error=log_error,
        success_checker=lambda: csv_iteration_is_valid(csv_path, iter_num),
        result_exists_checker=lambda: csv_has_iteration(csv_path, iter_num),
    )


def reset_msrun_log():
    runtime_helpers.clear_path(LMSV_ROOT / "msrun_log")
    (LMSV_ROOT / "msrun_log").mkdir(parents=True, exist_ok=True)


def init_workspace(result_dir_name=None):
    """清理本任务相关历史产物。"""
    log_step("初始化整网工作目录")
    targets = [
        LMSV_ROOT / "msrun_log",
        LMSV_ROOT / "res" / "training_log_pta",
        LMSV_ROOT / "res" / "training_log_msa",
        LMSV_ROOT / "res" / "analyse_report",
        LMSV_ROOT / "res" / "execution_pta.csv",
        LMSV_ROOT / "res" / "execution_msa.csv",
    ]
    if result_dir_name:
        targets.extend(
            [
                LMSV_ROOT / "ms" / result_dir_name,
                LMSV_ROOT / "pta" / result_dir_name,
                LMSV_ROOT / "res" / result_dir_name,
            ]
        )

    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        runtime_helpers.clear_path(target)

    (LMSV_ROOT / "res").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "msrun_log").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "res" / "training_log_pta").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "res" / "training_log_msa").mkdir(parents=True, exist_ok=True)

def build_pta_verify_stage_cmd(
    iter_num,
    mutate_args,
    load_path,
    pta_env,
    pta_path,
    shared_weight_path,
    shared_mode,
    train_iters,
    step_log_csv_path=None,
    result_csv_path=None,
    trace_dir=None,
    trace_record_dir=None,
    trace_run_name="pta-baseline",
    perturb=False,
    perturb_eps=None,
    trace_enabled=None,
    runtime_args_list=None,
    launcher_overrides=None,
    env_overrides=None,
    optimizer_env=None,
):
    train_iters = int(train_iters)
    dist_cfg = resolve_distributed_config(launcher_overrides)
    runtime_args_block = _build_runtime_args_block(runtime_args_list)
    expert_parallel_args_block = _build_expert_parallel_args_block(dist_cfg, mutate_args, runtime_args_list)
    num_query_groups_args_block = _build_num_query_groups_args_block(launcher_overrides, mutate_args, runtime_args_list)
    precision_dependency_args_block = _build_precision_dependency_args_block(runtime_args_list)
    variant_env_block = _build_variant_env_override_block(env_overrides, optimizer_env)
    context_parallel_arg = _build_context_parallel_args_block(dist_cfg, mutate_args, runtime_args_list)
    if step_log_csv_path:
        step_log_block = f"export LMSV_TRAINING_LOG_CSV={shlex.quote(str(Path(step_log_csv_path).resolve()))}"
    else:
        step_log_block = "unset LMSV_TRAINING_LOG_CSV"
    pta_csv_path = result_csv_path or (LMSV_ROOT / Config.PTA_CSV_PATH)
    trace_block = _build_trace_env_block(
        iter_num=iter_num,
        trace_dir=trace_dir or (Path(Config.PERSIST_ROOT) / "iters" / f"iter_{iter_num}" / "traces"),
        trace_record_dir=trace_record_dir,
        backend="pta",
        run_name=trace_run_name,
        perturb=perturb,
        perturb_eps=perturb_eps,
        trace_enabled=trace_enabled,
    )
    return f"""
    {build_conda_activate_block(pta_env, load_ascend=True)}
    export PTA_PATH={shlex.quote(pta_path)}
    export PTAPATH={shlex.quote(pta_path)}
    source scripts/envset/pta.sh
    export LMSV_ENABLE_SUBMODULE_SHARED_WEIGHT_PATCH=1
    export LMSV_PATCH_LOG=1
    export LMSV_SUBMODULE_TARGET_SCRIPT=mutate_and_forward/load_and_forward_graph.py
    export LMSV_SHARED_WEIGHT_TARGET_MODULES=core.graph,utils.runtime.core.graph

    {_build_distributed_deterministic_env_block(dist_cfg)}
    {variant_env_block}

    NPUS_PER_NODE={dist_cfg["npus_per_node"]}
    MASTER_ADDR={shlex.quote(dist_cfg["master_addr"])}
    MASTER_PORT={dist_cfg["master_port"]}
    NNODES={dist_cfg["nnodes"]}
    NODE_RANK={dist_cfg["node_rank"]}
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$MASTER_PORT"
    export NNODES="$NNODES"
    export NODE_RANK="$NODE_RANK"

    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    GPT_ARGS="
        --tensor-model-parallel-size {dist_cfg["tp"]} \
        --pipeline-model-parallel-size {dist_cfg["pp"]} \
        --expert-model-parallel-size {dist_cfg["ep"]} \
        {context_parallel_arg} \
        {expert_parallel_args_block} \
        {num_query_groups_args_block} \
        {precision_dependency_args_block} \
        --num-layers 16 \
        --hidden-size 928 \
        --ffn-hidden-size 1712 \
        --num-attention-heads 8 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path {TOKENIZER_BAICHUAN_REL} \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size 1 \
        --global-batch-size 8 \
        --make-vocab-size-divisible-by 1 \
        --seed 114514 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --position-embedding-type rope \
    "

    export MUTATE_ROUND={iter_num}
    export MUTATE_ARGS={shlex.quote(mutate_args)}
    export LMSV_SHARED_WEIGHT_PATH={shlex.quote(shared_weight_path)}
    export LMSV_SHARED_WEIGHT_MODE={shlex.quote(shared_mode)}
    export LMSV_PTA_CSV_PATH={shlex.quote(str(Path(pta_csv_path).resolve()))}
    export LMSV_TRAIN_ITERS={train_iters}
    {trace_block}
    {step_log_block}
    torchrun $DISTRIBUTED_ARGS {shlex.quote(f"{RUNTIME_SCRIPT_REL}/submodule_entry.py")} \
        $GPT_ARGS \
        {runtime_args_block} \
        $MUTATE_ARGS \
        --train-iters {train_iters} \
        --load-path {shlex.quote(load_path)}
    """


def run_pta_verify_stage(
    iter_num,
    mutate_args,
    load_path,
    exec_log_file,
    pta_env,
    pta_path,
    shared_weight_path,
    shared_mode,
    train_iters,
    step_log_csv_path=None,
    result_csv_path=None,
    script_output_path=None,
    trace_dir=None,
    trace_record_dir=None,
    trace_run_name="pta-baseline",
    perturb=False,
    perturb_eps=None,
    trace_enabled=None,
    runtime_args_list=None,
    launcher_overrides=None,
    env_overrides=None,
    optimizer_env=None,
):
    cmd = build_pta_verify_stage_cmd(
        iter_num,
        mutate_args,
        load_path,
        pta_env,
        pta_path,
        shared_weight_path,
        shared_mode,
        train_iters,
        step_log_csv_path=step_log_csv_path,
        result_csv_path=result_csv_path,
        trace_dir=trace_dir,
        trace_record_dir=trace_record_dir,
        trace_run_name=trace_run_name,
        perturb=perturb,
        perturb_eps=perturb_eps,
        trace_enabled=trace_enabled,
        runtime_args_list=runtime_args_list,
        launcher_overrides=launcher_overrides,
        env_overrides=env_overrides,
        optimizer_env=optimizer_env,
    )
    if script_output_path:
        write_runtime_script(script_output_path, cmd)
    result = run_shell_to_file(
        cmd,
        exec_log_file,
        check=False,
        timeout=Config.PTA_MAX_RUNTIME,
        timeout_label="PTA执行",
    )
    return result is not None and result.returncode == 0


def build_msa_verify_load_cmd(
    iter_num,
    mutate_args,
    load_path,
    msa_env,
    msa_path,
    shared_weight_path,
    train_iters,
    step_log_csv_path=None,
    result_csv_path=None,
    profile_output_dir=None,
    trace_dir=None,
    trace_record_dir=None,
    trace_run_name="msa-baseline",
    perturb=False,
    perturb_eps=None,
    trace_enabled=None,
    runtime_args_list=None,
    launcher_overrides=None,
    env_overrides=None,
    optimizer_env=None,
):
    train_iters = int(train_iters)
    dist_cfg = resolve_distributed_config(launcher_overrides)
    runtime_args_block = _build_runtime_args_block(runtime_args_list)
    expert_parallel_args_block = _build_expert_parallel_args_block(dist_cfg, mutate_args, runtime_args_list)
    num_query_groups_args_block = _build_num_query_groups_args_block(launcher_overrides, mutate_args, runtime_args_list)
    precision_dependency_args_block = _build_precision_dependency_args_block(runtime_args_list)
    variant_env_block = _build_variant_env_override_block(env_overrides, optimizer_env)
    context_parallel_arg = _build_context_parallel_args_block(dist_cfg, mutate_args, runtime_args_list)
    if step_log_csv_path:
        step_log_block = f"export LMSV_TRAINING_LOG_CSV={shlex.quote(str(Path(step_log_csv_path).resolve()))}"
    else:
        step_log_block = "unset LMSV_TRAINING_LOG_CSV"
    msa_csv_path = result_csv_path or (LMSV_ROOT / Config.MSA_CSV_PATH)
    trace_block = _build_trace_env_block(
        iter_num=iter_num,
        trace_dir=trace_dir or (Path(Config.PERSIST_ROOT) / "iters" / f"iter_{iter_num}" / "traces"),
        trace_record_dir=trace_record_dir,
        backend="msa",
        run_name=trace_run_name,
        perturb=perturb,
        perturb_eps=perturb_eps,
        trace_enabled=trace_enabled,
    )
    return f"""
    {build_conda_activate_block(msa_env, load_ascend=True)}
    export MSA_PATH={shlex.quote(msa_path)}
    export MSAPATH={shlex.quote(msa_path)}
    source scripts/envset/msa.sh
    {_build_msa_profile_dir_env_block(profile_output_dir)}
    export LMSV_ENABLE_SUBMODULE_SHARED_WEIGHT_PATCH=1
    export LMSV_PATCH_LOG=1
    export LMSV_SUBMODULE_TARGET_SCRIPT=ms_mutate_and_forward/load_and_forward_graph.py
    export LMSV_SHARED_WEIGHT_TARGET_MODULES=core.graph,utils.runtime.core.graph

    {_build_distributed_deterministic_env_block(dist_cfg)}
    {variant_env_block}

    NPUS_PER_NODE={dist_cfg["npus_per_node"]}
    MASTER_ADDR={shlex.quote(dist_cfg["master_addr"])}
    MASTER_PORT={dist_cfg["master_port"]}
    NNODES={dist_cfg["nnodes"]}
    NODE_RANK={dist_cfg["node_rank"]}
    WORLD_SIZE={dist_cfg["world_size"]}
    export MASTER_ADDR="$MASTER_ADDR"
    export MASTER_PORT="$MASTER_PORT"
    export NNODES="$NNODES"
    export NODE_RANK="$NODE_RANK"
    export WORLD_SIZE="$WORLD_SIZE"

    DISTRIBUTED_ARGS="
        --master_addr $MASTER_ADDR \
        --node_rank $NODE_RANK \
        --worker_num $WORLD_SIZE \
        --local_worker_num $NPUS_PER_NODE \
        --master_port $MASTER_PORT \
        --log_dir=msrun_log \
        --join=False \
        --cluster_time_out=300 \
        --bind_core=True
    "

    GPT_ARGS="
        --tensor-model-parallel-size {dist_cfg["tp"]} \
        --pipeline-model-parallel-size {dist_cfg["pp"]} \
        --expert-model-parallel-size {dist_cfg["ep"]} \
        {context_parallel_arg} \
        {expert_parallel_args_block} \
        {num_query_groups_args_block} \
        {precision_dependency_args_block} \
        --num-layers 16 \
        --hidden-size 928 \
        --ffn-hidden-size 1712 \
        --num-attention-heads 8 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path {TOKENIZER_BAICHUAN_REL} \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size 1 \
        --global-batch-size 8 \
        --make-vocab-size-divisible-by 1 \
        --seed 114514 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --position-embedding-type rope \
    "

    export MUTATE_ROUND={iter_num}
    export MUTATE_ARGS={shlex.quote(mutate_args)}
    export LMSV_SHARED_WEIGHT_PATH={shlex.quote(shared_weight_path)}
    export LMSV_SHARED_WEIGHT_MODE=load
    export LMSV_MSA_CSV_PATH={shlex.quote(str(Path(msa_csv_path).resolve()))}
    export LMSV_TRAIN_ITERS={train_iters}
    {trace_block}
    {step_log_block}
    msrun $DISTRIBUTED_ARGS {shlex.quote(f"{RUNTIME_SCRIPT_REL}/submodule_entry.py")} \
        $GPT_ARGS \
        {runtime_args_block} \
        $MUTATE_ARGS \
        --train-iters {train_iters} \
        --load-path {shlex.quote(load_path)}
    """


def run_msa_verify_load(
    iter_num,
    mutate_args,
    load_path,
    exec_log_file,
    msa_env,
    msa_path,
    shared_weight_path,
    train_iters,
    step_log_csv_path=None,
    result_csv_path=None,
    profile_output_dir=None,
    script_output_path=None,
    trace_dir=None,
    trace_record_dir=None,
    trace_run_name="msa-baseline",
    perturb=False,
    perturb_eps=None,
    trace_enabled=None,
    runtime_args_list=None,
    launcher_overrides=None,
    env_overrides=None,
    optimizer_env=None,
):
    cmd = build_msa_verify_load_cmd(
        iter_num,
        mutate_args,
        load_path,
        msa_env,
        msa_path,
        shared_weight_path,
        train_iters,
        step_log_csv_path=step_log_csv_path,
        result_csv_path=result_csv_path,
        profile_output_dir=profile_output_dir,
        trace_dir=trace_dir,
        trace_record_dir=trace_record_dir,
        trace_run_name=trace_run_name,
        perturb=perturb,
        perturb_eps=perturb_eps,
        trace_enabled=trace_enabled,
        runtime_args_list=runtime_args_list,
        launcher_overrides=launcher_overrides,
        env_overrides=env_overrides,
        optimizer_env=optimizer_env,
    )
    if script_output_path:
        write_runtime_script(script_output_path, cmd)
    result = run_shell_to_file(
        cmd,
        exec_log_file,
        check=False,
        timeout=Config.MSA_MAX_RUNTIME,
        timeout_label="MSA执行",
    )
    return result is not None and result.returncode == 0


def _apply_config(params):
    Config.MODE = "DEVELOP"
    Config.TOTAL_ITER = 1
    Config.TEST_ITERATIONS = Config.TOTAL_ITER
    Config.RUNTIME_ROUNDS = Config.TOTAL_ITER
    Config.BASE_SEED = 43
    Config.MUTNM = 0
    Config.SAVE_STEPS = 1
    Config.LOAD_STEPS = 1
    Config.FULLNET_ASSEMBLY_MODE = "single_model_fullnet"
    Config.PTA_MAX_RUNTIME = 6000
    Config.MSA_MAX_RUNTIME = 6000
    Config.LOG_INIT_WAIT = 240
    Config.LOG_STABLE_THRESHOLD = 300
    Config.TARGET_TENSOR_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_TENSOR_PARALLEL_SIZE", Config.TARGET_TENSOR_PARALLEL_SIZE),
    )
    Config.TARGET_PIPELINE_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_PIPELINE_PARALLEL_SIZE", Config.TARGET_PIPELINE_PARALLEL_SIZE),
    )
    Config.TARGET_EXPERT_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_EXPERT_PARALLEL_SIZE", Config.TARGET_EXPERT_PARALLEL_SIZE),
    )
    Config.TARGET_CONTEXT_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_CONTEXT_PARALLEL_SIZE", Config.TARGET_CONTEXT_PARALLEL_SIZE),
    )
    Config.ENABLE_DATA_PARALLEL = data_helpers.parse_bool(
        params.get("ENABLE_DATA_PARALLEL", Config.ENABLE_DATA_PARALLEL)
    )
    Config.TARGET_NPUS_PER_NODE = int(params.get("TARGET_NPUS_PER_NODE", Config.TARGET_NPUS_PER_NODE) or 0)
    Config.TARGET_WORLD_SIZE = int(params.get("TARGET_WORLD_SIZE", Config.TARGET_WORLD_SIZE) or 0)
    Config.TARGET_MASTER_ADDR = str(params.get("TARGET_MASTER_ADDR", Config.TARGET_MASTER_ADDR))
    Config.TARGET_MASTER_PORT = int(params.get("TARGET_MASTER_PORT", Config.TARGET_MASTER_PORT))
    Config.PTA_ENV = str(params.get("PTA_ENV", os.environ.get("PTA_NAME", Config.PTA_ENV)))
    Config.MSA_ENV = str(params.get("MSA_ENV", os.environ.get("MSA_NAME", Config.MSA_ENV)))
    Config.SAVE_ABNORMAL_WEIGHTS = True
    os.environ["BASE_SEED"] = str(Config.BASE_SEED)

    raw_persist_root = str(params.get("PERSIST_ROOT", os.environ.get("LMSV_OUTPATH", str(LMSV_ROOT / "output"))))
    persist_root_path = Path(raw_persist_root).expanduser()
    if not persist_root_path.is_absolute():
        persist_root_path = LMSV_ROOT / persist_root_path
    Config.PERSIST_ROOT = str(persist_root_path.resolve())
    raw_records_root = str(params.get("RECORDS_ROOT", os.environ.get("LMSV_RECORDS_ROOT", str(LMSV_ROOT / "records"))))
    records_root_path = Path(raw_records_root).expanduser()
    if not records_root_path.is_absolute():
        records_root_path = LMSV_ROOT / records_root_path
    Config.RECORDS_ROOT = str(records_root_path.resolve())

    raw_tmp_root = str(params.get("SHARED_WEIGHT_TMP_ROOT", Config.SHARED_WEIGHT_TMP_ROOT))
    tmp_root_path = Path(raw_tmp_root).expanduser()
    if not tmp_root_path.is_absolute():
        tmp_root_path = LMSV_ROOT / tmp_root_path
    Config.SHARED_WEIGHT_TMP_ROOT = str(tmp_root_path.resolve())

    trace_cfg = params.get("TRACE", {})
    if not isinstance(trace_cfg, dict):
        trace_cfg = {}
    Config.TRACE_ENABLED = data_helpers.parse_bool(
        trace_cfg.get("ENABLED", os.environ.get("LMSV_FULLNET_TRACE", Config.TRACE_ENABLED))
    )
    Config.TRACE_FULL_WEIGHTS = data_helpers.parse_bool(
        trace_cfg.get("EXPORT_FULL_WEIGHTS", trace_cfg.get("FULL_WEIGHTS", Config.TRACE_FULL_WEIGHTS))
    )
    Config.TRACE_PERTURBATION_RUNS = data_helpers.parse_bool(
        trace_cfg.get("PERTURBATION_RUNS", Config.TRACE_PERTURBATION_RUNS)
    )
    Config.TRACE_PERTURB_EPS = str(
        params.get("PERTURB_EPS", trace_cfg.get("PERTURB_EPS", Config.TRACE_PERTURB_EPS))
    )
    Config.TRACE_PERTURB_SEED = str(trace_cfg.get("PERTURB_SEED", "") or "")

    precision_cfg = params.get("PRECISION", {})
    if not isinstance(precision_cfg, dict):
        precision_cfg = {}
    Config.BASELINE_ALIGNMENT_REQUIRED = True
    try:
        Config.BASELINE_LOSS_TOLERANCE = float(
            params.get(
                "BASELINE_LOSS_TOLERANCE",
                precision_cfg.get("BASELINE_LOSS_TOLERANCE", Config.BASELINE_LOSS_TOLERANCE),
            )
        )
    except (TypeError, ValueError):
        Config.BASELINE_LOSS_TOLERANCE = 0.0


TRAIN_PREPARE = "prepare"
TRAIN_PTA_BASELINE = "pta-baseline"
TRAIN_MSA_BASELINE = "msa-baseline"
TRAIN_PTA_PRETURB = "pta-preturb"
TRAIN_MSA_PRETURB = "msa-preturb"
RUN_TRAININGS = (TRAIN_PTA_BASELINE, TRAIN_MSA_BASELINE, TRAIN_PTA_PRETURB, TRAIN_MSA_PRETURB)


def _model_name_from_path(model_path):
    return Path(str(model_path)).stem


def _variant_sort_key(path):
    name = path.name
    return (0 if name == "ancestor" else 1, name)


def list_model_variants(model_name):
    model_dir = MUTATED_CONFIG_DIR / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"未找到模型变体目录: {model_dir}")
    variants = sorted([path for path in model_dir.iterdir() if path.is_dir()], key=_variant_sort_key)
    if not variants:
        raise FileNotFoundError(f"模型变体目录为空: {model_dir}")
    if variants[0].name != "ancestor":
        raise FileNotFoundError(f"{model_dir} 缺少 ancestor 变体；ancestor 必须第一个执行")
    return variants


def resolve_variant_files(variant_dir):
    json_path = variant_dir / "mutating.json"
    yaml_path = variant_dir / "mutated_config.yaml"
    if not json_path.exists() or json_path.stat().st_size <= 0 or not yaml_path.exists() or yaml_path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"变体缺少 mutating.json/mutated_config.yaml: {variant_dir}"
        )
    return json_path, yaml_path


def stage_output_dir(output_root, model_name, variant_name, training_name, iteration, max_iterations):
    base = Path(output_root) / model_name / variant_name / training_name
    if max_iterations > 1:
        base = base / f"iter_{iteration}"
    return base


def fresh_stage_dir(output_root, model_name, variant_name, training_name, iteration, max_iterations):
    stage_dir = stage_output_dir(output_root, model_name, variant_name, training_name, iteration, max_iterations)
    runtime_helpers.clear_path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def stage_record_dir(records_root, model_name, variant_name, training_name, iteration, max_iterations):
    base = Path(records_root) / model_name / variant_name / training_name
    if max_iterations > 1:
        base = base / f"iter_{iteration}"
    return base


def fresh_stage_dirs(output_root, records_root, model_name, variant_name, training_name, iteration, max_iterations):
    output_stage_dir = fresh_stage_dir(output_root, model_name, variant_name, training_name, iteration, max_iterations)
    record_stage_dir = stage_record_dir(records_root, model_name, variant_name, training_name, iteration, max_iterations)
    runtime_helpers.clear_path(record_stage_dir)
    record_stage_dir.mkdir(parents=True, exist_ok=True)
    return output_stage_dir, record_stage_dir


def variant_output_dir(output_root, model_name, variant_name):
    path = Path(output_root) / model_name / variant_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_output_dir(output_root, model_name):
    path = Path(output_root) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def variant_record_dir(records_root, model_name, variant_name):
    path = Path(records_root) / model_name / variant_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def materialize_variant_for_runtime(model_name, variant_dir, iteration):
    """Copy a prepared variant into the legacy runtime layout expected by task3."""
    source_json, source_yaml = resolve_variant_files(variant_dir)
    runtime_dir = LMSV_ROOT / "res" / model_name
    runtime_helpers.clear_path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    json_dst = runtime_dir / f"mutating-{iteration}.json"
    yaml_dst = runtime_dir / f"mutated_config_iter_{iteration:03d}.yaml"
    shutil.copy2(source_json, json_dst)
    shutil.copy2(source_yaml, yaml_dst)
    load_path = f"res/{model_name}/mutating-{iteration}.json"
    return load_path, json_dst, yaml_dst, source_json, source_yaml


def copy_variant_inputs(stage_dir, source_json, source_yaml, runtime_json=None, runtime_yaml=None):
    variant_input_dir = Path(stage_dir) / "variant_inputs"
    variant_input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_json, variant_input_dir / "mutating.json")
    shutil.copy2(source_yaml, variant_input_dir / "mutated_config.yaml")
    if runtime_json and Path(runtime_json).exists():
        shutil.copy2(runtime_json, variant_input_dir / Path(runtime_json).name)
    if runtime_yaml and Path(runtime_yaml).exists():
        shutil.copy2(runtime_yaml, variant_input_dir / Path(runtime_yaml).name)


def copy_if_exists(src_path, dst_path):
    src = Path(src_path)
    if not src.exists():
        return False
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return True


def prune_empty_dirs(root):
    root = Path(root)
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def archive_stage_runtime(stage_dir, model_name, *, include_msrun=False):
    copy_if_exists(LMSV_ROOT / "res" / model_name / "verify_log.txt", Path(stage_dir) / "verify_log.txt")
    if include_msrun:
        copy_if_exists(LMSV_ROOT / "msrun_log", Path(stage_dir) / "msrun_log")


def write_variant_status(
    records_root,
    model_name,
    variant_name,
    iteration,
    max_iterations,
    overall_status,
    reason="",
    **stages,
):
    root = variant_record_dir(records_root, model_name, variant_name)
    status_path = root / ("status.json" if max_iterations == 1 else f"status_iter_{iteration}.json")
    payload = {
        "task_name": "fullnet",
        "model": model_name,
        "variant": variant_name,
        "iteration": iteration,
        "overall_status": overall_status,
        "reason": reason,
        "trainings": {
            TRAIN_PREPARE: stages.get(TRAIN_PREPARE, "SKIP"),
            TRAIN_PTA_BASELINE: stages.get(TRAIN_PTA_BASELINE, "SKIP"),
            TRAIN_MSA_BASELINE: stages.get(TRAIN_MSA_BASELINE, "SKIP"),
            TRAIN_PTA_PRETURB: stages.get(TRAIN_PTA_PRETURB, "SKIP"),
            TRAIN_MSA_PRETURB: stages.get(TRAIN_MSA_PRETURB, "SKIP"),
            "baseline-align": stages.get("baseline-align", "SKIP"),
        },
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if overall_status != "PASS":
        failure_path = root / ("failure_info.txt" if max_iterations == 1 else f"failure_info_iter_{iteration}.txt")
        failure_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def write_alignment_report_new(
    records_root,
    model_name,
    variant_name,
    iteration,
    max_iterations,
    *,
    issue,
    tolerance,
    required,
    pta_csv,
    msa_csv,
    pta_step_csv,
    msa_step_csv,
):
    root = variant_record_dir(records_root, model_name, variant_name)
    report_path = root / (
        "baseline_alignment.json" if max_iterations == 1 else f"baseline_alignment_iter_{iteration}.json"
    )
    payload = {
        "model": model_name,
        "variant": variant_name,
        "iteration": iteration,
        "aligned": issue is None,
        "required": bool(required),
        "tolerance": float(tolerance),
        "issue": issue or "",
        "pta_csv": str(pta_csv),
        "msa_csv": str(msa_csv),
        "pta_step_csv": str(pta_step_csv),
        "msa_step_csv": str(msa_step_csv),
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload, report_path


def build_final_overview_markdown(summary):
    lines = [
        "# FrameDiff FullNet Summary",
        "",
        f"- status: {summary.get('overall_status', 'UNKNOWN')}",
        f"- models: {len(summary.get('model_results', []))}",
        f"- iterations: {summary.get('iterations')}",
        f"- planned_runs: {summary.get('planned_variant_runs')}",
        f"- failed_runs: {summary.get('failed_variant_runs')}",
        f"- pta_baseline: {summary.get('pta-baseline_success')}/{summary.get('planned_variant_runs')}",
        f"- msa_baseline: {summary.get('msa-baseline_success')}/{summary.get('planned_variant_runs')}",
        "",
        "## Training Matrix",
        "",
        "| Model | Variant | Iter | Overall | prepare | pta-baseline | msa-baseline | baseline-align | pta-preturb | msa-preturb | Reason |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in summary.get("records", []):
        trainings = record.get("trainings", {})
        lines.append(
            "| {model} | {variant} | {iteration} | {overall} | {prepare} | {pta} | {msa} | {align} | {pta_p} | {msa_p} | {reason} |".format(
                model=record.get("model", ""),
                variant=record.get("variant", ""),
                iteration=record.get("iteration", ""),
                overall=record.get("overall_status", ""),
                prepare=trainings.get(TRAIN_PREPARE, ""),
                pta=trainings.get(TRAIN_PTA_BASELINE, ""),
                msa=trainings.get(TRAIN_MSA_BASELINE, ""),
                align=trainings.get("baseline-align", ""),
                pta_p=trainings.get(TRAIN_PTA_PRETURB, ""),
                msa_p=trainings.get(TRAIN_MSA_PRETURB, ""),
                reason=str(record.get("reason", "") or "").replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Model Overview",
            "",
            "| Model | Status | Planned | PTA OK | MSA OK | Failed | Reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in summary.get("model_results", []):
        lines.append(
            "| {model} | {status} | {planned} | {pta} | {msa} | {failed} | {reason} |".format(
                model=item.get("model", ""),
                status=item.get("status", ""),
                planned=item.get("planned_variant_runs", 0),
                pta=item.get("pta_baseline_success", 0),
                msa=item.get("msa_baseline_success", 0),
                failed=item.get("failed_variant_runs", 0),
                reason=str(item.get("reason", "") or "").replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def stage_files(stage_dir, record_stage_dir=None):
    stage_dir = Path(stage_dir)
    record_stage_dir = Path(record_stage_dir or stage_dir)
    return {
        "runtime_log": record_stage_dir / "runtime.log",
        "script": record_stage_dir / "run.sh",
        "trace_dir": stage_dir,
        "trace_record_dir": record_stage_dir,
        "step_csv": record_stage_dir / "training_log.csv",
        "result_csv": record_stage_dir / "execution.csv",
    }


def _has_valid_trace_payload(stage_dir):
    stage_dir = Path(stage_dir)
    if not stage_dir.is_dir():
        return False
    try:
        return any(
            path.is_file()
            and path.suffix == ".pt"
            and "final_output" in path.name
            and path.stat().st_size > 0
            for path in stage_dir.rglob("*final_output*.pt")
        )
    except OSError:
        return False


def stage_run_is_valid(output_root, records_root, model_name, variant_name, training_name, iteration, max_iterations):
    del records_root
    stage_dir = stage_output_dir(output_root, model_name, variant_name, training_name, iteration, max_iterations)
    return _has_valid_trace_payload(stage_dir)


def model_shared_weight_path(output_root, model_name):
    return Path(output_root) / model_name / "shared_weight.pth"


def shared_weight_is_valid(path):
    path = Path(path)
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def build_repair_plan(model_paths, output_root, records_root, max_iterations):
    plan = {
        "models": {},
        "missing_runs": [],
        "full_models": [],
        "errors": [],
    }
    for model_path in model_paths:
        model_name = _model_name_from_path(model_path)
        model_plan = {
            "shared_weight": str(model_shared_weight_path(output_root, model_name)),
            "shared_weight_ok": False,
            "full_model": False,
            "variants": {},
        }
        plan["models"][model_name] = model_plan
        try:
            variants = list_model_variants(model_name)
        except Exception as exc:
            reason = f"{model_name}: 变体读取失败: {exc}"
            plan["errors"].append(reason)
            continue
        sw_path = model_shared_weight_path(output_root, model_name)
        model_plan["shared_weight_ok"] = shared_weight_is_valid(sw_path)
        if not model_plan["shared_weight_ok"]:
            model_plan["full_model"] = True
            plan["full_models"].append(model_name)
        for variant_dir in variants:
            variant_name = variant_dir.name
            variant_plan = {}
            model_plan["variants"][variant_name] = variant_plan
            for iteration in range(1, max_iterations + 1):
                iter_plan = {}
                variant_plan[str(iteration)] = iter_plan
                for training_name in RUN_TRAININGS:
                    ok = stage_run_is_valid(output_root, records_root, model_name, variant_name, training_name, iteration, max_iterations)
                    iter_plan[training_name] = ok
                    if model_plan["full_model"] or not ok:
                        plan["missing_runs"].append({
                            "model": model_name,
                            "variant": variant_name,
                            "iteration": iteration,
                            "training": training_name,
                            "reason": "shared_weight_missing_full_model" if model_plan["full_model"] else "missing_or_invalid_stage",
                        })
    return plan


def _format_repair_plan(plan, *, title):
    lines = [f"[FullNet][补测扫描] {title}"]
    if plan.get("errors"):
        lines.append("  变体读取错误:")
        for item in plan["errors"]:
            lines.append(f"    - {item}")
    full_models = plan.get("full_models") or []
    if full_models:
        lines.append("  需要整模型完整流程补测（缺 shared_weight.pth）:")
        for model_name in full_models:
            lines.append(f"    - {model_name}")
    grouped = {}
    for item in plan.get("missing_runs", []):
        grouped.setdefault(item["model"], {}).setdefault(item["variant"], []).append(item)
    if not grouped and not full_models and not plan.get("errors"):
        lines.append("  未发现缺失或无效跑测。")
    else:
        lines.append("  缺失/无效跑测:")
        for model_name in sorted(grouped):
            lines.append(f"    {model_name}:")
            for variant_name in sorted(grouped[model_name], key=lambda name: (0 if name == "ancestor" else 1, name)):
                runs = grouped[model_name][variant_name]
                labels = [f"iter{run['iteration']}:{run['training']}" for run in runs]
                lines.append(f"      - {variant_name}: {', '.join(labels)}")
    return "\n".join(lines)


def _print_repair_plan(plan, *, title):
    message = _format_repair_plan(plan, title=title)
    print(message, flush=True)
    for line in message.splitlines():
        log_info(line)


def _stage_label(model_name, variant_name, training_name, iteration):
    return f"{model_name}/{variant_name}/iter{iteration}/{training_name}"


def _format_repaired_runs(repaired_runs):
    lines = ["[FullNet][补测结果] 本次实际补测完成的跑测:"]
    if not repaired_runs:
        lines.append("  无。")
        return "\n".join(lines)
    grouped = {}
    for item in repaired_runs:
        grouped.setdefault(item["model"], {}).setdefault(item["variant"], []).append(item)
    for model_name in sorted(grouped):
        lines.append(f"  {model_name}:")
        for variant_name in sorted(grouped[model_name], key=lambda name: (0 if name == "ancestor" else 1, name)):
            labels = [f"iter{item['iteration']}:{item['training']}" for item in grouped[model_name][variant_name]]
            lines.append(f"    - {variant_name}: {', '.join(labels)}")
    return "\n".join(lines)


def _print_repaired_runs(repaired_runs):
    message = _format_repaired_runs(repaired_runs)
    print(message, flush=True)
    for line in message.splitlines():
        log_info(line)


def _append_repaired_run(repaired_runs, model_name, variant_name, iteration, training_name):
    item = {
        "model": model_name,
        "variant": variant_name,
        "iteration": iteration,
        "training": training_name,
    }
    if item not in repaired_runs:
        repaired_runs.append(item)


def main(params):
    project_tmp_root = configure_project_tmp_env()
    utils.control.clean.kill_pretraingpt()
    _apply_config(params)

    model_paths = normalize_models(params.get("MODELS", Config.MODELS))
    if not model_paths:
        log_error("整网参数错误：MODELS 为空或格式非法")
        return 1

    pta_path = os.environ.get("PTA_PATH") or os.environ.get("PTAPATH")
    msa_path = os.environ.get("MSA_PATH") or os.environ.get("MSAPATH")
    if not pta_path:
        log_error("环境变量缺失：请先配置 PTA_PATH")
        return 1
    if not msa_path:
        log_error("环境变量缺失：请先配置 MSA_PATH")
        return 1

    max_iterations = max(1, int(Config.TOTAL_ITER))
    output_root = Path(Config.PERSIST_ROOT).resolve()
    records_root = Path(Config.RECORDS_ROOT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records_root.mkdir(parents=True, exist_ok=True)
    repair_missing = data_helpers.parse_bool(params.get("REPAIR_MISSING", False))

    log_step("整网链路启动")
    log_kv("配置", "迭代次数", max_iterations)
    log_kv("配置", "基础随机种子", Config.BASE_SEED)
    log_kv("配置", "模型配置", model_paths)
    log_kv("配置", "变体来源", MUTATED_CONFIG_DIR)
    log_kv("配置", "公开输出目录", output_root)
    log_kv("配置", "运行记录目录", records_root)
    log_kv("配置", "对比模式", "pta_msa")
    log_kv("配置", "训练命名", ", ".join((TRAIN_PREPARE,) + RUN_TRAININGS))
    log_kv("配置", "训练步数", f"PREPARE({Config.SAVE_STEPS}) | LOAD({Config.LOAD_STEPS})")
    log_kv("配置", "当前执行对", "PTA + MSA")
    log_kv("配置", "激活环境", f"PTA={Config.PTA_ENV} | MSA={Config.MSA_ENV}")
    log_kv("配置", "项目临时目录", project_tmp_root)
    log_kv("配置", "共享权重临时目录", Config.SHARED_WEIGHT_TMP_ROOT)
    log_kv("配置", "补测模式", repair_missing)
    log_kv("配置", "Trace导出", f"{Config.TRACE_ENABLED} | full_weights={Config.TRACE_FULL_WEIGHTS} | perturb_runs={Config.TRACE_PERTURBATION_RUNS}")
    log_kv(
        "配置",
        "Baseline精度门槛",
        f"ancestor_required={Config.BASELINE_ALIGNMENT_REQUIRED} | variants_required=False | loss_tolerance={Config.BASELINE_LOSS_TOLERANCE}",
    )
    log_kv("概览", "开始时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    initial_repair_plan = None
    full_repair_models = set()
    if repair_missing:
        initial_repair_plan = build_repair_plan(model_paths, output_root, records_root, max_iterations)
        full_repair_models = set(initial_repair_plan.get("full_models") or [])
        _print_repair_plan(initial_repair_plan, title="补测前完整扫描")
        if not initial_repair_plan.get("missing_runs") and not initial_repair_plan.get("errors"):
            log_info("补测扫描未发现缺失项，直接结束。")
            return 0
        print("[FullNet][补测扫描] 10秒后开始正式补测。", flush=True)
        time.sleep(10)

    init_workspace()
    shared_weight_tmp_run_dir = (
        Path(Config.SHARED_WEIGHT_TMP_ROOT) / f"fullnet_{output_root.name}_{os.getpid()}"
    ).resolve()
    shared_weight_tmp_run_dir.mkdir(parents=True, exist_ok=True)

    run_records = []
    pta_success_count = 0
    msa_success_count = 0
    variant_failure_count = 0
    planned_baseline_runs = 0
    exit_code = 0
    model_results = []
    repaired_runs = []

    try:
        for model_path in model_paths:
            model_name = _model_name_from_path(model_path)
            model_result = {
                "model": model_name,
                "status": "RUNNING",
                "reason": "",
                "planned_variant_runs": 0,
                "failed_variant_runs": 0,
                "pta_baseline_success": 0,
                "msa_baseline_success": 0,
            }
            model_results.append(model_result)
            skip_current_model = False
            assembly_model_paths = _assembly_model_paths([model_path])
            if not any(
                _parse_optional_positive_int(params.get(key)) > 0
                for key in (
                    "TARGET_TENSOR_PARALLEL_SIZE",
                    "TARGET_PIPELINE_PARALLEL_SIZE",
                    "TARGET_EXPERT_PARALLEL_SIZE",
                    "TARGET_CONTEXT_PARALLEL_SIZE",
                )
            ):
                Config.TARGET_TENSOR_PARALLEL_SIZE = 0
                Config.TARGET_PIPELINE_PARALLEL_SIZE = 0
                Config.TARGET_EXPERT_PARALLEL_SIZE = 0
                Config.TARGET_CONTEXT_PARALLEL_SIZE = 0
            configure_auto_parallel_from_models(assembly_model_paths)
            Config.MSA_MONITOR_LOG = resolve_msa_monitor_log()
            Config.NODE_NUM = _infer_fullnet_decoder_count(assembly_model_paths)
            Config.RUNTIME_ROUNDS = max_iterations
            mutate_args_common = build_mutate_args(
                assembly_model_paths,
                Config.NODE_NUM,
                Config.MUTNM,
                Config.RUNTIME_ROUNDS,
            )
            try:
                variants = list_model_variants(model_name)
            except Exception as exc:
                reason = f"模型变体读取失败: {exc}"
                log_error(f"模型 {model_name} {reason}，跳过当前模型")
                model_result["status"] = "SKIPPED"
                model_result["reason"] = reason
                exit_code = 1
                continue
            model_planned_runs = len(variants) * max_iterations
            model_result["planned_variant_runs"] = model_planned_runs
            planned_baseline_runs += model_planned_runs

            log_step(f"模型 {model_name} 整网执行准备")
            log_kv("配置", f"{model_name}.整网基准模型", assembly_model_paths)
            log_kv("配置", f"{model_name}.Decoder层数", Config.NODE_NUM)
            log_kv("配置", f"{model_name}.变体顺序", [path.name for path in variants])
            dist_cfg = resolve_distributed_config()
            log_kv(
                "配置",
                f"{model_name}.单机并行设置",
                f"TP={dist_cfg['tp']} | PP={dist_cfg['pp']} | EP={dist_cfg['ep']} | CP={dist_cfg['cp']} | NPUS_PER_NODE={dist_cfg['npus_per_node']} | WORLD_SIZE={dist_cfg['world_size']}",
            )

            for i in range(1, max_iterations + 1):
                if skip_current_model:
                    break
                log_step(f"{model_name} 开始第 {i}/{max_iterations} 轮")
                ancestor_shared_weight = (
                    shared_weight_tmp_run_dir / model_name / f"ancestor_shared_iter{i}.pth"
                ).resolve()
                ancestor_shared_weight.parent.mkdir(parents=True, exist_ok=True)
                cleanup_shared_weight_file(ancestor_shared_weight)
                model_requires_full_repair = repair_missing and model_name in full_repair_models
                if repair_missing and not model_requires_full_repair:
                    persisted_shared_weight = model_shared_weight_path(output_root, model_name)
                    if shared_weight_is_valid(persisted_shared_weight):
                        copy_if_exists(persisted_shared_weight, ancestor_shared_weight)
                        log_info(f"[{model_name}/iter{i}] 补测复用已有共享权重: {persisted_shared_weight}")
                    else:
                        model_requires_full_repair = True
                        full_repair_models.add(model_name)
                        log_warn(f"[{model_name}/iter{i}] shared_weight.pth 不存在或无效，退回整模型完整流程补测")

                for variant_dir in variants:
                    if skip_current_model:
                        break
                    variant_name = variant_dir.name
                    stage_results = {
                        TRAIN_PREPARE: "SKIP",
                        TRAIN_PTA_BASELINE: "SKIP",
                        TRAIN_MSA_BASELINE: "SKIP",
                        TRAIN_PTA_PRETURB: "SKIP",
                        TRAIN_MSA_PRETURB: "SKIP",
                        "baseline-align": "SKIP",
                    }
                    record = {
                        "model": model_name,
                        "variant": variant_name,
                        "iteration": i,
                        "overall_status": "RUNNING",
                        "reason": "",
                        "trainings": stage_results,
                    }
                    run_records.append(record)

                    def fail_current(reason):
                        nonlocal variant_failure_count
                        log_error(f"[{model_name}/{variant_name}/iter{i}] {reason}")
                        record["overall_status"] = "FAILED"
                        record["reason"] = reason
                        write_variant_status(
                            records_root,
                            model_name,
                            variant_name,
                            i,
                            max_iterations,
                            "FAILED",
                            reason,
                            **stage_results,
                        )
                        variant_failure_count += 1
                        model_result["failed_variant_runs"] += 1

                    try:
                        load_path, runtime_json, runtime_yaml, source_json, source_yaml = materialize_variant_for_runtime(
                            model_name,
                            variant_dir,
                            i,
                        )
                    except Exception as exc:
                        fail_current(f"变体输入准备失败: {exc}")
                        continue

                    variant_runtime = load_variant_runtime_context(source_json)
                    runtime_args_list = variant_runtime["runtime_args"]
                    launcher_overrides = apply_variant_tensor_parallel_constraints(
                        variant_runtime["launcher"],
                        runtime_yaml,
                    )
                    env_overrides = variant_runtime["env"]
                    optimizer_env = variant_runtime["optimizer_env"]
                    save_steps = _runtime_control_int(variant_runtime, "SAVE_STEPS", Config.SAVE_STEPS)
                    load_steps = _runtime_control_int(variant_runtime, "LOAD_STEPS", Config.LOAD_STEPS)
                    msa_monitor_log = resolve_msa_monitor_log(launcher_overrides)
                    if runtime_args_list or launcher_overrides or env_overrides or optimizer_env or variant_runtime["runtime_control"]:
                        log_kv(
                            "变体override",
                            f"{model_name}/{variant_name}",
                            {
                                "runtime_args": runtime_args_list,
                                "launcher": launcher_overrides,
                                "env": env_overrides,
                                "optimizer_env": optimizer_env,
                                "runtime_control": variant_runtime["runtime_control"],
                            },
                        )

                    shared_weight_path = str(ancestor_shared_weight)
                    stage_needs = {training_name: True for training_name in RUN_TRAININGS}
                    if repair_missing and not model_requires_full_repair:
                        stage_needs = {
                            training_name: not stage_run_is_valid(
                                output_root,
                                records_root,
                                model_name,
                                variant_name,
                                training_name,
                                i,
                                max_iterations,
                            )
                            for training_name in RUN_TRAININGS
                        }
                        if not any(stage_needs.values()):
                            stage_results.update({training_name: "OK" for training_name in RUN_TRAININGS})
                            stage_results["baseline-align"] = "SKIP"
                            write_variant_status(
                                records_root,
                                model_name,
                                variant_name,
                                i,
                                max_iterations,
                                "PASS",
                                "补测跳过：四次跑测均已有效",
                                **stage_results,
                            )
                            record["overall_status"] = "PASS"
                            record["reason"] = "补测跳过：四次跑测均已有效"
                            continue

                    if variant_name == "ancestor" and (not repair_missing or model_requires_full_repair):
                        utils.control.clean.kill_pretraingpt()
                        stage_dir = stage_output_dir(output_root, model_name, variant_name, TRAIN_PREPARE, i, max_iterations)
                        runtime_helpers.clear_path(stage_dir)
                        record_stage_dir = stage_record_dir(records_root, model_name, variant_name, TRAIN_PREPARE, i, max_iterations)
                        runtime_helpers.clear_path(record_stage_dir)
                        record_stage_dir.mkdir(parents=True, exist_ok=True)
                        files = stage_files(stage_dir, record_stage_dir)
                        copy_variant_inputs(record_stage_dir, source_json, source_yaml, runtime_json, runtime_yaml)
                        log_step(f"{model_name}/{variant_name} {TRAIN_PREPARE}: PTA 生成共享权重")
                        prepare_ok = run_pta_verify_stage(
                            i,
                            mutate_args_common,
                            load_path,
                            files["runtime_log"],
                            Config.PTA_ENV,
                            pta_path,
                            shared_weight_path,
                            "save",
                            save_steps,
                            step_log_csv_path=files["step_csv"],
                            result_csv_path=files["result_csv"],
                            trace_dir=files["trace_dir"],
                            trace_record_dir=files["trace_record_dir"],
                            trace_run_name=TRAIN_PREPARE,
                            trace_enabled=False,
                            script_output_path=files["script"],
                            runtime_args_list=runtime_args_list,
                            launcher_overrides=launcher_overrides,
                            env_overrides=env_overrides,
                            optimizer_env=optimizer_env,
                        )
                        archive_stage_runtime(record_stage_dir, model_name)
                        if not prepare_ok or not ancestor_shared_weight.exists() or ancestor_shared_weight.stat().st_size <= 0:
                            stage_results[TRAIN_PREPARE] = "ERROR"
                            fail_current(f"{TRAIN_PREPARE} 失败或未产出共享权重: {files['runtime_log']}")
                            model_result["status"] = "FAILED"
                            model_result["reason"] = f"{TRAIN_PREPARE} 失败或未产出共享权重"
                            skip_current_model = True
                            continue
                        copy_if_exists(ancestor_shared_weight, model_output_dir(output_root, model_name) / "shared_weight.pth")
                        stage_results[TRAIN_PREPARE] = "OK"
                    elif not ancestor_shared_weight.exists() or ancestor_shared_weight.stat().st_size <= 0:
                        fail_current("ancestor 共享权重不存在，无法执行后续变体")
                        continue

                    if not stage_needs.get(TRAIN_PTA_BASELINE, True):
                        pta_stage_dir = stage_output_dir(output_root, model_name, variant_name, TRAIN_PTA_BASELINE, i, max_iterations)
                        pta_record_dir = stage_record_dir(records_root, model_name, variant_name, TRAIN_PTA_BASELINE, i, max_iterations)
                        pta_files = stage_files(pta_stage_dir, pta_record_dir)
                        stage_results[TRAIN_PTA_BASELINE] = "OK"
                        if repair_missing and stage_needs.get(TRAIN_PTA_BASELINE, False):
                            _append_repaired_run(repaired_runs, model_name, variant_name, i, TRAIN_PTA_BASELINE)
                        pta_success_count += 1
                        model_result["pta_baseline_success"] += 1
                        log_info(f"[{_stage_label(model_name, variant_name, TRAIN_PTA_BASELINE, i)}] 补测跳过：已有有效插桩结果")
                    else:
                        utils.control.clean.kill_pretraingpt()
                        pta_stage_dir, pta_record_dir = fresh_stage_dirs(output_root, records_root, model_name, variant_name, TRAIN_PTA_BASELINE, i, max_iterations)
                        pta_files = stage_files(pta_stage_dir, pta_record_dir)
                        copy_variant_inputs(pta_record_dir, source_json, source_yaml, runtime_json, runtime_yaml)
                        log_step(f"{model_name}/{variant_name} {TRAIN_PTA_BASELINE}: PTA baseline")
                        pta_ok = run_pta_verify_stage(
                            i,
                            mutate_args_common,
                            load_path,
                            pta_files["runtime_log"],
                            Config.PTA_ENV,
                            pta_path,
                            shared_weight_path,
                            "load",
                            load_steps,
                            step_log_csv_path=pta_files["step_csv"],
                            result_csv_path=pta_files["result_csv"],
                            trace_dir=pta_files["trace_dir"],
                            trace_record_dir=pta_files["trace_record_dir"],
                            trace_run_name=TRAIN_PTA_BASELINE,
                            script_output_path=pta_files["script"],
                            runtime_args_list=runtime_args_list,
                            launcher_overrides=launcher_overrides,
                            env_overrides=env_overrides,
                            optimizer_env=optimizer_env,
                        )
                        archive_stage_runtime(pta_record_dir, model_name)
                        if not pta_ok or not csv_iteration_is_valid(pta_files["result_csv"], i):
                            stage_results[TRAIN_PTA_BASELINE] = "ERROR"
                            copy_if_exists(ancestor_shared_weight, pta_record_dir / "shared_weight_on_failure.pth")
                            fail_current(f"{TRAIN_PTA_BASELINE} 失败或结果无效: {pta_files['runtime_log']}")
                            continue
                        stage_results[TRAIN_PTA_BASELINE] = "OK"
                        pta_success_count += 1
                        model_result["pta_baseline_success"] += 1

                    if not stage_needs.get(TRAIN_MSA_BASELINE, True):
                        msa_stage_dir = stage_output_dir(output_root, model_name, variant_name, TRAIN_MSA_BASELINE, i, max_iterations)
                        msa_record_dir = stage_record_dir(records_root, model_name, variant_name, TRAIN_MSA_BASELINE, i, max_iterations)
                        msa_files = stage_files(msa_stage_dir, msa_record_dir)
                        stage_results[TRAIN_MSA_BASELINE] = "OK"
                        if repair_missing and stage_needs.get(TRAIN_MSA_BASELINE, False):
                            _append_repaired_run(repaired_runs, model_name, variant_name, i, TRAIN_MSA_BASELINE)
                        msa_success_count += 1
                        model_result["msa_baseline_success"] += 1
                        log_info(f"[{_stage_label(model_name, variant_name, TRAIN_MSA_BASELINE, i)}] 补测跳过：已有有效插桩结果")
                    else:
                        utils.control.clean.kill_pretraingpt()
                        msa_stage_dir, msa_record_dir = fresh_stage_dirs(output_root, records_root, model_name, variant_name, TRAIN_MSA_BASELINE, i, max_iterations)
                        msa_files = stage_files(msa_stage_dir, msa_record_dir)
                        msa_profile_dir = msa_record_dir / "profiler" / "raw"
                        msa_profile_report_dir = msa_record_dir / "profiler" / "report"
                        copy_variant_inputs(msa_record_dir, source_json, source_yaml, runtime_json, runtime_yaml)
                        reset_msrun_log()
                        log_step(f"{model_name}/{variant_name} {TRAIN_MSA_BASELINE}: MSA baseline")
                        msa_ok = run_msa_verify_load(
                            i,
                            mutate_args_common,
                            load_path,
                            msa_files["runtime_log"],
                            Config.MSA_ENV,
                            msa_path,
                            shared_weight_path,
                            load_steps,
                            step_log_csv_path=msa_files["step_csv"],
                            result_csv_path=msa_files["result_csv"],
                            profile_output_dir=msa_profile_dir,
                            trace_dir=msa_files["trace_dir"],
                            trace_record_dir=msa_files["trace_record_dir"],
                            trace_run_name=TRAIN_MSA_BASELINE,
                            script_output_path=msa_files["script"],
                            runtime_args_list=runtime_args_list,
                            launcher_overrides=launcher_overrides,
                            env_overrides=env_overrides,
                            optimizer_env=optimizer_env,
                        )
                        msa_finished = wait_msa_finish_csv(i, msa_files["result_csv"], TRAIN_MSA_BASELINE, monitor_log=msa_monitor_log) if msa_ok else False
                        archive_stage_runtime(msa_record_dir, model_name, include_msrun=True)
                        if msa_profile_dir.exists() and any(msa_profile_dir.rglob("*")):
                            generate_profile_report(
                                msa_profile_dir,
                                msa_profile_report_dir,
                                msa_files["step_csv"],
                                msa_files["runtime_log"],
                                f"FullNet-MSA-{model_name}-{variant_name}",
                                i,
                            )
                        if not msa_ok or not msa_finished or not csv_iteration_is_valid(msa_files["result_csv"], i):
                            stage_results[TRAIN_MSA_BASELINE] = "ERROR"
                            copy_if_exists(ancestor_shared_weight, msa_record_dir / "shared_weight_on_failure.pth")
                            fail_current(f"{TRAIN_MSA_BASELINE} 失败或结果无效: {msa_files['runtime_log']}")
                            continue
                        stage_results[TRAIN_MSA_BASELINE] = "OK"
                        msa_success_count += 1
                        model_result["msa_baseline_success"] += 1

                    precision_issue = find_preferred_loss_mismatch(
                        pta_files["result_csv"],
                        msa_files["result_csv"],
                        iteration=i,
                        tolerance=Config.BASELINE_LOSS_TOLERANCE,
                        pta_step_csv_path=pta_files["step_csv"],
                        msa_step_csv_path=msa_files["step_csv"],
                    )
                    alignment_required = Config.BASELINE_ALIGNMENT_REQUIRED and variant_name == "ancestor"
                    alignment_report, alignment_path = write_alignment_report_new(
                        records_root,
                        model_name,
                        variant_name,
                        i,
                        max_iterations,
                        issue=precision_issue,
                        tolerance=Config.BASELINE_LOSS_TOLERANCE,
                        required=alignment_required,
                        pta_csv=pta_files["result_csv"],
                        msa_csv=msa_files["result_csv"],
                        pta_step_csv=pta_files["step_csv"],
                        msa_step_csv=msa_files["step_csv"],
                    )
                    stage_results["baseline-align"] = (
                        "OK" if precision_issue is None else ("ERROR" if alignment_required else "WARN")
                    )
                    if precision_issue:
                        log_warn(f"[{model_name}/{variant_name}/iter{i}] Baseline未对齐: {precision_issue}")
                        if alignment_required:
                            copy_if_exists(ancestor_shared_weight, msa_record_dir / "shared_weight_on_alignment_failure.pth")
                            fail_current("Baseline精度未对齐，跳过当前模型")
                            exit_code = 1
                            model_result["status"] = "FAILED"
                            model_result["reason"] = "ancestor baseline 精度未对齐"
                            skip_current_model = True
                            break
                    else:
                        log_info(f"[{model_name}/{variant_name}/iter{i}] Baseline精度对齐通过: {alignment_report}")

                    if Config.TRACE_ENABLED and Config.TRACE_PERTURBATION_RUNS:
                        if not stage_needs.get(TRAIN_PTA_PRETURB, True):
                            pta_perturb_dir = stage_output_dir(output_root, model_name, variant_name, TRAIN_PTA_PRETURB, i, max_iterations)
                            pta_perturb_record_dir = stage_record_dir(records_root, model_name, variant_name, TRAIN_PTA_PRETURB, i, max_iterations)
                            pta_perturb_files = stage_files(pta_perturb_dir, pta_perturb_record_dir)
                            stage_results[TRAIN_PTA_PRETURB] = "OK"
                            log_info(f"[{_stage_label(model_name, variant_name, TRAIN_PTA_PRETURB, i)}] 补测跳过：已有有效插桩结果")
                        else:
                            utils.control.clean.kill_pretraingpt()
                            pta_perturb_dir, pta_perturb_record_dir = fresh_stage_dirs(output_root, records_root, model_name, variant_name, TRAIN_PTA_PRETURB, i, max_iterations)
                            pta_perturb_files = stage_files(pta_perturb_dir, pta_perturb_record_dir)
                            copy_variant_inputs(pta_perturb_record_dir, source_json, source_yaml, runtime_json, runtime_yaml)
                            log_step(f"{model_name}/{variant_name} {TRAIN_PTA_PRETURB}: PTA 输入扰动")
                            pta_perturb_ok = run_pta_verify_stage(
                                i,
                                mutate_args_common,
                                load_path,
                                pta_perturb_files["runtime_log"],
                                Config.PTA_ENV,
                                pta_path,
                                shared_weight_path,
                                "load",
                                load_steps,
                                step_log_csv_path=pta_perturb_files["step_csv"],
                                result_csv_path=pta_perturb_files["result_csv"],
                                trace_dir=pta_perturb_files["trace_dir"],
                                trace_record_dir=pta_perturb_files["trace_record_dir"],
                                trace_run_name=TRAIN_PTA_PRETURB,
                                perturb=True,
                                perturb_eps=Config.TRACE_PERTURB_EPS,
                                script_output_path=pta_perturb_files["script"],
                                runtime_args_list=runtime_args_list,
                                launcher_overrides=launcher_overrides,
                                env_overrides=env_overrides,
                                optimizer_env=optimizer_env,
                            )
                            archive_stage_runtime(pta_perturb_record_dir, model_name)
                            if not pta_perturb_ok or not csv_iteration_is_valid(pta_perturb_files["result_csv"], i):
                                stage_results[TRAIN_PTA_PRETURB] = "ERROR"
                                fail_current(f"{TRAIN_PTA_PRETURB} 失败或结果无效: {pta_perturb_files['runtime_log']}")
                                continue
                            stage_results[TRAIN_PTA_PRETURB] = "OK"
                            if repair_missing and stage_needs.get(TRAIN_PTA_PRETURB, False):
                                _append_repaired_run(repaired_runs, model_name, variant_name, i, TRAIN_PTA_PRETURB)

                        if not stage_needs.get(TRAIN_MSA_PRETURB, True):
                            msa_perturb_dir = stage_output_dir(output_root, model_name, variant_name, TRAIN_MSA_PRETURB, i, max_iterations)
                            msa_perturb_record_dir = stage_record_dir(records_root, model_name, variant_name, TRAIN_MSA_PRETURB, i, max_iterations)
                            msa_perturb_files = stage_files(msa_perturb_dir, msa_perturb_record_dir)
                            stage_results[TRAIN_MSA_PRETURB] = "OK"
                            log_info(f"[{_stage_label(model_name, variant_name, TRAIN_MSA_PRETURB, i)}] 补测跳过：已有有效插桩结果")
                        else:
                            utils.control.clean.kill_pretraingpt()
                            msa_perturb_dir, msa_perturb_record_dir = fresh_stage_dirs(output_root, records_root, model_name, variant_name, TRAIN_MSA_PRETURB, i, max_iterations)
                            msa_perturb_files = stage_files(msa_perturb_dir, msa_perturb_record_dir)
                            copy_variant_inputs(msa_perturb_record_dir, source_json, source_yaml, runtime_json, runtime_yaml)
                            reset_msrun_log()
                            log_step(f"{model_name}/{variant_name} {TRAIN_MSA_PRETURB}: MSA 输入扰动")
                            msa_perturb_ok = run_msa_verify_load(
                                i,
                                mutate_args_common,
                                load_path,
                                msa_perturb_files["runtime_log"],
                                Config.MSA_ENV,
                                msa_path,
                                shared_weight_path,
                                load_steps,
                                step_log_csv_path=msa_perturb_files["step_csv"],
                                result_csv_path=msa_perturb_files["result_csv"],
                                trace_dir=msa_perturb_files["trace_dir"],
                                trace_record_dir=msa_perturb_files["trace_record_dir"],
                                trace_run_name=TRAIN_MSA_PRETURB,
                                perturb=True,
                                perturb_eps=Config.TRACE_PERTURB_EPS,
                                script_output_path=msa_perturb_files["script"],
                                runtime_args_list=runtime_args_list,
                                launcher_overrides=launcher_overrides,
                                env_overrides=env_overrides,
                                optimizer_env=optimizer_env,
                            )
                            msa_perturb_finished = (
                                wait_msa_finish_csv(i, msa_perturb_files["result_csv"], TRAIN_MSA_PRETURB, monitor_log=msa_monitor_log)
                                if msa_perturb_ok
                                else False
                            )
                            archive_stage_runtime(msa_perturb_record_dir, model_name, include_msrun=True)
                            if not msa_perturb_ok or not msa_perturb_finished or not csv_iteration_is_valid(msa_perturb_files["result_csv"], i):
                                stage_results[TRAIN_MSA_PRETURB] = "ERROR"
                                fail_current(f"{TRAIN_MSA_PRETURB} 失败或结果无效: {msa_perturb_files['runtime_log']}")
                                continue
                            stage_results[TRAIN_MSA_PRETURB] = "OK"
                            if repair_missing and stage_needs.get(TRAIN_MSA_PRETURB, False):
                                _append_repaired_run(repaired_runs, model_name, variant_name, i, TRAIN_MSA_PRETURB)

                    write_variant_status(
                        records_root,
                        model_name,
                        variant_name,
                        i,
                        max_iterations,
                        "PASS",
                        "变体执行完成",
                        **stage_results,
                    )
                    record["overall_status"] = "PASS"
                    record["reason"] = "变体执行完成"
                    utils.control.clean.kill_pretraingpt()
            if model_result["status"] == "RUNNING":
                if model_result["failed_variant_runs"]:
                    model_result["status"] = "PARTIAL"
                    model_result["reason"] = "存在失败的训练阶段"
                else:
                    model_result["status"] = "PASS"
                    model_result["reason"] = "模型执行完成"
    finally:
        shutil.rmtree(shared_weight_tmp_run_dir, ignore_errors=True)

    prune_empty_dirs(output_root)

    if repair_missing:
        _print_repaired_runs(repaired_runs)
        final_repair_plan = build_repair_plan(model_paths, output_root, records_root, max_iterations)
        _print_repair_plan(final_repair_plan, title="补测后完整扫描")
        if final_repair_plan.get("missing_runs") or final_repair_plan.get("errors"):
            exit_code = 1

    if exit_code == 0 and any(item.get("status") != "PASS" for item in model_results):
        exit_code = 1
    overall_status = "PASS" if exit_code == 0 else "FAILED"
    summary = {
        "task_name": "fullnet",
        "overall_status": overall_status,
        "models": [_model_name_from_path(path) for path in model_paths],
        "iterations": max_iterations,
        "load_steps": Config.LOAD_STEPS,
        "trainings": [TRAIN_PREPARE, *RUN_TRAININGS],
        "planned_variant_runs": planned_baseline_runs,
        "failed_variant_runs": variant_failure_count,
        "pta-baseline_success": pta_success_count,
        "msa-baseline_success": msa_success_count,
        "model_results": model_results,
        "exit_code": exit_code,
        "records": run_records,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (records_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_md_path = records_root / "summary.md"
    summary_md_path.write_text(build_final_overview_markdown(summary), encoding="utf-8")

    log_step("整网链路结束")
    log_kv("统计", "总体状态", overall_status)
    log_kv("统计", "PTA baseline 成功", f"{pta_success_count}/{planned_baseline_runs}")
    log_kv("统计", "MSA baseline 成功", f"{msa_success_count}/{planned_baseline_runs}")
    log_kv("统计", "变体失败", variant_failure_count)
    for item in model_results:
        log_info(
            "[FullNet][总览] {model}: {status} | planned={planned} | PTA={pta} | MSA={msa} | failed={failed} | {reason}".format(
                model=item.get("model"),
                status=item.get("status"),
                planned=item.get("planned_variant_runs", 0),
                pta=item.get("pta_baseline_success", 0),
                msa=item.get("msa_baseline_success", 0),
                failed=item.get("failed_variant_runs", 0),
                reason=item.get("reason", ""),
            )
        )
    for record in run_records:
        trainings = record.get("trainings", {})
        log_info(
            "[FullNet][训练总览] {model}/{variant}/iter{iteration}: {overall} | prepare={prepare} | pta-baseline={pta} | msa-baseline={msa} | baseline-align={align} | pta-preturb={pta_p} | msa-preturb={msa_p} | {reason}".format(
                model=record.get("model"),
                variant=record.get("variant"),
                iteration=record.get("iteration"),
                overall=record.get("overall_status"),
                prepare=trainings.get(TRAIN_PREPARE, ""),
                pta=trainings.get(TRAIN_PTA_BASELINE, ""),
                msa=trainings.get(TRAIN_MSA_BASELINE, ""),
                align=trainings.get("baseline-align", ""),
                pta_p=trainings.get(TRAIN_PTA_PRETURB, ""),
                msa_p=trainings.get(TRAIN_MSA_PRETURB, ""),
                reason=record.get("reason", ""),
            )
        )
    log_kv("统计", "汇总文件", records_root / "summary.json")
    log_kv("统计", "总览文件", summary_md_path)
    log_kv("概览", "结束时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return exit_code
