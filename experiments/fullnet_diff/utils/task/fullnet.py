#!/usr/bin/env python3
"""
语言模型整网组装与差分验证链路。
重构自旧版 internal_exe_script.sh
"""

import json
import os
import subprocess
import shutil
import time
import shlex
from datetime import datetime
from pathlib import Path
import yaml

import utils
from utils.analyze.precision import find_preferred_loss_mismatch
from utils.runtime.paths import MODEL_CONFIG_DIR, MUTATION_SCRIPT_DIR, RUNTIME_SCRIPT_DIR, TOKENIZER_DIR, repo_rel
from utils.runtime.profiler_tools import generate_profile_report
from utils.task import data_helpers, runtime_helpers

LMSV_ROOT = Path(__file__).resolve().parents[2]
PROJECT_TMP_ROOT = LMSV_ROOT / "tmp"
FULLNET_TMP_ROOT = PROJECT_TMP_ROOT / "fullnet"
MODEL_CONFIG_REL = repo_rel(MODEL_CONFIG_DIR)
MUTATION_SCRIPT_REL = repo_rel(MUTATION_SCRIPT_DIR)
RUNTIME_SCRIPT_REL = repo_rel(RUNTIME_SCRIPT_DIR)
TOKENIZER_BAICHUAN_REL = repo_rel(TOKENIZER_DIR / "baichuan2")


class Config:
    # 任务参数
    MODE = "DEVELOP"
    TOTAL_ITER = 1
    TEST_ITERATIONS = 1
    BASE_SEED = 43
    MUTNM = 0
    MODELS = ["qwen2"]
    NODE_NUM = 0
    MUTATION_ROUNDS = TOTAL_ITER
    ARGS_PATH = "assets/runtime/configs/mutation_schema.yaml"
    SAVE_STEPS = 1
    LOAD_STEPS = 15
    FULLNET_ASSEMBLY_MODE = "single_model_fullnet"

    # 运行配置
    PTA_ENV = "mindspeed"
    MSA_ENV = "msadapter"
    PTA_MAX_RUNTIME = 3000
    MAX_MUTATION_WAIT = 600
    MSA_MAX_RUNTIME = 3000
    LOG_INIT_WAIT = 240
    LOG_STABLE_THRESHOLD = 150
    SAVE_ABNORMAL_WEIGHTS = True
    TARGET_TENSOR_PARALLEL_SIZE = 0
    TARGET_PIPELINE_PARALLEL_SIZE = 0
    TARGET_EXPERT_PARALLEL_SIZE = 0
    TARGET_NPUS_PER_NODE = 0
    TARGET_WORLD_SIZE = 0
    TARGET_MASTER_ADDR = "localhost"
    TARGET_MASTER_PORT = 6000

    # 路径配置
    LOG_PATH = "res/internal_execution.log"
    MSA_MONITOR_LOG = "msrun_log/worker_0.log"
    PTA_CSV_PATH = "res/execution_pta.csv"
    MSA_CSV_PATH = "res/execution_msa.csv"
    PERSIST_ROOT = ""
    SHARED_WEIGHT_TMP_ROOT = str(FULLNET_TMP_ROOT / "shared_weight")
    TRACE_ENABLED = True
    TRACE_FULL_WEIGHTS = True
    TRACE_PERTURBATION_RUNS = True
    TRACE_PERTURB_EPS = "1e-5"
    TRACE_PERTURB_SEED = ""
    BASELINE_ALIGNMENT_REQUIRED = True
    BASELINE_LOSS_TOLERANCE = 0.0


LOG_SCOPE = "FullNet"


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

    if not constraints:
        return max_tp

    for candidate in range(max_tp, 0, -1):
        if all(value % candidate == 0 for value in constraints):
            return candidate
    return 1


def resolve_distributed_config():
    inferred_cards = _infer_visible_device_count()
    tp = _parse_optional_positive_int(Config.TARGET_TENSOR_PARALLEL_SIZE)
    pp = _parse_optional_positive_int(Config.TARGET_PIPELINE_PARALLEL_SIZE)
    ep = _parse_optional_positive_int(Config.TARGET_EXPERT_PARALLEL_SIZE)

    if tp <= 0 and pp <= 0 and ep <= 0:
        tp = inferred_cards
        pp = 1
        ep = 1
    else:
        tp = max(1, tp)
        pp = max(1, pp)
        ep = max(1, ep)

    parallel_cards = max(1, tp * pp * ep)
    configured_world = int(Config.TARGET_WORLD_SIZE or 0)

    configured_npus = int(Config.TARGET_NPUS_PER_NODE or 0)
    if configured_npus <= 0:
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
        "inferred_cards": inferred_cards,
        "nnodes": 1,
        "node_rank": 0,
        "master_addr": str(Config.TARGET_MASTER_ADDR),
        "master_port": int(Config.TARGET_MASTER_PORT),
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
        )
    ):
        return

    visible_cards = _infer_visible_device_count()
    Config.TARGET_TENSOR_PARALLEL_SIZE = _infer_model_aware_tensor_parallel_size(model_paths, visible_cards)
    Config.TARGET_PIPELINE_PARALLEL_SIZE = 1
    Config.TARGET_EXPERT_PARALLEL_SIZE = 1


def resolve_msa_monitor_log():
    worker_index = max(0, resolve_distributed_config()["npus_per_node"] - 1)
    return f"msrun_log/worker_{worker_index}.log"


def _to_bool(value):
    return data_helpers.parse_bool(value)


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
    perturb=False,
    perturb_eps=None,
):
    enabled = "1" if Config.TRACE_ENABLED or os.environ.get("LMSV_FULLNET_TRACE") == "1" else "0"
    perturb_value = "1" if perturb else "0"
    eps = str(perturb_eps or Config.TRACE_PERTURB_EPS)
    lines = [
        f"export LMSV_FULLNET_TRACE={enabled}",
        f"export LMSV_FULLNET_TRACE_BACKEND={shlex.quote(str(backend))}",
        f"export LMSV_FULLNET_TRACE_RUN={shlex.quote(str(run_name))}",
        f"export LMSV_FULLNET_TRACE_ITER={int(iter_num)}",
        f"export LMSV_FULLNET_TRACE_DIR={shlex.quote(str(Path(trace_dir).resolve()))}",
        f"export LMSV_FULLNET_TRACE_FULL_WEIGHTS={'1' if Config.TRACE_FULL_WEIGHTS else '0'}",
        f"export LMSV_FULLNET_PERTURB={perturb_value}",
        f"export LMSV_FULLNET_PERTURB_EPS={shlex.quote(eps)}",
    ]
    if Config.TRACE_PERTURB_SEED:
        lines.append(f"export LMSV_FULLNET_PERTURB_SEED={shlex.quote(str(Config.TRACE_PERTURB_SEED))}")
    if Config.TRACE_ENABLED:
        lines.append("export LMSV_DEBUG_COMPARE=${LMSV_DEBUG_COMPARE:-1}")
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


def backup_artifact_to_output(src_path, run_dir, iter_num, category, dst_name=None, missing_log_level="warn"):
    return runtime_helpers.backup_artifact_to_output(
        src_path,
        run_dir,
        iter_num,
        category,
        LMSV_ROOT,
        log_info,
        log_warn,
        dst_name=dst_name,
        missing_log_level=missing_log_level,
    )


def backup_weight_on_pta_msa_failure(weight_path, run_dir, iter_num, reason):
    if not Config.SAVE_ABNORMAL_WEIGHTS:
        log_info(f"[iter{iter_num}] 已关闭异常迭代权重备份，跳过保存: {reason}")
        return False
    return runtime_helpers.backup_weight_on_pta_msa_failure(
        weight_path,
        run_dir,
        iter_num,
        reason,
        LMSV_ROOT,
        log_info,
        log_warn,
    )


def backup_weight_on_precision_issue(weight_path, run_dir, iter_num, reason):
    if not Config.SAVE_ABNORMAL_WEIGHTS:
        log_info(f"[iter{iter_num}] 已关闭异常迭代权重备份，跳过保存: {reason}")
        return False
    return runtime_helpers.backup_weight_on_precision_issue(
        weight_path,
        run_dir,
        iter_num,
        reason,
        LMSV_ROOT,
        log_info,
        log_warn,
    )


def backup_runtime_log_to_output(log_path, run_dir, iter_num, dst_name=None):
    return runtime_helpers.backup_runtime_log_to_output(
        log_path,
        run_dir,
        iter_num,
        LMSV_ROOT,
        log_info,
        log_warn,
        dst_name=dst_name,
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


def resolve_result_dir_name(model_paths, explicit=None):
    """
    推导 mutate 产物目录名。
    migrate 后的 mutate_graph-auto.py 在 module 为逗号列表时，
    最终目录会落在“最后一个模型”的 stem（例如 modelA,modelB -> res/modelB）。
    """
    if explicit:
        return Path(str(explicit)).stem
    if not model_paths:
        return ""
    return Path(model_paths[-1]).stem


def cleanup_shared_weight_file(weight_path):
    data_helpers.cleanup_shared_weight_file(weight_path)


def remove_iteration_rows(csv_path, iteration):
    return data_helpers.remove_iteration_rows(LMSV_ROOT / csv_path, iteration, log_warn, log_info)


def csv_has_iteration(csv_path, iteration):
    return data_helpers.csv_has_iteration(csv_path, iteration)


def csv_iteration_is_valid(csv_path, iteration):
    return data_helpers.csv_iteration_is_valid(csv_path, iteration)


def pta_result_csv_path(iteration):
    return LMSV_ROOT / Config.PTA_CSV_PATH


def wait_for_mutation_artifacts(iter_num, result_dir_name):
    """等待 mutate 产物生成（json + yaml）。"""
    json_path = LMSV_ROOT / "res" / result_dir_name / f"mutating-{iter_num}.json"
    yaml_path = LMSV_ROOT / "res" / result_dir_name / f"mutated_config_iter_{iter_num:03d}.yaml"

    deadline = time.time() + Config.MAX_MUTATION_WAIT
    while time.time() < deadline:
        if (
            json_path.exists()
            and json_path.stat().st_size > 0
            and yaml_path.exists()
            and yaml_path.stat().st_size > 0
        ):
            log_info(f"Mutation产物就绪: {json_path} | {yaml_path}")
            return True
        log_info(f"等待Mutation产物中... 迭代{iter_num}")
        time.sleep(10)

    log_error(f"等待Mutation产物超时（>{Config.MAX_MUTATION_WAIT}s）")
    return False


def wait_msa_finish(iter_num):
    """等待 MSA 校验完成。成功以日志稳定且结果 CSV 出现当前轮有效指标为准。"""
    log_step(f"等待MSA验证完成 | 迭代{iter_num}")
    log_path = LMSV_ROOT / Config.MSA_MONITOR_LOG
    csv_path = LMSV_ROOT / Config.MSA_CSV_PATH
    return runtime_helpers.wait_msa_finish(
        iter_num=iter_num,
        log_path=log_path,
        total_timeout=Config.MSA_MAX_RUNTIME,
        init_wait=Config.LOG_INIT_WAIT,
        stable_threshold=Config.LOG_STABLE_THRESHOLD,
        poll_interval=20,
        log_info=log_info,
        log_error=log_error,
        success_checker=lambda: csv_iteration_is_valid(csv_path, iter_num),
        result_exists_checker=lambda: csv_has_iteration(csv_path, iter_num),
    )


def wait_msa_finish_csv(iter_num, csv_path, label):
    log_step(f"等待MSA验证完成 | {label} | 迭代{iter_num}")
    log_path = LMSV_ROOT / Config.MSA_MONITOR_LOG
    csv_path = Path(csv_path)
    return runtime_helpers.wait_msa_finish(
        iter_num=iter_num,
        log_path=log_path,
        total_timeout=Config.MSA_MAX_RUNTIME,
        init_wait=Config.LOG_INIT_WAIT,
        stable_threshold=Config.LOG_STABLE_THRESHOLD,
        poll_interval=20,
        log_info=log_info,
        log_error=log_error,
        success_checker=lambda: csv_iteration_is_valid(csv_path, iter_num),
        result_exists_checker=lambda: csv_has_iteration(csv_path, iter_num),
    )


def init_workspace(result_dir_name):
    """清理本任务相关历史产物。"""
    log_step("初始化整网工作目录")
    targets = [
        LMSV_ROOT / "msrun_log",
        LMSV_ROOT / "ms" / result_dir_name,
        LMSV_ROOT / "pta" / result_dir_name,
        LMSV_ROOT / "res" / result_dir_name,
        LMSV_ROOT / "res" / "training_log_pta",
        LMSV_ROOT / "res" / "training_log_msa",
        LMSV_ROOT / "res" / "analyse_report",
        LMSV_ROOT / "res" / "execution_pta.csv",
        LMSV_ROOT / "res" / "execution_msa.csv",
    ]

    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        runtime_helpers.clear_path(target)

    (LMSV_ROOT / "res").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "msrun_log").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "res" / "training_log_pta").mkdir(parents=True, exist_ok=True)
    (LMSV_ROOT / "res" / "training_log_msa").mkdir(parents=True, exist_ok=True)

def snapshot_iter_artifacts(iter_num, run_dir, result_dir_name):
    """收集每轮关键产物，便于追溯。"""
    iter_dir = Path(run_dir) / f"iter_{iter_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    msrun_src = LMSV_ROOT / "msrun_log"
    if msrun_src.exists():
        shutil.copytree(msrun_src, iter_dir / "msrun_log", dirs_exist_ok=True)

    model_res_dir = LMSV_ROOT / "res" / result_dir_name
    if model_res_dir.exists():
        mutation_dir = iter_dir / "mutation_inputs"
        mutation_dir.mkdir(parents=True, exist_ok=True)
        mutation_files = [
            model_res_dir / f"mutating-{iter_num}.json",
            model_res_dir / f"mutating-{iter_num}-err.json",
            model_res_dir / f"mutated_config_iter_{iter_num:03d}.yaml",
            model_res_dir / "mutate_log.txt",
        ]
        for src in mutation_files:
            if src.exists():
                shutil.copy2(src, mutation_dir / src.name)


def _repo_rel_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        try:
            return path.relative_to(LMSV_ROOT).as_posix()
        except ValueError:
            return str(path)
    return path.as_posix()


def write_iteration_status(
    iter_num,
    run_dir,
    overall_status,
    reason="",
    *,
    mutate_result="SKIP",
    pta_save_result="SKIP",
    pta_load_result="SKIP",
    msa_load_result="SKIP",
    baseline_align_result="SKIP",
):
    iter_dir = Path(run_dir) / f"iter_{iter_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "task_name": "fullnet",
        "iteration": iter_num,
        "overall_status": overall_status,
        "reason": reason,
        "components": {
            "MUTATE": mutate_result,
            "PTA_SAVE": pta_save_result,
            "PTA_LOAD": pta_load_result,
            "MSA_LOAD": msa_load_result,
            "BASELINE_ALIGN": baseline_align_result,
        },
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(iter_dir / "status.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if overall_status != "PASS":
        with open(iter_dir / "FAILED_FLAG", "w", encoding="utf-8") as handle:
            handle.write(
                "MUTATE={MUTATE} PTA_SAVE={PTA_SAVE} PTA_LOAD={PTA_LOAD} "
                "MSA_LOAD={MSA_LOAD} BASELINE_ALIGN={BASELINE_ALIGN}\n".format(**payload["components"])
            )
        with open(iter_dir / "failure_info.txt", "w", encoding="utf-8") as handle:
            handle.write(
                "FAILED_COMPONENTS: "
                "MUTATE={MUTATE} PTA_SAVE={PTA_SAVE} PTA_LOAD={PTA_LOAD} "
                "MSA_LOAD={MSA_LOAD} BASELINE_ALIGN={BASELINE_ALIGN}\n".format(**payload["components"])
            )
            if reason:
                handle.write(f"REASON: {reason}\n")


def write_baseline_alignment_report(
    iter_num,
    run_dir,
    *,
    issue,
    tolerance,
    required,
    pta_csv,
    msa_csv,
    pta_step_csv,
    msa_step_csv,
):
    iter_dir = Path(run_dir) / f"iter_{iter_num}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iter_num,
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
    with open(iter_dir / "baseline_alignment.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def run_pta_mutate(iter_num, mutate_args, exec_log_file, pta_env, pta_path):
    cmd = f"""
    {build_conda_activate_block(pta_env, load_ascend=True)}
    export PTA_PATH={shlex.quote(pta_path)}
    export PTAPATH={shlex.quote(pta_path)}
    source scripts/envset/pta.sh
    export LMSV_MUTATE_SKIP_FORWARD=${{LMSV_MUTATE_SKIP_FORWARD:-1}}
    export LMSV_FULLNET_ASSEMBLY=1
    export LMSV_FULLNET_ASSEMBLY_MODE={shlex.quote(Config.FULLNET_ASSEMBLY_MODE)}
    export MUTATE_ROUND={iter_num}
    export MUTATE_ARGS={shlex.quote(mutate_args)}
    bash {shlex.quote(f"{MUTATION_SCRIPT_REL}/mutate-auto.sh")}
    """
    result = run_shell_to_file(
        cmd,
        exec_log_file,
        check=False,
        timeout=Config.PTA_MAX_RUNTIME,
        timeout_label="PTA执行",
    )
    return result is not None and result.returncode == 0


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
    trace_run_name="pta_baseline",
    perturb=False,
    perturb_eps=None,
):
    train_iters = int(train_iters)
    dist_cfg = resolve_distributed_config()
    if step_log_csv_path:
        step_log_block = f"export LMSV_TRAINING_LOG_CSV={shlex.quote(str(Path(step_log_csv_path).resolve()))}"
    else:
        step_log_block = "unset LMSV_TRAINING_LOG_CSV"
    pta_csv_path = result_csv_path or (LMSV_ROOT / Config.PTA_CSV_PATH)
    trace_block = _build_trace_env_block(
        iter_num=iter_num,
        trace_dir=trace_dir or (Path(Config.PERSIST_ROOT) / "iters" / f"iter_{iter_num}" / "traces"),
        backend="pta",
        run_name=trace_run_name,
        perturb=perturb,
        perturb_eps=perturb_eps,
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
    trace_run_name="pta_baseline",
    perturb=False,
    perturb_eps=None,
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
        trace_run_name=trace_run_name,
        perturb=perturb,
        perturb_eps=perturb_eps,
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
    trace_run_name="msa_baseline",
    perturb=False,
    perturb_eps=None,
):
    train_iters = int(train_iters)
    dist_cfg = resolve_distributed_config()
    if step_log_csv_path:
        step_log_block = f"export LMSV_TRAINING_LOG_CSV={shlex.quote(str(Path(step_log_csv_path).resolve()))}"
    else:
        step_log_block = "unset LMSV_TRAINING_LOG_CSV"
    msa_csv_path = result_csv_path or (LMSV_ROOT / Config.MSA_CSV_PATH)
    trace_block = _build_trace_env_block(
        iter_num=iter_num,
        trace_dir=trace_dir or (Path(Config.PERSIST_ROOT) / "iters" / f"iter_{iter_num}" / "traces"),
        backend="msa",
        run_name=trace_run_name,
        perturb=perturb,
        perturb_eps=perturb_eps,
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
    trace_run_name="msa_baseline",
    perturb=False,
    perturb_eps=None,
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
        trace_run_name=trace_run_name,
        perturb=perturb,
        perturb_eps=perturb_eps,
    )
    if script_output_path:
        write_runtime_script(script_output_path, cmd)
    result = run_shell_to_file(cmd, exec_log_file, check=False)
    return result is not None and result.returncode == 0


def _apply_config(params):
    Config.MODE = "DEVELOP"
    Config.TOTAL_ITER = 1
    Config.TEST_ITERATIONS = 1
    Config.MUTATION_ROUNDS = 1
    Config.BASE_SEED = 43
    Config.MUTNM = 0
    Config.SAVE_STEPS = 1
    Config.LOAD_STEPS = 15
    Config.FULLNET_ASSEMBLY_MODE = "single_model_fullnet"
    Config.PTA_MAX_RUNTIME = 3000
    Config.MAX_MUTATION_WAIT = 600
    Config.MSA_MAX_RUNTIME = 3000
    Config.LOG_INIT_WAIT = 240
    Config.LOG_STABLE_THRESHOLD = 150
    Config.TARGET_TENSOR_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_TENSOR_PARALLEL_SIZE", Config.TARGET_TENSOR_PARALLEL_SIZE),
    )
    Config.TARGET_PIPELINE_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_PIPELINE_PARALLEL_SIZE", Config.TARGET_PIPELINE_PARALLEL_SIZE),
    )
    Config.TARGET_EXPERT_PARALLEL_SIZE = _parse_optional_positive_int(
        params.get("TARGET_EXPERT_PARALLEL_SIZE", Config.TARGET_EXPERT_PARALLEL_SIZE),
    )
    Config.TARGET_NPUS_PER_NODE = int(params.get("TARGET_NPUS_PER_NODE", Config.TARGET_NPUS_PER_NODE) or 0)
    Config.TARGET_WORLD_SIZE = int(params.get("TARGET_WORLD_SIZE", Config.TARGET_WORLD_SIZE) or 0)
    Config.TARGET_MASTER_ADDR = str(params.get("TARGET_MASTER_ADDR", Config.TARGET_MASTER_ADDR))
    Config.TARGET_MASTER_PORT = int(params.get("TARGET_MASTER_PORT", Config.TARGET_MASTER_PORT))
    Config.ARGS_PATH = "assets/runtime/configs/mutation_schema.yaml"
    Config.PTA_ENV = str(params.get("PTA_ENV", os.environ.get("PTA_NAME", Config.PTA_ENV)))
    Config.MSA_ENV = str(params.get("MSA_ENV", os.environ.get("MSA_NAME", Config.MSA_ENV)))
    Config.SAVE_ABNORMAL_WEIGHTS = True
    os.environ["BASE_SEED"] = str(Config.BASE_SEED)

    raw_persist_root = str(params.get("PERSIST_ROOT", os.environ.get("LMSV_OUTPATH", str(LMSV_ROOT / "output"))))
    persist_root_path = Path(raw_persist_root).expanduser()
    if not persist_root_path.is_absolute():
        persist_root_path = LMSV_ROOT / persist_root_path
    Config.PERSIST_ROOT = str(persist_root_path.resolve())

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


def _refresh_analysis(run_dir, model_paths, max_iterations):
    try:
        from utils.analyze.fullnet_result import analyze_fullnet_run

        analyze_fullnet_run(
            output_root=Path(Config.PERSIST_ROOT).resolve(),
            run_dir=run_dir,
            model_name=",".join(Path(path).stem for path in model_paths),
            planned_iterations=max_iterations,
        )
    except Exception as exc:
        log_warn(f"整网分析刷新失败，已跳过: {exc}")


def _archive_result_csvs(iter_num, run_dir, *, pta_csv=None, include_msa=False):
    if pta_csv is not None:
        backup_artifact_to_output(pta_csv, run_dir, iter_num, "", "execution_pta.csv", missing_log_level="info")
    if include_msa:
        backup_artifact_to_output(LMSV_ROOT / Config.MSA_CSV_PATH, run_dir, iter_num, "", "execution_msa.csv", missing_log_level="info")


def main(params):
    project_tmp_root = configure_project_tmp_env()
    utils.control.clean.kill_pretraingpt()
    _apply_config(params)

    model_paths = normalize_models(params.get("MODELS", Config.MODELS))
    if not model_paths:
        log_error("整网参数错误：MODELS 为空或格式非法")
        return 1
    assembly_model_paths = _assembly_model_paths(model_paths)
    configure_auto_parallel_from_models(assembly_model_paths)
    Config.MSA_MONITOR_LOG = resolve_msa_monitor_log()

    inferred_decoder_count = _infer_fullnet_decoder_count(assembly_model_paths)
    Config.NODE_NUM = inferred_decoder_count
    Config.MUTATION_ROUNDS = 1

    pta_path = os.environ.get("PTA_PATH") or os.environ.get("PTAPATH")
    msa_path = os.environ.get("MSA_PATH") or os.environ.get("MSAPATH")
    if not pta_path:
        log_error("环境变量缺失：请先配置 PTA_PATH")
        return 1
    if not msa_path:
        log_error("环境变量缺失：请先配置 MSA_PATH")
        return 1

    max_iterations = 1
    mutate_args_common = build_mutate_args(assembly_model_paths, Config.NODE_NUM, Config.MUTNM, Config.MUTATION_ROUNDS)
    mutate_args_for_mutate = (
        f"{mutate_args_common} --args_path {Config.ARGS_PATH}" if Config.ARGS_PATH else mutate_args_common
    )
    result_dir_name = resolve_result_dir_name(assembly_model_paths, params.get("RESULT_DIR_NAME"))
    if not result_dir_name:
        log_error("整网参数错误：无法推导变异结果目录")
        return 1

    log_step("整网链路启动")
    log_kv("配置", "迭代次数", max_iterations)
    log_kv("配置", "基础随机种子", Config.BASE_SEED)
    log_kv("配置", "模型配置", model_paths)
    log_kv("配置", "整网基准模型", assembly_model_paths)
    log_kv("配置", "Decoder层数", Config.NODE_NUM)
    log_kv("配置", "变异结果目录", f"res/{result_dir_name}")
    log_kv("配置", "对比模式", "pta_msa")
    log_kv("配置", "MUTATE_ARGS(verify/load)", mutate_args_common)
    log_kv("配置", "MUTATE_ARGS(mutate)", mutate_args_for_mutate)
    log_kv("配置", "训练步数", f"SAVE({Config.SAVE_STEPS}) | LOAD({Config.LOAD_STEPS})")
    dist_cfg = resolve_distributed_config()
    log_kv(
        "配置",
        "单机并行设置",
        f"TP={dist_cfg['tp']} | PP={dist_cfg['pp']} | EP={dist_cfg['ep']} | NPUS_PER_NODE={dist_cfg['npus_per_node']} | WORLD_SIZE={dist_cfg['world_size']}",
    )
    log_kv("配置", "当前执行对", "PTA + MSA")
    log_kv("配置", "激活环境", f"PTA={Config.PTA_ENV} | MSA={Config.MSA_ENV}")
    log_kv("配置", "项目临时目录", project_tmp_root)
    log_kv("配置", "共享权重临时目录", Config.SHARED_WEIGHT_TMP_ROOT)
    log_kv("配置", "Trace导出", f"{Config.TRACE_ENABLED} | full_weights={Config.TRACE_FULL_WEIGHTS} | perturb_runs={Config.TRACE_PERTURBATION_RUNS}")
    log_kv("配置", "Baseline精度门槛", f"required={Config.BASELINE_ALIGNMENT_REQUIRED} | loss_tolerance={Config.BASELINE_LOSS_TOLERANCE}")
    log_kv("概览", "开始时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    init_workspace(result_dir_name)
    run_dir = (Path(Config.PERSIST_ROOT) / "iters").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    shared_weight_tmp_run_dir = (
        Path(Config.SHARED_WEIGHT_TMP_ROOT) / f"fullnet_{run_dir.parent.name}_{os.getpid()}"
    ).resolve()
    shared_weight_tmp_run_dir.mkdir(parents=True, exist_ok=True)

    pta_success_count = 0
    msa_success_count = 0
    exit_code = 0
    try:
        for i in range(1, max_iterations + 1):
            try:
                log_step(f"开始迭代 {i}/{max_iterations}")
                utils.control.clean.kill_pretraingpt()

                mutate_result = "SKIP"
                pta_save_result = "SKIP"
                pta_load_result = "SKIP"
                msa_load_result = "SKIP"

                runtime_log_dir = run_dir / f"iter_{i}" / "runtime_logs"
                runtime_log_dir.mkdir(parents=True, exist_ok=True)
                script_artifact_dir = run_dir / f"iter_{i}" / "scripts"
                trace_dir = run_dir / f"iter_{i}" / "traces"
                mutate_log = runtime_log_dir / f"pta_mutate_iter{i}.log"
                pta_save_log = runtime_log_dir / f"pta_save_iter{i}.log"
                pta_load_log = runtime_log_dir / f"pta_load_iter{i}.log"
                msa_load_log = runtime_log_dir / f"msa_load_iter{i}.log"
                pta_perturb_log = runtime_log_dir / f"pta_perturb_iter{i}.log"
                msa_perturb_log = runtime_log_dir / f"msa_perturb_iter{i}.log"
                msa_profile_dir = run_dir / f"iter_{i}" / "profiler" / "msa-load"
                msa_profile_report_dir = run_dir / f"iter_{i}" / "analysis" / "msa-profiler"
                pta_step_csv = LMSV_ROOT / "res" / "training_log_pta" / f"training_log-{i}.csv"
                msa_step_csv = LMSV_ROOT / "res" / "training_log_msa" / f"training_log-{i}.csv"
                pta_perturb_step_csv = LMSV_ROOT / "res" / "training_log_pta" / f"training_log-perturb-{i}.csv"
                msa_perturb_step_csv = LMSV_ROOT / "res" / "training_log_msa" / f"training_log-perturb-{i}.csv"
                pta_perturb_csv = LMSV_ROOT / "res" / "execution_pta_perturb.csv"
                msa_perturb_csv = LMSV_ROOT / "res" / "execution_msa_perturb.csv"
                pta_step_csv.unlink(missing_ok=True)
                msa_step_csv.unlink(missing_ok=True)
                pta_perturb_step_csv.unlink(missing_ok=True)
                msa_perturb_step_csv.unlink(missing_ok=True)
                remove_iteration_rows(pta_perturb_csv, i)
                remove_iteration_rows(msa_perturb_csv, i)

                shared_weight_file = (shared_weight_tmp_run_dir / f"iter{i}.pth").resolve()
                shared_weight_file.parent.mkdir(parents=True, exist_ok=True)
                cleanup_shared_weight_file(shared_weight_file)
                shared_weight_path = str(shared_weight_file)
                load_path = f"res/{result_dir_name}/mutating-{i}.json"
                load_path_abs = LMSV_ROOT / load_path

                log_step("1. PTA侧生成整网变异配置")
                mutate_ok = run_pta_mutate(i, mutate_args_for_mutate, mutate_log, Config.PTA_ENV, pta_path)
                backup_runtime_log_to_output(mutate_log, run_dir, i)
                if not mutate_ok or not wait_for_mutation_artifacts(i, result_dir_name):
                    mutate_result = "ERROR"
                    reason = "变异配置生成失败，整网实验已停止"
                    log_error(f"[iter{i}] {reason}，详见 {mutate_log}")
                    write_iteration_status(i, run_dir, "FAILED", reason, mutate_result=mutate_result)
                    snapshot_iter_artifacts(i, run_dir, result_dir_name)
                    cleanup_shared_weight_file(shared_weight_file)
                    exit_code = 1
                    break
                mutate_result = "OK"

                if not load_path_abs.exists() or load_path_abs.stat().st_size <= 0:
                    reason = f"变异JSON缺失: {load_path_abs}"
                    log_error(f"[iter{i}] {reason}，整网实验已停止")
                    write_iteration_status(i, run_dir, "FAILED", reason, mutate_result=mutate_result)
                    snapshot_iter_artifacts(i, run_dir, result_dir_name)
                    cleanup_shared_weight_file(shared_weight_file)
                    exit_code = 1
                    break

                log_step("2. PTA-SAVE：生成共享权重")
                pta_save_ok = run_pta_verify_stage(
                    i, mutate_args_common, load_path, pta_save_log, Config.PTA_ENV, pta_path,
                    shared_weight_path, "save", Config.SAVE_STEPS,
                    trace_dir=trace_dir,
                    trace_run_name="pta_save",
                    script_output_path=script_artifact_dir / f"pta-save_iter{i}.sh",
                )
                backup_runtime_log_to_output(pta_save_log, run_dir, i)
                if not pta_save_ok or not shared_weight_file.exists() or shared_weight_file.stat().st_size <= 0:
                    pta_save_result = "ERROR"
                    backup_weight_on_pta_msa_failure(shared_weight_file, run_dir, i, "PTA-SAVE失败或未产出共享权重")
                    write_iteration_status(i, run_dir, "FAILED", "PTA-SAVE失败", mutate_result=mutate_result, pta_save_result=pta_save_result)
                    snapshot_iter_artifacts(i, run_dir, result_dir_name)
                    cleanup_shared_weight_file(shared_weight_file)
                    exit_code = 1
                    break
                pta_save_result = "OK"
                remove_iteration_rows(Config.PTA_CSV_PATH, i)

                utils.control.clean.kill_pretraingpt()
                log_step("3. PTA-LOAD：整网基线回放")
                pta_load_ok = run_pta_verify_stage(
                    i, mutate_args_common, load_path, pta_load_log, Config.PTA_ENV, pta_path,
                    shared_weight_path, "load", Config.LOAD_STEPS,
                    step_log_csv_path=pta_step_csv,
                    trace_dir=trace_dir,
                    trace_run_name="pta_baseline",
                    script_output_path=script_artifact_dir / f"pta-load_iter{i}.sh",
                )
                backup_runtime_log_to_output(pta_load_log, run_dir, i)
                backup_artifact_to_output(pta_step_csv, run_dir, i, "", f"training_log_pta-{i}.csv", missing_log_level="info")
                pta_result_csv = pta_result_csv_path(i)
                _archive_result_csvs(i, run_dir, pta_csv=pta_result_csv)
                if not pta_load_ok or not csv_iteration_is_valid(pta_result_csv, i):
                    pta_load_result = "ERROR"
                    backup_weight_on_pta_msa_failure(shared_weight_file, run_dir, i, "PTA-LOAD失败或结果无效")
                    write_iteration_status(i, run_dir, "FAILED", "PTA-LOAD失败或结果无效", mutate_result=mutate_result, pta_save_result=pta_save_result, pta_load_result=pta_load_result)
                    snapshot_iter_artifacts(i, run_dir, result_dir_name)
                    cleanup_shared_weight_file(shared_weight_file)
                    exit_code = 1
                    break
                pta_load_result = "OK"
                pta_success_count += 1
                utils.control.clean.kill_pretraingpt()

                log_step("4. MSA-LOAD：整网差分回放")
                msa_load_ok = run_msa_verify_load(
                    i, mutate_args_common, load_path, msa_load_log, Config.MSA_ENV, msa_path,
                    shared_weight_path, Config.LOAD_STEPS,
                    step_log_csv_path=msa_step_csv,
                    profile_output_dir=msa_profile_dir,
                    trace_dir=trace_dir,
                    trace_run_name="msa_baseline",
                    script_output_path=script_artifact_dir / f"msa-load_iter{i}.sh",
                )
                backup_runtime_log_to_output(msa_load_log, run_dir, i)
                if not msa_load_ok or not wait_msa_finish(i) or not csv_iteration_is_valid(LMSV_ROOT / Config.MSA_CSV_PATH, i):
                    msa_load_result = "ERROR"
                    backup_weight_on_pta_msa_failure(shared_weight_file, run_dir, i, "MSA-LOAD失败或结果无效")
                    write_iteration_status(i, run_dir, "FAILED", "MSA-LOAD失败或结果无效", mutate_result=mutate_result, pta_save_result=pta_save_result, pta_load_result=pta_load_result, msa_load_result=msa_load_result)
                    snapshot_iter_artifacts(i, run_dir, result_dir_name)
                    cleanup_shared_weight_file(shared_weight_file)
                    exit_code = 1
                    break
                backup_artifact_to_output(msa_step_csv, run_dir, i, "", f"training_log_msa-{i}.csv", missing_log_level="info")
                _archive_result_csvs(i, run_dir, include_msa=True)
                if msa_profile_dir.exists() and any(msa_profile_dir.rglob("*")):
                    generate_profile_report(msa_profile_dir, msa_profile_report_dir, msa_step_csv, msa_load_log, "FullNet-MSA", i)
                msa_load_result = "OK"
                msa_success_count += 1

                log_step("5. Baseline精度对齐检查：PTA baseline vs MSA baseline")
                precision_issue = find_preferred_loss_mismatch(
                    pta_result_csv_path(i), LMSV_ROOT / Config.MSA_CSV_PATH,
                    iteration=i,
                    tolerance=Config.BASELINE_LOSS_TOLERANCE,
                    pta_step_csv_path=pta_step_csv,
                    msa_step_csv_path=msa_step_csv,
                )
                alignment_report = write_baseline_alignment_report(
                    i,
                    run_dir,
                    issue=precision_issue,
                    tolerance=Config.BASELINE_LOSS_TOLERANCE,
                    required=Config.BASELINE_ALIGNMENT_REQUIRED,
                    pta_csv=pta_result_csv_path(i),
                    msa_csv=LMSV_ROOT / Config.MSA_CSV_PATH,
                    pta_step_csv=pta_step_csv,
                    msa_step_csv=msa_step_csv,
                )
                if precision_issue:
                    log_warn(f"[iter{i}] Baseline未对齐，跳过扰动/RQ3数据采集: {precision_issue}")
                    backup_weight_on_precision_issue(shared_weight_file, run_dir, i, precision_issue)
                    backup_artifact_to_output(run_dir / f"iter_{i}" / "baseline_alignment.json", run_dir, i, "", "baseline_alignment.json", missing_log_level="info")
                    if Config.BASELINE_ALIGNMENT_REQUIRED:
                        write_iteration_status(
                            i,
                            run_dir,
                            "FAILED",
                            "Baseline精度未对齐，已跳过扰动/RQ3数据采集",
                            mutate_result=mutate_result,
                            pta_save_result=pta_save_result,
                            pta_load_result=pta_load_result,
                            msa_load_result=msa_load_result,
                            baseline_align_result="ERROR",
                        )
                        snapshot_iter_artifacts(i, run_dir, result_dir_name)
                        cleanup_shared_weight_file(shared_weight_file)
                        exit_code = 1
                        break
                else:
                    log_info(f"[iter{i}] Baseline精度对齐通过: {alignment_report}")

                if Config.TRACE_ENABLED and Config.TRACE_PERTURBATION_RUNS:
                    utils.control.clean.kill_pretraingpt()
                    log_step("6. PTA-PERTURB：整网输入扰动回放（备份蜕变数据）")
                    pta_perturb_ok = run_pta_verify_stage(
                        i, mutate_args_common, load_path, pta_perturb_log, Config.PTA_ENV, pta_path,
                        shared_weight_path, "load", Config.LOAD_STEPS,
                        step_log_csv_path=pta_perturb_step_csv,
                        result_csv_path=pta_perturb_csv,
                        trace_dir=trace_dir,
                        trace_run_name="pta_perturb",
                        perturb=True,
                        perturb_eps=Config.TRACE_PERTURB_EPS,
                        script_output_path=script_artifact_dir / f"pta-perturb_iter{i}.sh",
                    )
                    backup_runtime_log_to_output(pta_perturb_log, run_dir, i, dst_name=f"pta_perturb_iter{i}.log")
                    backup_artifact_to_output(pta_perturb_step_csv, run_dir, i, "", f"training_log_pta_perturb-{i}.csv", missing_log_level="info")
                    backup_artifact_to_output(pta_perturb_csv, run_dir, i, "", "execution_pta_perturb.csv", missing_log_level="info")
                    if not pta_perturb_ok or not csv_iteration_is_valid(pta_perturb_csv, i):
                        log_warn(f"[iter{i}] PTA扰动回放未产出有效结果，基线结果继续保留")

                    utils.control.clean.kill_pretraingpt()
                    log_step("7. MSA-PERTURB：整网输入扰动回放（蜕变数据）")
                    msa_perturb_ok = run_msa_verify_load(
                        i, mutate_args_common, load_path, msa_perturb_log, Config.MSA_ENV, msa_path,
                        shared_weight_path, Config.LOAD_STEPS,
                        step_log_csv_path=msa_perturb_step_csv,
                        result_csv_path=msa_perturb_csv,
                        trace_dir=trace_dir,
                        trace_run_name="msa_perturb",
                        perturb=True,
                        perturb_eps=Config.TRACE_PERTURB_EPS,
                        script_output_path=script_artifact_dir / f"msa-perturb_iter{i}.sh",
                    )
                    msa_perturb_finished = wait_msa_finish_csv(i, msa_perturb_csv, "扰动") if msa_perturb_ok else False
                    backup_runtime_log_to_output(msa_perturb_log, run_dir, i, dst_name=f"msa_perturb_iter{i}.log")
                    backup_artifact_to_output(msa_perturb_step_csv, run_dir, i, "", f"training_log_msa_perturb-{i}.csv", missing_log_level="info")
                    backup_artifact_to_output(msa_perturb_csv, run_dir, i, "", "execution_msa_perturb.csv", missing_log_level="info")
                    if not msa_perturb_ok or not msa_perturb_finished or not csv_iteration_is_valid(msa_perturb_csv, i):
                        log_warn(f"[iter{i}] MSA扰动回放未产出有效结果，基线结果继续保留")
                utils.control.clean.kill_pretraingpt()

                peer_ok = msa_load_result == "OK"
                write_iteration_status(
                    i, run_dir, "PASS" if peer_ok else "PASS_WITH_WARNINGS", "迭代执行完成",
                    mutate_result=mutate_result, pta_save_result=pta_save_result,
                    pta_load_result=pta_load_result, msa_load_result=msa_load_result,
                    baseline_align_result="OK" if precision_issue is None else "WARN",
                )
                snapshot_iter_artifacts(i, run_dir, result_dir_name)
                cleanup_shared_weight_file(shared_weight_file)
            finally:
                _refresh_analysis(run_dir, model_paths, max_iterations)
    finally:
        shutil.rmtree(shared_weight_tmp_run_dir, ignore_errors=True)

    log_step("整网链路结束")
    log_kv("统计", "PTA 成功", f"{pta_success_count}/{max_iterations}")
    log_kv("统计", "MSA 成功", f"{msa_success_count}/{max_iterations}")
    log_step("开始自动分析整网结果")
    analysis = None
    try:
        from utils.analyze.fullnet_result import analyze_fullnet_run

        analysis = analyze_fullnet_run(
            output_root=Path(Config.PERSIST_ROOT).resolve(),
            run_dir=run_dir,
            model_name=",".join(Path(path).stem for path in model_paths),
            planned_iterations=max_iterations,
        )
        log_info(f"分析目录: {analysis.analysis_dir}")
        log_info(f"HTML报告: {analysis.report_html}")
        log_info(f"JSON汇总: {analysis.summary_json}")
    except Exception as exc:
        log_warn(f"自动分析失败，已跳过: {exc}")
    log_kv("概览", "结束时间", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return exit_code
