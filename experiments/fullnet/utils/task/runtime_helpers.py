#!/usr/bin/env python3
"""Shared runtime helpers used by task executors."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


def configure_project_tmp_env(project_tmp_root: Path) -> str:
    """Prepare repo-local runtime space and keep Python temp files off NFS."""
    override_root = os.environ.get("LMSV_PROJECT_TMP_ROOT", "").strip()
    if override_root:
        project_tmp_root = Path(override_root).expanduser()

    project_tmp_root.mkdir(parents=True, exist_ok=True)
    process_tmp_root = Path(
        os.environ.get("LMSV_PROCESS_TMP_ROOT", f"/tmp/framediff-fullnet-{os.getuid()}")
    ).expanduser()
    process_tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_root = str(process_tmp_root.resolve())
    for env_name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[env_name] = tmp_root
    prepare_repo_runtime_workspace(project_tmp_root, Path(__file__).resolve().parents[2])
    return str(project_tmp_root.resolve())


def build_sigterm_shield_block(env_var_name: str = "LMSV_IGNORE_PTA_SIGTERM") -> str:
    """Return a shell snippet that ignores SIGTERM when the given env var is enabled."""
    return "\n".join(
        [
            f'if [ "${{{env_var_name}:-0}}" = "1" ]; then',
            '  echo "[LMSV] SIGTERM shielding enabled for this run"',
            "  trap '' TERM",
            "fi",
        ]
    )


def prepare_repo_runtime_workspace(project_tmp_root: Path, repo_root: Path) -> Path:
    """Route repo-root runtime directories into the repo-local tmp workspace."""
    workspace_root = (project_tmp_root / "runtime_workspace").resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    os.environ["LMSV_RUNTIME_ROOT"] = str(workspace_root)

    for name in ("res", "msrun_log", "ms", "pta"):
        target = workspace_root / name
        target.mkdir(parents=True, exist_ok=True)
        link_path = repo_root / name

        if link_path.is_symlink():
            try:
                if link_path.resolve() == target:
                    continue
            except FileNotFoundError:
                pass
            link_path.unlink()
        elif link_path.exists():
            if link_path.is_dir():
                for child in link_path.iterdir():
                    shutil.move(str(child), target / child.name)
                link_path.rmdir()
            else:
                shutil.move(str(link_path), target / link_path.name)

        link_path.symlink_to(target, target_is_directory=True)

    return workspace_root


def clear_path(path: str | Path) -> None:
    """Remove a file/dir while keeping repo-root runtime symlinks intact."""
    candidate = Path(path)
    if candidate.is_symlink():
        target = candidate.resolve(strict=False)
        if not target.exists():
            return
        for child in list(target.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        return

    if not candidate.exists():
        return

    if candidate.is_dir():
        shutil.rmtree(candidate, ignore_errors=True)
    else:
        candidate.unlink(missing_ok=True)


def run_shell_to_file(
    cmd: str,
    log_file: str | Path,
    repo_root: Path,
    log_error,
    check: bool = False,
    timeout: int | float | None = None,
    timeout_label: str | None = None,
):
    """Run a bash command and persist merged stdout/stderr to a log file."""
    shell_cmd = f"set -o pipefail\n{cmd}"
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[START] {start_ts}\n")
        handle.write("[COMMAND]\n")
        handle.write(f"{cmd}\n\n")
        handle.flush()

        process = subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            cwd=str(repo_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            label = timeout_label or "命令执行"
            handle.write(f"\n[TIMEOUT] {label}超时（>{timeout}s），已终止当前进程组\n")
            handle.flush()
            log_error(f"{label}超时（>{timeout}s），已终止当前进程，按失败处理")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

        end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        handle.write(f"\n[END] {end_ts}\n")
        if timed_out:
            handle.write("[TIMED_OUT] 1\n")
        handle.write(f"[RETURNCODE] {process.returncode}\n")

    result = subprocess.CompletedProcess(process.args, process.returncode)
    result.timed_out = timed_out

    if check and result.returncode != 0:
        log_error(f"命令执行失败，日志: {log_file}")
        return None
    return result


def backup_artifact_to_output(
    src_path: str | Path,
    run_dir: str | Path,
    iter_num: int,
    category: str,
    repo_root: Path,
    log_info,
    log_warn,
    dst_name: str | None = None,
    missing_log_level: str = "warn",
) -> bool:
    """Archive a runtime artifact into the per-iteration output directory."""
    src = Path(src_path)
    if not src.is_absolute():
        src = repo_root / src

    if not src.exists():
        message = f"[iter{iter_num}] 产物不存在，跳过output备份: {src}"
        if missing_log_level == "info":
            log_info(message)
        else:
            log_warn(message)
        return False

    artifact_dir = Path(run_dir) / f"iter_{iter_num}" / category
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dst = artifact_dir / (dst_name or src.name)

    try:
        if src.resolve() == dst.resolve():
            log_info(f"[iter{iter_num}] 产物已位于归档目录，跳过重复备份: {dst}")
            return True
    except FileNotFoundError:
        pass

    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

    log_info(f"[iter{iter_num}] 已备份到output: {src} -> {dst}")
    return True


def backup_weight_on_pta_msa_failure(
    weight_path: str | Path,
    run_dir: str | Path,
    iter_num: int,
    reason: str,
    repo_root: Path,
    log_info,
    log_warn,
) -> bool:
    log_warn(f"[iter{iter_num}] 检测到PTA/MSA异常，尝试备份共享权重: {reason}")
    return backup_artifact_to_output(
        weight_path,
        run_dir,
        iter_num,
        "weights/pta-save",
        repo_root,
        log_info,
        log_warn,
    )


def backup_weight_on_precision_issue(
    weight_path: str | Path,
    run_dir: str | Path,
    iter_num: int,
    reason: str,
    repo_root: Path,
    log_info,
    log_warn,
) -> bool:
    log_warn(f"[iter{iter_num}] 检测到精度问题，尝试备份共享权重: {reason}")
    return backup_artifact_to_output(
        weight_path,
        run_dir,
        iter_num,
        "weights/pta-save",
        repo_root,
        log_info,
        log_warn,
    )


def backup_runtime_log_to_output(
    log_path: str | Path,
    run_dir: str | Path,
    iter_num: int,
    repo_root: Path,
    log_info,
    log_warn,
    dst_name: str | None = None,
) -> bool:
    """Archive runtime logs into the iteration directory."""
    return backup_artifact_to_output(
        log_path,
        run_dir,
        iter_num,
        "runtime_logs",
        repo_root,
        log_info,
        log_warn,
        dst_name=dst_name,
    )


def wait_msa_finish(
    *,
    iter_num: int,
    log_path: str | Path,
    total_timeout: int,
    init_wait: int,
    stable_threshold: int,
    poll_interval: int,
    log_info,
    log_error,
    success_checker,
    result_exists_checker=None,
) -> bool:
    """Wait for an MSA run to finish only after log activity stays stable long enough."""
    monitor_log = Path(log_path)
    monitor_name = monitor_log.name
    start_time = time.time()
    total_deadline = start_time + max(1, int(total_timeout))
    init_deadline = min(total_deadline, start_time + max(1, int(init_wait)))
    poll_seconds = max(1, int(poll_interval))
    stable_seconds_required = max(1, int(stable_threshold))

    started = False
    last_size = -1
    last_update_at = None
    last_progress_log_at = 0.0

    while time.time() < total_deadline:
        now = time.time()
        log_exists = monitor_log.exists()
        current_size = 0
        if log_exists:
            try:
                current_size = int(monitor_log.stat().st_size)
            except OSError:
                current_size = 0

        if not started:
            if log_exists:
                started = True
                last_size = current_size
                last_update_at = now
                log_info(f"检测到 MSA 日志文件生成 | {monitor_name}={current_size} bytes")
            elif now >= init_deadline:
                log_error(f"初始等待超时（>{int(init_wait)}s），未检测到MSA日志")
                return False
            else:
                log_info("等待 msrun 日志文件生成...")
                time.sleep(poll_seconds)
                continue
        else:
            if current_size != last_size:
                last_size = current_size
                last_update_at = now
                log_info(f"MSA运行中，日志持续更新 | {monitor_name}={current_size} bytes")
            else:
                stable_for = int(now - (last_update_at or now))
                if now - last_progress_log_at >= poll_seconds:
                    log_info(f"MSA日志已稳定 {stable_for}s，阈值 {stable_seconds_required}s")
                    last_progress_log_at = now
                if stable_for >= stable_seconds_required:
                    if success_checker():
                        log_info(f"检测到MSA第{iter_num}轮有效结果，验证完成")
                        return True
                    if result_exists_checker and result_exists_checker():
                        log_error(f"MSA第{iter_num}轮结果已写出但无有效指标")
                    else:
                        log_error(f"MSA日志已稳定，但第{iter_num}轮结果仍未写出")
                    return False

        time.sleep(poll_seconds)

    log_error(f"等待MSA结果超时（>{int(total_timeout)}s）")
    return False


def write_runtime_script(script_path: str | Path, cmd: str) -> Path:
    """Persist generated runtime commands as executable scripts for tracing."""
    script = Path(script_path)
    script.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "#!/usr/bin/env bash\n"
        "set -o pipefail\n"
        ': "${LMSV_PRETRAIN_GPT:=pretrain_gpt.py}"\n'
        f"{cmd.strip()}\n"
    )
    script.write_text(content, encoding="utf-8")
    script.chmod(0o755)
    return script


def build_conda_activate_block(env_name: str, load_ascend: bool = False) -> str:
    """Build a reusable shell snippet that activates a conda env."""
    lines = [
        "# 非交互/远程 shell 场景下，conda 可能未注入 PATH；先尝试加载常见 profile。",
        'if [ -f "$HOME/.bashrc" ]; then',
        '  source "$HOME/.bashrc" >/dev/null 2>&1 || true',
        "fi",
        'if [ -f "$HOME/.bash_profile" ]; then',
        '  source "$HOME/.bash_profile" >/dev/null 2>&1 || true',
        "fi",
        "",
        "# 优先用 conda info --base；失败时回退到常见安装目录。",
        "CONDA_PATH=$(conda info --base 2>/dev/null || true)",
        'if [ -z "$CONDA_PATH" ]; then',
        '  for _cand in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/usr/local/miniconda3"; do',
        '    if [ -x "$_cand/bin/conda" ]; then',
        '      CONDA_PATH="$_cand"',
        "      break",
        "    fi",
        "  done",
        "fi",
        'if [ -z "$CONDA_PATH" ]; then',
        '  echo "ERROR: conda base path not found" >&2',
        "  exit 1",
        "fi",
        'source "$CONDA_PATH/etc/profile.d/conda.sh"',
    ]
    if load_ascend:
        lines.extend(
            [
                'if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then',
                "  source /usr/local/Ascend/ascend-toolkit/set_env.sh",
                "fi",
            ]
        )
    if env_name:
        lines.append(f"conda activate {shlex.quote(env_name)}")
    return "\n".join(lines)


def normalize_models(raw_models, model_config_dir: Path) -> list[str]:
    """Normalize model names or yaml paths to runtime model config paths."""
    if isinstance(raw_models, str):
        items = [part.strip() for part in raw_models.split(",") if part.strip()]
    elif isinstance(raw_models, (list, tuple)):
        items = [str(item).strip() for item in raw_models if str(item).strip()]
    else:
        return []

    model_config_rel = model_config_dir.as_posix()
    paths = []
    for item in items:
        if item.endswith(".yaml"):
            model_path = item if "/" in item else f"{model_config_rel}/{item}"
        else:
            model_path = f"{model_config_rel}/{item}.yaml"
        paths.append(model_path)
    return paths
