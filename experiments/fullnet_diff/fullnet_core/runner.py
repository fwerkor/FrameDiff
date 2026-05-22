from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import utils

from .config import load_config
from .paths import CONFIG_PATH, OUTPUT_ROOT, PROJECT_ROOT


TASK_LABEL = "FrameDiff language-model full-network assembly"


def _log(tag: str, message: str) -> str:
    return f"[FrameDiff-FullNet][{tag}] {message}" if tag else f"[FrameDiff-FullNet] {message}"


def _handle_sigint(_signum, _frame) -> None:
    print("\n[FrameDiff-FullNet] 已中断。", flush=True)
    raise SystemExit(130)


signal.signal(signal.SIGINT, _handle_sigint)


def _output_root(config: dict[str, Any]) -> Path:
    raw = os.environ.get("FRAMEDIFF_FULLNET_OUTPUT_ROOT") or os.environ.get("LMSV_OUTPUT_ROOT")
    if raw:
        root = Path(raw).expanduser()
        return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()
    config_root = config.get("OUTPUT_ROOT")
    if config_root:
        root = Path(str(config_root)).expanduser()
        return root if root.is_absolute() else (PROJECT_ROOT / root).resolve()
    return OUTPUT_ROOT


def create_output_dir(config: dict[str, Any]) -> Path:
    output_root = _output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        suffix = "" if attempt == 0 else f"-{attempt}"
        output_dir = output_root / f"{_dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}{suffix}"
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("创建 output 目录失败")

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output_dir / "log.txt").write_text("", encoding="utf-8")
    os.environ["LMSV_LOGPATH"] = str(output_dir / "log.txt")
    os.environ["LMSV_OUTPATH"] = str(output_dir)
    return output_dir


def export_runtime_env(config: dict[str, Any]) -> None:
    env_map = {
        "PTA_PATH": ("PTAPATH", "PTA path"),
        "MSA_PATH": ("MSAPATH", "MSA path"),
    }
    for key, (legacy_key, _label) in env_map.items():
        if key not in config:
            continue
        value = str(config[key])
        os.environ[key] = value
        os.environ[legacy_key] = value

    os.environ.setdefault("PTA_NAME", "mindspeed")
    os.environ.setdefault("PTANAME", os.environ["PTA_NAME"])
    os.environ.setdefault("MSA_NAME", "msadapter")
    os.environ.setdefault("MSANAME", os.environ["MSA_NAME"])
    os.environ.setdefault("SAVE_ABNORMAL_WEIGHTS", "1")
    os.environ.setdefault("LMSV_DEBUG_COMPARE", "1")
    os.environ.setdefault("LMSV_FULLNET_TRACE", "1")
    os.environ.setdefault("LMSV_FULLNET_TRACE_FULL_WEIGHTS", "1")

    fullnet = config.get("fullnet") if isinstance(config.get("fullnet"), dict) else {}
    if "PERTURB_EPS" in fullnet:
        os.environ.setdefault("LMSV_FULLNET_PERTURB_EPS", str(fullnet["PERTURB_EPS"]))


def configure_tmp_defaults() -> None:
    if os.environ.get("LMSV_PROJECT_TMP_ROOT"):
        return
    tmp_root = PROJECT_ROOT / "tmp"
    os.environ["LMSV_PROJECT_TMP_ROOT"] = str(tmp_root.resolve())


def check_other_instances() -> None:
    current_pid = os.getpid()
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    if result.returncode != 0:
        return

    current_root = str(PROJECT_ROOT)
    running = []
    for raw_line in result.stdout.splitlines():
        parts = raw_line.strip().split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid == current_pid:
            continue
        comm, args = parts[1], parts[2]
        if not comm.startswith("python") or "fullnet.py" not in args:
            continue
        try:
            pid_cwd = str(Path(f"/proc/{pid}/cwd").resolve())
        except (OSError, ValueError):
            pid_cwd = ""
        if current_root in pid_cwd:
            running.append(pid)
    if running:
        raise RuntimeError(f"已有同目录 fullnet.py 正在运行: {running}")


def _fullnet_config(config: dict[str, Any]) -> dict[str, Any]:
    fullnet = config.get("fullnet")
    if isinstance(fullnet, dict):
        return fullnet
    raise ValueError("配置缺少 fullnet")


def run_fullnet(config: dict[str, Any]) -> int:
    fullnet = _fullnet_config(config)

    check_other_instances()
    configure_tmp_defaults()
    export_runtime_env(config)
    output_dir = create_output_dir(config)

    params = dict(fullnet)

    utils.log.write.info(_log("任务", f"开始执行 {TASK_LABEL}"))
    utils.log.write.info(_log("输出", str(output_dir)))
    from utils.task import fullnet as fullnet_module

    result = int(fullnet_module.main(params) or 0)
    if result != 0:
        utils.log.write.error(_log("失败", f"整网链路失败，退出码 {result}。详见 {output_dir / 'log.txt'} 和 iters/iter_1/failure_info.txt"))
    return result


def copy_example_if_missing() -> None:
    if CONFIG_PATH.exists():
        return
    from .config import write_config

    write_config(load_config(CONFIG_PATH), CONFIG_PATH)


def main() -> int:
    copy_example_if_missing()
    config = load_config(CONFIG_PATH)
    return run_fullnet(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        utils.log.write.exception(_log("异常", "任务执行失败"), exc, default_component="FrameDiff full-network runner")
        raise SystemExit(1)
