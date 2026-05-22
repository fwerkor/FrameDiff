#!/usr/bin/env python

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import utils


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TMP_ROOT = PROJECT_ROOT / "tmp"
DEFAULT_MASTER_PORT = 6000


def _resolve_tmp_root():
    raw_tmp_root = os.environ.get("TMPDIR") or str(DEFAULT_TMP_ROOT)
    tmp_root = Path(raw_tmp_root).expanduser()
    if not tmp_root.is_absolute():
        tmp_root = PROJECT_ROOT / tmp_root
    return tmp_root.resolve()


def _cleanup_runtime_tmp_dirs():
    tmp_root = _resolve_tmp_root()
    tmp_root.mkdir(parents=True, exist_ok=True)
    for pattern in ("hccl*", "torch_distributed*"):
        for candidate in tmp_root.glob(pattern):
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
            else:
                candidate.unlink(missing_ok=True)


def _read_cmdline(pid: int) -> str:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def _runtime_target_terms() -> tuple[str, ...]:
    return (
        "pretrain_gpt.py",
        "pretrain_vlm",
        "pretrain_sora",
        "run_mindformer.py",
        "submodule_entry.py",
        "load_and_forward",
        "torchrun",
        "msrun",
        "hccl",
        "pta_memory_wrapper",
    )


def _proc_matches_runtime(cmdline: str) -> bool:
    return bool(cmdline) and any(term in cmdline for term in _runtime_target_terms())


def _iter_runtime_pids():
    current_pid = os.getpid()

    proc_root = Path("/proc")
    if not proc_root.exists():
        return

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue

        pid = int(entry.name)
        if pid == current_pid:
            continue

        cmdline = _read_cmdline(pid)
        if not _proc_matches_runtime(cmdline):
            continue

        yield pid, cmdline


def _run_command(command: list[str], *, timeout: float = 2.0) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""
    output = (completed.stdout or completed.stderr or "").strip()
    return completed.returncode == 0, output


def _split_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _split_columns(cell: str) -> list[str]:
    return [item for item in re.split(r"\s{2,}", cell.strip()) if item]


def _parse_npu_smi_process_pids(output: str) -> dict[int, dict[str, str]]:
    processes: dict[int, dict[str, str]] = {}
    in_process_table = False

    for line in output.splitlines():
        if "Process id" in line and "Process name" in line:
            in_process_table = True
            continue
        if not in_process_table or not line.strip().startswith("|"):
            continue

        cells = _split_table_cells(line)
        if len(cells) < 4:
            continue

        left_columns = _split_columns(cells[0])
        if len(left_columns) < 2 or not left_columns[0].isdigit():
            continue

        pid_text = cells[1].strip()
        if not pid_text.isdigit():
            continue

        pid = int(pid_text)
        processes[pid] = {
            "npu_id": left_columns[0],
            "chip_id": left_columns[1],
            "name": cells[2].strip() or "-",
            "memory": cells[3].strip() or "-",
        }

    return processes


def _collect_npu_smi_processes() -> dict[int, dict[str, str]]:
    binary = shutil.which("npu-smi")
    if not binary:
        return {}
    ok, output = _run_command([binary, "info"])
    if not ok or not output:
        return {}
    return _parse_npu_smi_process_pids(output)


def _iter_proc_fds(proc_entry: Path):
    fd_dir = proc_entry / "fd"
    try:
        yield from fd_dir.iterdir()
    except OSError:
        return


def _collect_socket_inodes_for_port(port: int) -> set[str]:
    target_hex = f"{port:04X}"
    socket_inodes: set[str] = set()

    for net_name in ("tcp", "tcp6", "udp", "udp6"):
        net_path = Path("/proc/net") / net_name
        try:
            lines = net_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue

            local_address = parts[1]
            inode = parts[9]
            _, _, local_port = local_address.rpartition(":")
            if local_port.upper() != target_hex:
                continue
            if inode.isdigit():
                socket_inodes.add(inode)

    return socket_inodes


def _collect_port_processes(port: int = DEFAULT_MASTER_PORT) -> dict[int, dict[str, str]]:
    socket_inodes = _collect_socket_inodes_for_port(port)
    if not socket_inodes:
        return {}

    current_pid = os.getpid()
    candidates: dict[int, dict[str, str]] = {}
    proc_root = Path("/proc")
    if not proc_root.exists():
        return candidates

    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue

        pid = int(entry.name)
        if pid == current_pid:
            continue

        for fd_entry in _iter_proc_fds(entry):
            try:
                target = os.readlink(fd_entry)
            except OSError:
                continue

            match = re.fullmatch(r"socket:\[(\d+)\]", target)
            if not match or match.group(1) not in socket_inodes:
                continue

            cmdline = _read_cmdline(pid)
            candidates[pid] = {
                "cmdline": cmdline or f"pid={pid} (port {port})",
                "source": f"port:{port}",
            }
            break

    return candidates


def _collect_runtime_processes() -> dict[int, dict[str, str]]:
    candidates: dict[int, dict[str, str]] = {}

    for pid, cmdline in list(_iter_runtime_pids()):
        candidates[pid] = {
            "cmdline": cmdline,
            "source": "cmdline",
        }

    npu_processes = _collect_npu_smi_processes()
    for pid, meta in npu_processes.items():
        if pid == os.getpid():
            continue

        cmdline = _read_cmdline(pid)
        process_name = meta.get("name", "")
        is_python_worker = bool(re.search(r"(?:^|/)(python(?:\d+(?:\.\d+)?)?)$", process_name))
        if not is_python_worker and "python" not in cmdline:
            continue

        candidates.setdefault(
            pid,
            {
                "cmdline": cmdline or process_name or f"pid={pid}",
                "source": "npu-smi",
            },
        )

    runtime_ports = {DEFAULT_MASTER_PORT}
    for env_name in ("MASTER_PORT", "LMSV_MSRUN_MASTER_PORT", "TARGET_MASTER_PORT"):
        try:
            value = int(os.environ.get(env_name, "") or 0)
        except ValueError:
            value = 0
        if value > 0:
            runtime_ports.add(value)
    runtime_ports.update({6000, 6001, 6002})

    for port in runtime_ports:
        for pid, meta in _collect_port_processes(port).items():
            candidates.setdefault(pid, meta)

    return candidates


def _kill_runtime_processes():
    killed = []
    for pid, meta in _collect_runtime_processes().items():
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append((pid, meta.get("cmdline", ""), meta.get("source", "unknown")))
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            utils.log.write.warn(f"无权限终止进程 {pid}: {exc}")
    return killed


def _cleanup_npu_state():
    """清理NPU设备状态，尽可能释放残留资源"""
    # 1. 尝试同步NPU流并清空缓存
    try:
        import torch_npu
        torch_npu.npu.synchronize()
        torch_npu.npu.empty_cache()
        utils.log.write.info("NPU同步与缓存清空完成")
    except Exception:
        pass

    # 2. 清理共享内存中的残留
    shm_dir = Path("/dev/shm")
    if shm_dir.exists():
        for pattern in ("torch_*", "hccl_*", "npu_*", "sem.torch*", "psm_*"):
            for candidate in shm_dir.glob(pattern):
                try:
                    if candidate.is_dir():
                        shutil.rmtree(candidate, ignore_errors=True)
                    else:
                        candidate.unlink(missing_ok=True)
                except Exception:
                    pass

    # 3. 再次检查并清理残留进程
    remaining = _collect_runtime_processes()
    if remaining:
        for pid, meta in remaining.items():
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        time.sleep(5)

    # 4. 再次尝试NPU同步
    try:
        import torch_npu
        torch_npu.npu.synchronize()
        torch_npu.npu.empty_cache()
    except Exception:
        pass


def kill_pretraingpt():
    killed = _kill_runtime_processes()
    _cleanup_runtime_tmp_dirs()

    # 第一次等待：让进程完全退出
    time.sleep(15)

    # 深度清理NPU状态
    _cleanup_npu_state()

    # 第二次等待：确保NPU硬件状态恢复
    time.sleep(15)

    remaining_procs = len(_collect_runtime_processes())
