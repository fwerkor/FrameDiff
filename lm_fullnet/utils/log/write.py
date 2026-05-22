#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

try:
    from colorama import Fore, Style, init
except ImportError:
    class _NoColor:
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""
        RESET_ALL = ""
        BRIGHT = ""
        DIM = ""
        NORMAL = ""

    Fore = Style = _NoColor()

    def init(*_args, **_kwargs):
        return None


init()


def _append_log_line(timestamp: str, level: str, text: str) -> None:
    log_path = os.environ.get("LMSV_LOGPATH")
    if not log_path:
        return

    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] [{level}] {text}\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[{timestamp}] [LOGGER] 写入日志文件失败: {exc}", file=sys.stderr)


def _emit(level: str, color: str, text) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = str(text)
    print(f"{Fore.BLUE}[{timestamp}]{Style.RESET_ALL} {color}[{level}]{Style.RESET_ALL} {message}")
    _append_log_line(timestamp, level, message)


def _relative_location(file_name: str) -> str:
    try:
        return os.path.relpath(file_name, Path.cwd())
    except Exception:  # noqa: BLE001
        return file_name


def _extract_trace_location(exc: Exception) -> str:
    tb = traceback.extract_tb(exc.__traceback__)
    if not tb:
        return ""
    last = tb[-1]
    return f"{_relative_location(last.filename)}:{last.lineno} ({last.name})"


def _build_diagnostic(exc: Exception, default_component: str | None = None) -> dict:
    message = str(exc).strip() or repr(exc)
    lower_text = f"{type(exc).__name__} {message}".lower()

    component = default_component or "未知组件"
    reason = f"默认按 {type(exc).__name__} 处理"
    advice = [
        "先结合异常位置与上下文日志，确认失败发生在配置解析、任务调度还是外部脚本执行阶段。",
        "保留当前 output 目录下的 log.txt、runtime_logs 与相关脚本，便于复现和二次定位。",
    ]

    if isinstance(exc, FileNotFoundError) or any(token in lower_text for token in ("not found", "不存在", "缺少", "找不到")):
        component = "文件/产物定位"
        reason = "命中缺失类关键词或 FileNotFoundError"
        advice = [
            "检查报错路径是否真实存在，重点核对 config.json 中的路径配置和 output 目录结构。",
            "如果缺的是训练产物，回看上游步骤是否成功生成了日志、权重、analysis 或中间文件。",
            "确认运行用户对目标目录具备读写权限，且没有被清理流程提前删除。",
        ]
    if isinstance(exc, PermissionError) or "permission" in lower_text or "权限" in lower_text:
        component = "文件权限/环境权限"
        reason = "命中权限类关键词或 PermissionError"
        advice = [
            "检查当前用户对目标路径、脚本和目录是否有读写执行权限。",
            "如果路径位于挂载盘或共享目录，确认挂载状态和目录属主没有变化。",
        ]
    if any(token in lower_text for token in ("config", "json", "fullnet", "配置", "请求体")):
        component = "配置解析/参数校验"
        reason = "命中配置或请求体相关关键词"
        advice = [
            "对照 config.json.example 检查必填字段、字段类型和 fullnet 配置。",
            "确认数值字段没有写成空串或非法文本，列表字段数量关系满足约束。",
            "如果是 WebUI 请求异常，确认前端提交的是合法 JSON，且字段名与表单 schema 一致。",
        ]
    if any(token in lower_text for token in ("conda", "环境", "msa_path", "pta_path", "set_env.sh")):
        component = "运行环境准备"
        reason = "命中 conda/环境变量/代码路径关键词"
        advice = [
            "检查 PTA、MSA 的 conda 环境名和代码路径是否配置正确。",
            "确认环境初始化脚本可执行，依赖包完整，必要环境变量已经导出。",
            "如果是新机器或新容器，先单独验证对应训练脚本能否脱离 LMSV 独立启动。",
        ]
    if isinstance(exc, TimeoutError) or any(token in lower_text for token in ("timeout", "timed out", "超时")):
        component = "外部工具执行/超时"
        reason = "命中超时类关键词"
        advice = [
            "先确认是启动慢还是执行卡住，再结合相关 worker 日志判断阻塞点。",
            "检查机器资源、NPU/GPU 状态和数据路径可达性，必要时适当放宽超时阈值。",
            "如果同一脚本独立执行也会卡住，优先定位脚本或框架侧问题，而不是 WebUI/主控层。",
        ]
    if any(token in lower_text for token in ("pgrep", "pid", "进程", "process", "signal", "sigterm", "sigkill")):
        component = "进程管理"
        reason = "命中进程与信号相关关键词"
        advice = [
            "检查是否存在残留 do.py 或训练子进程，避免多实例竞争同一 output 或资源。",
            "确认进程组创建与回收正常，必要时先清理残留训练进程后再重试。",
        ]
    if any(token in lower_text for token in ("msrun_log", "training_log", "ckpt", "权重", "analysis", "output/")):
        component = "训练产物归档/结果分析"
        reason = "命中日志、权重或 analysis 相关关键词"
        advice = [
            "确认上游阶段确实产出了对应日志、权重或 analysis 文件，再检查归档路径是否正确。",
            "如果问题只出现在归档或重分析阶段，可先复用已有 output 目录单独重跑分析流程。",
        ]
    if any(token in lower_text for token in ("接口不存在", "/api/", "webui", "json 对象")):
        component = "WebUI 接口层"
        reason = "命中 WebUI/API 相关关键词"
        advice = [
            "确认请求路径、方法和请求体格式正确，必要时在浏览器开发者工具里复查接口响应。",
            "如果后端返回的是业务异常，继续查看 payload 中的组件定位和主日志内容。",
        ]

    location = _extract_trace_location(exc)
    return {
        "error_type": type(exc).__name__,
        "message": message,
        "component": component,
        "reason": reason,
        "location": location,
        "advice": advice,
    }


def diagnose_exception(exc: Exception, default_component: str | None = None) -> dict:
    diagnostic = _build_diagnostic(exc, default_component=default_component)
    advice_summary = " ".join(item.rstrip("。；") + "。" for item in diagnostic["advice"])
    lines = [
        f"问题组件: {diagnostic['component']}",
        f"异常类型: {diagnostic['error_type']}",
        f"定位依据: {diagnostic['reason']}",
    ]
    if diagnostic["location"]:
        lines.append(f"代码位置: {diagnostic['location']}")
    lines.append("解决思路: " + advice_summary)
    diagnostic["summary"] = "\n".join(lines)
    return diagnostic


def exception(context, exc: Exception, default_component: str | None = None) -> dict:
    diagnostic = diagnose_exception(exc, default_component=default_component)
    _emit("ERROR", Fore.RED, f"{context}: {diagnostic['error_type']}: {diagnostic['message']}")
    _emit("ERROR", Fore.RED, f"[诊断] 问题组件: {diagnostic['component']}")
    _emit("ERROR", Fore.RED, f"[诊断] 定位依据: {diagnostic['reason']}")
    if diagnostic["location"]:
        _emit("ERROR", Fore.RED, f"[诊断] 代码位置: {diagnostic['location']}")
    for index, item in enumerate(diagnostic["advice"], start=1):
        _emit("ERROR", Fore.RED, f"[诊断] 解决思路{index}: {item}")
    return diagnostic


def info(text):
    _emit("INFO", Fore.GREEN, text)


def warn(text):
    _emit("WARN", Fore.YELLOW, text)


def error(text):
    _emit("ERROR", Fore.RED, text)
