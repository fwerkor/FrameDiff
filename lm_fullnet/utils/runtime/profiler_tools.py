#!/usr/bin/env python3
"""Helpers for enabling torch_npu profiling and generating simple tuning reports."""

from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _parse_ranks(text: str) -> list[int]:
    raw = (text or "").strip()
    if not raw:
        return [-1]
    result = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.append(int(part))
        except ValueError:
            continue
    return result or [-1]


@dataclass(frozen=True)
class ProfilerSettings:
    enabled: bool = False
    output_dir: str = ""
    level: str = "level1"
    with_cpu: bool = False
    with_stack: bool = False
    record_shapes: bool = False
    with_memory: bool = True
    export_type: str = "text"
    data_simplification: bool = True
    step_start: int = 1
    step_end: int = 5
    ranks: tuple[int, ...] = (-1,)


def load_profiler_settings_from_env() -> ProfilerSettings:
    ranks = tuple(_parse_ranks(os.getenv("LMSV_MSA_PROFILE_RANKS", "-1")))
    step_start = max(1, _env_int("LMSV_MSA_PROFILE_STEP_START", 1))
    step_end = max(step_start, _env_int("LMSV_MSA_PROFILE_STEP_END", 5))
    return ProfilerSettings(
        enabled=_env_bool("LMSV_MSA_PROFILE", False),
        output_dir=_env_text("LMSV_MSA_PROFILE_DIR", ""),
        level=_env_text("LMSV_MSA_PROFILE_LEVEL", "level1"),
        with_cpu=_env_bool("LMSV_MSA_PROFILE_WITH_CPU", False),
        with_stack=_env_bool("LMSV_MSA_PROFILE_WITH_STACK", False),
        record_shapes=_env_bool("LMSV_MSA_PROFILE_RECORD_SHAPES", False),
        with_memory=_env_bool("LMSV_MSA_PROFILE_WITH_MEMORY", True),
        export_type=_env_text("LMSV_MSA_PROFILE_EXPORT_TYPE", "text"),
        data_simplification=_env_bool("LMSV_MSA_PROFILE_DATA_SIMPLIFICATION", True),
        step_start=step_start,
        step_end=step_end,
        ranks=ranks,
    )


class NullProfiler:
    enabled = False

    def start(self) -> None:
        return None

    def step(self) -> None:
        return None

    def stop(self) -> None:
        return None


def create_torch_npu_profiler_from_env(
    current_iteration: int = 0,
    rank: int | None = None,
    world_size: int | None = None,
    metadata: dict[str, Any] | None = None,
):
    settings = load_profiler_settings_from_env()
    if not settings.enabled or not settings.output_dir:
        return None

    if settings.ranks != (-1,) and rank is not None and rank not in settings.ranks:
        return None

    try:
        import torch_npu
    except Exception:
        return None

    output_dir = Path(settings.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    level_map = {
        "level_none": torch_npu.profiler.ProfilerLevel.Level_none,
        "level0": torch_npu.profiler.ProfilerLevel.Level0,
        "level1": torch_npu.profiler.ProfilerLevel.Level1,
        "level2": torch_npu.profiler.ProfilerLevel.Level2,
    }
    export_map = {
        "text": torch_npu.profiler.ExportType.Text,
        "db": torch_npu.profiler.ExportType.Db,
    }

    profiler_level = level_map.get(settings.level.lower(), torch_npu.profiler.ProfilerLevel.Level1)
    export_type = export_map.get(settings.export_type.lower(), torch_npu.profiler.ExportType.Text)

    step_start = max(1, settings.step_start)
    step_end = max(step_start, settings.step_end)
    active = max(1, step_end - step_start + 1)

    if step_start == current_iteration + 1:
        warmup = 0
    elif step_start > current_iteration + 1:
        warmup = 1
    else:
        warmup = 0
    skip_first = max(0, step_start - current_iteration - 2)

    activities = [torch_npu.profiler.ProfilerActivity.NPU]
    if settings.with_cpu:
        activities.append(torch_npu.profiler.ProfilerActivity.CPU)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=profiler_level,
        export_type=export_type,
        data_simplification=settings.data_simplification,
    )

    profiler = torch_npu.profiler.profile(
        with_stack=settings.with_stack,
        record_shapes=settings.record_shapes,
        profile_memory=settings.with_memory,
        activities=activities,
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=warmup,
            active=active,
            repeat=1,
            skip_first=skip_first,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir)),
        experimental_config=experimental_config,
    )

    merged_metadata = {
        "current_iteration": current_iteration,
        "rank": rank,
        "world_size": world_size,
        "settings": {
            "level": settings.level,
            "with_cpu": settings.with_cpu,
            "with_stack": settings.with_stack,
            "record_shapes": settings.record_shapes,
            "with_memory": settings.with_memory,
            "export_type": settings.export_type,
            "data_simplification": settings.data_simplification,
            "step_start": settings.step_start,
            "step_end": settings.step_end,
            "ranks": list(settings.ranks),
        },
    }
    if metadata:
        merged_metadata.update(metadata)

    try:
        profiler.add_metadata_json("lmsv_profiler", json.dumps(merged_metadata, ensure_ascii=False))
    except Exception:
        pass
    return profiler


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_step_metrics(step_csv_path: str | Path | None) -> list[dict[str, float]]:
    if not step_csv_path:
        return []
    path = Path(step_csv_path)
    if not path.exists() or path.stat().st_size <= 0:
        return []

    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = _safe_float(row.get("Step") or row.get("step"))
            elapsed = _safe_float(row.get("Execution Time (s)") or row.get("execution_time_s"))
            memory = _safe_float(row.get("NPU Memory (MB)") or row.get("memory_mb"))
            loss = _safe_float(row.get("loss"))
            if step is None or elapsed is None or memory is None:
                continue
            rows.append(
                {
                    "step": step,
                    "execution_time_s": elapsed,
                    "memory_mb": memory,
                    "loss": loss if loss is not None else float("nan"),
                }
            )
    return rows


def _profile_file_summary(profile_dir: Path) -> dict[str, Any]:
    if not profile_dir.exists():
        return {"exists": False, "file_count": 0, "total_size_bytes": 0, "top_files": []}

    files = [path for path in profile_dir.rglob("*") if path.is_file()]
    files_sorted = sorted(files, key=lambda item: item.stat().st_size, reverse=True)
    return {
        "exists": True,
        "file_count": len(files),
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "top_files": [
            {
                "path": str(path.relative_to(profile_dir)),
                "size_bytes": path.stat().st_size,
            }
            for path in files_sorted[:10]
        ],
    }


def _build_advice(step_rows: list[dict[str, float]], settings: ProfilerSettings) -> list[str]:
    advice: list[str] = []
    if not step_rows:
        advice.append("未解析到逐 step 指标，先确认 profiling 与训练日志 CSV 是否都已生成。")
        return advice

    times = [row["execution_time_s"] for row in step_rows]
    memories = [row["memory_mb"] for row in step_rows]
    avg_time = statistics.mean(times)
    max_time = max(times)
    avg_mem = statistics.mean(memories)
    max_mem = max(memories)
    stdev_time = statistics.pstdev(times) if len(times) > 1 else 0.0
    cov = stdev_time / avg_time if avg_time > 0 else 0.0

    if cov >= 0.15:
        advice.append(
            "step 时间抖动较大，建议优先排查数据准备、主机侧同步、首次图编译/缓存未命中和 H2D 传输抖动。"
        )
    if max_time >= avg_time * 1.3:
        advice.append(
            "存在明显慢 step，建议结合 profiler 时间线定位是否有算子回退、通信等待或重复图构建。"
        )
    if max_mem >= 24 * 1024:
        advice.append(
            "显存峰值较高，建议优先尝试降低 micro-batch-size、开启/增强重计算，或缩短 seq-length。"
        )
    elif max_mem >= avg_mem * 1.2:
        advice.append(
            "显存峰值高于均值较多，建议检查是否存在个别 step 的临时张量膨胀或异常缓存增长。"
        )
    if not settings.with_memory:
        advice.append("当前未开启 memory profiling，如需定位内存热点，建议在外部 MSA profiler 配置中补充显存采样。")
    if not settings.with_cpu:
        advice.append("当前未采集 CPU 活动，如怀疑调度或数据侧瓶颈，建议在外部 MSA profiler 配置中补充 CPU 活动采样。")
    if settings.level.lower() in {"level0", "level_none"}:
        advice.append("当前 profiling level 偏轻，若需要更细粒度热点定位，建议提升到 `level1` 或 `level2`。")
    if settings.step_end - settings.step_start + 1 < 3:
        advice.append("当前采样窗口较短，建议至少覆盖 3-5 个稳定 step，以便判断波动和趋势。")
    if not advice:
        advice.append("采样窗口内 step 时间和显存较稳定，可进一步针对时间线中的热点算子做定向优化。")
    return advice


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _build_findings(
    step_rows: list[dict[str, float]],
    settings: ProfilerSettings,
    file_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not file_summary.get("exists"):
        findings.append(
            {
                "severity": "high",
                "category": "profiling_data",
                "title": "未发现 profiler 原始数据目录",
                "evidence": ["profile_dir 不存在或未生成任何 profiling 文件。"],
                "recommendation": "先确认外部 MSA 侧已正确开启 profiler，并将产物写入约定目录。",
            }
        )
        return findings

    if not step_rows:
        findings.append(
            {
                "severity": "high",
                "category": "step_metrics",
                "title": "缺少逐 step 指标，智能分析证据不足",
                "evidence": ["未解析到 step 级执行时间/显存统计。"],
                "recommendation": "检查训练日志 CSV 是否正常归档，并确认 profiler 采样窗口覆盖了有效训练 step。",
            }
        )
        return findings

    times = [row["execution_time_s"] for row in step_rows]
    memories = [row["memory_mb"] for row in step_rows]
    avg_time = statistics.mean(times)
    max_time = max(times)
    min_time = min(times)
    avg_mem = statistics.mean(memories)
    max_mem = max(memories)
    stdev_time = statistics.pstdev(times) if len(times) > 1 else 0.0
    cov = stdev_time / avg_time if avg_time > 0 else 0.0
    slow_ratio = max_time / avg_time if avg_time > 0 else 0.0
    mem_ratio = max_mem / avg_mem if avg_mem > 0 else 0.0

    if cov >= 0.20:
        findings.append(
            {
                "severity": "high",
                "category": "performance_jitter",
                "title": "step 耗时波动明显，疑似存在主机侧或调度侧抖动",
                "evidence": [
                    f"avg_step_time={_format_float(avg_time)}s",
                    f"max_step_time={_format_float(max_time)}s",
                    f"min_step_time={_format_float(min_time)}s",
                    f"time_cov={_format_float(cov)}",
                ],
                "recommendation": "优先检查数据准备、主机侧同步、H2D 传输、首次图编译与缓存命中情况。",
            }
        )
    elif cov >= 0.12:
        findings.append(
            {
                "severity": "medium",
                "category": "performance_jitter",
                "title": "step 耗时存在一定波动",
                "evidence": [
                    f"avg_step_time={_format_float(avg_time)}s",
                    f"max_step_time={_format_float(max_time)}s",
                    f"time_cov={_format_float(cov)}",
                ],
                "recommendation": "建议结合时间线确认是否有间歇性通信等待、数据加载不均衡或图缓存抖动。",
            }
        )

    if slow_ratio >= 1.50:
        findings.append(
            {
                "severity": "high",
                "category": "slow_step",
                "title": "存在明显慢 step，疑似有热点算子或等待事件",
                "evidence": [
                    f"avg_step_time={_format_float(avg_time)}s",
                    f"max_step_time={_format_float(max_time)}s",
                    f"slow_step_ratio={_format_float(slow_ratio)}x",
                ],
                "recommendation": "结合 profiler 时间线重点排查算子回退、通信等待、重复构图或异常同步点。",
            }
        )
    elif slow_ratio >= 1.25:
        findings.append(
            {
                "severity": "medium",
                "category": "slow_step",
                "title": "检测到相对均值更慢的 step",
                "evidence": [
                    f"avg_step_time={_format_float(avg_time)}s",
                    f"max_step_time={_format_float(max_time)}s",
                    f"slow_step_ratio={_format_float(slow_ratio)}x",
                ],
                "recommendation": "建议抽查慢 step 附近时间线，确认是否存在局部热点或同步等待。",
            }
        )

    if max_mem >= 24 * 1024:
        findings.append(
            {
                "severity": "high",
                "category": "memory_pressure",
                "title": "显存峰值较高，存在内存压力风险",
                "evidence": [
                    f"avg_memory_mb={_format_float(avg_mem, 2)}",
                    f"max_memory_mb={_format_float(max_mem, 2)}",
                ],
                "recommendation": "优先尝试降低 micro-batch-size、增强重计算、缩短 seq-length，并检查大临时张量分配。",
            }
        )
    elif mem_ratio >= 1.20:
        findings.append(
            {
                "severity": "medium",
                "category": "memory_spike",
                "title": "显存峰值相对均值偏高，存在瞬时内存尖峰",
                "evidence": [
                    f"avg_memory_mb={_format_float(avg_mem, 2)}",
                    f"max_memory_mb={_format_float(max_mem, 2)}",
                    f"memory_peak_ratio={_format_float(mem_ratio)}x",
                ],
                "recommendation": "建议排查个别 step 的临时张量膨胀、缓存增长和重计算边界。",
            }
        )

    if not settings.with_cpu:
        findings.append(
            {
                "severity": "low",
                "category": "observability_gap",
                "title": "当前未采集 CPU 活动，主机侧瓶颈判断能力有限",
                "evidence": ["with_cpu=false"],
                "recommendation": "如怀疑数据侧、调度侧或主机侧瓶颈，建议外部 profiler 配置中开启 CPU 活动采集。",
            }
        )
    if not settings.with_memory:
        findings.append(
            {
                "severity": "low",
                "category": "observability_gap",
                "title": "当前未采集显存信息，内存热点判断能力有限",
                "evidence": ["with_memory=false"],
                "recommendation": "如需定位显存峰值和内存尖峰，建议外部 profiler 配置中开启 memory profiling。",
            }
        )
    if settings.step_end - settings.step_start + 1 < 3:
        findings.append(
            {
                "severity": "low",
                "category": "sampling_window",
                "title": "采样窗口偏短，结论稳定性有限",
                "evidence": [f"step_window={settings.step_start}-{settings.step_end}"],
                "recommendation": "建议至少覆盖 3-5 个稳定 step，再观察整体趋势和波动。",
            }
        )
    if not findings:
        findings.append(
            {
                "severity": "info",
                "category": "overall_health",
                "title": "采样窗口内未发现明显异常",
                "evidence": [
                    f"avg_step_time={_format_float(avg_time)}s",
                    f"time_cov={_format_float(cov)}",
                    f"max_memory_mb={_format_float(max_mem, 2)}",
                ],
                "recommendation": "可以继续结合时间线聚焦热点算子，做更细粒度的定向优化。",
            }
        )
    return findings


def _build_intelligent_summary(findings: list[dict[str, Any]]) -> dict[str, Any]:
    severity_rank = {"high": 3, "medium": 2, "low": 1, "info": 0}
    highest = "info"
    for finding in findings:
        if severity_rank.get(finding.get("severity", "info"), 0) > severity_rank[highest]:
            highest = finding.get("severity", "info")

    if highest == "high":
        status = "需优先处理"
        conclusion = "当前采样结果显示存在较明显性能或可观测性问题，建议优先按报告中的高优先级项排查。"
    elif highest == "medium":
        status = "建议关注"
        conclusion = "当前采样结果显示存在可优化项，建议结合时间线和训练日志做进一步定位。"
    else:
        status = "整体稳定"
        conclusion = "当前采样窗口内未发现明显异常，性能表现整体较稳定。"

    priorities = [
        item["title"]
        for item in sorted(findings, key=lambda finding: severity_rank.get(finding.get("severity", "info"), 0), reverse=True)
        if item.get("severity") in {"high", "medium"}
    ][:3]

    return {
        "status": status,
        "highest_severity": highest,
        "conclusion": conclusion,
        "top_priorities": priorities,
    }


def generate_profile_report(
    profile_dir: str | Path,
    report_dir: str | Path,
    step_csv_path: str | Path | None = None,
    exec_log_path: str | Path | None = None,
    task_label: str = "MSA",
    iter_num: int | None = None,
) -> dict[str, Any]:
    settings = load_profiler_settings_from_env()
    profile_root = Path(profile_dir).resolve()
    report_root = Path(report_dir).resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    step_rows = _load_step_metrics(step_csv_path)
    file_summary = _profile_file_summary(profile_root)
    times = [row["execution_time_s"] for row in step_rows]
    memories = [row["memory_mb"] for row in step_rows]
    findings = _build_findings(step_rows, settings, file_summary)
    intelligent_summary = _build_intelligent_summary(findings)

    summary = {
        "task_label": task_label,
        "iteration": iter_num,
        "profile_dir": str(profile_root),
        "step_csv_path": str(step_csv_path) if step_csv_path else "",
        "exec_log_path": str(exec_log_path) if exec_log_path else "",
        "settings": {
            "enabled": settings.enabled,
            "level": settings.level,
            "with_cpu": settings.with_cpu,
            "with_stack": settings.with_stack,
            "record_shapes": settings.record_shapes,
            "with_memory": settings.with_memory,
            "export_type": settings.export_type,
            "data_simplification": settings.data_simplification,
            "step_start": settings.step_start,
            "step_end": settings.step_end,
            "ranks": list(settings.ranks),
        },
        "profile_files": file_summary,
        "step_metrics": {
            "count": len(step_rows),
            "avg_execution_time_s": statistics.mean(times) if times else None,
            "min_execution_time_s": min(times) if times else None,
            "max_execution_time_s": max(times) if times else None,
            "avg_memory_mb": statistics.mean(memories) if memories else None,
            "max_memory_mb": max(memories) if memories else None,
            "time_stdev_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
        },
        "intelligent_analysis": intelligent_summary,
        "findings": findings,
        "advice": _build_advice(step_rows, settings),
    }

    summary_path = report_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {task_label} Profiler Summary",
        "",
        f"- iteration: {iter_num if iter_num is not None else '-'}",
        f"- profile_dir: {profile_root}",
        f"- step_csv_path: {step_csv_path or '-'}",
        f"- export_type: {settings.export_type}",
        f"- profile_level: {settings.level}",
        f"- profile_files: {file_summary['file_count']}",
        f"- profile_total_size_bytes: {file_summary['total_size_bytes']}",
        "",
        "## Intelligent Analysis",
        "",
        f"- status: {intelligent_summary['status']}",
        f"- highest_severity: {intelligent_summary['highest_severity']}",
        f"- conclusion: {intelligent_summary['conclusion']}",
    ]
    if intelligent_summary["top_priorities"]:
        lines.extend(
            [
                "",
                "### Top Priorities",
                "",
            ]
        )
        for item in intelligent_summary["top_priorities"]:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Findings",
            "",
        ]
    )
    for index, finding in enumerate(findings, start=1):
        lines.append(f"### {index}. {finding['title']}")
        lines.append("")
        lines.append(f"- severity: {finding['severity']}")
        lines.append(f"- category: {finding['category']}")
        for evidence in finding.get("evidence", []):
            lines.append(f"- evidence: {evidence}")
        lines.append(f"- recommendation: {finding['recommendation']}")
        lines.append("")

    lines.extend(
        [
        "## Step Stats",
        "",
        f"- count: {summary['step_metrics']['count']}",
        f"- avg_execution_time_s: {summary['step_metrics']['avg_execution_time_s']}",
        f"- max_execution_time_s: {summary['step_metrics']['max_execution_time_s']}",
        f"- avg_memory_mb: {summary['step_metrics']['avg_memory_mb']}",
        f"- max_memory_mb: {summary['step_metrics']['max_memory_mb']}",
        "",
        "## Advice",
        "",
        ]
    )
    for item in summary["advice"]:
        lines.append(f"- {item}")
    lines.append("")

    markdown_path = report_root / "summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return summary
