#!/usr/bin/env python3
"""Lightweight full-network analysis for FrameDiff language-model experiments."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AnalysisArtifacts:
    analysis_dir: Path
    report_html: Path
    summary_json: Path
    report_md: Path
    executed_iterations: int
    variant_success_count: int
    variant_success_rate: float
    functional_failures: int
    precision_failures: int
    repro_root: Path | None = None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_iteration_loss(csv_path: Path, iteration: int) -> float | None:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_iter = row.get("Iteration") or row.get("iteration") or row.get("iter")
                if raw_iter is not None:
                    try:
                        if int(float(str(raw_iter))) != int(iteration):
                            continue
                    except ValueError:
                        continue
                for key in ("Loss", "loss", "lm loss", "lm_loss"):
                    if key in row and str(row[key]).strip() not in {"", "-"}:
                        try:
                            return float(row[key])
                        except ValueError:
                            return None
    except OSError:
        return None
    return None


def _loss_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def _has_precision_hint(iter_dir: Path) -> bool:
    for path in iter_dir.rglob("*"):
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        name = path.name.lower()
        if name.endswith((".log", ".txt", ".md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(token in text for token in ("精度", "precision", "loss mismatch", "不一致")):
                return True
    return False


def _read_trace_summary(iter_dir: Path) -> dict[str, Any]:
    trace_root = iter_dir / "traces"
    summary: dict[str, Any] = {
        "enabled": trace_root.exists(),
        "tensor_count": 0,
        "weight_count": 0,
        "event_count": 0,
        "tensor_file_count": 0,
        "weight_file_count": 0,
        "runs": [],
        "backends": [],
        "component_coverage": {},
        "losses": [],
        "variant_event_count": 0,
    }
    if not trace_root.exists():
        return summary

    runs: set[str] = set()
    backends: set[str] = set()
    components: dict[str, dict[str, Any]] = {}

    for path in trace_root.rglob("trace_index.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(item.get("kind", ""))
            if kind == "tensor":
                summary["tensor_count"] += 1
            elif kind == "weights":
                summary["weight_count"] += 1
            elif kind == "event":
                summary["event_count"] += 1
                if item.get("event") == "loss":
                    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
                    summary["losses"].append(
                        {
                            "run": item.get("run"),
                            "backend": item.get("backend"),
                            "step": item.get("step"),
                            "name": payload.get("name"),
                            "value": payload.get("value"),
                            "extra": payload.get("extra", {}),
                        }
                    )
                if item.get("event") == "variant_record":
                    summary["variant_event_count"] += 1

            if item.get("run") is not None:
                runs.add(str(item.get("run")))
            if item.get("backend") is not None:
                backends.add(str(item.get("backend")))

            component_id = item.get("component_id")
            component_name = item.get("component_name")
            if component_id is not None or component_name is not None:
                key = f"{component_id}:{component_name}"
                bucket = components.setdefault(
                    key,
                    {
                        "component_id": component_id,
                        "component_name": component_name,
                        "tensor_count": 0,
                        "weight_count": 0,
                        "event_count": 0,
                    },
                )
                if kind == "tensor":
                    bucket["tensor_count"] += 1
                elif kind == "weights":
                    bucket["weight_count"] += 1
                elif kind == "event":
                    bucket["event_count"] += 1

    summary["runs"] = sorted(runs)
    summary["backends"] = sorted(backends)
    summary["component_coverage"] = dict(sorted(components.items()))
    trace_files = list(trace_root.rglob("*"))
    summary["tensor_file_count"] = len(
        [
            path
            for path in trace_files
            if path.is_file() and path.suffix in {".pt", ".npy"} and "/tensors/" in path.as_posix()
        ]
    )
    summary["weight_file_count"] = len(
        [path for path in trace_files if path.is_file() and path.suffix == ".pt" and "/weights/" in path.as_posix()]
    )
    return summary


def _baseline_alignment(iter_dir: Path) -> dict[str, Any]:
    data = _load_json(iter_dir / "baseline_alignment.json")
    if data:
        return data
    return {"aligned": None, "required": None, "tolerance": None, "issue": ""}


def _merge_trace_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "enabled": any(item.get("enabled") for item in summaries),
        "tensor_count": 0,
        "weight_count": 0,
        "event_count": 0,
        "tensor_file_count": 0,
        "weight_file_count": 0,
        "runs": [],
        "backends": [],
        "component_coverage": {},
        "losses": [],
        "variant_event_count": 0,
    }
    runs: set[str] = set()
    backends: set[str] = set()
    components: dict[str, dict[str, Any]] = {}
    for item in summaries:
        for key in ("tensor_count", "weight_count", "event_count", "tensor_file_count", "weight_file_count", "variant_event_count"):
            merged[key] += int(item.get(key, 0) or 0)
        runs.update(str(value) for value in item.get("runs", []) if value is not None)
        backends.update(str(value) for value in item.get("backends", []) if value is not None)
        merged["losses"].extend(item.get("losses", []) if isinstance(item.get("losses"), list) else [])
        coverage = item.get("component_coverage") if isinstance(item.get("component_coverage"), dict) else {}
        for key, value in coverage.items():
            bucket = components.setdefault(
                key,
                {
                    "component_id": value.get("component_id") if isinstance(value, dict) else None,
                    "component_name": value.get("component_name") if isinstance(value, dict) else None,
                    "tensor_count": 0,
                    "weight_count": 0,
                    "event_count": 0,
                },
            )
            if isinstance(value, dict):
                bucket["tensor_count"] += int(value.get("tensor_count", 0) or 0)
                bucket["weight_count"] += int(value.get("weight_count", 0) or 0)
                bucket["event_count"] += int(value.get("event_count", 0) or 0)
    merged["runs"] = sorted(runs)
    merged["backends"] = sorted(backends)
    merged["component_coverage"] = dict(sorted(components.items()))
    return merged


def _has_precision_hint_in_dirs(paths: list[Path]) -> bool:
    return any(path.exists() and _has_precision_hint(path) for path in paths)


def _stage_dir(variant_dir: Path, training: str, iteration: int) -> Path:
    base = variant_dir / training
    iter_dir = base / f"iter_{iteration}"
    return iter_dir if iter_dir.is_dir() else base


def _variant_iterations(variant_dir: Path) -> list[int]:
    values: set[int] = set()
    if (variant_dir / "status.json").exists():
        values.add(1)
    for path in variant_dir.glob("status_iter_*.json"):
        match = re.search(r"status_iter_(\d+)\.json$", path.name)
        if match:
            values.add(int(match.group(1)))
    for path in (variant_dir / "pta-baseline").glob("iter_*"):
        if path.is_dir():
            match = re.search(r"iter_(\d+)$", path.name)
            if match:
                values.add(int(match.group(1)))
    if not values and (variant_dir / "pta-baseline" / "execution.csv").exists():
        values.add(1)
    return sorted(values)


def _variant_dirs(output_root: Path) -> list[tuple[str, str, Path]]:
    variants: list[tuple[str, str, Path]] = []
    for model_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        if model_dir.name in {"analysis", "tmp"}:
            continue
        for variant_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if (variant_dir / "pta-baseline").exists() or (variant_dir / "status.json").exists() or list(variant_dir.glob("status_iter_*.json")):
                variants.append((model_dir.name, variant_dir.name, variant_dir))
    return variants


def _new_variant_payload(model: str, variant: str, variant_dir: Path, iteration: int) -> dict[str, Any]:
    status = _load_json(variant_dir / ("status.json" if iteration == 1 else f"status_iter_{iteration}.json"))
    trainings = status.get("trainings") if isinstance(status.get("trainings"), dict) else {}
    pta_dir = _stage_dir(variant_dir, "pta-baseline", iteration)
    msa_dir = _stage_dir(variant_dir, "msa-baseline", iteration)
    pta_preturb_dir = _stage_dir(variant_dir, "pta-preturb", iteration)
    msa_preturb_dir = _stage_dir(variant_dir, "msa-preturb", iteration)
    stage_dirs = [pta_dir, msa_dir, pta_preturb_dir, msa_preturb_dir]
    pta_loss = _read_iteration_loss(pta_dir / "execution.csv", iteration)
    msa_loss = _read_iteration_loss(msa_dir / "execution.csv", iteration)
    pta_preturb_loss = _read_iteration_loss(pta_preturb_dir / "execution.csv", iteration)
    msa_preturb_loss = _read_iteration_loss(msa_preturb_dir / "execution.csv", iteration)
    baseline_alignment = (
        _load_json(variant_dir / ("baseline_alignment.json" if iteration == 1 else f"baseline_alignment_iter_{iteration}.json"))
        or _baseline_alignment(msa_dir)
    )
    trace_summary = _merge_trace_summaries([_read_trace_summary(path) for path in stage_dirs])
    return {
        "model": model,
        "variant": variant,
        "iteration": iteration,
        "status": status.get("overall_status", "UNKNOWN"),
        "reason": status.get("reason", ""),
        "prepare": str(trainings.get("prepare", "SKIP")),
        "pta_load": str(trainings.get("pta-baseline", "UNKNOWN")),
        "msa_load": str(trainings.get("msa-baseline", "UNKNOWN")),
        "pta_preturb": str(trainings.get("pta-preturb", "UNKNOWN")),
        "msa_preturb": str(trainings.get("msa-preturb", "UNKNOWN")),
        "baseline_align": str(trainings.get("baseline-align", "UNKNOWN")),
        "baseline_alignment": baseline_alignment,
        "pta_loss": pta_loss,
        "msa_loss": msa_loss,
        "pta_preturb_loss": pta_preturb_loss,
        "msa_preturb_loss": msa_preturb_loss,
        "pta_msa_abs_loss_delta": _loss_delta(pta_loss, msa_loss),
        "pta_metamorphic_abs_loss_delta": _loss_delta(pta_loss, pta_preturb_loss),
        "msa_metamorphic_abs_loss_delta": _loss_delta(msa_loss, msa_preturb_loss),
        "has_precision_hint": _has_precision_hint_in_dirs(stage_dirs),
        "baseline_aligned": baseline_alignment.get("aligned"),
        "trace": trace_summary,
        "artifacts": {
            "variant_root": str(variant_dir),
            "pta-baseline": str(pta_dir),
            "msa-baseline": str(msa_dir),
            "pta-preturb": str(pta_preturb_dir),
            "msa-preturb": str(msa_preturb_dir),
        },
    }


def _new_layout_payloads(output_root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for model, variant, variant_dir in _variant_dirs(output_root):
        for iteration in _variant_iterations(variant_dir):
            payloads.append(_new_variant_payload(model, variant, variant_dir, iteration))
    return payloads


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FrameDiff Full-Network Summary",
        "",
        f"- model: {payload.get('model_name') or '-'}",
        f"- planned iterations: {payload.get('planned_iterations')}",
        f"- executed variant runs: {payload.get('executed_iterations')}",
        f"- successful variant runs: {payload.get('variant_success_count')}",
        f"- functional failures: {payload.get('functional_failures')}",
        f"- precision hints: {payload.get('precision_failures')}",
        f"- baseline alignment failures: {payload.get('baseline_alignment_failures')}",
        f"- traced tensors: {payload.get('trace_tensor_count')}",
        f"- traced weights: {payload.get('trace_weight_count')}",
        "",
        "| model | variant | iter | status | baseline aligned | PTA loss | MSA loss | |PTA-MSA| | MSA preturb | |MSA-MSA'| | trace tensors | trace weights |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload.get("iterations", []):
        trace = item.get("trace", {}) if isinstance(item.get("trace"), dict) else {}
        row = dict(item)
        row["model"] = item.get("model", "-")
        row["variant"] = item.get("variant", "-")
        row["tensor_count"] = trace.get("tensor_count", 0)
        row["weight_count"] = trace.get("weight_count", 0)
        lines.append(
            "| {model} | {variant} | {iteration} | {status} | {baseline_aligned} | {pta_loss} | {msa_loss} | {pta_msa_abs_loss_delta} | "
            "{msa_preturb_loss} | {msa_metamorphic_abs_loss_delta} | {tensor_count} | {weight_count} |".format(**row)
        )
    lines.append("")
    lines.append("Runtime logs, step-level CSVs, variant inputs, and full tensor/weight traces are kept in each training directory.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for item in payload.get("iterations", []):
        trace = item.get("trace", {}) if isinstance(item.get("trace"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{item.get('model', '-')}</td>"
            f"<td>{item.get('variant', '-')}</td>"
            f"<td>{item['iteration']}</td>"
            f"<td>{item['status']}</td>"
            f"<td>{item.get('baseline_aligned')}</td>"
            f"<td>{item['pta_loss']}</td>"
            f"<td>{item['msa_loss']}</td>"
            f"<td>{item['pta_msa_abs_loss_delta']}</td>"
            f"<td>{item.get('msa_preturb_loss')}</td>"
            f"<td>{item.get('msa_metamorphic_abs_loss_delta')}</td>"
            f"<td>{trace.get('tensor_count', 0)}</td>"
            f"<td>{trace.get('weight_count', 0)}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FrameDiff Full-Network Summary</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>FrameDiff Full-Network Summary</h1>
  <p>Model: {payload.get('model_name') or '-'}</p>
  <p>Executed variant runs: {payload.get('executed_iterations')} / Planned iterations: {payload.get('planned_iterations')}</p>
  <p>Baseline alignment failures: {payload.get('baseline_alignment_failures')}</p>
  <p>Traced tensors: {payload.get('trace_tensor_count')} | Traced weights: {payload.get('trace_weight_count')}</p>
  <table>
    <thead><tr><th>model</th><th>variant</th><th>iter</th><th>status</th><th>baseline aligned</th><th>PTA loss</th><th>MSA loss</th><th>|PTA-MSA|</th><th>MSA preturb</th><th>|MSA-MSA'|</th><th>trace tensors</th><th>trace weights</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def analyze_fullnet_run(
    output_root: str | Path,
    model_name: str | None = None,
    planned_iterations: int | None = None,
) -> AnalysisArtifacts:
    output_root = Path(output_root).resolve()
    analysis_dir = output_root / "analysis"
    data_dir = analysis_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    run_summary = _load_json(output_root / "summary.json")
    if model_name is None and isinstance(run_summary.get("models"), list):
        model_name = ",".join(str(item) for item in run_summary["models"])
    if planned_iterations is None and run_summary.get("iterations") is not None:
        planned_iterations = int(run_summary.get("iterations") or 0)

    iterations = _new_layout_payloads(output_root)
    executed = len(iterations)
    variant_success = sum(1 for item in iterations if item["status"] == "PASS")
    functional_failures = sum(1 for item in iterations if str(item["status"]).startswith("FAILED"))
    precision_failures = sum(1 for item in iterations if item["has_precision_hint"])
    baseline_alignment_failures = sum(1 for item in iterations if item.get("baseline_aligned") is False)
    trace_tensor_count = sum(int((item.get("trace") or {}).get("tensor_count", 0)) for item in iterations)
    trace_weight_count = sum(int((item.get("trace") or {}).get("weight_count", 0)) for item in iterations)

    payload = {
        "analysis_type": "frame_diff_fullnet",
        "model_name": model_name,
        "planned_iterations": planned_iterations,
        "executed_iterations": executed,
        "variant_success_count": variant_success,
        "variant_success_rate": variant_success / executed if executed else 0.0,
        "functional_failures": functional_failures,
        "precision_failures": precision_failures,
        "baseline_alignment_failures": baseline_alignment_failures,
        "trace_tensor_count": trace_tensor_count,
        "trace_weight_count": trace_weight_count,
        "iterations": iterations,
    }

    summary_json = data_dir / "summary.json"
    report_md = analysis_dir / "summary.md"
    report_html = analysis_dir / "report.html"
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report_md, payload)
    _write_html(report_html, payload)

    return AnalysisArtifacts(
        analysis_dir=analysis_dir,
        report_html=report_html,
        summary_json=summary_json,
        report_md=report_md,
        executed_iterations=executed,
        variant_success_count=variant_success,
        variant_success_rate=payload["variant_success_rate"],
        functional_failures=functional_failures,
        precision_failures=precision_failures,
        repro_root=None,
    )
