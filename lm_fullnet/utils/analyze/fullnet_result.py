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
    mutation_success_count: int
    mutation_success_rate: float
    functional_failures: int
    precision_failures: int
    repro_root: Path | None = None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _iter_dirs(run_dir: Path) -> list[Path]:
    def key(path: Path) -> int:
        match = re.search(r"iter_(\d+)$", path.name)
        return int(match.group(1)) if match else 0

    return sorted(
        [path for path in run_dir.glob("iter_*") if path.is_dir()],
        key=key,
    )


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
        "mutation_event_count": 0,
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
                if item.get("event") == "mutation_record":
                    summary["mutation_event_count"] += 1

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


def _component_state(status: dict[str, Any], component: str) -> str:
    components = status.get("components")
    if not isinstance(components, dict):
        return "UNKNOWN"
    return str(components.get(component, "UNKNOWN"))


def _iteration_payload(iter_dir: Path) -> dict[str, Any]:
    match = re.search(r"iter_(\d+)$", iter_dir.name)
    iteration = int(match.group(1)) if match else 0
    status = _load_json(iter_dir / "status.json")
    pta_loss = _read_iteration_loss(iter_dir / "execution_pta.csv", iteration)
    msa_loss = _read_iteration_loss(iter_dir / "execution_msa.csv", iteration)
    pta_perturb_loss = _read_iteration_loss(iter_dir / "execution_pta_perturb.csv", iteration)
    msa_perturb_loss = _read_iteration_loss(iter_dir / "execution_msa_perturb.csv", iteration)
    trace_summary = _read_trace_summary(iter_dir)
    return {
        "iteration": iteration,
        "status": status.get("overall_status", "UNKNOWN"),
        "reason": status.get("reason", ""),
        "mutate": _component_state(status, "MUTATE"),
        "pta_save": _component_state(status, "PTA_SAVE"),
        "pta_load": _component_state(status, "PTA_LOAD"),
        "msa_load": _component_state(status, "MSA_LOAD"),
        "pta_loss": pta_loss,
        "msa_loss": msa_loss,
        "pta_perturb_loss": pta_perturb_loss,
        "msa_perturb_loss": msa_perturb_loss,
        "pta_msa_abs_loss_delta": _loss_delta(pta_loss, msa_loss),
        "pta_metamorphic_abs_loss_delta": _loss_delta(pta_loss, pta_perturb_loss),
        "msa_metamorphic_abs_loss_delta": _loss_delta(msa_loss, msa_perturb_loss),
        "has_precision_hint": _has_precision_hint(iter_dir),
        "trace": trace_summary,
        "artifacts": {
            "runtime_logs": str(iter_dir / "runtime_logs"),
            "mutation_inputs": str(iter_dir / "mutation_inputs"),
            "traces": str(iter_dir / "traces"),
            "pta_step_log": str(iter_dir / f"training_log_pta-{iteration}.csv"),
            "msa_step_log": str(iter_dir / f"training_log_msa-{iteration}.csv"),
            "pta_perturb_step_log": str(iter_dir / f"training_log_pta_perturb-{iteration}.csv"),
            "msa_perturb_step_log": str(iter_dir / f"training_log_msa_perturb-{iteration}.csv"),
        },
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FrameDiff Full-Network Summary",
        "",
        f"- model: {payload.get('model_name') or '-'}",
        f"- planned iterations: {payload.get('planned_iterations')}",
        f"- executed iterations: {payload.get('executed_iterations')}",
        f"- successful mutations: {payload.get('mutation_success_count')}",
        f"- functional failures: {payload.get('functional_failures')}",
        f"- precision hints: {payload.get('precision_failures')}",
        f"- traced tensors: {payload.get('trace_tensor_count')}",
        f"- traced weights: {payload.get('trace_weight_count')}",
        "",
        "| iter | status | PTA loss | MSA loss | |PTA-MSA| | MSA perturb | |MSA-MSA'| | trace tensors | trace weights |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload.get("iterations", []):
        trace = item.get("trace", {}) if isinstance(item.get("trace"), dict) else {}
        lines.append(
            "| {iteration} | {status} | {pta_loss} | {msa_loss} | {pta_msa_abs_loss_delta} | "
            "{msa_perturb_loss} | {msa_metamorphic_abs_loss_delta} | {tensor_count} | {weight_count} |".format(
                tensor_count=trace.get("tensor_count", 0),
                weight_count=trace.get("weight_count", 0),
                **item,
            )
        )
    lines.append("")
    lines.append("Runtime logs, step-level CSVs, mutation inputs, and full tensor/weight traces are kept in each `iter_<n>` directory.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for item in payload.get("iterations", []):
        trace = item.get("trace", {}) if isinstance(item.get("trace"), dict) else {}
        rows.append(
            "<tr>"
            f"<td>{item['iteration']}</td>"
            f"<td>{item['status']}</td>"
            f"<td>{item['pta_loss']}</td>"
            f"<td>{item['msa_loss']}</td>"
            f"<td>{item['pta_msa_abs_loss_delta']}</td>"
            f"<td>{item.get('msa_perturb_loss')}</td>"
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
  <p>Executed: {payload.get('executed_iterations')} / Planned: {payload.get('planned_iterations')}</p>
  <p>Traced tensors: {payload.get('trace_tensor_count')} | Traced weights: {payload.get('trace_weight_count')}</p>
  <table>
    <thead><tr><th>iter</th><th>status</th><th>PTA loss</th><th>MSA loss</th><th>|PTA-MSA|</th><th>MSA perturb</th><th>|MSA-MSA'|</th><th>trace tensors</th><th>trace weights</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def analyze_fullnet_run(
    output_root: str | Path,
    run_dir: str | Path | None = None,
    model_name: str | None = None,
    planned_iterations: int | None = None,
) -> AnalysisArtifacts:
    output_root = Path(output_root).resolve()
    run_dir = Path(run_dir).resolve() if run_dir else output_root / "iters"
    analysis_dir = output_root / "analysis"
    data_dir = analysis_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    iterations = [_iteration_payload(path) for path in _iter_dirs(run_dir)]
    executed = len(iterations)
    mutation_success = sum(1 for item in iterations if item["mutate"] == "OK")
    functional_failures = sum(1 for item in iterations if str(item["status"]).startswith("FAILED"))
    precision_failures = sum(1 for item in iterations if item["has_precision_hint"])
    trace_tensor_count = sum(int((item.get("trace") or {}).get("tensor_count", 0)) for item in iterations)
    trace_weight_count = sum(int((item.get("trace") or {}).get("weight_count", 0)) for item in iterations)

    payload = {
        "analysis_type": "frame_diff_fullnet",
        "model_name": model_name,
        "planned_iterations": planned_iterations,
        "executed_iterations": executed,
        "mutation_success_count": mutation_success,
        "mutation_success_rate": mutation_success / executed if executed else 0.0,
        "functional_failures": functional_failures,
        "precision_failures": precision_failures,
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
        mutation_success_count=mutation_success,
        mutation_success_rate=payload["mutation_success_rate"],
        functional_failures=functional_failures,
        precision_failures=precision_failures,
        repro_root=None,
    )
