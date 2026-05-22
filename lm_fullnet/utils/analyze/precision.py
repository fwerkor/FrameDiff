#!/usr/bin/env python3
"""Shared helpers for runtime precision mismatch detection."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

ITERATION_COLUMN_NAMES = (
    "Iteration",
    "iteration",
    "step",
)

LOSS_COLUMN_NAMES = (
    "loss",
    "Loss",
)


def _pick_optional_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    names = list(fieldnames)
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in names:
            return candidate
        resolved = lowered.get(candidate.lower())
        if resolved:
            return resolved
    return None


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _read_loss_series(csv_path: Path) -> dict[int, float]:
    if not csv_path.exists() or csv_path.stat().st_size <= 0:
        return {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}

        iteration_key = _pick_optional_column(reader.fieldnames, ITERATION_COLUMN_NAMES)
        loss_key = _pick_optional_column(reader.fieldnames, LOSS_COLUMN_NAMES)
        if not iteration_key or not loss_key:
            return {}

        rows: dict[int, float] = {}
        for row in reader:
            iteration = _parse_float(row.get(iteration_key))
            loss = _parse_float(row.get(loss_key))
            if iteration is None or loss is None:
                continue
            rows[int(iteration)] = loss
        return rows


def _read_iteration_loss(csv_path: Path, iteration: int) -> Optional[float]:
    if not csv_path.exists() or csv_path.stat().st_size <= 0:
        return None

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None

        iteration_key = _pick_optional_column(reader.fieldnames, ITERATION_COLUMN_NAMES)
        loss_key = _pick_optional_column(reader.fieldnames, LOSS_COLUMN_NAMES)
        if not iteration_key or not loss_key:
            return None

        for row in reader:
            row_iteration = _parse_float(row.get(iteration_key))
            if row_iteration is None or int(row_iteration) != int(iteration):
                continue
            return _parse_float(row.get(loss_key))
    return None


def find_series_loss_mismatch(
    pta_csv_path: str | Path,
    msa_csv_path: str | Path,
    tolerance: float = 0.0,
) -> Optional[str]:
    pta_index = _read_loss_series(Path(pta_csv_path))
    msa_index = _read_loss_series(Path(msa_csv_path))
    common_steps = sorted(set(pta_index) & set(msa_index))
    if not common_steps:
        return None

    mismatches: list[tuple[int, float, float, float]] = []
    for step in common_steps:
        pta_loss = pta_index[step]
        msa_loss = msa_index[step]
        diff = abs(pta_loss - msa_loss)
        if diff > tolerance:
            mismatches.append((step, pta_loss, msa_loss, diff))

    if not mismatches:
        return None

    first_step, first_pta, first_msa, first_diff = mismatches[0]
    max_diff = max(item[3] for item in mismatches)
    return (
        f"检测到loss不一致，共{len(mismatches)}/{len(common_steps)}个公共step异常；"
        f"首个异常step={first_step}，PTA={first_pta:.10g}，MSA={first_msa:.10g}，"
        f"diff={first_diff:.10g}，最大diff={max_diff:.10g}"
    )


def find_iteration_loss_mismatch(
    pta_csv_path: str | Path,
    msa_csv_path: str | Path,
    iteration: int,
    tolerance: float = 0.0,
) -> Optional[str]:
    pta_loss = _read_iteration_loss(Path(pta_csv_path), iteration)
    msa_loss = _read_iteration_loss(Path(msa_csv_path), iteration)
    if pta_loss is None or msa_loss is None:
        return None

    diff = abs(pta_loss - msa_loss)
    if diff <= tolerance:
        return None

    return (
        f"检测到loss不一致，iter={iteration}，PTA={pta_loss:.10g}，"
        f"MSA={msa_loss:.10g}，diff={diff:.10g}"
    )


def find_preferred_loss_mismatch(
    pta_csv_path: str | Path,
    msa_csv_path: str | Path,
    *,
    iteration: int,
    tolerance: float = 0.0,
    pta_step_csv_path: str | Path | None = None,
    msa_step_csv_path: str | Path | None = None,
) -> Optional[str]:
    """Match task analysis preference: step CSV first, single-iteration CSV fallback."""
    if pta_step_csv_path and msa_step_csv_path:
        series_issue = find_series_loss_mismatch(
            pta_step_csv_path,
            msa_step_csv_path,
            tolerance=tolerance,
        )
        if series_issue:
            return series_issue
        return None

    return find_iteration_loss_mismatch(
        pta_csv_path,
        msa_csv_path,
        iteration,
        tolerance=tolerance,
    )
