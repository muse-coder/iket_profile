#!/usr/bin/env python3
"""Validate joint-profile artifacts and their attribution boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pm-csv", type=Path, required=True)
    parser.add_argument("--memory-csv", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-role", action="append", default=[])
    parser.add_argument("--max-duration-delta-pct", type=float, default=5.0)
    parser.add_argument("--min-active-samples", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text())
    manifest = json.loads(args.manifest.read_text())
    trace = json.loads(args.trace.read_text())
    with args.pm_csv.open(newline="") as handle:
        pm_rows = list(csv.DictReader(handle))
    with args.memory_csv.open(newline="") as handle:
        memory_rows = list(csv.DictReader(handle))

    failures: list[str] = []
    warnings: list[str] = []
    iket_us = float(summary["iket_duration_us"])
    ncu_us = float(summary["ncu_duration_us"])
    duration_delta_pct = abs(iket_us - ncu_us) / ncu_us * 100.0
    if duration_delta_pct > args.max_duration_delta_pct:
        failures.append(
            f"duration delta {duration_delta_pct:.3f}% exceeds "
            f"{args.max_duration_delta_pct:.3f}%"
        )
    active_samples = int(summary["ncu_active_sample_count"])
    if active_samples < args.min_active_samples:
        failures.append(
            f"only {active_samples} active PM samples; need {args.min_active_samples}"
        )
    if len(pm_rows) != active_samples or len(memory_rows) != active_samples:
        failures.append("CSV row counts do not match summary active sample count")
    roles = summary.get("roles", {})
    if not roles:
        failures.append("no IKET ranges were included")
    for role in args.require_role:
        if role not in roles or int(roles[role].get("count", 0)) == 0:
            failures.append(f"missing required IKET role {role}")
    if not trace.get("traceEvents"):
        failures.append("Perfetto/Chrome trace contains no events")
    kernel = manifest.get("kernel", {})
    if not kernel.get("launch_dimensions_match", False):
        failures.append("IKET and NCU grid/block dimensions do not match")
    if manifest.get("alignment") != summary.get("alignment"):
        failures.append("manifest and summary alignment policies differ")
    interpretation = manifest.get("interpretation", {})
    if not interpretation.get("ncu_metrics_are_gpu_or_hardware_domain_aggregates"):
        failures.append("manifest does not preserve NCU aggregate attribution boundary")
    quality = summary.get("memory_timeline_quality", {})
    if quality.get("l1_hit_valid_samples", 0) == 0:
        warnings.append(
            "L1 hit has no lookup-active samples; do not interpret zero as 0% hit"
        )
    if quality.get("l2_hit_valid_samples", 0) == 0:
        warnings.append("L2 hit has no activity-valid samples")
    if summary.get("missing_metrics"):
        warnings.append(f"missing PM metrics: {', '.join(summary['missing_metrics'])}")

    result = {
        "ok": not failures,
        "duration_delta_pct": duration_delta_pct,
        "active_samples": active_samples,
        "pm_rows": len(pm_rows),
        "memory_rows": len(memory_rows),
        "trace_events": len(trace.get("traceEvents", [])),
        "launch_dimensions_match": kernel.get("launch_dimensions_match"),
        "failures": failures,
        "warnings": warnings,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
