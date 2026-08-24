#!/usr/bin/env python3
"""Align an external IKET trace with Nsight Compute PM Sampling data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


METRICS = {
    "tensor_core_pct": (
        "TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "tensor_memory_pct": (
        "SM_A.TriageCompute.sm__mem_tensor_cycles_active_realtime.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "l1_hit_pct": "SM_B.TriageCompute.l1tex__t_sector_hit_rate.pct",
    "l1_data_pipe_pct": (
        "SM_A.TriageCompute.l1tex__data_pipe_lsu_wavefronts.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "l1_lookup_hit_throughput_pct": (
        "SM_B.TriageCompute.l1tex__t_sectors_lookup_hit.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "l1_lookup_miss_throughput_pct": (
        "SM_B.TriageCompute.l1tex__t_sectors_lookup_miss.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "gmem_lgds_wavefronts_avg": (
        "SM_A.TriageCompute.l1tex__data_pipe_lsu_wavefronts_mem_lgds.avg"
    ),
    "smem_pipe_pct": (
        "SM_A.TriageCompute.l1tex__data_pipe_lsu_wavefronts_mem_shared.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "l2_hit_pct": "LTS.TriageCompute.lts__average_t_sector_hit_rate_realtime.pct",
    "l2_throughput_pct": (
        "LTS.TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed"
    ),
    "l2_from_l1_throughput_pct": (
        "LTS.TriageCompute.lts__t_sector_throughput_srcunit_tex.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "l2_to_dram_throughput_pct": (
        "LTS.TriageCompute.lts__t_sector_throughput_srcnode_fbp.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "dram_read_pct": (
        "FBSP.TriageCompute.dram__read_throughput.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "dram_write_pct": (
        "FBSP.TriageCompute.dram__write_throughput.avg."
        "pct_of_peak_sustained_elapsed"
    ),
    "dram_throughput_pct": (
        "FBSP.TriageCompute.dram__throughput.avg.pct_of_peak_sustained_elapsed"
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iket", type=Path, required=True, help="NVIDIA IKET *.trace.json")
    parser.add_argument("--ncu", type=Path, required=True, help="Nsight Compute *.ncu-rep")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--event-prefix",
        default="FA4.",
        help="only IKET ranges with this prefix select the launch; empty means all",
    )
    parser.add_argument("--kernel-regex", help="optional kernel-name selector")
    parser.add_argument("--workload-metadata", type=Path)
    parser.add_argument("--dram-peak-gbps", type=float)
    return parser.parse_args(argv)


def import_ncu_report():
    candidates = sorted(
        Path("/opt/nvidia/nsight-compute").glob("*/extras/python"), reverse=True
    )
    if not candidates:
        raise RuntimeError("cannot find Nsight Compute extras/python/ncu_report.py")
    sys.path.insert(0, str(candidates[0]))
    import ncu_report  # type: ignore

    return ncu_report


def metric_value(action: Any, name: str) -> Any:
    try:
        metric = action.metric_by_name(name)
        return metric.value() if metric is not None else None
    except Exception:
        return None


def pm_pass_groups(action: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return PM pass-group provenance and a raw-metric-to-group lookup."""
    metric = action.metric_by_name("profiler__pmsampler_pass_groups")
    if metric is None:
        return [], {}
    groups = []
    metric_to_group = {}
    correlations = metric.correlation_ids()
    for index in range(metric.num_instances()):
        group_id = int(correlations.value(index))
        names = [name for name in str(metric.value(index)).split(",") if name]
        groups.append({"group_id": group_id, "metrics": names})
        metric_to_group.update({name: group_id for name in names})
    return groups, metric_to_group


def load_iket(path: Path, event_prefix: str, kernel_regex: str | None):
    data = json.loads(path.read_text())
    strings = data["stringTable"]
    pattern = re.compile(kernel_regex) if kernel_regex else None
    launches = []
    for launch in data.get("launches", []):
        if pattern and not pattern.search(launch.get("kernelName", "")):
            continue
        names = [
            strings[item["rangeNameIdx"]]
            for item in launch.get("ranges", [])
            if "rangeNameIdx" in item
        ]
        if names and (not event_prefix or any(name.startswith(event_prefix) for name in names)):
            launches.append(launch)
    if len(launches) != 1:
        raise RuntimeError(f"expected one matching IKET launch, found {len(launches)}")
    launch = launches[0]
    lifetimes = launch.get("warpLifetimes", [])
    if not lifetimes:
        raise RuntimeError("matching IKET launch has no warp lifetimes")
    start_ns = min(item["startTs"] for item in lifetimes)
    end_ns = max(item["endTs"] for item in lifetimes)
    return data, launch, start_ns, end_ns


def load_ncu(path: Path, kernel_regex: str | None):
    report = import_ncu_report().load_report(path)
    pattern = re.compile(kernel_regex) if kernel_regex else None
    actions = [
        action
        for result_range in report
        for action in result_range
        if pattern is None or pattern.search(action.name())
    ]
    if len(actions) != 1:
        raise RuntimeError(f"expected one matching NCU action, found {len(actions)}")
    action = actions[0]
    start_metric = action.metric_by_name("profiler__timestamp_workload_start_0")
    end_metric = action.metric_by_name("profiler__timestamp_workload_end_0")
    start_ns = start_metric.correlation_ids().value(0)
    end_ns = end_metric.correlation_ids().value(0)
    interval_ns = metric_value(action, "profiler__pmsampler_interval_time")
    samples: dict[int, dict[str, float]] = {}
    units: dict[str, str] = {}
    missing = []
    for label, metric_name in METRICS.items():
        metric = action.metric_by_name(metric_name)
        if metric is None:
            missing.append(label)
            continue
        units[label] = metric.unit()
        timestamps = metric.correlation_ids()
        for index in range(metric.num_instances()):
            samples.setdefault(timestamps.value(index), {})[label] = metric.value(index)
    return action, start_ns, end_ns, interval_ns, samples, units, missing


def field_name(role: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    iket_data, launch, iket_start, iket_end = load_iket(
        args.iket, args.event_prefix, args.kernel_regex
    )
    action, ncu_start, ncu_end, interval_ns, samples, units, missing = load_ncu(
        args.ncu, args.kernel_regex
    )
    pass_groups, raw_metric_to_group = pm_pass_groups(action)
    label_to_group = {
        label: raw_metric_to_group.get(metric_name)
        for label, metric_name in METRICS.items()
    }
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "pm": prefix.with_suffix(".pm_samples.csv"),
        "memory": prefix.with_suffix(".memory_timeline.csv"),
        "trace": prefix.with_suffix(".perfetto.json"),
        "summary": prefix.with_suffix(".summary.json"),
        "manifest": prefix.with_suffix(".run_manifest.json"),
    }
    paths["trace_gz"] = Path(f"{paths['trace']}.gz")

    strings = iket_data["stringTable"]
    locations = iket_data["locationTable"]
    ranges = []
    for item in launch.get("ranges", []):
        role = strings[item["rangeNameIdx"]]
        if args.event_prefix and not role.startswith(args.event_prefix):
            continue
        location_indices = item.get("warpLocIdxs", [])
        if not location_indices:
            continue
        location = locations[location_indices[0]]
        ranges.append(
            {
                "role": role,
                "start_us": (item["startTs"] - iket_start) / 1e3,
                "end_us": (item["endTs"] - iket_start) / 1e3,
                "location": location,
            }
        )
    if not ranges:
        raise RuntimeError("matching IKET launch has no selected ranges")
    roles = sorted({item["role"] for item in ranges})
    active_samples = {
        timestamp: values
        for timestamp, values in samples.items()
        if ncu_start <= timestamp <= ncu_end
    }
    interval_us = float(interval_ns) / 1e3 if interval_ns else None
    combined_samples: dict[int, dict[str, Any]] = {}
    for timestamp, source_values in active_samples.items():
        window_end_us = (timestamp - ncu_start) / 1e3
        window_start_us = max(0.0, window_end_us - (interval_us or 0.0))
        values: dict[str, Any] = dict(source_values)
        values["source_pass_groups"] = ",".join(
            str(group)
            for group in sorted(
                {
                    label_to_group[label]
                    for label in source_values
                    if label_to_group.get(label) is not None
                }
            )
        )
        l1_activity = values.get("l1_lookup_hit_throughput_pct", 0.0) + values.get(
            "l1_lookup_miss_throughput_pct", 0.0
        )
        values["l1_hit_valid"] = int(l1_activity > 1e-12)
        values["l2_hit_valid"] = int(
            values.get("l2_from_l1_throughput_pct", 0.0) > 1e-12
            or values.get("l2_throughput_pct", 0.0) > 1e-12
        )
        values["dram_active"] = int(values.get("dram_throughput_pct", 0.0) > 1e-12)
        values["smem_active"] = int(values.get("smem_pipe_pct", 0.0) > 1e-12)
        if args.dram_peak_gbps is not None:
            for direction in ("read", "write"):
                values[f"dram_{direction}_gbps_est"] = (
                    values.get(f"dram_{direction}_pct", 0.0)
                    * args.dram_peak_gbps
                    / 100.0
                )
            values["dram_total_gbps_est"] = (
                values.get("dram_throughput_pct", 0.0) * args.dram_peak_gbps / 100.0
            )
        for role in roles:
            overlapping = [
                item
                for item in ranges
                if item["role"] == role
                and item["start_us"] < window_end_us
                and item["end_us"] > window_start_us
            ]
            key = field_name(role)
            values[f"active_{key}_warps"] = len(overlapping)
            values[f"active_{key}_sms"] = len(
                {item["location"]["smId"] for item in overlapping}
            )
        combined_samples[timestamp] = values

    quality_fields = ["l1_hit_valid", "l2_hit_valid", "dram_active", "smem_active"]
    estimated_fields = (
        ["dram_read_gbps_est", "dram_write_gbps_est", "dram_total_gbps_est"]
        if args.dram_peak_gbps is not None
        else []
    )
    activity_fields = [
        field
        for role in roles
        for field in (f"active_{field_name(role)}_warps", f"active_{field_name(role)}_sms")
    ]
    pm_fields = [
        "time_us",
        "window_start_us",
        "sample_interval_us",
        "source_pass_groups",
        *METRICS,
        *quality_fields,
        *estimated_fields,
        *activity_fields,
    ]
    with paths["pm"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pm_fields)
        writer.writeheader()
        for timestamp, values in sorted(combined_samples.items()):
            time_us = (timestamp - ncu_start) / 1e3
            writer.writerow(
                {
                    "time_us": time_us,
                    "window_start_us": max(0.0, time_us - (interval_us or 0.0)),
                    "sample_interval_us": interval_us or "",
                    **values,
                }
            )

    memory_metric_fields = [
        "l1_hit_pct",
        "l1_lookup_hit_throughput_pct",
        "l1_lookup_miss_throughput_pct",
        "l1_data_pipe_pct",
        "gmem_lgds_wavefronts_avg",
        "l2_hit_pct",
        "l2_throughput_pct",
        "l2_from_l1_throughput_pct",
        "l2_to_dram_throughput_pct",
        "smem_pipe_pct",
        "dram_read_pct",
        "dram_write_pct",
        "dram_throughput_pct",
    ]
    memory_fields = [
        "time_us",
        "window_start_us",
        "sample_interval_us",
        "source_pass_groups",
        *memory_metric_fields,
        *quality_fields,
        *estimated_fields,
    ]
    with paths["memory"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=memory_fields)
        writer.writeheader()
        for timestamp, values in sorted(combined_samples.items()):
            time_us = (timestamp - ncu_start) / 1e3
            writer.writerow(
                {
                    "time_us": time_us,
                    "window_start_us": max(0.0, time_us - (interval_us or 0.0)),
                    "sample_interval_us": interval_us or "",
                    **{field: values.get(field, "") for field in memory_fields[3:]},
                }
            )

    events: list[dict[str, Any]] = []
    for sm_id in sorted({item["location"]["smId"] for item in ranges}):
        events.append(
            {"ph": "M", "name": "process_name", "pid": 1000 + sm_id, "tid": 0,
             "args": {"name": f"IKET SM {sm_id}"}}
        )
    for item in ranges:
        location = item["location"]
        events.append(
            {
                "ph": "X",
                "cat": "IKET",
                "name": item["role"],
                "pid": 1000 + location["smId"],
                "tid": location["warpId"],
                "ts": item["start_us"],
                "dur": item["end_us"] - item["start_us"],
                "args": {
                    "sm_id": location["smId"],
                    "warp_id": location["warpId"],
                    "cta": location.get("ctaId"),
                    "gpc_id": location.get("gpcId"),
                    "tpc_id": location.get("tpcId"),
                },
            }
        )
    events.extend(
        [
            {"ph": "M", "name": "process_name", "pid": 2, "tid": 0,
             "args": {"name": "NCU PM Sampling (GPU aggregate)"}},
            {"ph": "X", "cat": "NCU", "name": "NCU profiled kernel", "pid": 2,
             "tid": 0, "ts": 0, "dur": (ncu_end - ncu_start) / 1e3,
             "args": {"kernel": action.name()}},
        ]
    )
    counter_groups = {
        "NCU compute utilization": ["tensor_core_pct", "tensor_memory_pct"],
        "NCU cache hit rate": ["l1_hit_pct", "l2_hit_pct"],
        "NCU memory throughput": [
            "l1_data_pipe_pct", "l2_throughput_pct", "smem_pipe_pct",
            "dram_read_pct", "dram_write_pct", "dram_throughput_pct", *estimated_fields,
        ],
        "NCU metric validity": quality_fields,
        "IKET overlapping activity": activity_fields,
    }
    for timestamp, values in sorted(combined_samples.items()):
        for track_id, (name, fields) in enumerate(counter_groups.items(), start=1):
            events.append(
                {
                    "ph": "C", "cat": "Joint timeline", "name": name,
                    "pid": 2, "tid": track_id, "ts": (timestamp - ncu_start) / 1e3,
                    "args": {field: values[field] for field in fields if field in values},
                }
            )
    trace = {
        "displayTimeUnit": "us",
        "traceEvents": events,
        "metadata": {
            "alignment": "kernel-start-relative, separate equivalent launches, no scaling",
            "pm_window_model": "(sample_timestamp - interval, sample_timestamp]",
        },
    }
    trace_text = json.dumps(trace, separators=(",", ":"))
    paths["trace"].write_text(trace_text)
    with gzip.open(paths["trace_gz"], "wt") as handle:
        handle.write(trace_text)

    by_role: dict[str, list[float]] = {}
    for item in ranges:
        by_role.setdefault(item["role"], []).append(item["end_us"] - item["start_us"])
    metric_summary = {}
    for label in METRICS:
        values = [row[label] for row in active_samples.values() if label in row]
        if values:
            metric_summary[label] = {
                "unit": units.get(label, ""),
                "mean": statistics.fmean(values),
                "max": max(values),
                "samples": len(values),
            }
    summary = {
        "alignment": "kernel-start-relative, separate equivalent launches, no scaling",
        "pm_window_model": "(sample_timestamp - interval, sample_timestamp]",
        "iket_duration_us": (iket_end - iket_start) / 1e3,
        "ncu_duration_us": (ncu_end - ncu_start) / 1e3,
        "ncu_sampling_interval_us": interval_us,
        "ncu_active_sample_count": len(active_samples),
        "roles": {
            role: {
                "count": len(values), "p50_us": percentile(values, 0.5),
                "p95_us": percentile(values, 0.95), "max_us": max(values),
            }
            for role, values in sorted(by_role.items())
        },
        "metrics_during_kernel": metric_summary,
        "memory_timeline_quality": {
            "l1_hit_valid_samples": sum(row["l1_hit_valid"] for row in combined_samples.values()),
            "l2_hit_valid_samples": sum(row["l2_hit_valid"] for row in combined_samples.values()),
            "dram_active_samples": sum(row["dram_active"] for row in combined_samples.values()),
            "smem_active_samples": sum(row["smem_active"] for row in combined_samples.values()),
            "dram_peak_gbps_for_estimate": args.dram_peak_gbps,
        },
        "missing_metrics": missing,
        "limitations": [
            "IKET provides SM/CTA/warp locations; NCU PM values remain GPU/domain aggregates.",
            "IKET and NCU are separate equivalent executions, not one simultaneous capture.",
        ],
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")

    ncu_grid = [int(metric_value(action, f"launch__grid_dim_{axis}")) for axis in "xyz"]
    ncu_block = [int(metric_value(action, f"launch__block_dim_{axis}")) for axis in "xyz"]
    iket_grid = [launch[f"gridDim{axis}"] for axis in "XYZ"]
    iket_block = [launch[f"blockDim{axis}"] for axis in "XYZ"]
    workload_metadata = (
        json.loads(args.workload_metadata.read_text()) if args.workload_metadata else None
    )
    manifest = {
        "format_version": 2,
        "sources": {"iket": str(args.iket.resolve()), "ncu": str(args.ncu.resolve())},
        "alignment": summary["alignment"],
        "pm_window_model": summary["pm_window_model"],
        "workload_metadata": workload_metadata,
        "kernel": {
            "iket_name": launch["kernelName"], "ncu_name": action.name(),
            "iket_grid": iket_grid, "ncu_grid": ncu_grid,
            "iket_block": iket_block, "ncu_block": ncu_block,
            "launch_dimensions_match": iket_grid == ncu_grid and iket_block == ncu_block,
            "iket_duration_us": summary["iket_duration_us"],
            "ncu_duration_us": summary["ncu_duration_us"],
        },
        "sampling": {
            "interval_us": interval_us,
            "active_samples": len(active_samples),
            "pm_pass_groups": metric_value(action, "profiler__pmsampler_pass_groups"),
            "pm_pass_group_details": pass_groups,
            "replayer_passes": metric_value(action, "profiler__replayer_passes"),
            "merged_samples": metric_value(action, "profiler__pmsampler_merged_samples"),
            "metrics": {label: units.get(label, "") for label in METRICS},
            "missing_metrics": missing,
        },
        "interpretation": {
            "ncu_metrics_are_gpu_or_hardware_domain_aggregates": True,
            "per_sm_cta_warp_location_comes_from_iket": True,
            "cross_pass_values_may_not_be_exactly_simultaneous": True,
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    for path in paths.values():
        print(f"wrote {path}")
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
