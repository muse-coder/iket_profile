# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Decode semantic payloads emitted by the automatic CuTe DSL hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PIPELINE_STAGE_BITS = 8
PIPELINE_PHASE_SHIFT = PIPELINE_STAGE_BITS
PIPELINE_COUNT_SHIFT = PIPELINE_PHASE_SHIFT + 1
TILE_COORD_BITS = 16
TILE_COORD_MASK = (1 << TILE_COORD_BITS) - 1
LOOP_SITE_BITS = 16
LOOP_INDEX_BITS = 64 - LOOP_SITE_BITS
LOOP_INDEX_MASK = (1 << LOOP_INDEX_BITS) - 1

LOOP_EVENT_NAME = "auto.loop.tile_seq"
SCHEDULER_TILE_EVENT_NAME = "auto.scheduler.tile"


def _unsigned_64(value: int) -> int:
    return value & ((1 << 64) - 1)


def decode_pipeline_payload(value: int) -> dict[str, int | str]:
    raw = _unsigned_64(value)
    return {
        "source": "pipeline_state",
        "sequence": raw >> PIPELINE_COUNT_SHIFT,
        "phase": (raw >> PIPELINE_PHASE_SHIFT) & 1,
        "stage": raw & ((1 << PIPELINE_STAGE_BITS) - 1),
    }


def decode_tile_payload(value: int) -> dict[str, int | str]:
    raw = _unsigned_64(value)
    coordinates = [
        (raw >> (axis * TILE_COORD_BITS)) & TILE_COORD_MASK
        for axis in reversed(range(4))
    ]
    return {
        "source": "scheduler",
        "tile_0": coordinates[0],
        "tile_1": coordinates[1],
        "tile_2": coordinates[2],
        "tile_3": coordinates[3],
    }


def decode_loop_payload(value: int) -> dict[str, int | str]:
    raw = _unsigned_64(value)
    index = raw & LOOP_INDEX_MASK
    if index & (1 << (LOOP_INDEX_BITS - 1)):
        index -= 1 << LOOP_INDEX_BITS
    return {
        "source": "loop_induction",
        "source_line": raw >> LOOP_INDEX_BITS,
        "iteration": index,
    }


def decode_event_payload(event_name: str, value: int) -> dict[str, int | str]:
    if event_name == LOOP_EVENT_NAME:
        return decode_loop_payload(value)
    if event_name == SCHEDULER_TILE_EVENT_NAME:
        return decode_tile_payload(value)
    return decode_pipeline_payload(value)


def _payload_value(item: dict[str, Any]) -> int | None:
    for event in item.get("internalEvents", ()):
        if "payloadVal" in event:
            return int(event["payloadVal"])
    if "payloadVal" in item:
        return int(item["payloadVal"])
    return None


def _location(
    item: dict[str, Any], location_table: list[dict[str, Any]]
) -> dict[str, Any]:
    indices = (
        item.get("warpLocIdxs")
        or item.get("warpLocIdx")
        or item.get("locIdx")
    )
    if isinstance(indices, list):
        index = indices[0] if indices else None
    else:
        index = indices
    if isinstance(index, int) and 0 <= index < len(location_table):
        return location_table[index]
    location = item.get("location")
    if isinstance(location, dict):
        return location
    warp_locations = item.get("warpLocs")
    if isinstance(warp_locations, list) and warp_locations:
        return warp_locations[0]
    return {}


def _semantic_items(
    items: Iterable[dict[str, Any]],
    string_table: list[str],
    location_table: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decoded = []
    for item in items:
        name_index = item.get("rangeNameIdx", item.get("markerNameIdx"))
        if not isinstance(name_index, int) or not 0 <= name_index < len(string_table):
            continue
        payload = _payload_value(item)
        if payload is None:
            continue
        event_name = string_table[name_index]
        if not event_name.startswith("auto."):
            continue
        location = _location(item, location_table)
        decoded.append(
            {
                "event": event_name,
                "start_ns": item.get("startTs", item.get("timestamp")),
                "end_ns": item.get("endTs", item.get("timestamp")),
                "cta": location.get("ctaId"),
                "warp": location.get("warpId"),
                **decode_event_payload(event_name, payload),
            }
        )
    return decoded


def decode_trace(trace: dict[str, Any]) -> dict[str, Any]:
    strings = trace.get("stringTable", [])
    locations = trace.get("locationTable", [])
    launches = []
    for launch in trace.get("launches", []):
        events = _semantic_items(launch.get("ranges", []), strings, locations)
        events.extend(_semantic_items(launch.get("markers", []), strings, locations))
        launches.append(
            {
                "kernel": launch.get("kernelName"),
                "grid": [launch.get(f"gridDim{axis}") for axis in "XYZ"],
                "block": [launch.get(f"blockDim{axis}") for axis in "XYZ"],
                "events": sorted(
                    events,
                    key=lambda event: event.get("start_ns") or 0,
                ),
            }
        )
    return {"schema": "iket-cutedsl-semantics-v1", "launches": launches}


def write_semantic_trace(input_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = input_path.with_suffix(".semantic.json")
    trace = json.loads(input_path.read_text())
    output_path.write_text(json.dumps(decode_trace(trace), indent=2) + "\n")
    return output_path


def write_semantic_sidecars(output_dir: Path) -> list[Path]:
    return [
        write_semantic_trace(path)
        for path in sorted(output_dir.glob("iket_pid_*.trace.json"))
    ]
