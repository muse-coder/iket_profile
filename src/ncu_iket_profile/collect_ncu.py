#!/usr/bin/env python3
"""Run Nsight Compute PM Sampling for exactly one selected kernel launch."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def target_command(values: Sequence[str]) -> list[str]:
    command = list(values)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("target command is required after --")
    return command


def build_command(args: argparse.Namespace) -> list[str]:
    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("ncu was not found on PATH")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ncu,
        "--force-overwrite",
        "--section",
        args.section,
        "--kernel-name",
        f"regex:{args.kernel_regex}",
        "--launch-count",
        "1",
        "--pm-sampling-interval",
        str(args.interval_ns),
        "--pm-sampling-max-passes",
        str(args.max_passes),
        "--replay-mode",
        args.replay_mode,
        "--cache-control",
        args.cache_control,
        "--export",
        str(output),
    ]
    if args.disable_pm_warp_sampling:
        command.append("--disable-pm-warp-sampling")
    command.extend(("--", *target_command(args.command)))
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-regex", required=True)
    parser.add_argument("--interval-ns", type=int, default=10000)
    parser.add_argument("--section", default="PmSampling")
    parser.add_argument("--max-passes", type=int, default=0)
    parser.add_argument("--replay-mode", choices=("kernel", "application"), default="kernel")
    parser.add_argument("--cache-control", choices=("all", "none"), default="all")
    parser.add_argument(
        "--disable-pm-warp-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        command = build_command(args)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.print_command:
        print(shlex.join(command))
        return 0
    result = subprocess.run(command, check=False)
    manifest = {
        "command": command,
        "returncode": result.returncode,
        "requested_interval_ns": args.interval_ns,
        "kernel_regex": args.kernel_regex,
        "section": args.section,
        "replay_mode": args.replay_mode,
        "cache_control": args.cache_control,
    }
    manifest_path = args.output.resolve().with_suffix(".collect.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")
    return result.returncode


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
