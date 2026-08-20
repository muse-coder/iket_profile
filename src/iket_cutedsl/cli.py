# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line interface for standalone CuTe DSL IKET profiling."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence

from .bootstrap import parse_detailed_cta
from .semantics import write_semantic_sidecars, write_semantic_trace


def _target(command: Sequence[str]) -> list[str]:
    result = list(command)
    if result and result[0] == "--":
        result.pop(0)
    if not result:
        raise ValueError("a Python target is required after --")
    return result


def _selected_cta(value: str) -> str:
    coordinates = parse_detailed_cta(value)
    if coordinates is None:
        raise argparse.ArgumentTypeError("use --all-ctas instead of 'all'")
    return ",".join(str(coordinate) for coordinate in coordinates)


def build_profile_command(args: argparse.Namespace) -> list[str]:
    """Build the underlying run-iket command without invoking a shell."""
    command = [sys.executable, "-m", "iket.cli.main"]
    if args.working_dir:
        command.extend(("--working-dir", args.working_dir))
    command.extend(("--output-dir", args.output_dir))
    if args.clobber:
        command.append("--clobber")
    if args.log_level:
        command.extend(("--log-level", args.log_level))
    if args.context_buffer_size:
        command.extend(("--context-buffer-size", args.context_buffer_size))

    command.extend(("profile", "--postprocess", args.postprocess))
    if args.max_ts_count_per_warp is not None:
        command.extend(
            ("--max-ts-cnt-per-warp", str(args.max_ts_count_per_warp))
        )
    if args.keep is not None:
        command.append("--keep" if args.keep else "--no-keep")

    detailed_cta = "all" if args.all_ctas else args.detailed_cta
    command.extend(
        (
            "--",
            sys.executable,
            "-m",
            "iket_cutedsl.bootstrap",
            "--detailed-cta",
            detailed_cta,
            "--",
            *_target(args.command),
        )
    )
    return command


def _add_target_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--detailed-cta",
        type=_selected_cta,
        default="0,0,0",
        metavar="X,Y,Z",
        help="CTA that records detailed internal ranges (default: 0,0,0)",
    )
    selection.add_argument(
        "--all-ctas",
        action="store_true",
        help="record detailed internal ranges from every CTA",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Python script or `python -m module` command after --",
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iket-cutedsl",
        description="Automatically instrument and profile a CuTe DSL Python target.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    profile = subparsers.add_parser(
        "profile", help="instrument a Python target and collect an IKET trace"
    )
    profile.add_argument("--output-dir", "-o", default="iket_output")
    profile.add_argument("--working-dir")
    profile.add_argument("--clobber", action="store_true")
    profile.add_argument(
        "--postprocess",
        choices=("perfetto", "json", "html", "none", "all"),
        default="all",
    )
    profile.add_argument(
        "--log-level", choices=("error", "warn", "info", "debug", "trace")
    )
    profile.add_argument("--context-buffer-size")
    profile.add_argument("--max-ts-count-per-warp", type=int)
    keep = profile.add_mutually_exclusive_group()
    keep.add_argument("--keep", action="store_true", dest="keep")
    keep.add_argument("--no-keep", action="store_false", dest="keep")
    profile.set_defaults(keep=None)
    profile.add_argument(
        "--print-command",
        action="store_true",
        help="print the generated run-iket command without executing it",
    )
    _add_target_arguments(profile)

    run = subparsers.add_parser(
        "run", help="instrument and run a target without launching run-iket"
    )
    _add_target_arguments(run)

    decode = subparsers.add_parser(
        "decode", help="decode tile/stage/loop payloads in an IKET JSON trace"
    )
    decode.add_argument("trace", type=Path)
    decode.add_argument("--output", "-o", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    try:
        if args.subcommand == "profile":
            command = build_profile_command(args)
            if args.print_command:
                print(shlex.join(command))
                return 0
            result = subprocess.run(command, check=False)
            if result.returncode == 0 and args.postprocess in ("json", "all"):
                for path in write_semantic_sidecars(Path(args.output_dir)):
                    print(f"[iket-cutedsl] Wrote semantic trace to {path}")
            return result.returncode

        if args.subcommand == "decode":
            output = write_semantic_trace(args.trace, args.output)
            print(output)
            return 0

        from .bootstrap import main as bootstrap_main

        detailed_cta = "all" if args.all_ctas else args.detailed_cta
        return bootstrap_main(
            ["--detailed-cta", detailed_cta, "--", *_target(args.command)]
        )
    except ValueError as error:
        parser.error(str(error))
    return 2


def entrypoint() -> None:
    raise SystemExit(main())
