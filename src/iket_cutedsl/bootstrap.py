# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a Python target while the CuTe DSL instrumentation patch is active."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
from typing import Sequence

from .auto_ops import patch_cute_iket_ops


def parse_detailed_cta(value: str) -> tuple[int, int, int] | None:
    if value == "all":
        return None
    try:
        coordinates = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected x,y,z or all") from error
    if len(coordinates) != 3 or any(coordinate < 0 for coordinate in coordinates):
        raise argparse.ArgumentTypeError("expected non-negative x,y,z or all")
    return coordinates


def _strip_separator(command: Sequence[str]) -> list[str]:
    result = list(command)
    if result and result[0] == "--":
        result.pop(0)
    if not result:
        raise ValueError("a Python script, module, or command is required after --")
    return result


def _is_python_executable(value: str) -> bool:
    return Path(value).name.lower().startswith("python")


def run_python_target(command: Sequence[str]) -> None:
    """Execute a Python target in this process so the patch covers compilation."""
    target = _strip_separator(command)
    if _is_python_executable(target[0]):
        target.pop(0)
    if not target:
        raise ValueError("the Python executable must be followed by a target")

    if target[0] == "-m":
        if len(target) < 2:
            raise ValueError("-m requires a module name")
        module = target[1]
        sys.argv = [module, *target[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return

    if target[0] == "-c":
        if len(target) < 2:
            raise ValueError("-c requires Python source")
        sys.argv = ["-c", *target[2:]]
        namespace = {"__name__": "__main__", "__package__": None}
        exec(compile(target[1], "<string>", "exec"), namespace)
        return

    if target[0].startswith("-"):
        raise ValueError(
            f"unsupported Python option {target[0]!r}; pass a script, -m, or -c"
        )

    script = Path(target[0]).resolve()
    sys.argv = [str(script), *target[1:]]
    sys.path[0] = str(script.parent)
    runpy.run_path(str(script), run_name="__main__")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detailed-cta",
        type=parse_detailed_cta,
        default=(0, 0, 0),
        metavar="X,Y,Z|all",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    # Instrumented code must be compiled in this process, not loaded from a
    # pre-instrumentation CuTe DSL cache entry.
    os.environ["CUTE_DSL_NO_CACHE"] = "1"
    with patch_cute_iket_ops(detailed_cta=args.detailed_cta):
        run_python_target(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
