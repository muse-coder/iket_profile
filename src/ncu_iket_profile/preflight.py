#!/usr/bin/env python3
"""Check NCU + IKET joint-profile prerequisites."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


def command_version(command: list[str]) -> dict:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        first = next((line for line in result.stdout.splitlines() if line.strip()), "")
        return {"ok": result.returncode == 0, "version": first}
    except Exception as error:
        return {"ok": False, "error": str(error)}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ncu = shutil.which("ncu")
    ncu_report = sorted(
        Path("/opt/nvidia/nsight-compute").glob("*/extras/python/ncu_report.py"),
        reverse=True,
    )
    checks: dict[str, dict] = {
        "ncu": {
            "available": ncu is not None,
            "path": ncu,
            **(command_version([ncu, "--version"]) if ncu else {}),
        },
        "ncu_report": {
            "available": bool(ncu_report),
            "path": str(ncu_report[0]) if ncu_report else None,
        },
        "iket_json_input": {
            "available": True,
            "note": "The merger accepts NVIDIA IKET trace JSON from an external collector.",
        },
        "tool_files": {
            "available": all(
                (root / path).is_file()
                for path in (
                    "ncu_iket_profile/collect_ncu.py",
                    "ncu_iket_profile/merge.py",
                    "ncu_iket_profile/validate.py",
                )
            )
        },
    }
    try:
        import torch

        cuda = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cuda.update(
                {
                    "device": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "sm_count": props.multi_processor_count,
                }
            )
        checks["cuda"] = cuda
    except Exception as error:
        checks["cuda"] = {"available": False, "error": str(error)}

    checks["cutlass_iket_api"] = {
        "available": importlib.util.find_spec("cutlass.cute.experimental.iket") is not None
    }
    required = ("ncu", "ncu_report", "tool_files", "cuda")
    ok = all(checks[name].get("available", False) for name in required)
    result = {
        "tool_root": str(root),
        "python": sys.version.split()[0],
        "checks": checks,
        "ok": ok,
    }
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
