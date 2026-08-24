#!/usr/bin/env python3
"""Deterministic FA4 workload for joint IKET + NCU experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--metadata-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    repo = args.repo.resolve()
    if not (repo / "flash_attn").is_dir():
        raise RuntimeError(f"not a FlashAttention source tree: {repo}")
    # NCU must profile the kernel launch, not spend its sampling pass compiling
    # a fresh CuTeDSL specialization. Preserve an explicit caller override.
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    sys.path.insert(0, str(repo))
    from flash_attn.cute import flash_attn_func

    torch.manual_seed(args.seed)
    shape = (args.batch, args.seqlen, args.heads, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        for _ in range(args.warmup):
            flash_attn_func(q, k, v, causal=args.causal)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = None
        for _ in range(args.iterations):
            out = flash_attn_func(q, k, v, causal=args.causal)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3 / args.iterations

    assert out is not None
    output = out[0] if isinstance(out, tuple) else out
    metadata: dict[str, object] = {
        "workload": "FA4 forward",
        "repo": str(repo),
        "shape": list(shape),
        "dtype": "bfloat16",
        "causal": args.causal,
        "seed": args.seed,
        "output_checksum": output.float().sum().item(),
        "host_elapsed_ms": elapsed_ms,
    }
    if args.check:
        reference = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=args.causal,
        ).transpose(1, 2)
        error = (output.float() - reference.float()).abs()
        metadata["max_abs_error"] = error.max().item()
        metadata["mean_abs_error"] = error.mean().item()
    if args.metadata_out is not None:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
