---
name: iket-profile-cutedsl
description: Profile NVIDIA CuTe DSL kernels with the standalone iket-cutedsl CLI, select detailed CTAs, generate Perfetto or JSON traces, interpret per-warp MMA/TMA/wait timelines, and troubleshoot missing events or oversized traces. Use when Codex needs to check IKET dependency compatibility, profile CUTLASS/FlashInfer/FlashAttention CuTe DSL kernels, investigate which warps wait, inspect prologue/mainloop/epilogue behavior, or compare source-level IKET evidence with Nsight Compute.
---

# Profile CuTe DSL with IKET

Use this repository's `iket-cutedsl` package to instrument an existing Python
CuTe DSL launcher without editing its kernel source.

## Workflow

1. Locate the Python environment that already runs the target kernel.
2. Require Python 3.10 or newer and `nvidia-cutlass-dsl>=4.7,<4.8`. Keep
   `nvidia-cutlass-dsl-libs-base`, core, and CUDA-specific wheels on the same
   4.7.x release. Do not upgrade to 4.8 without adapter and real-kernel tests.
3. Install this repository into that same environment if needed:

   ```bash
   /path/to/python -m pip install --no-deps -e /path/to/iket_profile
   ```

4. Inspect the launcher before profiling. Ensure it imports/JIT-compiles the
   kernel in the same Python process. Prefer a launcher that performs one target
   invocation; setup and unrelated kernels otherwise create additional Grids.
5. Start with one detailed CTA. Default to `(0,0,0)` unless the grid mapping or
   an earlier trace identifies a better representative CTA.
6. Choose output based on the task:
   - use `perfetto` for interactive chronology;
   - use `json` for statistics, pattern inference, or automated comparison;
   - use `all` only when both views are required.
7. Run the profile:

   ```bash
   CUDA_VISIBLE_DEVICES=0 /path/to/iket-cutedsl profile \
     --output-dir /path/to/iket_output \
     --clobber \
     --postprocess perfetto \
     --detailed-cta 0,0,0 \
     -- \
     python /path/to/kernel_launcher.py --launcher-args
   ```

8. Verify that the target reports success and that the output contains
   `iket_pid_*.pftrace` or `iket_pid_*.trace.json`.
9. Inspect event coverage before drawing conclusions. Confirm warp lifetimes
   and expected MMA/TMA/wait families are present.
10. With `json` or `all`, inspect the automatically written
    `*.trace.semantic.json`. Decode an existing JSON trace with:

    ```bash
    iket-cutedsl decode /path/to/iket_pid_123.trace.json
    ```

## Automatic tile and pipeline semantics

Interpret the semantic sidecar in this order:

- `auto.scheduler.tile` contains the scheduler's first four logical coordinate
  axes as `tile_0..tile_3`. Standard CUTLASS schedulers and third-party classes exposing
  `get_current_work`, `initial_work_tile_info`, or `advance_to_next_work` are
  discovered automatically.
- Pipeline wait/commit/release ranges decode to `sequence`, `stage`, and
  `phase`. Correlate adjacent MMA/TMA ranges on the same warp by time; payload
  SSA values are deliberately not carried across control-flow regions.
- `auto.loop.tile_seq` records the source line and dynamic induction value for
  each balanced `cutlass.range` iteration. It is iteration evidence, not an
  unconditional claim that the loop is one output tile.

Loops containing `break`, `continue`, or `return` are skipped so range endpoints
remain balanced. Scheduler coordinates remain the stronger tile identity when
both forms are present. Axis meanings are scheduler-defined; standard GEMM is
normally `(m,n,l,0)`. Payloads retain four non-negative 16-bit axes.

## CTA selection

Keep `--detailed-cta 0,0,0` for the first run. This retains entry/exit lifetime
for every warp in every CTA, but writes internal ranges only for the selected
logical CTA.

Use another CTA when the all-CTA warp envelopes show an outlier:

```bash
iket-cutedsl profile --detailed-cta 3,1,0 -o ./iket_output -- \
  python kernel_launcher.py
```

Use `--all-ctas` only for small grids or when explicitly requested. Warn that
it can multiply trace size and instrumentation overhead.

## Repository launchers

Wrap the repository's existing Python entry point; do not patch the kernel
source merely to invoke the profiler.

For FlashInfer GDN or MoE:

```bash
cd /path/to/flashinfer
iket-cutedsl profile --postprocess perfetto -o ./iket_output \
  --detailed-cta 0,0,0 -- python your_launcher.py
```

For FlashAttention prefill or decode:

```bash
cd /path/to/flash-attention
iket-cutedsl profile --postprocess perfetto -o ./iket_output \
  --detailed-cta 0,0,0 -- python your_launcher.py --mode prefill
```

Preserve application arguments and input shapes so IKET and NCU comparisons
measure the same invocation.

## Python API

Use the context manager only when the caller needs explicit control over the
compilation region:

```python
from iket_cutedsl import patch_cute_iket_ops

with patch_cute_iket_ops(detailed_cta=(0, 0, 0)):
    compile_and_run_kernel()
```

Ensure compilation happens inside the context. Prefer the CLI for ordinary
profiling because it installs hooks before importing the target and sets
`CUTE_DSL_NO_CACHE=1` and `QUACK_CACHE_ENABLED=0`.

## Interpretation guardrails

- Treat `*.issue` as a software-visible issue interval, not asynchronous
  hardware completion latency.
- Treat `mma_wait` as a combined inline-PTX wait plus MMA bundle, not pure MMA
  latency.
- Do not equate API event counts with SASS instruction counts; lowering and
  warp-specialized issuer roles can change the mapping.
- Use IKET wait ranges to localize which warp and source phase waited. Use NCU
  stall metrics for aggregate hardware pressure; do not equate their units.
- Use NCU or an uninstrumented benchmark for final runtime and throughput.

## Troubleshooting

- If Perfetto shows only `Grid`, expand
  `Grid -> GPC -> TPC|VSM -> CTA -> Warp` and open the selected CTA.
- If ranges are missing, confirm compilation occurs in the profiled process,
  the selected CTA exists, no old binary/cache is used, and custom PTX passes
  through CuTe DSL `llvm.inline_asm`.
- If the trace is too large, replace `--all-ctas` with one detailed CTA. All
  CTA warp lifetimes remain available for choosing an outlier CTA.
- Increase timestamp or context buffer options only after IKET reports an
  insufficient capacity error.

## Validation after adapter changes

Run both unit suites and a real kernel before claiming compatibility:

```bash
python -m unittest discover -s tests -v
ruff check src tests
```

Then profile at least one unmodified GEMM. When changing inline-PTX or legacy
pipeline hooks, also profile one FlashAttention and one FlashInfer kernel.
