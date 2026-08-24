---
name: ncu-iket-joint-profile
description: Run Nsight Compute PM Sampling and correlate its time-windowed hardware metrics with an already collected NVIDIA IKET fine-grained kernel timeline. Use for joint Tensor/L1/L2/DRAM/SMEM and SM/CTA/warp analysis, not ordinary benchmarking or an aggregate NCU-only report.
---

# NCU + IKET Joint Timeline Analysis

Use IKET for exact event timing and SM/CTA/warp location. Use Nsight Compute
PM Sampling for time-windowed hardware metrics. Combine them by time overlap
while preserving the fact that most NCU values are GPU or hardware-domain
aggregates rather than per-SM measurements.

## Prepare equivalent executions

Select exactly one deterministic kernel launch. Keep the IKET and NCU runs
equivalent in kernel specialization, input shape/dtype/seed, grid/block,
dynamic shared memory, launch order, output checksum, clock policy, and cache
policy. Precompile or warm the JIT cache before the NCU run so compilation does
not replace the intended profiled launch.

IKET and NCU commonly use CUDA injection, so collect them in separate runs.
Do not claim same-execution simultaneity.

## Run NCU PM Sampling

First query PM-sampleable metrics on the target device:

```bash
ncu --query-metrics-collection pmsampling
```

Metric names and counter domains vary by GPU architecture and NCU release.
Prefer the shipped `PmSampling` section for broad analysis; use explicit
`pmsampling:<metric>` names only after confirming availability.

Use this collection template:

```bash
ncu --force-overwrite \
  --section PmSampling \
  --kernel-name 'regex:<KERNEL_REGEX>' \
  --launch-skip <MATCHING_LAUNCHES_TO_SKIP> \
  --launch-count 1 \
  --pm-sampling-interval <INTERVAL> \
  --pm-sampling-buffer-size 0 \
  --pm-sampling-max-passes 0 \
  --disable-pm-warp-sampling \
  --replay-mode kernel \
  --cache-control all \
  --export <OUTPUT_PREFIX> \
  -- <TARGET_COMMAND> <TARGET_ARGUMENTS>
```

Interpret the options deliberately:

- `--pm-sampling-interval 0` lets NCU choose. On architectures with time-based
  sampling, values such as `5000` or `10000` request roughly 5 or 10 us;
  architectures may instead use cycles. Read the actual interval from the
  report rather than trusting the request.
- Keep `--disable-pm-warp-sampling` when only PM counters are needed. Remove it
  when PM Warp Sampling and stall-state correlation are required, and record
  the additional collection behavior.
- `--cache-control all` gives isolated, reproducible replay behavior.
  `--cache-control none` is appropriate when preserving application-warm cache
  state matters. Never compare runs without recording this choice.
- `--pm-sampling-max-passes 0` lets NCU determine the required passes. Fewer
  metrics and compatible counter domains reduce replay and cross-pass
  alignment uncertainty.
- Export `.ncu-rep`. Ordinary console or CSV aggregate output is insufficient
  because joint analysis needs metric instances and their correlation
  timestamps.

If the report contains no in-kernel PM instances, verify kernel selection,
sampling support, context activity, JIT/cache state, interval, and buffer
pressure before drawing conclusions.

## Extract and validate NCU samples

Use the NCU report API to extract each PM metric's instance values and
correlation timestamps. Also extract:

- workload start/end timestamps;
- actual interval in ns or cycles;
- PM pass groups and their metric membership;
- replay count;
- buffer size and merged/dropped samples when present;
- grid/block, kernel name, duration, and metric units.

Require launch metadata to match the IKET run. Warn when durations differ by
more than 3%; reject time correlation by default above 5%. Require at least 10
in-kernel PM samples for timeline conclusions.

## Align the timelines

Convert both sources to kernel-relative time and align kernel start at zero.
Do not rescale one duration to fit the other. Preserve original timestamps,
sample interval, and pass group.

Treat each PM sample as a time window, not an instantaneous event. For a
time-based interval, use `(sample_timestamp - interval, sample_timestamp]`
unless the report proves another convention. Join every IKET range that
overlaps the window and derive:

- active SM, CTA, and warp counts per semantic phase;
- MMA QK/PV issue count and issue coverage;
- TMA/load issue count and coverage;
- data-wait, buffer-wait, barrier-wait, and synchronization coverage;
- per-SM busy time, CTA count, last-finish time, and tail imbalance.

Keep different NCU pass groups separate or label aligned values with their
source pass. Values from different replay passes are not exactly simultaneous.

## Correlate hardware metrics with IKET activity

Use these relationships as diagnostic evidence:

| IKET observation | NCU observation | Likely interpretation |
| --- | --- | --- |
| Few MMA-active SMs | Low Tensor throughput | Parallelism shortage or tail imbalance |
| Many MMA-active SMs, sparse MMA issue | Low Tensor throughput | Dependency, wait, or issue-density bottleneck |
| Dense MMA issue | High Tensor/TMEM throughput | Compute pipeline is well utilized |
| High TMA/load issue or data wait | L2 hit falls and DRAM rises | Data supply pressure |
| Long barrier waits | Tensor and SM activity fall | Producer/consumer or warp-specialization imbalance |
| Late work on a small SM subset | GPU utilization falls near kernel end | Tail effect or scheduler imbalance |

Distinguish MMA issue time from Tensor Core execution. IKET can measure issue
density and waits, but NCU measures achieved hardware throughput relative to a
peak definition. Do not substitute one for the other.

## Interpretation boundaries

- Never assign aggregate Tensor/L1/L2/DRAM values to one SM, CTA, warp, or
  IKET range. An overlapping value must retain a `gpu_aggregate` or equivalent
  label.
- Correlation explains what activity coexisted with a PM window; it does not
  prove ownership when multiple phases or SMs overlap.
- Interpret L1/L2 hit rate only when the matching lookup/activity denominator
  is nonzero. Otherwise report it as unavailable, not 0% hit.
- L1 metrics may not represent TMA or traffic that bypasses conventional L1
  lookup accounting.
- Preserve percent-of-peak and other native units. Derive GB/s only from an
  explicitly supplied sustained peak matching the metric definition and mark
  the result estimated.
- Report both `pct_of_peak_sustained_elapsed` and
  `pct_of_peak_sustained_active` semantics when used; their denominators differ.

Read [references/joint_schema.md](references/joint_schema.md) when defining a
joint window table or timeline output. Read
[references/metric_routing.md](references/metric_routing.md) when selecting or
interpreting PM metrics on a new architecture.
