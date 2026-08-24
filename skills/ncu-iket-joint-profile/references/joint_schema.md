# Joint profile schema

The canonical analysis unit is a PM sampling window joined to every IKET event
that overlaps it.

```json
{
  "time_us": 12.512,
  "window_start_us": 2.512,
  "sample_interval_us": 10.0,
  "source_pass_groups": "0",
  "ncu_gpu_aggregate": {
    "tensor_core_pct": 97.96,
    "l2_hit_pct": 44.35,
    "dram_throughput_pct": 4.58
  },
  "iket_activity": {
    "active_mma_sms": 148,
    "active_mma_warps": 148,
    "qk_issue_count": 4096,
    "pv_issue_count": 4096
  },
  "quality": {
    "l1_hit_valid": false
  }
}
```

Required artifacts:

- `run_manifest.json`: tool/GPU versions, workload, launch, checksum, policies.
- `*.pm_samples.csv`: all PM metrics plus IKET activity counts.
- `*.memory_timeline.csv`: cache validity/activity, L1/L2, SMEM, DRAM tracks.
- `*.perfetto.json[.gz]`: per-SM IKET ranges and separate NCU counter tracks.
- `*.summary.json`: duration agreement, sample coverage, role distributions,
  metric statistics, data-quality flags, and limitations.

Use explicit suffixes: `_pct`, `_gbps_est`, `_count`, `_sms`, `_warps`, and
`_valid`. Preserve raw units. Keep absolute timestamps in source artifacts and
use kernel-relative microseconds in the joint output.

Model a time-based PM sample as `(sample_timestamp - interval,
sample_timestamp]` unless the report proves a different convention. Record the
pass group for every exported window and include the raw metric membership of
each pass group in the run manifest. Never imply exact simultaneity across
different pass groups.
