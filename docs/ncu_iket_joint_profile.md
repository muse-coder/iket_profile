# NCU + IKET Joint Profile

NVIDIA Nsight Compute PM Sampling 采集、解析和 IKET 时间线合并工具。

## 数据模型

- IKET JSON 提供事件起止时间以及 SM/CTA/warp 位置。
- NCU `.ncu-rep` 提供 Tensor Core、TMEM、L1/L2、SMEM、DRAM 的 PM
  Sampling 时间序列。
- 两者使用相同 workload 分别运行，按 kernel 相对起点对齐，不缩放时间。
- NCU 指标保持 `gpu_aggregate` 语义，不归属到单个 SM、CTA 或 warp。
- PM 样本按 `(timestamp - interval, timestamp]` 时间窗口与 IKET range 求重叠。

## 预检

```bash
ncu-iket-preflight
```

## 采集 NCU PM Sampling

以下命令中的路径均为示例：

```bash
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 ncu-iket-collect \
  --output runs/fa4/ncu/fa4_pm \
  --kernel-regex '.*FlashAttentionForwardSm100.*' \
  --interval-ns 10000 \
  --disable-pm-warp-sampling \
  -- \
  python adapters/flash_attention/profile_fa4.py \
    --repo <FLASH_ATTENTION_REPO> \
    --seqlen 8192 --heads 32 --warmup 0 --iterations 1 \
    --metadata-out runs/fa4/ncu/workload.json
```

FA4 adapter 也会默认启用 CuTeDSL cache。若在 NCU sampling pass 内重新 JIT
specialization，报告可能只有 profiler metadata 而没有 PM metric instances；验证器
会把有效样本数为零判为失败。

如果只想检查 NCU 命令而不运行：

```bash
ncu-iket-collect \
  --output runs/example/report --kernel-regex '.*Kernel.*' \
  --print-command -- python demo.py
```

## 合并外部 IKET trace

IKET 采集由外部流程完成；本工具只要求 NVIDIA IKET `*.trace.json`。例如已有
`FA4.*` range 时：

```bash
ncu-iket-merge \
  --iket runs/fa4/iket/iket_pid_0xNNNN.trace.json \
  --ncu runs/fa4/ncu/fa4_pm.ncu-rep \
  --event-prefix 'FA4.' \
  --kernel-regex '.*FlashAttentionForwardSm100.*' \
  --workload-metadata runs/fa4/ncu/workload.json \
  --output-prefix runs/fa4/joint/fa4_b300
```

其他 IKET 命名体系可修改 `--event-prefix`；传空字符串表示接收该 launch 的
全部 range。

## 输出

- `*.perfetto.json[.gz]`：SM/warp IKET ranges 与独立 NCU counter tracks。
- `*.pm_samples.csv`：PM 指标及每个窗口重叠的 IKET 活动数。
- `*.memory_timeline.csv`：L1/L2、SMEM、DRAM 和有效性标记。
- `*.summary.json`：持续时间、range 分布、PM 指标统计和警告。
- `*.run_manifest.json`：来源、启动维度、采样间隔、replay/pass 和归因边界。

验证：

```bash
ncu-iket-validate \
  --summary runs/fa4/joint/fa4_b300.summary.json \
  --pm-csv runs/fa4/joint/fa4_b300.pm_samples.csv \
  --memory-csv runs/fa4/joint/fa4_b300.memory_timeline.csv \
  --trace runs/fa4/joint/fa4_b300.perfetto.json \
  --manifest runs/fa4/joint/fa4_b300.run_manifest.json \
  --require-role FA4.MMA
```

## 限制

- IKET 与 NCU 都使用 CUDA injection，因此通常不能在同一次 kernel execution
  中同时采集；必须使用等价的确定性运行。
- NCU 多个 replay pass 的指标不保证完全同时，manifest 会记录 replay/pass。
- L1 hit 只有在 lookup hit/miss activity 非零时才可解释。
- DRAM/SMEM 百分比不能直接称为 GB/s；只有显式提供 sustained peak 后才能输出
  带 `_gbps_est` 后缀的估算值。

## 测试

```bash
python -m unittest discover -s tests -v
python -m py_compile src/ncu_iket_profile/*.py adapters/flash_attention/profile_fa4.py
```
