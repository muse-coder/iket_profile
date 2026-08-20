# iket-cutedsl

`iket-cutedsl` 是一个可独立安装的 CuTe DSL 细粒度 profiling 工具。它在
CuTe DSL kernel 导入、JIT 编译期间自动安装 IKET hook，因此已有 kernel
通常不需要添加手工插桩代码。

它提供两类互补数据：

- 所有 CTA 中每个 warp 的 kernel 进入和退出时间；
- 选定 CTA 中每个 warp 的 prologue、mainloop、MMA、TMA、wait、barrier 和
  epilogue 源码/API 级时间线。

本仓库只包含自动插桩适配器和 CLI。CuTe DSL、IKET device/runtime API 以及
底层 `run-iket` profiler 由 `nvidia-cutlass-dsl` 4.7.x 提供，并受其自身许可
条款约束。

## 1. 环境要求

- Linux 和受 IKET/CuTe DSL 支持的 NVIDIA GPU；
- Python 3.10 或更高版本；
- `nvidia-cutlass-dsl>=4.7,<4.8`；
- kernel 必须由目标 Python 进程导入并编译，而不是完全在另一个子进程编译。

### 版本兼容矩阵

| 组件 | 支持范围 | 本次验证版本 |
| --- | --- | --- |
| Python | `>=3.10` | `3.12.3` |
| `iket-cutedsl` | `0.1.x` | `0.1.0` |
| `nvidia-cutlass-dsl` | `>=4.7,<4.8` | `4.7.0` |
| `nvidia-cutlass-dsl-libs-base` | 与 DSL 保持相同 4.7.x | `4.7.0` |
| `nvidia-cutlass-dsl-libs-core` | 与 DSL 保持相同 4.7.x | `4.7.0` |
| CUDA 13 DSL wheel | 与 DSL 保持相同 4.7.x | `nvidia-cutlass-dsl-libs-cu13 4.7.0` |
| CUDA Python | CuTe DSL 的传递依赖 | `13.3.1` |
| PyTorch | 仅由目标 launcher 决定 | `2.13.0+cu130` |
| NVIDIA Driver | 满足目标 GPU/kernel | `580.105.08` |
| Nsight Compute | 可选，仅用于交叉验证 | `2026.1.0.0` |

工具依赖 CuTe DSL 的部分私有操作类型和 MLIR 入口，因此当前明确限定在
4.7.x。不要在没有运行单测和真实 kernel 验证的情况下直接放宽到 4.8。

CUDA 13 环境可以先安装匹配的 DSL wheel：

```bash
python -m pip install 'nvidia-cutlass-dsl[cu13]>=4.7,<4.8'
python -m pip install --no-deps -e .
```

建议安装到运行 kernel 的同一个 Python 环境。先确认底层工具存在：

```bash
python -c 'import cutlass.cute, iket; print("CuTe DSL / IKET available")'
python -m iket.cli.main --help
```

## 2. 安装

普通安装：

```bash
git clone git@github.com:muse-coder/iket_profile.git
cd iket_profile
python -m pip install .
```

开发时使用 editable install：

```bash
cd /path/to/iket_profile
python -m pip install --no-deps -e .
```

`--no-deps` 适用于已经安装正确 CuTe DSL wheel 的环境。安装后检查：

```bash
iket-cutedsl --help
iket-cutedsl profile --help
```

## 3. 最短使用方式

假设现有 CuTe DSL 启动脚本为 `my_kernel.py`：

```bash
CUDA_VISIBLE_DEVICES=0 iket-cutedsl profile \
  --output-dir ./iket_output \
  --clobber \
  --postprocess perfetto \
  --detailed-cta 0,0,0 \
  -- \
  python my_kernel.py --kernel-argument value
```

CLI 会自动执行：

1. 启动底层 `run-iket`；
2. 在目标脚本导入和编译 kernel 之前安装 CuTe DSL API/inline-PTX hook；
3. 设置 `CUTE_DSL_NO_CACHE=1`，避免加载插桩前生成的 kernel cache；
4. 完成 IKET dry-run、buffer 规划、正式采集和后处理。

命令中 `--` 之后必须是 Python 脚本、`python -m module` 或 `python -c`：

```bash
iket-cutedsl profile -o ./iket_output -- \
  python -m package.kernel --shape 1024,1024,512
```

如果需要在另一个源码目录运行：

```bash
iket-cutedsl profile \
  --working-dir /path/to/kernel/repository \
  -o /path/to/iket_output \
  --detailed-cta 0,0,0 \
  -- \
  python examples/run_kernel.py
```

## 4. 选择采集范围

默认值为：

```text
--detailed-cta 0,0,0
```

这表示：

- 每个 CTA 的每个 warp 都保留完整 kernel lifetime；
- 只有逻辑 CTA `(blockIdx.x, blockIdx.y, blockIdx.z) = (0,0,0)` 写入内部
  MMA/TMA/wait 等时间戳；
- 非目标 CTA 不跳过任何 kernel 计算，只是不写内部 range。

选择其他 CTA：

```bash
iket-cutedsl profile --detailed-cta 3,1,0 -o ./iket_output -- \
  python my_kernel.py
```

对于很小的 grid，可以记录所有 CTA：

```bash
iket-cutedsl profile --all-ctas -o ./iket_output -- python my_kernel.py
```

`--all-ctas` 可能使 trace 大幅膨胀。对 attention、MoE 或大 GEMM，建议先用
默认 CTA 获取完整 warp 时间线，再按需要选择异常 CTA 重跑。

## 5. 输出格式

### Perfetto

```bash
iket-cutedsl profile --postprocess perfetto -o ./iket_output -- \
  python my_kernel.py
```

输出为 `iket_pid_*.pftrace`。将文件拖入 <https://ui.perfetto.dev/>。轨道结构为：

```text
Root
└── Grid
    └── GPC
        └── TPC | VSM
            └── CTA(x, y, z)
                └── WarpXX
                    ├── Warp Life Time
                    ├── auto.prologue.*
                    ├── auto.main.*
                    ├── auto.pipeline.*wait
                    └── auto.epilogue.*
```

Perfetto 默认会折叠硬件层级。若只看到 `Grid1`，继续展开 GPC、TPC/VSM、
`CTA(0, 0, 0)` 和 Warp，并放大微秒级时间窗口。非目标 CTA 只有 warp
lifetime，这是 CTA 选择策略的预期结果。

### JSON

```bash
iket-cutedsl profile --postprocess json -o ./iket_output -- python my_kernel.py
```

`.trace.json` 是 IKET 的结构化分析格式，包含：

- `launches`：kernel、grid/block、ranges 和 warp lifetimes；
- `locationTable`：GPC/TPC/SM、CTA、warp 映射；
- `stringTable`：事件名称表。

它适合脚本统计和模式推断，不是 Perfetto/Chrome trace JSON，不能直接拖入
Perfetto UI。

也可以使用 `--postprocess html` 或 `--postprocess all`。

## 6. 能识别什么

当前自动 hook 覆盖：

- native CuTe `cute.copy`、`cute.gemm`；
- TMA load/store/reduce issue；
- `tcgen05`、warp-group 和 warp MMA；
- producer/consumer pipeline acquire、wait、commit、release 和 tail；
- `cp.async` load、commit、wait；
- TMEM allocate/load/free；
- scheduler response wait/issue；
- CTA/named/cluster barrier；
- 通过 `cutlass._mlir.dialects.llvm.inline_asm` 传递的 selected inline PTX：
  `tcgen05.mma`、`wgmma.mma_async`、`mma.sync`、TMA、mbarrier、async wait
  和 CTA/cluster barrier。

事件名前缀表示大致阶段：

| 前缀 | 含义 |
| --- | --- |
| `auto.prologue.*` | descriptor prefetch、cluster 初始化等 |
| `auto.main.*` | mainloop TMA/MMA、accumulator 和 pipeline |
| `auto.scheduler.*` | persistent scheduler 请求和等待 |
| `auto.epilogue.*` | TMEM/SMEM copy、TMA store 和释放 |
| `auto.cpasync.*` | `cp.async` kernel 的 load/commit/wait |
| `auto.sync.*` | CTA、named、mbarrier 等同步 |

一个 inline-PTX block 可能静态包含多条相同指令，例如：

```text
auto.main.mma.tcgen05.ptx.x8
```

表示该 inline-asm block 静态包含 8 条匹配的 MMA，不表示整次 kernel 只执行
8 条 MMA。若 block 同时包含 wait loop 和 MMA，会标记为：

```text
auto.main.mma_wait.tcgen.ptx.x8
```

该 range 的持续时间包含等待，不能当作纯 MMA latency。

## 7. FlashInfer 和 FlashAttention

工具不需要针对仓库修改 kernel。只需包装原有的 Python launcher：

```bash
cd /path/to/flashinfer
iket-cutedsl profile --postprocess perfetto \
  -o ./iket_gdn --detailed-cta 0,0,0 -- \
  python your_gdn_launcher.py

iket-cutedsl profile --postprocess perfetto \
  -o ./iket_moe --detailed-cta 0,0,0 -- \
  python your_moe_launcher.py
```

```bash
cd /path/to/flash-attention
iket-cutedsl profile --postprocess perfetto \
  -o ./iket_prefill --detailed-cta 0,0,0 -- \
  python your_flash_attention_launcher.py --mode prefill

iket-cutedsl profile --postprocess perfetto \
  -o ./iket_decode --detailed-cta 0,0,0 -- \
  python your_flash_attention_launcher.py --mode decode
```

端到端验证过的 workload 包括 CUTLASS FP16 GEMM、FlashInfer W4A16 grouped
MoE GEMM、FlashInfer GDN decode，以及 FlashAttention/Quack CuTe DSL 的 prefill
和 decode。FlashAttention 的自定义 inline-PTX MMA/TMA 和 legacy pipeline wait
均能被识别。

建议 launcher 只执行需要分析的一次目标调用。若脚本包含多个 kernel launch，
Perfetto 中会出现多个 Grid，JSON 中也会出现多个 `launches`。

## 8. Python API

需要精确控制编译范围时，可以使用 context manager：

```python
from iket_cutedsl import patch_cute_iket_ops

with patch_cute_iket_ops(detailed_cta=(0, 0, 0)):
    compile_and_run_kernel()
```

编译必须发生在 context 内。已经预编译或从旧 cache 加载的 kernel 不会自动
包含 range。CLI 模式通常更安全，因为它会在导入目标脚本前安装 hook。

`run` 子命令只安装 hook 并运行脚本，不启动底层 profiler，适合检查 kernel
在插桩状态下能否正常编译：

```bash
iket-cutedsl run --detailed-cta 0,0,0 -- python my_kernel.py
```

## 9. 常用参数

| 参数 | 作用 |
| --- | --- |
| `-o`, `--output-dir` | 输出目录，默认 `iket_output` |
| `--clobber` | 覆盖已有输出目录 |
| `--detailed-cta X,Y,Z` | 选择记录内部 range 的 CTA |
| `--all-ctas` | 所有 CTA 都记录内部 range |
| `--postprocess` | `perfetto`、`json`、`html`、`all` 或 `none` |
| `--working-dir` | 目标脚本运行目录 |
| `--max-ts-count-per-warp` | 手工提供每个 warp 的最大时间戳数量提示 |
| `--context-buffer-size` | 覆盖 IKET context buffer 大小 |
| `--keep` / `--no-keep` | 保留或删除中间 tracker/config 数据 |
| `--log-level` | `error`、`warn`、`info`、`debug`、`trace` |
| `--print-command` | 只打印生成的底层 `run-iket` 命令 |

## 10. 数据含义与限制

- IKET range 是 source/API/IR-builder 级事件，不是每条 SASS 指令的 PC trace；
- TMA/MMA `issue` range 表示软件发射区间，不等于异步硬件操作完成时间；
- wait range 能回答哪个 warp 在哪个同步点等待以及等待多久，但不能直接等同
  于 Nsight Compute 的 aggregate stall cycles；
- 一个 `cute.copy` 可能 lower 成多条 SASS 指令，因此 API range 次数与硬件
  instruction count 不一定一一对应；
- instrumentation 会带来运行时扰动。精确的未插桩 kernel 时间、吞吐率和硬件
  stall 原因仍应结合 Nsight Compute；
- 如果 kernel 在另一个 Python/native 子进程中才被编译，当前进程中的 monkey
  patch 无法覆盖它；
- 未识别的自定义编译入口或新的 CuTe DSL 私有 API 需要新增 adapter。

## 11. 常见问题

### Trace 只有 Grid

展开 Perfetto 的 `Grid -> GPC -> TPC|VSM -> CTA -> Warp`，并寻找通过
`--detailed-cta` 指定的 CTA。其他 CTA 默认只有 warp lifetime。

### Trace 没有 MMA/TMA/wait

检查：

1. kernel 是否在 `iket-cutedsl` 包装的同一 Python 进程内编译；
2. 是否打开了正确的目标 CTA；
3. 是否使用了预编译 binary 或子进程；
4. 自定义指令是否经过 CuTe DSL 的 `llvm.inline_asm` 入口；
5. 是否属于当前尚未覆盖的新 API。

### Trace 太大

不要使用 `--all-ctas`，先选择一个代表性 CTA。所有 CTA 的 warp lifetime 仍然
会被保留，可据此判断调度分布和异常 CTA。

## 12. 开发和测试

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
ruff check src tests
```

## 13. Codex Skill

仓库内置 `iket-profile-cutedsl` Skill：

```text
skills/iket-profile-cutedsl/
├── SKILL.md
└── agents/openai.yaml
```

将该目录安装或链接到 Codex skills 目录后，可以通过以下提示调用：

```text
Use $iket-profile-cutedsl to profile this FlashAttention CuTe DSL kernel and
explain which warps are waiting.
```

Skill 会要求 agent 先确认依赖版本，再选择 CTA 和输出格式，验证 trace 中的
warp/range 覆盖，并遵守 IKET 与 NCU 不能直接等值比较的解释边界。

## License

本仓库代码使用 BSD-3-Clause。`nvidia-cutlass-dsl` 和 IKET runtime 使用它们
各自的 NVIDIA 许可条款。
