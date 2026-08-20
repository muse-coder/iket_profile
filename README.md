# iket-cutedsl

`iket-cutedsl` 为 NVIDIA CuTe DSL kernel 自动添加 IKET 细粒度 profiling，
用于观察每个 warp 的执行时间线、MMA/TMA、wait、barrier，以及
prologue/mainloop/epilogue 阶段。

用户不需要修改 kernel 或手工添加 range。CLI 会在目标 Python 脚本外层自动
建立 instrumentation context；工具仍会插桩，因此存在一定性能扰动。

## 依赖

| 组件 | 版本 |
| --- | --- |
| Python | `>=3.10` |
| `nvidia-cutlass-dsl` | `>=4.7,<4.8` |
| `nvidia-cutlass-dsl-libs-*` | 与 DSL 保持相同的 4.7.x 版本 |

当前实现使用了部分 CuTe DSL 私有 API，升级到 4.8 或更高版本前需要重新运行
单测和真实 kernel 验证。

本工具验证环境为 Python 3.12.3、CuTe DSL 4.7.0、CUDA 13.0、PyTorch
2.13.0+cu130 和 NVIDIA Driver 580.105.08。PyTorch 与 Nsight Compute 不是
工具的硬依赖，它们由目标 launcher 或交叉验证需求决定。

## 安装

安装到运行 CuTe DSL kernel 的同一个 Python 环境：

```bash
git clone git@github.com:muse-coder/iket_profile.git
cd iket_profile

# 已经安装正确 CuTe DSL wheel 时
python -m pip install --no-deps -e .
```

CUDA 13 环境也可以先安装匹配的 DSL wheel：

```bash
python -m pip install 'nvidia-cutlass-dsl[cu13]>=4.7,<4.8'
python -m pip install --no-deps -e .
```

## Profile 一个现有 kernel

假设 kernel 启动脚本为 `my_kernel.py`：

```bash
CUDA_VISIBLE_DEVICES=0 iket-cutedsl profile \
  --output-dir ./iket_output \
  --clobber \
  --postprocess perfetto \
  --detailed-cta 0,0,0 \
  -- \
  python my_kernel.py --kernel-arguments
```

CLI 实际执行：

```python
with patch_cute_iket_ops(detailed_cta=(0, 0, 0)):
    import_and_run_target_script()
```

因此 kernel 的导入、JIT 编译和执行都发生在自动插桩范围内。目标 kernel
必须在该 Python 进程中编译；如果真正的编译发生在子进程，应直接包装那个
子进程的启动脚本。

## CTA 采集范围

默认使用：

```text
--detailed-cta 0,0,0
```

此时：

- 所有 CTA 都记录每个 warp 的 kernel 开始和结束时间；
- 只有 CTA `(0,0,0)` 记录内部 MMA、TMA、wait 等细粒度 range；
- 非目标 CTA 的计算不会被跳过。

可以选择其他 CTA：

```bash
iket-cutedsl profile --detailed-cta 3,1,0 -o ./iket_output -- \
  python my_kernel.py
```

`--all-ctas` 会记录所有 CTA 的内部 range，只建议用于小 grid，否则 trace 和
profiling 开销可能明显增大。

## 输出格式

Perfetto：

```bash
iket-cutedsl profile --postprocess perfetto -o ./iket_output -- \
  python my_kernel.py
```

打开 `iket_pid_*.pftrace` 时，在 Perfetto 中依次展开：

```text
Grid -> GPC -> TPC|VSM -> CTA -> Warp -> event
```

如果只看到 `Grid1`，说明轨道仍处于折叠状态。只有通过
`--detailed-cta` 选中的 CTA 包含内部事件。

结构化 JSON：

```bash
iket-cutedsl profile --postprocess json -o ./iket_output -- \
  python my_kernel.py
```

`.trace.json` 用于脚本统计和模式分析，不是 Perfetto JSON，不能直接拖入
Perfetto UI。也可以使用 `--postprocess html` 或 `--postprocess all`。

## Python API

需要手工控制插桩范围时：

```python
from iket_cutedsl import patch_cute_iket_ops

with patch_cute_iket_ops(detailed_cta=(0, 0, 0)):
    import my_kernel
    my_kernel.compile_and_run()
```

kernel 的导入和编译应位于 context 内。普通场景优先使用 CLI。

## 覆盖范围

当前自动识别：

- native CuTe copy、MMA、TMA 和 pipeline API；
- `cp.async`、TMEM、scheduler 和 barrier；
- 通过 CuTe DSL `llvm.inline_asm` 发出的 MMA、TMA、mbarrier 和 async wait；
- FlashInfer GDN/MoE 和 FlashAttention/Quack 使用的典型路径。

## 数据边界

- IKET range 是 source/API 级事件，不是逐条 SASS PC trace；
- MMA/TMA `issue` 不等于异步硬件操作完成时间；
- `mma_wait` 表示同一个 inline-PTX block 同时包含 wait 和 MMA；
- wait range 可以定位哪个 warp 在哪里等待，但不等同于 NCU stall cycles；
- API range 次数不一定等于 SASS 指令次数；
- 精确的未插桩性能仍应使用正常 benchmark 或 Nsight Compute。

## 测试

```bash
python -m unittest discover -s tests -v
ruff check src tests
```

## Codex Skill

仓库包含精简的 `skills/iket-profile-cutedsl/SKILL.md`。安装该 Skill 后可使用：

```text
Use $iket-profile-cutedsl to profile this CuTe DSL kernel and explain which
warps are waiting.
```

## License

本仓库代码使用 BSD-3-Clause。CuTe DSL 和 IKET runtime 使用各自的 NVIDIA
许可条款。
