# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Automatic IKET instrumentation for a CuTe DSL warp timeline.

This module instruments major prologue, mainloop, scheduler, epilogue, and
selected inline-PTX operations without tracing individual SASS instructions.
"""

from contextlib import contextmanager
from functools import wraps
import re
import threading
from types import ModuleType
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.atom import CopyAtom, MmaAtom
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.common import CopyR2SOp
from cutlass.cute.nvgpu.cpasync.copy import (
    CopyG2SOp,
    CopyBulkTensorIm2ColS2GOp,
    CopyBulkTensorTileS2GOp,
    CopyReduceBulkTensorTileS2GOp,
    TmaCopyOp,
)
from cutlass.cute.nvgpu.tcgen05.copy import _LdBase as TmemLoadOp
from cutlass.cute.nvgpu.tcgen05.mma import Tcgen05MmaOp
from cutlass.cute.nvgpu.warp.copy import StMatrix16x8x8bOp, StMatrix8x8x16bOp
from cutlass.cute.nvgpu.warp.mma import WarpMmaOp
from cutlass.cute.nvgpu.warpgroup.mma import WarpGroupMmaOp
from cutlass.cute.typing import AddressSpace, Tensor


_TMA_STORE_OPS = (CopyBulkTensorTileS2GOp, CopyBulkTensorIm2ColS2GOp)
_TMA_REDUCE_STORE_OPS = (CopyReduceBulkTensorTileS2GOp,)
_SMEM_STORE_OPS = (CopyR2SOp, StMatrix8x8x16bOp, StMatrix16x8x8bOp)

_patch_lock = threading.RLock()
_patch_depth = 0
_saved_targets: Optional[List[Tuple[Any, str, Callable[..., Any]]]] = None
_active_detailed_cta: Optional[Tuple[int, int, int]] = None
_trace_state = threading.local()

_PTX_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_PTX_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
_INLINE_PTX_MMA_PATTERNS = (
    (
        re.compile(r"\btcgen05\.mma\b"),
        "auto.main.mma.tcgen05.ptx",
        "auto.main.mma_wait.tcgen.ptx",
    ),
    (
        re.compile(r"\bwgmma\.mma_async\b"),
        "auto.main.mma.wgmma.ptx",
        "auto.main.mma_wait.wgmma.ptx",
    ),
    (
        re.compile(r"(?<![\w.])mma\.sync\b"),
        "auto.main.mma.warp.ptx",
        "auto.main.mma_wait.warp.ptx",
    ),
)
_INLINE_PTX_TMA_REDUCE_RE = re.compile(r"\bcp\.reduce\.async\.bulk\.tensor\b")
_INLINE_PTX_TMA_RE = re.compile(r"\bcp\.async\.bulk\.tensor[^\s;]*")
_INLINE_PTX_WAIT_PATTERNS = (
    (
        re.compile(r"\bmbarrier\.(?:try_wait|test_wait)\b"),
        "auto.sync.mbarrier.wait.ptx",
    ),
    (
        re.compile(r"\bcp\.async(?:\.bulk)?\.wait_group\b"),
        "auto.cpasync.wait.ptx",
    ),
    (
        re.compile(r"\b(?:wgmma|tcgen05)\.wait_group\b"),
        "auto.main.mma.wait.ptx",
    ),
    (
        re.compile(r"\bbar(?:rier)?\.(?:sync|cluster\.wait)\b"),
        "auto.sync.barrier.wait.ptx",
    ),
)

EventSelector = Union[str, Callable[..., Optional[str]]]
DetailedCta = Optional[Tuple[int, int, int]]


def _normalize_detailed_cta(detailed_cta: Any) -> DetailedCta:
    if detailed_cta is None:
        return None
    if not isinstance(detailed_cta, tuple) or len(detailed_cta) != 3:
        raise TypeError("detailed_cta must be a three-integer tuple or None")
    if any(type(coord) is not int or coord < 0 for coord in detailed_cta):
        raise ValueError("detailed_cta coordinates must be non-negative integers")
    return detailed_cta


@dsl_user_op
@cute.jit
def _range_start_for_cta(
    event_name: str,
    detailed_cta: Tuple[int, int, int],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Any:
    """Start a real range only in ``detailed_cta`` and a sentinel elsewhere."""
    block_x, block_y, block_z = cute.arch.block_idx()
    is_target = (
        (block_x == detailed_cta[0])
        & (block_y == detailed_cta[1])
        & (block_z == detailed_cta[2])
    )
    return (
        cute.experimental.iket.range_start(event_name, loc=loc, ip=ip)
        if is_target
        else cute.experimental.iket.sentinel_token(event_name, loc=loc, ip=ip)
    )


def _range_start(
    event_name: str,
    detailed_cta: DetailedCta,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Any:
    if detailed_cta is None:
        return cute.experimental.iket.range_start(event_name, loc=loc, ip=ip)
    return _range_start_for_cta(event_name, detailed_cta, loc=loc, ip=ip)


def _copy_event_name(op: Any) -> Optional[str]:
    """Return a phase-qualified IKET label for a selected copy operation."""
    if isinstance(op, _TMA_REDUCE_STORE_OPS):
        return "auto.epilogue.tma.reduce_issue"
    if isinstance(op, _TMA_STORE_OPS):
        return "auto.epilogue.tma.store_issue"
    if isinstance(op, TmaCopyOp):
        return "auto.main.tma.load_issue"
    if isinstance(op, CopyG2SOp):
        return "auto.cpasync.load_issue"
    if isinstance(op, TmemLoadOp):
        return "auto.epilogue.tmem.load"
    if isinstance(op, _SMEM_STORE_OPS):
        return "auto.epilogue.smem.store"
    return None


def _copy_call_event_name(
    atom: CopyAtom,
    src: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    dst: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
) -> Optional[str]:
    event_name = _copy_event_name(atom.op)
    if event_name is not None:
        return event_name

    src_primary = src[0] if isinstance(src, (list, tuple)) else src
    dst_primary = dst[0] if isinstance(dst, (list, tuple)) else dst
    try:
        if (
            src_primary.memspace == AddressSpace.rmem
            and dst_primary.memspace == AddressSpace.smem
        ):
            return "auto.epilogue.smem.store"
    except (AttributeError, ValueError):
        pass
    return None


def _mma_event_name(op: Any) -> str:
    """Return a phase-qualified IKET label for an MMA operation."""
    if isinstance(op, Tcgen05MmaOp):
        return "auto.main.mma.tcgen05_issue"
    if isinstance(op, WarpGroupMmaOp):
        return "auto.main.mma.wgmma_issue"
    if isinstance(op, WarpMmaOp):
        return "auto.main.mma.warp_issue"
    return "auto.main.mma.issue"


def _inline_ptx_event_name(asm_string: Any) -> Optional[str]:
    """Classify profiled instruction families in one LLVM inline-PTX block.

    A single ``llvm.inline_asm`` call can emit an unrolled sequence of the same
    instruction. Keep that static count in the label instead of presenting the
    whole call as one hardware instruction.
    """
    if not isinstance(asm_string, str):
        return None

    uncommented = _PTX_BLOCK_COMMENT_RE.sub("", asm_string)
    uncommented = _PTX_LINE_COMMENT_RE.sub("", uncommented)
    wait_matches = [
        (len(pattern.findall(uncommented)), event_name)
        for pattern, event_name in _INLINE_PTX_WAIT_PATTERNS
    ]
    wait_count = sum(count for count, _ in wait_matches)
    for pattern, event_name, mma_wait_event_name in _INLINE_PTX_MMA_PATTERNS:
        instruction_count = len(pattern.findall(uncommented))
        if instruction_count:
            base_name = mma_wait_event_name if wait_count else event_name
            return f"{base_name}.x{instruction_count}"

    reduce_count = len(_INLINE_PTX_TMA_REDUCE_RE.findall(uncommented))
    if reduce_count:
        return f"auto.epilogue.tma.red.ptx.x{reduce_count}"

    tma_instructions = _INLINE_PTX_TMA_RE.findall(uncommented)
    if tma_instructions:
        load_count = sum(
            instruction.find("shared") >= 0
            and instruction.find("global") >= 0
            and instruction.find("shared") < instruction.find("global")
            for instruction in tma_instructions
        )
        store_count = sum(
            instruction.find("shared") >= 0
            and instruction.find("global") >= 0
            and instruction.find("global") < instruction.find("shared")
            for instruction in tma_instructions
        )
        if load_count == len(tma_instructions):
            return f"auto.main.tma.load.ptx.x{load_count}"
        if store_count == len(tma_instructions):
            return f"auto.epilogue.tma.store.ptx.x{store_count}"
        return f"auto.tma.mixed.ptx.x{len(tma_instructions)}"

    active_waits = [(count, name) for count, name in wait_matches if count]
    if len(active_waits) == 1:
        count, event_name = active_waits[0]
        return f"{event_name}.x{count}"
    if active_waits:
        return f"auto.sync.mixed.wait.ptx.x{wait_count}"
    return None


def _make_traced_inline_asm(
    original_inline_asm: Callable[..., Any], detailed_cta: DetailedCta
) -> Callable[..., Any]:
    """Trace selected PTX instructions emitted through LLVM InlineAsmOp."""

    @wraps(original_inline_asm)
    def traced_inline_asm(
        res: Any,
        operands_: Any,
        asm_string: Any,
        constraints: Any,
        *,
        has_side_effects: Any = None,
        is_align_stack: Any = None,
        tail_call_kind: Any = None,
        asm_dialect: Any = None,
        operand_attrs: Any = None,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Any:
        event_name = _inline_ptx_event_name(asm_string)
        if event_name is None or getattr(_trace_state, "suppress_inline_hook", False):
            return original_inline_asm(
                res,
                operands_,
                asm_string,
                constraints,
                has_side_effects=has_side_effects,
                is_align_stack=is_align_stack,
                tail_call_kind=tail_call_kind,
                asm_dialect=asm_dialect,
                operand_attrs=operand_attrs,
                loc=loc,
                ip=ip,
            )

        # IKET lowering can itself use low-level operations. Suppress recursive
        # classification while inserting this inline-PTX range.
        previous_suppression = getattr(_trace_state, "suppress_inline_hook", False)
        _trace_state.suppress_inline_hook = True
        try:
            token = _range_start(event_name, detailed_cta, loc=loc, ip=ip)
            try:
                return original_inline_asm(
                    res,
                    operands_,
                    asm_string,
                    constraints,
                    has_side_effects=has_side_effects,
                    is_align_stack=is_align_stack,
                    tail_call_kind=tail_call_kind,
                    asm_dialect=asm_dialect,
                    operand_attrs=operand_attrs,
                    loc=loc,
                    ip=ip,
                )
            finally:
                cute.experimental.iket.range_end(token, loc=loc, ip=ip)
        finally:
            _trace_state.suppress_inline_hook = previous_suppression

    return traced_inline_asm


def _participant_origin(participant: Any) -> Any:
    if isinstance(participant, pipeline.PipelineProducer):
        return participant._PipelineProducer__pipeline
    if isinstance(participant, pipeline.PipelineConsumer):
        return participant._PipelineConsumer__pipeline
    return participant.get_origin()


def _producer_acquire_event(producer: pipeline.PipelineProducer, *_: Any) -> str:
    origin = _participant_origin(producer)
    if isinstance(origin, pipeline.PipelineTmaUmma):
        return "auto.main.tma.buffer_wait"
    if isinstance(origin, pipeline.PipelineUmmaAsync):
        return "auto.main.acc.buffer_wait"
    return "auto.main.pipeline.buffer_wait"


def _consumer_wait_event(consumer: pipeline.PipelineConsumer, *_: Any) -> str:
    origin = _participant_origin(consumer)
    if isinstance(origin, pipeline.PipelineTmaUmma):
        return "auto.main.tma.data_wait"
    if isinstance(origin, pipeline.PipelineUmmaAsync):
        return "auto.epilogue.acc.wait"
    return "auto.main.pipeline.data_wait"


def _producer_commit_event(handle: Any, *_: Any) -> Optional[str]:
    if isinstance(_participant_origin(handle), pipeline.PipelineUmmaAsync):
        return "auto.main.acc.commit"
    return None


def _consumer_release_event(handle: Any, *_: Any) -> Optional[str]:
    origin = _participant_origin(handle)
    if isinstance(origin, pipeline.PipelineTmaUmma):
        return "auto.main.tma.release"
    if isinstance(origin, pipeline.PipelineUmmaAsync):
        return "auto.epilogue.acc.release"
    return None


def _named_barrier_event(barrier: pipeline.NamedBarrier, *_: Any) -> Optional[str]:
    # The tutorial reserves barrier 1 for the epilogue's four participating warps.
    if barrier.barrier_id == 1:
        return "auto.epilogue.sync.wait"
    return None


def _make_traced_call(
    original: Callable[..., Any],
    event_selector: EventSelector,
    detailed_cta: DetailedCta,
) -> Callable[..., Any]:
    @dsl_user_op
    @wraps(original)
    def traced_call(
        *args: Any,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
        **kwargs: Any,
    ) -> Any:
        event_name = (
            event_selector(*args, **kwargs)
            if callable(event_selector)
            else event_selector
        )
        if event_name is None:
            return original(*args, loc=loc, ip=ip, **kwargs)

        token = _range_start(event_name, detailed_cta, loc=loc, ip=ip)
        try:
            return original(*args, loc=loc, ip=ip, **kwargs)
        finally:
            cute.experimental.iket.range_end(token, loc=loc, ip=ip)

    return traced_call


def _make_traced_copy(
    original_copy: Callable[..., Any], detailed_cta: DetailedCta
) -> Callable[..., Any]:
    @dsl_user_op
    def traced_copy(
        atom: CopyAtom,
        src: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        dst: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        *,
        pred: Optional[Tensor] = None,
        unroll_factor: Optional[int] = None,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
        **kwargs: Any,
    ) -> Any:
        event_name = _copy_call_event_name(atom, src, dst)
        if event_name is None:
            return original_copy(
                atom,
                src,
                dst,
                pred=pred,
                unroll_factor=unroll_factor,
                loc=loc,
                ip=ip,
                **kwargs,
            )

        token = _range_start(event_name, detailed_cta, loc=loc, ip=ip)
        previous_suppression = getattr(_trace_state, "suppress_inline_hook", False)
        _trace_state.suppress_inline_hook = True
        try:
            return original_copy(
                atom,
                src,
                dst,
                pred=pred,
                unroll_factor=unroll_factor,
                loc=loc,
                ip=ip,
                **kwargs,
            )
        finally:
            _trace_state.suppress_inline_hook = previous_suppression
            cute.experimental.iket.range_end(token, loc=loc, ip=ip)

    return traced_copy


def _make_traced_gemm(
    original_gemm: Callable[..., Any], detailed_cta: DetailedCta
) -> Callable[..., Any]:
    @dsl_user_op
    def traced_gemm(
        atom: MmaAtom,
        d: Tensor,
        a: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        b: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        c: Tensor,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
        **kwargs: Any,
    ) -> Any:
        token = _range_start(_mma_event_name(atom.op), detailed_cta, loc=loc, ip=ip)
        previous_suppression = getattr(_trace_state, "suppress_inline_hook", False)
        _trace_state.suppress_inline_hook = True
        try:
            return original_gemm(
                atom,
                d,
                a,
                b,
                c,
                loc=loc,
                ip=ip,
                **kwargs,
            )
        finally:
            _trace_state.suppress_inline_hook = previous_suppression
            cute.experimental.iket.range_end(token, loc=loc, ip=ip)

    return traced_gemm


def _install_patch(
    saved_targets: List[Tuple[Any, str, Callable[..., Any]]],
    owner: Any,
    name: str,
    event_selector: EventSelector,
    detailed_cta: DetailedCta,
) -> None:
    original = getattr(owner, name)
    saved_targets.append((owner, name, original))
    setattr(owner, name, _make_traced_call(original, event_selector, detailed_cta))


def _install_timeline_patches(
    kernel_module: Optional[ModuleType], detailed_cta: DetailedCta
) -> None:
    assert _saved_targets is not None

    original_copy = cute.copy
    original_gemm = cute.gemm
    original_inline_asm = llvm.inline_asm
    _saved_targets.extend(
        [
            (cute, "copy", original_copy),
            (cute, "gemm", original_gemm),
            (llvm, "inline_asm", original_inline_asm),
        ]
    )
    cute.copy = _make_traced_copy(original_copy, detailed_cta)
    cute.gemm = _make_traced_gemm(original_gemm, detailed_cta)
    llvm.inline_asm = _make_traced_inline_asm(original_inline_asm, detailed_cta)

    targets = (
        (
            pipeline.PipelineTmaAsync,
            "producer_acquire",
            "auto.main.tma.buffer_wait",
        ),
        (
            pipeline.PipelineTmaUmma,
            "producer_acquire",
            "auto.main.tma.buffer_wait",
        ),
        (
            pipeline.PipelineAsync,
            "producer_acquire",
            "auto.pipeline.producer_wait",
        ),
        (
            pipeline.PipelineAsync,
            "consumer_wait",
            "auto.pipeline.consumer_wait",
        ),
        (
            pipeline.PipelineProducer,
            "acquire_and_advance",
            _producer_acquire_event,
        ),
        (
            pipeline.PipelineConsumer,
            "wait_and_advance",
            _consumer_wait_event,
        ),
        (
            pipeline.PipelineProducer.ImmutableResourceHandle,
            "commit",
            _producer_commit_event,
        ),
        (
            pipeline.PipelineConsumer.ImmutableResourceHandle,
            "release",
            _consumer_release_event,
        ),
        (pipeline.PipelineProducer, "tail", "auto.main.pipeline.tail"),
        (
            pipeline.PipelineClcFetchAsync,
            "consumer_wait",
            "auto.scheduler.response_wait",
        ),
        (
            utils.ClcDynamicPersistentTileScheduler,
            "advance_to_next_work",
            "auto.scheduler.issue",
        ),
        (pipeline.PipelineTmaStore, "producer_commit", "auto.epilogue.store.commit"),
        (pipeline.PipelineTmaStore, "producer_acquire", "auto.epilogue.store.wait"),
        (pipeline.PipelineTmaStore, "producer_tail", "auto.epilogue.store.tail"),
        (utils.TmemAllocator, "allocate", "auto.epilogue.tmem.alloc"),
        (utils.TmemAllocator, "wait_for_alloc", "auto.setup.tmem.alloc_wait"),
        (
            utils.TmemAllocator,
            "relinquish_alloc_permit",
            "auto.epilogue.tmem.teardown",
        ),
        (utils.TmemAllocator, "free", "auto.epilogue.tmem.teardown"),
        (pipeline.NamedBarrier, "arrive_and_wait", _named_barrier_event),
        (cpasync, "prefetch_descriptor", "auto.prologue.tma.prefetch"),
        (cute.arch, "cp_async_commit_group", "auto.cpasync.commit"),
        (cute.arch, "cp_async_wait_group", "auto.cpasync.wait"),
        (cute.arch, "barrier", "auto.sync.cta.wait"),
    )
    for owner, name, event_selector in targets:
        _install_patch(_saved_targets, owner, name, event_selector, detailed_cta)

    if kernel_module is not None:
        _install_patch(
            _saved_targets,
            kernel_module,
            "pipeline_init_wait",
            "auto.prologue.cluster.wait",
            detailed_cta,
        )


@contextmanager
def patch_cute_iket_ops(
    kernel_module: Optional[ModuleType] = None,
    *,
    detailed_cta: DetailedCta = None,
) -> Iterator[None]:
    """Temporarily install the automatic warp-timeline instrumentation.

    ``kernel_module`` is optional. Pass it when the kernel imported DSL helpers
    directly (rather than resolving them through a module), so those aliases can
    be patched and restored too. When ``detailed_cta=(x, y, z)`` is supplied,
    internal ranges are emitted only by that logical CTA; IKET warp lifetimes
    still retain entry/exit timing for every CTA. A re-entrant lock serializes
    patch users, nested contexts are supported, and all targets are restored if
    compilation fails.
    """
    global _active_detailed_cta, _patch_depth, _saved_targets

    normalized_cta = _normalize_detailed_cta(detailed_cta)

    _patch_lock.acquire()
    try:
        if _patch_depth == 0:
            _saved_targets = []
            _active_detailed_cta = normalized_cta
            try:
                _install_timeline_patches(kernel_module, normalized_cta)
            except Exception:
                for owner, name, original in reversed(_saved_targets):
                    setattr(owner, name, original)
                _saved_targets = None
                _active_detailed_cta = None
                raise
        else:
            if kernel_module is not None:
                raise ValueError(
                    "kernel_module can only be supplied to the outer context"
                )
            if normalized_cta is not None and normalized_cta != _active_detailed_cta:
                raise ValueError("nested patch cannot select a different detailed_cta")
        _patch_depth += 1

        try:
            yield
        finally:
            _patch_depth -= 1
            if _patch_depth == 0:
                assert _saved_targets is not None
                for owner, name, original in reversed(_saved_targets):
                    setattr(owner, name, original)
                _saved_targets = None
                _active_detailed_cta = None
    finally:
        _patch_lock.release()
