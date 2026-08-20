# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Automatic IKET instrumentation for a CuTe DSL warp timeline.

This module instruments major prologue, mainloop, scheduler, epilogue, and
selected inline-PTX operations without tracing individual SASS instructions.
"""

import ast
import builtins
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import re
import sys
import threading
from types import ModuleType
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl import ast_helpers
from cutlass.base_dsl.ast_preprocessor import DSLPreprocessor, _create_module_attribute
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
_saved_targets: Optional[List[Tuple[Any, str, Any]]] = None
_active_detailed_cta: Optional[Tuple[int, int, int]] = None
_trace_state = threading.local()
_MISSING = object()

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

@dataclass(frozen=True)
class _EventSpec:
    name: str
    payload: Any = None


EventSelection = Optional[Union[str, _EventSpec]]
EventSelector = Union[str, Callable[..., EventSelection]]
DetailedCta = Optional[Tuple[int, int, int]]

_PIPELINE_STAGE_BITS = 8
_PIPELINE_PHASE_SHIFT = _PIPELINE_STAGE_BITS
_PIPELINE_COUNT_SHIFT = _PIPELINE_PHASE_SHIFT + 1
_TILE_COORD_BITS = 16
_TILE_COORD_MASK = (1 << _TILE_COORD_BITS) - 1
_LOOP_SITE_BITS = 16
_LOOP_INDEX_BITS = 64 - _LOOP_SITE_BITS
_LOOP_INDEX_MASK = (1 << _LOOP_INDEX_BITS) - 1
_LOOP_EVENT_NAME = "auto.loop.tile_seq"
_SCHEDULER_TILE_EVENT_NAME = "auto.scheduler.tile"
_LOOP_START_HELPER = "_iket_auto_loop_range_start"
_LOOP_END_HELPER = "_iket_auto_loop_range_end"
_SCHEDULER_METHOD_NAMES = (
    "get_current_work",
    "initial_work_tile_info",
    "advance_to_next_work",
)


def _normalize_detailed_cta(detailed_cta: Any) -> DetailedCta:
    if detailed_cta is None:
        return None
    if not isinstance(detailed_cta, tuple) or len(detailed_cta) != 3:
        raise TypeError("detailed_cta must be a three-integer tuple or None")
    if any(type(coord) is not int or coord < 0 for coord in detailed_cta):
        raise ValueError("detailed_cta coordinates must be non-negative integers")
    return detailed_cta


def _as_event_spec(selection: EventSelection) -> Optional[_EventSpec]:
    if selection is None:
        return None
    if isinstance(selection, _EventSpec):
        return selection
    return _EventSpec(selection)


@dsl_user_op
@cute.jit
def _pack_pipeline_state(
    state: pipeline.PipelineState,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cutlass.Int64:
    """Pack ``count/index/phase`` into one IKET payload.

    Layout: ``count << 9 | phase << 8 | stage``.  The stage field is eight
    bits, which is comfortably above the number of stages supported by CuTe
    pipelines.  ``count`` is the exact pipeline progression counter and is a
    useful regular-loop tile sequence when no scheduler coordinate exists.
    """
    return (
        (cutlass.Int64(state.count) << _PIPELINE_COUNT_SHIFT)
        | (cutlass.Int64(state.phase) << _PIPELINE_PHASE_SHIFT)
        | cutlass.Int64(state.index)
    )


@dsl_user_op
@cute.jit
def _pack_tile_coord(
    m: Any,
    n: Any,
    batch: Any,
    extra: Any,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cutlass.Int64:
    """Pack four non-negative scheduler coordinate axes into 64 bits."""
    return (
        ((cutlass.Int64(m) & _TILE_COORD_MASK) << (3 * _TILE_COORD_BITS))
        | ((cutlass.Int64(n) & _TILE_COORD_MASK) << (2 * _TILE_COORD_BITS))
        | ((cutlass.Int64(batch) & _TILE_COORD_MASK) << _TILE_COORD_BITS)
        | (cutlass.Int64(extra) & _TILE_COORD_MASK)
    )


@dsl_user_op
@cute.jit
def _pack_loop_iteration(
    source_line: int,
    induction: Any,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cutlass.Int64:
    """Pack a 16-bit source line and 48-bit dynamic induction value."""
    return (
        (cutlass.Int64(source_line & ((1 << _LOOP_SITE_BITS) - 1)) << _LOOP_INDEX_BITS)
        | (cutlass.Int64(induction) & _LOOP_INDEX_MASK)
    )


@dsl_user_op
@cute.jit
def _range_start_for_cta(
    event_name: str,
    detailed_cta: Tuple[int, int, int],
    payload: Any = None,
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
    if cutlass.const_expr(payload is None):
        return (
            cute.experimental.iket.range_start(event_name, loc=loc, ip=ip)
            if is_target
            else cute.experimental.iket.sentinel_token(event_name, loc=loc, ip=ip)
        )
    return (
        cute.experimental.iket.range_start(event_name, payload, loc=loc, ip=ip)
        if is_target
        else cute.experimental.iket.sentinel_token(event_name, loc=loc, ip=ip)
    )


def _range_start(
    event_name: str,
    detailed_cta: DetailedCta,
    payload: Any = None,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Any:
    if detailed_cta is None:
        if payload is None:
            return cute.experimental.iket.range_start(event_name, loc=loc, ip=ip)
        return cute.experimental.iket.range_start(event_name, payload, loc=loc, ip=ip)
    return _range_start_for_cta(
        event_name, detailed_cta, payload, loc=loc, ip=ip
    )


def _range_end(
    token: Any,
    payload: Any = None,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    if payload is None:
        cute.experimental.iket.range_end(token, loc=loc, ip=ip)
    else:
        cute.experimental.iket.range_end(token, payload, loc=loc, ip=ip)


@dsl_user_op
@cute.jit
def _mark_for_cta(
    event_name: str,
    payload: Any,
    detailed_cta: Tuple[int, int, int],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    block_x, block_y, block_z = cute.arch.block_idx()
    is_target = (
        (block_x == detailed_cta[0])
        & (block_y == detailed_cta[1])
        & (block_z == detailed_cta[2])
    )
    if is_target:
        cute.experimental.iket.mark(event_name, payload, loc=loc, ip=ip)


def _mark(
    event_name: str,
    payload: Any,
    detailed_cta: DetailedCta,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    if detailed_cta is None:
        cute.experimental.iket.mark(event_name, payload, loc=loc, ip=ip)
    else:
        _mark_for_cta(event_name, payload, detailed_cta, loc=loc, ip=ip)


@dsl_user_op
@cute.jit
def _mark_valid_scheduler_tile(
    payload: Any,
    is_valid: Any,
    detailed_cta: DetailedCta,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    if is_valid:
        _mark(
            _SCHEDULER_TILE_EVENT_NAME,
            payload,
            detailed_cta,
            loc=loc,
            ip=ip,
        )


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
                _range_end(token, loc=loc, ip=ip)
        finally:
            _trace_state.suppress_inline_hook = previous_suppression

    return traced_inline_asm


def _participant_origin(participant: Any) -> Any:
    if isinstance(participant, pipeline.PipelineProducer):
        return participant._PipelineProducer__pipeline
    if isinstance(participant, pipeline.PipelineConsumer):
        return participant._PipelineConsumer__pipeline
    return participant.get_origin()


def _pipeline_state(subject: Any, args: Tuple[Any, ...] = ()) -> Any:
    """Return the dynamic PipelineState carried by a participant or call."""
    if isinstance(subject, pipeline.PipelineState):
        return subject
    if isinstance(subject, pipeline.PipelineProducer):
        return subject._PipelineProducer__state
    if isinstance(subject, pipeline.PipelineConsumer):
        return subject._PipelineConsumer__state

    immutable_state = getattr(
        subject, "_ImmutableResourceHandle__immutable_state", None
    )
    if immutable_state is not None:
        return immutable_state

    for value in args:
        if isinstance(value, pipeline.PipelineState):
            return value
    return None


def _pipeline_spec(
    event_name: str,
    subject: Any,
    args: Tuple[Any, ...] = (),
) -> _EventSpec:
    state = _pipeline_state(subject, args)
    payload = _pack_pipeline_state(state) if state is not None else None
    return _EventSpec(event_name, payload)


def _fixed_pipeline_event(
    event_name: str,
) -> Callable[..., _EventSpec]:
    def select(subject: Any, *args: Any, **_: Any) -> _EventSpec:
        return _pipeline_spec(event_name, subject, args)

    return select


def _producer_acquire_event(
    producer: pipeline.PipelineProducer, *args: Any
) -> _EventSpec:
    origin = _participant_origin(producer)
    if isinstance(origin, pipeline.PipelineTmaUmma):
        name = "auto.main.tma.buffer_wait"
    elif isinstance(origin, pipeline.PipelineUmmaAsync):
        name = "auto.main.acc.buffer_wait"
    else:
        name = "auto.main.pipeline.buffer_wait"
    return _pipeline_spec(name, producer, args)


def _consumer_wait_event(
    consumer: pipeline.PipelineConsumer, *args: Any
) -> _EventSpec:
    origin = _participant_origin(consumer)
    if isinstance(origin, pipeline.PipelineTmaUmma):
        name = "auto.main.tma.data_wait"
    elif isinstance(origin, pipeline.PipelineUmmaAsync):
        name = "auto.epilogue.acc.wait"
    else:
        name = "auto.main.pipeline.data_wait"
    return _pipeline_spec(name, consumer, args)


def _producer_commit_event(handle: Any, *args: Any) -> Optional[_EventSpec]:
    if isinstance(_participant_origin(handle), pipeline.PipelineUmmaAsync):
        return _pipeline_spec("auto.main.acc.commit", handle, args)
    return None


def _consumer_release_event(handle: Any, *args: Any) -> Optional[_EventSpec]:
    origin = _participant_origin(handle)
    if isinstance(origin, pipeline.PipelineTmaUmma):
        name = "auto.main.tma.release"
    elif isinstance(origin, pipeline.PipelineUmmaAsync):
        name = "auto.epilogue.acc.release"
    else:
        return None
    return _pipeline_spec(name, handle, args)


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
        selection = (
            event_selector(*args, **kwargs)
            if callable(event_selector)
            else event_selector
        )
        event = _as_event_spec(selection)
        if event is None:
            return original(*args, loc=loc, ip=ip, **kwargs)

        token = _range_start(
            event.name, detailed_cta, event.payload, loc=loc, ip=ip
        )
        try:
            return original(*args, loc=loc, ip=ip, **kwargs)
        finally:
            _range_end(token, event.payload, loc=loc, ip=ip)

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
            _range_end(token, loc=loc, ip=ip)

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
        event_name = _mma_event_name(atom.op)
        token = _range_start(event_name, detailed_cta, loc=loc, ip=ip)
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
            _range_end(token, loc=loc, ip=ip)

    return traced_gemm


def _flatten_coord(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(component for item in value for component in _flatten_coord(item))
    return (0 if value is None else value,)


def _tile_coord_components(
    work_tile: Any,
) -> Optional[Tuple[Any, Any, Any, Any]]:
    try:
        tile_idx = work_tile.tile_idx
        values = _flatten_coord(tile_idx)
    except (AttributeError, TypeError):
        return None
    if not values:
        return None
    padded = (*values[:4], 0, 0, 0, 0)
    return padded[0], padded[1], padded[2], padded[3]


def _make_traced_scheduler_work(
    original: Callable[..., Any], detailed_cta: DetailedCta
) -> Callable[..., Any]:
    @dsl_user_op
    @wraps(original)
    def traced_scheduler_work(
        *args: Any,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
        **kwargs: Any,
    ) -> Any:
        if getattr(_trace_state, "suppress_scheduler_hook", False):
            return original(*args, loc=loc, ip=ip, **kwargs)

        previous = getattr(_trace_state, "suppress_scheduler_hook", False)
        _trace_state.suppress_scheduler_hook = True
        try:
            work_tile = original(*args, loc=loc, ip=ip, **kwargs)
        finally:
            _trace_state.suppress_scheduler_hook = previous

        components = _tile_coord_components(work_tile)
        if components is not None:
            payload = _pack_tile_coord(*components, loc=loc, ip=ip)
            _mark_valid_scheduler_tile(
                payload,
                work_tile.is_valid_tile,
                detailed_cta,
                loc=loc,
                ip=ip,
            )
        return work_tile

    return traced_scheduler_work


def _is_scheduler_class(value: Any, module_name: str) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == module_name
        and value.__name__.endswith("Scheduler")
        and any(name in value.__dict__ for name in _SCHEDULER_METHOD_NAMES)
    )


def _patch_scheduler_classes_in_module(
    module: ModuleType, detailed_cta: DetailedCta
) -> None:
    """Patch scheduler protocol implementations defined by a newly loaded module."""
    assert _saved_targets is not None
    patched = {(id(owner), name) for owner, name, _ in _saved_targets}
    for attribute_name, value in vars(module).items():
        if not attribute_name.endswith("Scheduler"):
            continue
        if not _is_scheduler_class(value, module.__name__):
            continue
        for method_name in _SCHEDULER_METHOD_NAMES:
            original = value.__dict__.get(method_name)
            if original is None or (id(value), method_name) in patched:
                continue
            _set_patch(
                _saved_targets,
                value,
                method_name,
                _make_traced_scheduler_work(original, detailed_cta),
            )
            patched.add((id(value), method_name))


def _make_scheduler_aware_import(
    original_import: Callable[..., Any], detailed_cta: DetailedCta
) -> Callable[..., Any]:
    """Discover third-party scheduler classes after their module is imported."""
    @wraps(original_import)
    def traced_import(*args: Any, **kwargs: Any) -> Any:
        result = original_import(*args, **kwargs)
        candidates = set()
        if isinstance(result, ModuleType):
            candidates.add(result)
        if args and isinstance(args[0], str):
            candidates.add(sys.modules.get(args[0]))
        fromlist = args[3] if len(args) > 3 else kwargs.get("fromlist", ())
        if isinstance(result, ModuleType) and fromlist:
            for item in fromlist:
                if isinstance(item, str):
                    candidates.add(sys.modules.get(f"{result.__name__}.{item}"))
        for module in candidates:
            if isinstance(module, ModuleType):
                _patch_scheduler_classes_in_module(module, detailed_cta)
        return result

    return traced_import


@dsl_user_op
def _loop_range_start(
    source_line: int,
    induction: Any,
    detailed_cta: DetailedCta,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Any:
    payload = _pack_loop_iteration(source_line, induction, loc=loc, ip=ip)
    return _range_start(
        _LOOP_EVENT_NAME, detailed_cta, payload, loc=loc, ip=ip
    )


@dsl_user_op
def _loop_range_end(
    token: Any,
    source_line: int,
    induction: Any,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    payload = _pack_loop_iteration(source_line, induction, loc=loc, ip=ip)
    _range_end(token, payload, loc=loc, ip=ip)


def _should_instrument_loop(preprocessor: DSLPreprocessor, node: ast.For) -> bool:
    if not isinstance(node.target, ast.Name) or not node.body:
        return False
    if any(
        isinstance(item, (ast.Break, ast.Continue, ast.Return))
        for item in ast.walk(node)
    ):
        return False
    file_name = str(preprocessor.session_data.file_name)
    return "nvidia_cutlass_dsl" not in file_name and "iket_cutedsl" not in file_name


def _make_loop_transform(
    original: Callable[..., Any], detailed_cta: DetailedCta
) -> Callable[..., Any]:
    @wraps(original)
    def transform_for_loop(
        preprocessor: DSLPreprocessor,
        node: ast.For,
        active_symbols: Any,
        active_callables: Any,
    ) -> Any:
        if _should_instrument_loop(preprocessor, node):
            source_line = node.lineno & ((1 << _LOOP_SITE_BITS) - 1)
            token_name = (
                f"__iket_auto_loop_token_{source_line}_"
                f"{preprocessor.session_data.counter}"
            )
            helper_location = {
                "lineno": node.lineno,
                "col_offset": node.col_offset,
            }
            cta_node: ast.expr
            if detailed_cta is None:
                cta_node = ast.Constant(value=None)
            else:
                cta_node = ast.Tuple(
                    elts=[ast.Constant(value=value) for value in detailed_cta],
                    ctx=ast.Load(),
                )
            start = ast.Assign(
                targets=[ast.Name(id=token_name, ctx=ast.Store())],
                value=ast.Call(
                    func=_create_module_attribute(
                        _LOOP_START_HELPER, **helper_location
                    ),
                    args=[
                        ast.Constant(value=source_line),
                        ast.Name(id=node.target.id, ctx=ast.Load()),
                        cta_node,
                    ],
                    keywords=[],
                ),
            )
            end = ast.Expr(
                value=ast.Call(
                    func=_create_module_attribute(
                        _LOOP_END_HELPER, **helper_location
                    ),
                    args=[
                        ast.Name(id=token_name, ctx=ast.Load()),
                        ast.Constant(value=source_line),
                        ast.Name(id=node.target.id, ctx=ast.Load()),
                    ],
                    keywords=[],
                )
            )
            node.body = [
                ast.copy_location(ast.fix_missing_locations(start), node),
                *node.body,
                ast.copy_location(ast.fix_missing_locations(end), node),
            ]
        return original(preprocessor, node, active_symbols, active_callables)

    return transform_for_loop


def _install_patch(
    saved_targets: List[Tuple[Any, str, Any]],
    owner: Any,
    name: str,
    event_selector: EventSelector,
    detailed_cta: DetailedCta,
) -> None:
    original = getattr(owner, name)
    saved_targets.append((owner, name, original))
    setattr(owner, name, _make_traced_call(original, event_selector, detailed_cta))


def _set_patch(
    saved_targets: List[Tuple[Any, str, Any]], owner: Any, name: str, value: Any
) -> None:
    saved_targets.append((owner, name, getattr(owner, name, _MISSING)))
    setattr(owner, name, value)


def _restore_targets(saved_targets: List[Tuple[Any, str, Any]]) -> None:
    for owner, name, original in reversed(saved_targets):
        if original is _MISSING:
            delattr(owner, name)
        else:
            setattr(owner, name, original)


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

    _set_patch(
        _saved_targets,
        ast_helpers,
        _LOOP_START_HELPER,
        _loop_range_start,
    )
    _set_patch(
        _saved_targets,
        ast_helpers,
        _LOOP_END_HELPER,
        _loop_range_end,
    )
    original_loop_transform = DSLPreprocessor.transform_for_loop
    _set_patch(
        _saved_targets,
        DSLPreprocessor,
        "transform_for_loop",
        _make_loop_transform(original_loop_transform, detailed_cta),
    )

    for scheduler_type in (
        utils.ClcDynamicPersistentTileScheduler,
        utils.StaticPersistentTileScheduler,
    ):
        for method_name in ("get_current_work", "initial_work_tile_info"):
            original = getattr(scheduler_type, method_name)
            _set_patch(
                _saved_targets,
                scheduler_type,
                method_name,
                _make_traced_scheduler_work(original, detailed_cta),
            )

    # Grouped GEMM overrides get_current_work instead of inheriting the base
    # implementation. Runtime schedulers inherit the already-patched static
    # methods and therefore need no separate entry here.
    _set_patch(
        _saved_targets,
        utils.StaticPersistentGroupTileScheduler,
        "get_current_work",
        _make_traced_scheduler_work(
            utils.StaticPersistentGroupTileScheduler.get_current_work,
            detailed_cta,
        ),
    )

    # FlashAttention, Quack, and downstream projects commonly provide their
    # own scheduler classes. Discover those classes when their modules finish
    # importing and wrap the same protocol methods without repository-specific
    # imports or source changes.
    _set_patch(
        _saved_targets,
        builtins,
        "__import__",
        _make_scheduler_aware_import(builtins.__import__, detailed_cta),
    )

    targets = (
        (
            pipeline.PipelineTmaAsync,
            "producer_acquire",
            _fixed_pipeline_event("auto.main.tma.buffer_wait"),
        ),
        (
            pipeline.PipelineTmaUmma,
            "producer_acquire",
            _fixed_pipeline_event("auto.main.tma.buffer_wait"),
        ),
        (
            pipeline.PipelineAsync,
            "producer_acquire",
            _fixed_pipeline_event("auto.pipeline.producer_wait"),
        ),
        (
            pipeline.PipelineAsync,
            "consumer_wait",
            _fixed_pipeline_event("auto.pipeline.consumer_wait"),
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
        (
            pipeline.PipelineProducer,
            "tail",
            _fixed_pipeline_event("auto.main.pipeline.tail"),
        ),
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
        (
            pipeline.PipelineTmaStore,
            "producer_commit",
            _fixed_pipeline_event("auto.epilogue.store.commit"),
        ),
        (
            pipeline.PipelineTmaStore,
            "producer_acquire",
            _fixed_pipeline_event("auto.epilogue.store.wait"),
        ),
        (
            pipeline.PipelineTmaStore,
            "producer_tail",
            _fixed_pipeline_event("auto.epilogue.store.tail"),
        ),
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
            _trace_state.suppress_scheduler_hook = False
            try:
                _install_timeline_patches(kernel_module, normalized_cta)
            except Exception:
                _restore_targets(_saved_targets)
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
                _restore_targets(_saved_targets)
                _saved_targets = None
                _active_detailed_cta = None
    finally:
        _patch_lock.release()
