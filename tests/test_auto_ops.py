# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import ast
import inspect
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest import mock

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu.cpasync.copy import (
    CopyG2SOp,
    CopyBulkTensorTileG2SOp,
    CopyBulkTensorTileS2GOp,
    CopyReduceBulkTensorTileS2GOp,
)
from cutlass.cute.nvgpu.tcgen05.copy import Ld32x32bOp
from cutlass.cute.nvgpu.tcgen05.mma import MmaF16BF16Op
from cutlass.cute.nvgpu.warp.copy import StMatrix8x8x16bOp
from cutlass.cute.typing import AddressSpace


PACKAGE_SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))


from iket_cutedsl import auto_ops  # noqa: E402


@cute.jit
def _dynamic_loop_fixture(stop: cutlass.Int32):
    for tile in cutlass.range(stop):
        stop = tile
    return stop


class IketAutoOpsTest(unittest.TestCase):
    def test_classifies_tma_copy_direction(self):
        load = object.__new__(CopyBulkTensorTileG2SOp)
        store = object.__new__(CopyBulkTensorTileS2GOp)
        reduce_store = object.__new__(CopyReduceBulkTensorTileS2GOp)
        cpasync_load = object.__new__(CopyG2SOp)

        self.assertEqual(auto_ops._copy_event_name(load), "auto.main.tma.load_issue")
        self.assertEqual(
            auto_ops._copy_event_name(store), "auto.epilogue.tma.store_issue"
        )
        self.assertEqual(
            auto_ops._copy_event_name(reduce_store),
            "auto.epilogue.tma.reduce_issue",
        )
        self.assertEqual(
            auto_ops._copy_event_name(cpasync_load), "auto.cpasync.load_issue"
        )
        self.assertIsNone(auto_ops._copy_event_name(object()))

    def test_classifies_epilogue_copies(self):
        tmem_load = object.__new__(Ld32x32bOp)
        smem_store = object.__new__(StMatrix8x8x16bOp)

        self.assertEqual(
            auto_ops._copy_event_name(tmem_load), "auto.epilogue.tmem.load"
        )
        self.assertEqual(
            auto_ops._copy_event_name(smem_store), "auto.epilogue.smem.store"
        )

        atom = SimpleNamespace(op=object())
        src = SimpleNamespace(memspace=AddressSpace.rmem)
        dst = SimpleNamespace(memspace=AddressSpace.smem)
        self.assertEqual(
            auto_ops._copy_call_event_name(atom, src, dst),
            "auto.epilogue.smem.store",
        )

    def test_classifies_tcgen05_mma(self):
        op = object.__new__(MmaF16BF16Op)
        self.assertEqual(auto_ops._mma_event_name(op), "auto.main.mma.tcgen05_issue")

    def test_classifies_inline_ptx_instruction_bundles(self):
        tcgen05_bundle = """
        // tcgen05.mma in a comment must not count
        @p tcgen05.mma.cta_group::1.kind::f16 [a], b, c, d, 1;
        tcgen05.mma.cta_group::1.kind::f16 [a], b, c, d, 1;
        /* wgmma.mma_async in a block comment must not win classification */
        """
        self.assertEqual(
            auto_ops._inline_ptx_event_name(tcgen05_bundle),
            "auto.main.mma.tcgen05.ptx.x2",
        )
        self.assertEqual(
            auto_ops._inline_ptx_event_name("wgmma.mma_async.sync.aligned.m64n64k16;"),
            "auto.main.mma.wgmma.ptx.x1",
        )
        self.assertEqual(
            auto_ops._inline_ptx_event_name(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32;"
            ),
            "auto.main.mma.warp.ptx.x1",
        )
        self.assertEqual(
            auto_ops._inline_ptx_event_name(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes;"
            ),
            "auto.main.tma.load.ptx.x1",
        )
        self.assertEqual(
            auto_ops._inline_ptx_event_name(
                "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group;"
            ),
            "auto.epilogue.tma.store.ptx.x1",
        )
        self.assertEqual(
            auto_ops._inline_ptx_event_name(
                "cp.reduce.async.bulk.tensor.2d.global.shared::cta.bulk_group.add.f32;"
            ),
            "auto.epilogue.tma.red.ptx.x1",
        )
        self.assertEqual(
            auto_ops._inline_ptx_event_name("bar.sync 0;"),
            "auto.sync.barrier.wait.ptx.x1",
        )
        self.assertEqual(
            auto_ops._inline_ptx_event_name(
                "mbarrier.try_wait.parity.shared::cta.b64 p, [addr], phase;"
            ),
            "auto.sync.mbarrier.wait.ptx.x1",
        )
        self.assertEqual(
            auto_ops._inline_ptx_event_name(
                "mbarrier.try_wait.parity.shared::cta.b64 p, [addr], phase;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [a], b, c, d, 1;"
            ),
            "auto.main.mma_wait.tcgen.ptx.x1",
        )
        self.assertIsNone(auto_ops._inline_ptx_event_name(object()))

        classified = (
            auto_ops._inline_ptx_event_name(tcgen05_bundle),
            auto_ops._inline_ptx_event_name(
                "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group;"
            ),
        )
        self.assertTrue(all(len(name) <= 32 for name in classified if name))

    def test_inline_ptx_wrapper_places_range_around_original_call(self):
        original = mock.Mock(return_value="inline-result")
        traced = auto_ops._make_traced_inline_asm(original, (3, 2, 1))
        token = object()

        with mock.patch.object(auto_ops, "_range_start", return_value=token) as start:
            with mock.patch.object(auto_ops, "_range_end") as end:
                result = traced(
                    None,
                    ["operand"],
                    "tcgen05.mma.cta_group::1.kind::f16 [a], b, c, d, 1;",
                    "r",
                )

        self.assertEqual(result, "inline-result")
        start.assert_called_once_with(
            "auto.main.mma.tcgen05.ptx.x1",
            (3, 2, 1),
            loc=None,
            ip=None,
        )
        original.assert_called_once()
        end.assert_called_once_with(token, loc=None, ip=None)

    def test_patch_is_nested_and_restores_originals(self):
        original_copy = cute.copy
        original_gemm = cute.gemm
        original_wait = pipeline.PipelineConsumer.wait_and_advance
        original_legacy_wait = pipeline.PipelineAsync.consumer_wait
        original_cpasync_wait = cute.arch.cp_async_wait_group
        original_inline_asm = llvm.inline_asm
        original_loop_transform = auto_ops.DSLPreprocessor.transform_for_loop
        original_scheduler_work = (
            auto_ops.utils.ClcDynamicPersistentTileScheduler.get_current_work
        )

        with auto_ops.patch_cute_iket_ops():
            patched_copy = cute.copy
            patched_gemm = cute.gemm
            patched_wait = pipeline.PipelineConsumer.wait_and_advance
            patched_legacy_wait = pipeline.PipelineAsync.consumer_wait
            patched_cpasync_wait = cute.arch.cp_async_wait_group
            patched_inline_asm = llvm.inline_asm
            patched_loop_transform = auto_ops.DSLPreprocessor.transform_for_loop
            patched_scheduler_work = (
                auto_ops.utils.ClcDynamicPersistentTileScheduler.get_current_work
            )
            self.assertIsNot(patched_copy, original_copy)
            self.assertIsNot(patched_gemm, original_gemm)
            self.assertIsNot(patched_wait, original_wait)
            self.assertIsNot(patched_legacy_wait, original_legacy_wait)
            self.assertIsNot(patched_cpasync_wait, original_cpasync_wait)
            self.assertIsNot(patched_inline_asm, original_inline_asm)
            self.assertIsNot(patched_loop_transform, original_loop_transform)
            self.assertIsNot(patched_scheduler_work, original_scheduler_work)
            self.assertIs(
                getattr(auto_ops.ast_helpers, auto_ops._LOOP_START_HELPER),
                auto_ops._loop_range_start,
            )
            with auto_ops.patch_cute_iket_ops():
                self.assertIs(cute.copy, patched_copy)
                self.assertIs(cute.gemm, patched_gemm)
                self.assertIs(pipeline.PipelineConsumer.wait_and_advance, patched_wait)
                self.assertIs(
                    pipeline.PipelineAsync.consumer_wait, patched_legacy_wait
                )
                self.assertIs(cute.arch.cp_async_wait_group, patched_cpasync_wait)
                self.assertIs(llvm.inline_asm, patched_inline_asm)
                self.assertIs(
                    auto_ops.DSLPreprocessor.transform_for_loop,
                    patched_loop_transform,
                )

        self.assertIs(cute.copy, original_copy)
        self.assertIs(cute.gemm, original_gemm)
        self.assertIs(pipeline.PipelineConsumer.wait_and_advance, original_wait)
        self.assertIs(pipeline.PipelineAsync.consumer_wait, original_legacy_wait)
        self.assertIs(cute.arch.cp_async_wait_group, original_cpasync_wait)
        self.assertIs(llvm.inline_asm, original_inline_asm)
        self.assertIs(
            auto_ops.DSLPreprocessor.transform_for_loop, original_loop_transform
        )
        self.assertIs(
            auto_ops.utils.ClcDynamicPersistentTileScheduler.get_current_work,
            original_scheduler_work,
        )
        self.assertFalse(
            hasattr(auto_ops.ast_helpers, auto_ops._LOOP_START_HELPER)
        )

    def test_detailed_cta_is_validated_and_inherited_by_nested_patch(self):
        with self.assertRaisesRegex(TypeError, "three-integer tuple"):
            with auto_ops.patch_cute_iket_ops(detailed_cta=[0, 0, 0]):
                pass
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            with auto_ops.patch_cute_iket_ops(detailed_cta=(0, -1, 0)):
                pass

        with auto_ops.patch_cute_iket_ops(detailed_cta=(2, 1, 0)):
            self.assertEqual(auto_ops._active_detailed_cta, (2, 1, 0))
            with auto_ops.patch_cute_iket_ops():
                self.assertEqual(auto_ops._active_detailed_cta, (2, 1, 0))
            with self.assertRaisesRegex(ValueError, "different detailed_cta"):
                with auto_ops.patch_cute_iket_ops(detailed_cta=(3, 1, 0)):
                    pass

        self.assertIsNone(auto_ops._active_detailed_cta)

    def test_patch_restores_originals_after_exception(self):
        original_copy = cute.copy
        original_gemm = cute.gemm
        original_wait = pipeline.PipelineConsumer.wait_and_advance

        with self.assertRaisesRegex(RuntimeError, "compile failed"):
            with auto_ops.patch_cute_iket_ops():
                raise RuntimeError("compile failed")

        self.assertIs(cute.copy, original_copy)
        self.assertIs(cute.gemm, original_gemm)
        self.assertIs(pipeline.PipelineConsumer.wait_and_advance, original_wait)

    def test_loop_transform_wraps_each_dynamic_iteration(self):
        node = ast.parse("for tile in cutlass.range(2, 7):\n    consume(tile)\n").body[0]
        self.assertIsInstance(node, ast.For)
        preprocessor = SimpleNamespace(
            session_data=SimpleNamespace(file_name="/tmp/kernel.py", counter=13)
        )
        original = mock.Mock(return_value=[node])

        transformed = auto_ops._make_loop_transform(original, (2, 1, 0))(
            preprocessor, node, [set()], [set()]
        )

        self.assertEqual(transformed, [node])
        self.assertEqual(len(node.body), 3)
        start, _, end = node.body
        self.assertIsInstance(start, ast.Assign)
        self.assertIsInstance(end, ast.Expr)
        self.assertEqual(start.value.func.attr, auto_ops._LOOP_START_HELPER)
        self.assertEqual(end.value.func.attr, auto_ops._LOOP_END_HELPER)
        self.assertEqual(ast.literal_eval(start.value.args[2]), (2, 1, 0))
        self.assertEqual(len(start.value.args), 3)
        self.assertEqual(len(end.value.args), 3)
        self.assertEqual(start.targets[0].id, end.value.args[0].id)
        self.assertEqual(start.value.args[1].id, "tile")
        self.assertEqual(end.value.args[2].id, "tile")

    def test_real_preprocessor_accepts_injected_loop_helpers(self):
        preprocessor = auto_ops.DSLPreprocessor(["cutlass"])
        with auto_ops.patch_cute_iket_ops(detailed_cta=(0, 0, 0)):
            with preprocessor.get_session():
                tree = preprocessor.transform(
                    inspect.unwrap(_dynamic_loop_fixture), globals()
                )

        transformed = ast.unparse(tree)
        self.assertIn(auto_ops._LOOP_START_HELPER, transformed)
        self.assertIn(auto_ops._LOOP_END_HELPER, transformed)

    def test_loop_transform_skips_unbalanced_control_flow(self):
        node = ast.parse(
            "for tile in cutlass.range(8):\n"
            "    if tile == 3:\n"
            "        continue\n"
            "    consume(tile)\n"
        ).body[0]
        self.assertIsInstance(node, ast.For)
        preprocessor = SimpleNamespace(
            session_data=SimpleNamespace(file_name="/tmp/kernel.py", counter=7)
        )
        original = mock.Mock(return_value=[node])

        auto_ops._make_loop_transform(original, None)(
            preprocessor, node, [set()], [set()]
        )

        self.assertEqual(len(node.body), 2)

    def test_discovers_and_restores_third_party_scheduler(self):
        module = ModuleType("downstream.tile_scheduler")

        def get_current_work(self, *, loc=None, ip=None):
            return self.work

        scheduler = type(
            "DownstreamTileScheduler",
            (),
            {
                "__module__": module.__name__,
                "get_current_work": get_current_work,
            },
        )
        module.DownstreamTileScheduler = scheduler
        original = scheduler.get_current_work

        with auto_ops.patch_cute_iket_ops(detailed_cta=(0, 0, 0)):
            auto_ops._patch_scheduler_classes_in_module(module, (0, 0, 0))
            self.assertIsNot(scheduler.get_current_work, original)

        self.assertIs(scheduler.get_current_work, original)

    def test_import_hook_rescans_completed_requested_module(self):
        module_name = "downstream.completed_scheduler"
        module = ModuleType(module_name)

        def get_current_work(self, *, loc=None, ip=None):
            return self.work

        scheduler = type(
            "CompletedTileScheduler",
            (),
            {"__module__": module_name, "get_current_work": get_current_work},
        )
        module.CompletedTileScheduler = scheduler
        original = scheduler.get_current_work

        def fake_import(name, *args, **kwargs):
            sys.modules[name] = module
            return ModuleType("downstream")

        try:
            with auto_ops.patch_cute_iket_ops(detailed_cta=(0, 0, 0)):
                traced_import = auto_ops._make_scheduler_aware_import(
                    fake_import, (0, 0, 0)
                )
                traced_import(module_name)
                self.assertIsNot(scheduler.get_current_work, original)
        finally:
            sys.modules.pop(module_name, None)

        self.assertIs(scheduler.get_current_work, original)

    def test_flattens_four_axis_and_nested_scheduler_coordinates(self):
        work = SimpleNamespace(tile_idx=(4, None, (7, 9)))
        self.assertEqual(auto_ops._tile_coord_components(work), (4, 0, 7, 9))


if __name__ == "__main__":
    unittest.main()
