# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

from iket_cutedsl.semantics import (
    decode_loop_payload,
    decode_pipeline_payload,
    decode_tile_payload,
    decode_trace,
)


class SemanticPayloadTest(unittest.TestCase):
    def test_decodes_pipeline_state(self):
        payload = (37 << 9) | (1 << 8) | 3
        self.assertEqual(
            decode_pipeline_payload(payload),
            {
                "source": "pipeline_state",
                "sequence": 37,
                "phase": 1,
                "stage": 3,
            },
        )

    def test_decodes_scheduler_tile(self):
        payload = (11 << 48) | (7 << 32) | (2 << 16) | 5
        self.assertEqual(
            decode_tile_payload(payload),
            {
                "source": "scheduler",
                "tile_0": 11,
                "tile_1": 7,
                "tile_2": 2,
                "tile_3": 5,
            },
        )

    def test_decodes_signed_loop_iteration(self):
        payload = (1234 << 48) | ((1 << 48) - 2)
        self.assertEqual(
            decode_loop_payload(payload),
            {
                "source": "loop_induction",
                "source_line": 1234,
                "iteration": -2,
            },
        )

    def test_decodes_range_into_semantic_trace(self):
        payload = (9 << 9) | (1 << 8) | 2
        trace = {
            "stringTable": ["auto.main.tma.data_wait"],
            "locationTable": [{"ctaId": [0, 0, 0], "warpId": 4}],
            "launches": [
                {
                    "kernelName": "kernel",
                    "gridDimX": 1,
                    "gridDimY": 1,
                    "gridDimZ": 1,
                    "blockDimX": 32,
                    "blockDimY": 1,
                    "blockDimZ": 1,
                    "ranges": [
                        {
                            "rangeNameIdx": 0,
                            "startTs": 10,
                            "endTs": 20,
                            "warpLocIdxs": [0, 0],
                            "internalEvents": [{"payloadVal": payload}],
                        }
                    ],
                    "markers": [],
                }
            ],
        }
        event = decode_trace(trace)["launches"][0]["events"][0]
        self.assertEqual(event["stage"], 2)
        self.assertEqual(event["phase"], 1)
        self.assertEqual(event["sequence"], 9)
        self.assertEqual(event["cta"], [0, 0, 0])
        self.assertEqual(event["warp"], 4)

    def test_decodes_scheduler_marker_location(self):
        payload = (5 << 48) | (3 << 32) | (1 << 16) | 9
        trace = {
            "stringTable": ["auto.scheduler.tile"],
            "locationTable": [{"ctaId": [1, 0, 0], "warpId": 7}],
            "launches": [
                {
                    "kernelName": "kernel",
                    "markers": [
                        {
                            "markerNameIdx": 0,
                            "timestamp": 42,
                            "locIdx": 0,
                            "payloadVal": payload,
                        }
                    ],
                }
            ],
        }

        event = decode_trace(trace)["launches"][0]["events"][0]
        self.assertEqual(
            (event["tile_0"], event["tile_1"], event["tile_2"], event["tile_3"]),
            (5, 3, 1, 9),
        )
        self.assertEqual(event["cta"], [1, 0, 0])
        self.assertEqual(event["warp"], 7)


if __name__ == "__main__":
    unittest.main()
