from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from ncu_iket_profile.collect_ncu import build_command, target_command
from ncu_iket_profile.merge import field_name


class NcuJointCoreTest(unittest.TestCase):
    def test_field_name_is_csv_safe(self):
        self.assertEqual(field_name("FA4.MMA/QK Issue"), "fa4_mma_qk_issue")

    def test_target_command_removes_separator(self):
        self.assertEqual(target_command(["--", "python", "x.py"]), ["python", "x.py"])

    def test_ncu_command_keeps_target_as_argv(self):
        args = argparse.Namespace(
            output=Path("out/report"),
            kernel_regex=".*Kernel.*",
            interval_ns=5000,
            section="PmSampling",
            max_passes=0,
            replay_mode="kernel",
            cache_control="all",
            disable_pm_warp_sampling=True,
            command=["--", "python", "demo.py", "--size", "4"],
        )
        command = build_command(args)
        self.assertEqual(command[-5:], ["--", "python", "demo.py", "--size", "4"])
        self.assertIn("5000", command)


if __name__ == "__main__":
    unittest.main()
