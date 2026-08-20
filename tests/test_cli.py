# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import sys
import tempfile
import unittest

from iket_cutedsl.bootstrap import parse_detailed_cta, run_python_target
from iket_cutedsl.cli import build_profile_command, get_parser


class CliTest(unittest.TestCase):
    def test_cta_parser(self):
        self.assertEqual(parse_detailed_cta("2,1,0"), (2, 1, 0))
        self.assertIsNone(parse_detailed_cta("all"))
        with self.assertRaisesRegex(Exception, "non-negative"):
            parse_detailed_cta("0,-1,0")

    def test_profile_command_wraps_target_in_bootstrap(self):
        args = get_parser().parse_args(
            [
                "profile",
                "-o",
                "trace",
                "--clobber",
                "--detailed-cta",
                "3,2,1",
                "--",
                "python",
                "kernel.py",
                "--shape",
                "128",
            ]
        )
        command = build_profile_command(args)
        self.assertEqual(command[:3], [sys.executable, "-m", "iket.cli.main"])
        self.assertIn("iket_cutedsl.bootstrap", command)
        self.assertIn("3,2,1", command)
        self.assertEqual(command[-4:], ["python", "kernel.py", "--shape", "128"])

    def test_python_script_runs_in_process(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "result.txt"
            script = Path(directory) / "target.py"
            script.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "Path(sys.argv[1]).write_text(sys.argv[2])\n"
            )
            run_python_target(["python", str(script), str(marker), "profiled"])
            self.assertEqual(marker.read_text(), "profiled")


if __name__ == "__main__":
    unittest.main()
