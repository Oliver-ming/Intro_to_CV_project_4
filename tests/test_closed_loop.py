"""Checks the resumed 3DGS segments for the two generation rounds."""

import unittest
from pathlib import Path

from src.run.closed_loop import closed_loop_segments, segment_command


class ClosedLoopCommandTest(unittest.TestCase):
    def test_second_segment_resumes_from_the_first_checkpoint(self):
        segments = closed_loop_segments()
        self.assertEqual([item["after"] for item in segments], [
            "insert_round_1",
            "insert_round_2",
            "finish",
        ])
        command = segment_command(
            Path("source"),
            Path("model"),
            "python",
            Path("train.py"),
            segments[1]["iterations"],
            segments[1]["checkpoint_iterations"],
            Path("model/chkpnt1500.pth"),
        )
        self.assertIn("--start_checkpoint", command)
        self.assertIn("3000", command)
        self.assertIn("--disable_viewer", command)


if __name__ == "__main__":
    unittest.main()
