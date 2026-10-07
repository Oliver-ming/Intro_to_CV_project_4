"""Checks that generation rounds sit between the resumed 3DGS segments."""

import unittest
from pathlib import Path

from src.run.execute_closed_loop import run_closed_loop


class ExecuteClosedLoopTest(unittest.TestCase):
    def test_inserts_between_segments(self):
        calls = []
        run_closed_loop(
            Path("source"),
            Path("model"),
            "python",
            Path("train.py"),
            lambda command: calls.append(("run", command)),
            lambda round_index: calls.append(("insert", round_index)),
        )
        kinds = [item[0] for item in calls]
        self.assertEqual(kinds, ["run", "insert", "run", "insert", "run"])
        self.assertEqual(calls[1][1], 1)
        self.assertEqual(calls[3][1], 2)
        self.assertIn("chkpnt1500.pth", calls[2][1][-1])
        self.assertIn("chkpnt3000.pth", calls[4][1][-1])
        self.assertNotIn("--start_checkpoint", calls[0][1])

    def test_first_segment_loads_the_i3dgs_export(self):
        calls = []
        run_closed_loop(
            Path("source"),
            Path("model"),
            "python",
            Path("train.py"),
            lambda command: calls.append(command),
            lambda _round_index: None,
            Path("scene/chkpnt0.pth"),
        )
        self.assertIn("--start_checkpoint", calls[0])
        self.assertEqual(calls[0][-1], "scene/chkpnt0.pth")


if __name__ == "__main__":
    unittest.main()
