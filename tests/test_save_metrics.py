"""Checks the metrics record and the Part 3 schedule export."""

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.save_metrics import save_metrics
from src.run.part3 import closed_loop_actions, control_command, control_iterations


class SaveMetricsTest(unittest.TestCase):
    def test_appends_one_json_line(self):
        record = {
            "scene": "waymo-405841",
            "run": "colmap",
            "iteration": 7000,
            "psnr": 1.0,
            "ssim": 1.0,
            "lpips_vgg": 0.0,
            "pose_error": 0.1,
            "dropped_views": 0,
            "initialization_seconds": 1.0,
            "reconstruction_seconds": 2.0,
            "generation_seconds": 0.0,
            "joint_reconstruction_seconds": 0.0,
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "metrics.jsonl"
            save_metrics(path, record)
            loaded = json.loads(path.read_text().strip())
            self.assertEqual(loaded["run"], "colmap")

    def test_closed_loop_has_both_insertions(self):
        actions = closed_loop_actions()
        self.assertEqual(actions[1500], "insert_round_1")
        self.assertEqual(actions[3000], "record_round_1_and_insert_round_2")
        self.assertEqual(control_iterations(), [1500, 3000, 4500, 7000, 30000])

    def test_control_checkpoints_every_comparison_iteration(self):
        command = control_command(Path("sparse"), Path("model"), "python", Path("train.py"))
        text = " ".join(command)
        self.assertIn("--checkpoint_iterations 1500 3000 4500 7000 30000", text)
        self.assertNotIn("gen_", text)
        resumed = control_command(
            Path("sparse"),
            Path("model"),
            "python",
            Path("train.py"),
            Path("scene/chkpnt0.pth"),
            3,
        )
        self.assertEqual(resumed[-1], "scene/chkpnt0.pth")
        self.assertIn("--sh_degree", resumed)


if __name__ == "__main__":
    unittest.main()
