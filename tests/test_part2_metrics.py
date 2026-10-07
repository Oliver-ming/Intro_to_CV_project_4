"""Checks i3dgs pose summaries ignore unregistered training views."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.eval.part2_metrics import part2_pose_summary
from src.eval.score_part2 import joint_reconstruction_seconds


class Part2MetricsTest(unittest.TestCase):
    def test_dropped_view_is_excluded_from_the_error(self):
        def pose(x):
            matrix = np.eye(4)
            matrix[0, 3] = -x
            return matrix.tolist()

        metadata = {
            "keyframes": [
                {"info": {"name": "000000.jpg", "is_test": False}, "Rt": pose(0.0)},
                {"info": {"name": "000001.jpg", "is_test": False}, "Rt": pose(1.0)},
                {"info": {"name": "000002.jpg", "is_test": False}, "Rt": pose(2.0)},
            ]
        }
        centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        summary = part2_pose_summary(metadata, ["000000.jpg", "000001.jpg", "000002.jpg", "000003.jpg"], [0, 1, 2, 3], centers)
        self.assertEqual(summary["dropped_views"], 1)
        self.assertEqual(summary["registered_views"], 3)
        self.assertAlmostEqual(summary["pose_error"], 0.0, places=5)

    def test_joint_clock_is_read_by_split_name(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "pose_metrics.jsonl"
            path.write_text(
                json.dumps({"run": "i3dgs-sparse", "joint_reconstruction_seconds": 3.5}) + "\n"
            )
            self.assertEqual(joint_reconstruction_seconds(path, "sparse"), 3.5)


if __name__ == "__main__":
    unittest.main()
