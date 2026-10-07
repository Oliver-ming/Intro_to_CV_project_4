"""Checks test-view scoring from the nearest training pose."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.eval.evaluate_run import initial_test_poses, record_test_views


class EvaluateRunTest(unittest.TestCase):
    def test_nearest_pose_and_saved_record(self):
        centers = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        poses = [torch.eye(4), torch.eye(4), torch.eye(4)]
        poses[1][0, 3] = 10
        poses[2][0, 3] = 20
        chosen = initial_test_poses({5: torch.tensor([9.0, 0.0, 0.0])}, centers, poses)
        self.assertAlmostEqual(float(chosen[5][0, 3]), 10.0)

        image = torch.ones(1, 3, 8, 8)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "metrics.jsonl"
            record = record_test_views(
                path,
                {5: image},
                chosen,
                lambda pose: image + pose[0, 3] * 0,
                centers,
                centers,
                [0, 10, 20],
                [0, 10, 20],
                "waymo-405841",
                "colmap",
                7000,
                1.0,
                2.0,
                0.0,
                0.0,
                lambda prediction, target: torch.zeros(prediction.shape[0]),
                steps=1,
            )
            body = path.read_text()
        self.assertGreater(record["psnr"], 40)
        self.assertEqual(record["dropped_views"], 0)
        self.assertIn("waymo-405841", body)


if __name__ == "__main__":
    unittest.main()
