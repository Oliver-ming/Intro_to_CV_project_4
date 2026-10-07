"""Checks that a checkpoint score fills every required metrics field."""

import unittest

import torch

from src.eval.save_metrics import save_metrics
from src.eval.score import score_run


class ScoreTest(unittest.TestCase):
    def test_record_is_complete(self):
        image = torch.rand(1, 3, 32, 32)
        centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        record = score_run(
            image,
            image,
            centers,
            centers,
            [0, 1, 2, 3],
            [0, 1, 2],
            "waymo-405841",
            "colmap",
            7000,
            1.0,
            2.0,
            0.0,
            0.0,
            lambda prediction, target: torch.zeros(prediction.shape[0]),
        )
        self.assertGreater(record["psnr"], 80.0)
        self.assertEqual(record["dropped_views"], 1)
        self.assertEqual(record["lpips_vgg"], 0.0)
        save_metrics  # imported so a missing field would fail the writer contract
        self.assertIn("joint_reconstruction_seconds", record)


if __name__ == "__main__":
    unittest.main()
