"""Checks for PSNR and Sim(3) pose error."""

import unittest

import torch

from src.eval.pose_error import pose_error
from src.eval.rendering import psnr


class MetricTest(unittest.TestCase):
    def test_identical_images_have_high_psnr(self):
        image = torch.rand(2, 3, 32, 32)
        scores = psnr(image, image)
        self.assertGreater(float(scores.min()), 80.0)

    def test_sim3_removes_a_known_transform(self):
        ground_truth = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.0, 0.3], [1.0, 1.0, 0.5]],
        )
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        estimated = (ground_truth - torch.tensor([0.4, -0.2, 0.7])) @ rotation / 2.5
        self.assertLess(pose_error(estimated, ground_truth), 1e-4)


if __name__ == "__main__":
    unittest.main()
