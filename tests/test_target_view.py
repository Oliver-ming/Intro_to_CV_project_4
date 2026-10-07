"""Checks midpoint cameras between two estimated poses."""

import unittest

import torch

from src.generate.target_view import midpoint_pose


class TargetViewTest(unittest.TestCase):
    def test_translation_midpoint_keeps_rotation(self):
        first = torch.eye(4)
        second = torch.eye(4)
        second[0, 3] = 2.0
        second[1, 3] = 4.0
        midpoint = midpoint_pose(first, second)
        self.assertTrue(torch.allclose(midpoint[:3, :3], torch.eye(3), atol=1e-5))
        self.assertTrue(torch.allclose(midpoint[:3, 3], torch.tensor([1.0, 2.0, 0.0]), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
