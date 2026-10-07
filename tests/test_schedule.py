"""Checks for depth scale alignment and the closed-loop iteration schedule."""

import unittest

import torch

from src.generate.depth_scale import align_reference_depth
from src.reconstruct.schedule import closed_loop_event


class DepthScaleTest(unittest.TestCase):
    def test_median_ratio_matches_rendered_depth(self):
        rendered = torch.tensor([1.0, 2.0, 4.0, 0.0])
        estimated = rendered / 2
        estimated[3] = 5.0
        aligned, ratio = align_reference_depth(rendered, estimated)
        self.assertAlmostEqual(float(ratio), 2.0)
        self.assertTrue(torch.allclose(aligned[:3], rendered[:3]))


class ScheduleTest(unittest.TestCase):
    def test_two_rounds_share_one_run(self):
        self.assertEqual(closed_loop_event(1500), "insert_round_1")
        self.assertEqual(closed_loop_event(3000), "record_round_1_and_insert_round_2")
        self.assertEqual(closed_loop_event(4500), "record_round_2")
        self.assertEqual(closed_loop_event(7000), "report")
        self.assertEqual(closed_loop_event(30000), "report")
        self.assertIsNone(closed_loop_event(1499))


if __name__ == "__main__":
    unittest.main()
