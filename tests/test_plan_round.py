"""Checks that a round places one target between each adjacent pair."""

import unittest

import torch

from src.generate.plan_round import plan_round


class PlanRoundTest(unittest.TestCase):
    def test_one_target_between_three_views(self):
        poses = {}
        for index, x in ((0, 0.0), (10, 2.0), (20, 6.0)):
            pose = torch.eye(4)
            pose[0, 3] = x
            poses[index] = pose
        planned = plan_round([20, 0, 10], poses)
        self.assertEqual([(item[0], item[1]) for item in planned], [(0, 10), (10, 20)])
        self.assertAlmostEqual(float(planned[0][2][0, 3]), 1.0)
        self.assertEqual(plan_round([3], poses), [])


if __name__ == "__main__":
    unittest.main()
