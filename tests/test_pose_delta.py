"""Checks the evaluation-pose update and the 3DGS camera center."""

import unittest

import numpy as np
import torch

from src.eval.evaluation_pose import align_evaluation_pose
from src.eval.pose_delta import apply_pose_delta
from src.reconstruct.gaussian_camera import camera_center


class PoseDeltaTest(unittest.TestCase):
    def test_zero_delta_keeps_the_pose(self):
        pose = torch.eye(4)
        pose[0, 3] = 2
        updated = apply_pose_delta(pose, torch.zeros(6))
        self.assertTrue(torch.allclose(updated, pose, atol=1e-6))

    def test_photometric_steps_move_the_camera_toward_the_image(self):
        pose = torch.eye(4)

        def render(delta):
            moved = apply_pose_delta(pose, delta)
            return moved[0, 3].expand(1, 3, 2, 2)

        target = torch.ones(1, 3, 2, 2)
        delta = align_evaluation_pose(target, target, render, steps=40)
        rendered = render(delta)
        self.assertLess(torch.mean((rendered - target) ** 2).item(), 1.0)

    def test_detached_render_uses_a_finite_difference(self):
        def render(delta):
            return (1 - delta.detach()[3]).expand(1, 3, 2, 2).clone()

        target = torch.zeros(1, 3, 2, 2)
        delta = align_evaluation_pose(target, target, render, steps=30)
        self.assertGreater(float(delta[3]), 0.05)


class CameraCenterTest(unittest.TestCase):
    def test_center_matches_the_translation(self):
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = np.array([1.0, -2.0, 3.5])
        center = camera_center(camera_to_world)
        self.assertTrue(np.allclose(center.numpy(), [1.0, -2.0, 3.5], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
