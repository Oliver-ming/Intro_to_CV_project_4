"""Checks the 3DGS render camera and i3dgs pose export."""

import unittest

import numpy as np

from src.reconstruct.gaussian_camera import make_minicam
from src.run.i3dgs_export import registered_camera_to_world


class MiniCamTest(unittest.TestCase):
    def test_center_follows_the_camera_translation(self):
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = np.array([1.0, 2.0, 3.0])
        camera = make_minicam(camera_to_world, 8, 4, 4.0, 4.0)
        center = camera.camera_center.detach().cpu().numpy()
        self.assertTrue(np.allclose(center, [1.0, 2.0, 3.0], atol=1e-5))


class I3dgsExportTest(unittest.TestCase):
    def test_world_to_camera_is_inverted_and_tests_are_dropped(self):
        world_to_camera = np.eye(4)
        world_to_camera[0, 3] = -5
        metadata = {
            "keyframes": [
                {"info": {"name": "000000.jpg", "is_test": False}, "Rt": world_to_camera.tolist()},
                {"info": {"name": "000005.jpg", "is_test": True}, "Rt": np.eye(4).tolist()},
            ]
        }
        poses = registered_camera_to_world(metadata)
        self.assertEqual(list(poses), ["000000.jpg"])
        self.assertAlmostEqual(float(poses["000000.jpg"][0, 3]), 5.0)


if __name__ == "__main__":
    unittest.main()
