"""Checks that our camera matrix matches official 3DGS."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "gaussian-splatting"))
from utils.graphics_utils import getWorld2View2  # noqa: E402

from src.reconstruct.gaussian_camera import gaussian_rotation_translation, world_view_transform


class GaussianCameraTest(unittest.TestCase):
    def test_matches_official_world_to_view(self):
        camera_to_world = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [-1.0, 0.0, 0.0, 2.0],
                [0.0, -1.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        rotation, translation = gaussian_rotation_translation(camera_to_world)
        official = getWorld2View2(rotation, translation)
        ours = world_view_transform(camera_to_world).numpy().T
        self.assertTrue(np.allclose(ours, official, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
