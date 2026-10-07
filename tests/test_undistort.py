"""Checks Waymo frames can be resampled onto a PINHOLE camera."""

import unittest

import numpy as np

from src.data.intrinsics import intrinsics_for_prepared
from src.data.undistort import read_pinhole, undistort_waymo


class UndistortTest(unittest.TestCase):
    def test_reads_pinhole_intrinsics(self):
        text = "1 PINHOLE 8 4 4 4 4 2\n"
        self.assertEqual(read_pinhole(text), (8, 4, 4.0, 4.0, 4.0, 2.0))

    def test_same_camera_keeps_a_flat_image(self):
        image = np.full((4, 8, 3), 200, dtype=np.uint8)
        focal_x, focal_y, principal_x, principal_y = intrinsics_for_prepared("waymo-405841", 8, 4)
        cameras = f"1 PINHOLE 8 4 {focal_x} {focal_y} {principal_x} {principal_y}\n"
        output = undistort_waymo(image, 8, 4, cameras)
        self.assertEqual(output.shape, (4, 8, 3))
        self.assertGreater(int(output.mean()), 150)


if __name__ == "__main__":
    unittest.main()
