"""Checks which COLMAP camera model 3DGS is allowed to train on."""

import unittest

from src.init.colmap import camera_model_name


class CameraModelTest(unittest.TestCase):
    def test_reads_the_first_camera(self):
        text = "# header\n1 OPENCV 1600 1067 1 1 0 0 0 0 0 0\n"
        self.assertEqual(camera_model_name(text), "OPENCV")
        self.assertEqual(camera_model_name("1 PINHOLE 256 256 200 200 128 128\n"), "PINHOLE")


if __name__ == "__main__":
    unittest.main()
