"""Checks that a generated view is appended as a frozen COLMAP camera."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.generate.colmap_view import append_generated_view, read_camera_to_world


class ColmapViewTest(unittest.TestCase):
    def test_identity_pose_is_written(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "images.txt"
            path.write_text("# header\n")
            append_generated_view(path, "gen_000.jpg", np.eye(4), camera_id=1, image_id=7)
            body = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
            self.assertEqual(len(body), 1)
            parts = body[0].split()
            self.assertEqual(parts[0], "7")
            self.assertEqual(parts[-1], "gen_000.jpg")
            self.assertAlmostEqual(float(parts[1]), 1.0, places=5)

    def test_written_pose_reads_back(self):
        pose = np.eye(4)
        pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        pose[:3, 3] = np.array([1.5, -2.0, 3.25])
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "images.txt"
            path.write_text("# header\n")
            append_generated_view(path, "gen_000.jpg", pose, camera_id=1, image_id=7)
            recovered = read_camera_to_world(path)["gen_000.jpg"]
        self.assertTrue(np.allclose(recovered, pose, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
