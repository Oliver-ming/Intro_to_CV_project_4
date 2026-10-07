"""Checks VGGT's sparse layout and depth unprojection."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.generate.colmap_view import read_camera_to_world
from src.init.export_colmap import depth_to_points, export_omega_scene, write_text_scene
from src.init.sparse_layout import nest_sparse_model


class SparseLayoutTest(unittest.TestCase):
    def test_bins_move_into_sparse_zero(self):
        with tempfile.TemporaryDirectory() as raw:
            scene = Path(raw)
            sparse = scene / "sparse"
            sparse.mkdir()
            (sparse / "cameras.bin").write_bytes(b"cam")
            nested = nest_sparse_model(scene)
            self.assertTrue((nested / "cameras.bin").is_file())
            self.assertFalse((sparse / "cameras.bin").exists())


class ExportColmapTest(unittest.TestCase):
    def test_center_pixel_unprojects_along_the_camera_axis(self):
        depth = np.zeros((3, 3))
        depth[1, 1] = 4
        intrinsics = np.array([[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1.0]])
        points = depth_to_points(depth, intrinsics, np.eye(4), stride=1)
        self.assertEqual(points.shape, (1, 3))
        self.assertTrue(np.allclose(points[0], [0, 0, 4]))

    def test_written_pose_reads_back(self):
        pose = np.eye(4)[None]
        pose[0, 0, 3] = 3
        with tempfile.TemporaryDirectory() as raw:
            scene = Path(raw)
            write_text_scene(
                scene,
                ["000000.jpg"],
                pose,
                np.eye(3),
                8,
                4,
                np.array([[1.0, 2.0, 3.0]]),
                np.array([[9, 8, 7]]),
            )
            recovered = read_camera_to_world(scene / "sparse" / "0" / "images.txt")
            points = (scene / "sparse" / "0" / "points3D.txt").read_text()
        self.assertAlmostEqual(float(recovered["000000.jpg"][0, 3]), 3)
        self.assertIn("9 8 7", points)


class FeedForwardTest(unittest.TestCase):
    def test_vggt_launch_nests_the_model(self):
        from src.init.vggt import launch_vggt

        with tempfile.TemporaryDirectory() as raw:
            scene = Path(raw)

            def run(command):
                self.assertNotIn("--use_ba", command)
                (scene / "sparse").mkdir()
                (scene / "sparse" / "cameras.bin").write_bytes(b"cam")

            nested = launch_vggt(scene, "python", Path("demo_colmap.py"), run)
            self.assertTrue((nested / "cameras.bin").is_file())

    def test_omega_uses_dataset_intrinsics_at_image_resolution(self):
        depth = np.zeros((1, 2, 2))
        depth[0, 0, 0] = 2
        with tempfile.TemporaryDirectory() as raw:
            scene = Path(raw)
            export_omega_scene(
                scene,
                ["000000.jpg"],
                depth,
                np.eye(4)[None],
                np.array([[4.0, 0, 2.0], [0, 4.0, 1.0], [0, 0, 1.0]]),
                4,
                2,
            )
            cameras = (scene / "sparse" / "0" / "cameras.txt").read_text()
        self.assertIn("PINHOLE 4 2 4.0 4.0 2.0 1.0", cameras)


if __name__ == "__main__":
    unittest.main()
