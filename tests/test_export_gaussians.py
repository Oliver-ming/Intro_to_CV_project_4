"""Checks that exported Gaussians survive a 3DGS checkpoint round trip."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.reconstruct.export_gaussians import write_gaussian_checkpoint
from src.run.i3dgs_export import dropped_training_names


class ExportGaussiansTest(unittest.TestCase):
    def test_checkpoint_keeps_positions(self):
        xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "chkpnt0.pth"
            write_gaussian_checkpoint(
                path,
                xyz,
                torch.zeros(2, 1, 3),
                torch.zeros(2, 0, 3),
                torch.zeros(2, 3),
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
                torch.zeros(2, 1),
                sh_degree=0,
            )
            captured, iteration = torch.load(path, weights_only=False)
        self.assertEqual(iteration, 0)
        self.assertTrue(torch.equal(captured[1], xyz))

    def test_unregistered_training_names_are_dropped(self):
        metadata = {"keyframes": [{"info": {"name": "000000.jpg", "is_test": False}, "Rt": torch.eye(4).tolist()}]}
        dropped = dropped_training_names(["000000.jpg", "000010.jpg"], metadata)
        self.assertEqual(dropped, ["000010.jpg"])

    def test_large_coordinates_are_scaled_near_the_origin(self):
        from src.run.i3dgs_export import normalize_reconstruction

        poses = np.stack([np.eye(4), np.eye(4)])
        poses[1, 0, 3] = 1100.0
        xyz = torch.tensor([[1100.0, 0.0, 0.0]])
        scaling = torch.zeros(1, 3)
        normalized, points, log_scales, frame = normalize_reconstruction(poses, xyz, scaling)
        self.assertLess(abs(float(normalized[:, 0, 3].mean())), 1e-6)
        self.assertLess(float(np.abs(normalized[:, :3, 3]).max()), 1.01)
        self.assertAlmostEqual(float(points[0, 0]), float(normalized[1, 0, 3]), places=4)
        self.assertAlmostEqual(float(log_scales[0, 0]), -math.log(frame["radius"]), places=4)

    def test_far_gaussians_are_left_out(self):
        from src.run.i3dgs_export import keep_trainable_gaussians

        poses = np.stack([np.eye(4), np.eye(4)])
        poses[1, 0, 3] = 2.0
        xyz = torch.tensor([[1.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
        scaling = torch.zeros(2, 3)
        keep = keep_trainable_gaussians(poses, xyz, scaling)
        self.assertEqual(keep.tolist(), [True, False])

    def test_two_registered_views_write_a_checkpoint(self):
        from PIL import Image

        from src.run.i3dgs_export import export_part3_scene

        def pose(x):
            matrix = np.eye(4)
            matrix[0, 3] = x
            return np.linalg.inv(matrix).tolist()

        metadata = {
            "keyframes": [
                {"info": {"name": "000000.jpg", "is_test": False}, "Rt": pose(0)},
                {"info": {"name": "000010.jpg", "is_test": False}, "Rt": pose(4)},
            ]
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (4, 2)).save(images / "000000.jpg")
            Image.new("RGB", (4, 2)).save(images / "000010.jpg")
            result = export_part3_scene(
                root / "scene",
                metadata,
                ["000000.jpg", "000010.jpg", "000020.jpg"],
                images,
                np.eye(3),
                4,
                2,
                torch.zeros(2, 3),
                torch.zeros(2, 1, 3),
                torch.zeros(2, 0, 3),
                torch.zeros(2, 3),
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
                torch.zeros(2, 1),
                0,
            )
            self.assertIsNone(result["failure"])
            self.assertEqual(result["dropped_views"], 1)
            self.assertTrue(result["checkpoint"].is_file())
            self.assertTrue((root / "scene" / "images" / "000000.jpg").is_symlink())


if __name__ == "__main__":
    unittest.main()
