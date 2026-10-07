"""Checks reading of S3PO ground-truth camera centers."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.eval.ground_truth import dl3dv_camera_centers, waymo_camera_centers


class GroundTruthTest(unittest.TestCase):
    def test_waymo_and_dl3dv_centers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            matrix = np.eye(4)
            matrix[:3, 3] = [1.5, 2.5, 3.5]
            np.savetxt(root / "000.txt", matrix)
            centers = waymo_camera_centers(root)
            self.assertTrue(torch.allclose(centers[0], torch.tensor([1.5, 2.5, 3.5])))
            cameras = root / "cameras.json"
            cameras.write_text(json.dumps([
                {"image_name": "frame_00002.png", "cam_trans": [7.0, 8.0, 9.0]},
                {"image_name": "frame_00001.png", "cam_trans": [4.0, 5.0, 6.0]},
            ]))
            loaded = dl3dv_camera_centers(cameras, end=1)
            self.assertTrue(torch.allclose(loaded[0], torch.tensor([4.0, 5.0, 6.0])))


if __name__ == "__main__":
    unittest.main()
