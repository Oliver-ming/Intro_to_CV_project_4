"""Checks that a generated view lands in both the image folder and COLMAP."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.generate.write_back import write_generated_view


class WriteBackTest(unittest.TestCase):
    def test_image_and_pose_are_both_written(self):
        with tempfile.TemporaryDirectory() as raw:
            scene = Path(raw)
            sparse = scene / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "images.txt").write_text("# header\n")
            image = Image.new("RGB", (8, 8), (9, 8, 7))
            path = write_generated_view(scene, image, "gen_000.jpg", np.eye(4), 1, 4)
            self.assertTrue(path.is_file())
            body = (sparse / "images.txt").read_text()
            self.assertIn("gen_000.jpg", body)


if __name__ == "__main__":
    unittest.main()
