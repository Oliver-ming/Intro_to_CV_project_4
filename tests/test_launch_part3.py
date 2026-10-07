"""Checks loading COLMAP training views and running both generation rounds."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.generate.colmap_view import append_generated_view
from src.generate.state import GenerationState
from src.run.launch_part3 import launch_part3
from src.run.scene_views import load_training_views, read_intrinsics


class SceneViewsTest(unittest.TestCase):
    def test_intrinsics_and_training_views(self):
        with tempfile.TemporaryDirectory() as raw:
            scene = Path(raw)
            sparse = scene / "sparse" / "0"
            images = scene / "images"
            sparse.mkdir(parents=True)
            images.mkdir()
            (sparse / "cameras.txt").write_text("# header\n1 PINHOLE 8 4 10 12 4 2\n")
            (sparse / "images.txt").write_text("# header\n")
            append_generated_view(sparse / "images.txt", "000000.jpg", np.eye(4), 1, 1)
            shifted = np.eye(4)
            shifted[0, 3] = 4
            append_generated_view(sparse / "images.txt", "000010.jpg", shifted, 1, 2)
            append_generated_view(sparse / "images.txt", "gen_r1_0000.jpg", np.eye(4), 1, 3)
            Image.new("RGB", (8, 4), (1, 2, 3)).save(images / "000000.jpg")
            Image.new("RGB", (8, 4), (4, 5, 6)).save(images / "000010.jpg")
            intrinsics = read_intrinsics(sparse / "cameras.txt")
            indices, poses, loaded = load_training_views(scene)
        self.assertAlmostEqual(intrinsics[0, 0], 10)
        self.assertAlmostEqual(intrinsics[1, 1], 12)
        self.assertCountEqual(indices, [0, 10])
        self.assertAlmostEqual(float(poses[10][0, 3]), 4)
        self.assertEqual(loaded[0].size, (8, 4))


class LaunchPart3Test(unittest.TestCase):
    def test_two_rounds_between_three_segments(self):
        left = torch.eye(4)
        right = torch.eye(4)
        right[0, 3] = 10
        with tempfile.TemporaryDirectory() as raw:
            scene = Path(raw)
            sparse = scene / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "images.txt").write_text("# header\n")
            state = GenerationState(
                [0, 10],
                {0: left, 10: right},
                {0: Image.new("RGB", (4, 4)), 10: Image.new("RGB", (4, 4))},
                np.eye(3),
                scene,
                1,
                3,
                lambda _pose: Image.new("RGB", (4, 4)),
                lambda _index: torch.ones(1, 2),
                lambda _image: torch.ones(1, 2),
                lambda *_args: Image.new("RGB", (4, 4)),
            )
            calls = []
            result = launch_part3(scene, Path("model"), "python", state, lambda command: calls.append(command))
            names = sorted(path.name for path in (scene / "images").iterdir())
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(names), 3)
        self.assertIsNone(result["failure"])
        self.assertGreater(result["generation_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
