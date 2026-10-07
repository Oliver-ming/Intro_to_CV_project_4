"""Checks one generation round writes a scaled-depth restoration."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.generate.round import run_generation_round


class GenerationRoundTest(unittest.TestCase):
    def test_one_pair_writes_one_view_between_them(self):
        left = torch.eye(4)
        right = torch.eye(4)
        right[0, 3] = 10
        seen = {}

        def render_color(pose):
            seen["pose_x"] = float(pose[0, 3])
            return Image.new("RGB", (4, 4), (1, 2, 3))

        def render_depth(_index):
            return torch.tensor([[2.0, 4.0]])

        def estimate_depth(_image):
            return torch.tensor([[1.0, 2.0]])

        def restore(_target, _reference, depth, *_rest):
            seen["depth"] = depth.copy()
            return Image.new("RGB", (4, 4), (9, 9, 9))

        with tempfile.TemporaryDirectory() as raw:
            scene = Path(raw)
            sparse = scene / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "images.txt").write_text("# header\n")
            result = run_generation_round(
                1,
                [0, 10],
                {0: left, 10: right},
                {0: Image.new("RGB", (4, 4)), 10: Image.new("RGB", (4, 4))},
                np.eye(3),
                scene,
                1,
                3,
                render_color,
                render_depth,
                estimate_depth,
                restore,
            )
        self.assertEqual(result["written"], ["gen_r1_0000.jpg"])
        self.assertIn(5, result["indices"])
        self.assertAlmostEqual(seen["pose_x"], 5.0)
        self.assertTrue(np.allclose(seen["depth"], [[2.0, 4.0]]))
        self.assertGreater(result["generation_seconds"], 0.0)

    def test_depth_anything_map_is_resized_before_scaling(self):
        left = torch.eye(4)
        right = torch.eye(4)
        right[0, 3] = 10
        seen = {}

        def restore(_target, _reference, depth, *_rest):
            seen["depth"] = depth.copy()
            return Image.new("RGB", (4, 4))

        with tempfile.TemporaryDirectory() as raw:
            scene = Path(raw)
            sparse = scene / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "images.txt").write_text("# header\n")
            run_generation_round(
                1,
                [0, 10],
                {0: left, 10: right},
                {0: Image.new("RGB", (4, 4)), 10: Image.new("RGB", (4, 4))},
                np.eye(3),
                scene,
                1,
                3,
                lambda _pose: Image.new("RGB", (4, 4)),
                lambda _index: torch.tensor([[2.0, 4.0]]),
                lambda _image: torch.ones(2, 2),
                restore,
            )
        self.assertEqual(seen["depth"].shape, (1, 2))
        self.assertTrue(np.allclose(seen["depth"], [[2.0, 2.0]]))

    def test_one_registered_view_does_not_write(self):
        with tempfile.TemporaryDirectory() as raw:
            result = run_generation_round(
                1,
                [0],
                {0: torch.eye(4)},
                {0: Image.new("RGB", (2, 2))},
                np.eye(3),
                Path(raw),
                1,
                1,
                lambda _pose: Image.new("RGB", (2, 2)),
                lambda _index: torch.ones(1, 1),
                lambda _image: torch.ones(1, 1),
                lambda *_args: Image.new("RGB", (2, 2)),
            )
        self.assertEqual(result["written"], [])
        self.assertIn("failure", result)


if __name__ == "__main__":
    unittest.main()
