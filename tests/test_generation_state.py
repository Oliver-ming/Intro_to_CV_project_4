"""Checks that round 2 pairs the view written by round 1."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.generate.state import GenerationState


class GenerationStateTest(unittest.TestCase):
    def test_second_round_uses_the_inserted_view(self):
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
            state(1)
            state(2)
            names = sorted(path.name for path in (scene / "images").iterdir())
        self.assertIsNone(state.failure)
        self.assertIn(5, state.poses)
        self.assertCountEqual(state.indices, [0, 2, 5, 7, 10])
        self.assertEqual(len(names), 3)


if __name__ == "__main__":
    unittest.main()
