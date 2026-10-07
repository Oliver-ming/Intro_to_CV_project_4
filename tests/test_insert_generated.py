"""Checks that a generation round renders the checkpoint from that iteration."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.generate.state import GenerationState
from src.reconstruct.export_gaussians import write_gaussian_checkpoint
from src.run.insert_generated import checkpoint_insert


class CheckpointInsertTest(unittest.TestCase):
    def test_round_one_writes_a_view_from_the_checkpoint(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        left = torch.eye(4)
        right = torch.eye(4)
        right[0, 3] = 10
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scene = root / "scene"
            sparse = scene / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "images.txt").write_text("# header\n")
            model_dir = root / "model"
            write_gaussian_checkpoint(
                model_dir / "chkpnt1500.pth",
                torch.tensor([[0.0, 0.0, 4.0]]),
                torch.ones(1, 1, 3),
                torch.zeros(1, 0, 3),
                torch.zeros(1, 3),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[5.0]]),
                0,
            )
            state = GenerationState(
                [0, 10],
                {0: left, 10: right},
                {0: Image.new("RGB", (32, 16)), 10: Image.new("RGB", (32, 16))},
                np.array([[16.0, 0, 16.0], [0, 16.0, 8.0], [0, 0, 1.0]]),
                scene,
                1,
                3,
                lambda _pose: Image.new("RGB", (32, 16)),
                lambda _index: torch.ones(16, 32),
                lambda _image: torch.ones(16, 32),
                lambda *_args: Image.new("RGB", (32, 16)),
            )
            checkpoint_insert(model_dir, state, 32, 16, 16.0, 16.0, 0)(1)
            written = list((scene / "images").iterdir())
        self.assertIsNone(state.failure)
        self.assertEqual(len(written), 1)
        self.assertIn(5, state.indices)


if __name__ == "__main__":
    unittest.main()
