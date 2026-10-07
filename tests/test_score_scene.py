"""Checks Part 1 metric lines from registered COLMAP poses."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.eval.score_scene import nearest_training_pose, write_checkpoint_metrics
from src.generate.colmap_view import append_generated_view


class ScoreSceneTest(unittest.TestCase):
    def test_nearest_frame_is_the_closer_index(self):
        poses = {0: torch.eye(4), 10: torch.eye(4)}
        chosen = nearest_training_pose(5, [0, 10], poses)
        self.assertEqual(float(chosen[0, 3]), 0.0)
        poses[10][0, 3] = 4
        chosen = nearest_training_pose(8, [0, 10], poses)
        self.assertAlmostEqual(float(chosen[0, 3]), 4.0)

    def test_metrics_line_uses_registered_views_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prepared = root / "prepared"
            images = prepared / "images"
            images.mkdir(parents=True)
            frames = [f"{index:06d}.jpg" for index in range(4)]
            for name in frames:
                Image.new("RGB", (8, 4), (255, 255, 255)).save(images / name)
            (prepared / "split.json").write_text(json.dumps({
                "frames": frames,
                "dense": [0, 1, 2],
                "test": [3],
                "width": 8,
                "height": 4,
            }))
            sparse = root / "source" / "sparse" / "0"
            sparse.mkdir(parents=True)
            (sparse / "images.txt").write_text("# header\n")
            for image_id, index, shift in ((1, 0, 0.0), (2, 1, 1.0), (3, 2, 2.0)):
                pose = np.eye(4)
                pose[0, 3] = shift
                append_generated_view(sparse / "images.txt", frames[index], pose, 1, image_id)
            centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
            image = torch.ones(1, 3, 4, 8)
            record = write_checkpoint_metrics(
                "waymo-405841",
                prepared,
                root / "source",
                root / "model",
                root / "metrics.jsonl",
                centers,
                7000,
                1.5,
                9.0,
                steps=1,
                render_image=lambda pose: image + pose[0, 3] * 0,
                lpips_fn=lambda prediction, target: torch.zeros(prediction.shape[0]),
            )
        self.assertEqual(record["dropped_views"], 0)
        self.assertEqual(record["iteration"], 7000)
        self.assertGreater(record["psnr"], 40)
        self.assertAlmostEqual(record["initialization_seconds"], 1.5)


if __name__ == "__main__":
    unittest.main()
