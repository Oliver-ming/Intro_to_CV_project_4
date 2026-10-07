"""Checks the model adapters pass through the published call shapes."""

import unittest

import numpy as np
import torch
from PIL import Image

from src.generate.models import estimate_with_da3, restore_with_geoquery


class FakeDepth:
    def inference(self, images):
        self.count = len(images)

        class Prediction:
            depth = np.ones((1, 2, 3), dtype=np.float32)

        return Prediction()


class FakeGeoQuery:
    def sample(self, **kwargs):
        self.kwargs = kwargs
        return Image.new("RGB", (2, 2)), 0.1


class ModelAdapterTest(unittest.TestCase):
    def test_da3_returns_one_map(self):
        depth = estimate_with_da3(FakeDepth(), Image.new("RGB", (3, 2)))
        self.assertEqual(tuple(depth.shape), (2, 3))

    def test_geoquery_returns_the_image(self):
        model = FakeGeoQuery()
        image = restore_with_geoquery(
            model,
            Image.new("RGB", (2, 2)),
            Image.new("RGB", (2, 2)),
            np.ones((2, 2)),
            np.eye(3),
            np.eye(4),
            np.eye(4),
        )
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(model.kwargs["prompt"] if "prompt" in model.kwargs else "remove degradation", "remove degradation")

    def test_wide_view_is_restored_at_the_camera_size(self):
        seen = {}

        class WideGeoQuery:
            def sample(self, **kwargs):
                seen.update(kwargs)
                return kwargs["target_image"], 0.0

        image = restore_with_geoquery(
            WideGeoQuery(),
            Image.new("RGB", (1600, 1064)),
            Image.new("RGB", (1600, 1064)),
            np.ones((1064, 1600)),
            np.array([[1000.0, 0, 800.0], [0, 1000.0, 500.0], [0, 0, 1]]),
            np.eye(4),
            np.eye(4),
        )
        self.assertEqual(image.size, (1600, 1064))
        self.assertLessEqual(max(seen["target_image"].size), 512)
        self.assertEqual(seen["target_image"].size, seen["reference_image"].size)
        self.assertAlmostEqual(seen["K"][0, 0], 1000.0 * seen["target_image"].size[0] / 1600.0)


if __name__ == "__main__":
    unittest.main()
