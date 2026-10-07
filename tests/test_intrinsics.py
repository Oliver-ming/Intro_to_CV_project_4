"""Checks normalized DL3DV intrinsics and already-pixel Waymo intrinsics."""

import json
import tempfile
import unittest
from pathlib import Path

from src.data.intrinsics import colmap_reader_args, focal_for_prepared, pixel_intrinsics
from src.run.launch_part1 import reader_args_for_prepared


class IntrinsicsTest(unittest.TestCase):
    def test_dl3dv_fractions_become_pixels(self):
        focal_x, focal_y, principal_x, principal_y = pixel_intrinsics(
            0.7972654457072755, 0.7967596537558975, 0.5, 0.5, 256, 256
        )
        self.assertAlmostEqual(focal_x, 204.09995410106253)
        self.assertAlmostEqual(principal_x, 128.0)

    def test_waymo_pixels_stay_pixels(self):
        focal_x, _, principal_x, _ = pixel_intrinsics(2066.697564417299, 2066.697564417299, 950.5512774150723, 641.1870541472169, 1920, 1280)
        self.assertAlmostEqual(focal_x, 2066.697564417299)
        self.assertAlmostEqual(principal_x, 950.5512774150723)

    def test_waymo_focal_follows_the_width_cap(self):
        focal = focal_for_prepared("waymo-405841", 1600, 1067)
        self.assertAlmostEqual(focal, 2066.697564417299 * 1600 / 1920)

    def test_dl3dv_focal_uses_the_prepared_width(self):
        focal = focal_for_prepared(
            "dl3dv-2",
            256,
            256,
            (0.7972654457072755, 0.7967596537558975, 0.5, 0.5),
        )
        self.assertAlmostEqual(focal, 204.09995410106253)

    def test_waymo_colmap_keeps_distortion_and_scales_the_focal(self):
        args = colmap_reader_args("waymo-405841", 1600, 1067)
        params = args[args.index("--ImageReader.camera_params") + 1]
        self.assertEqual(args[args.index("--ImageReader.camera_model") + 1], "OPENCV")
        self.assertTrue(params.startswith(str(2066.697564417299 * 1600 / 1920)))
        self.assertIn("-0.35323300768499216", params)
        self.assertEqual(params.count(","), 7)

    def test_dl3dv_prepared_camera_is_pinhole(self):
        with tempfile.TemporaryDirectory() as raw:
            prepared = Path(raw)
            (prepared / "split.json").write_text(json.dumps({"width": 256, "height": 256}))
            (prepared / "cameras.json").write_text(json.dumps([
                {"fx": 0.7972654457072755, "fy": 0.7967596537558975, "cx": 0.5, "cy": 0.5}
            ]))
            args = reader_args_for_prepared("dl3dv-2", prepared)
        self.assertEqual(args[args.index("--ImageReader.camera_model") + 1], "PINHOLE")
        params = args[args.index("--ImageReader.camera_params") + 1]
        self.assertTrue(params.startswith("204.09995"))
        self.assertTrue(params.endswith("128.0,128.0"))


if __name__ == "__main__":
    unittest.main()
