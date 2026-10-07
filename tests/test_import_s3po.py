"""Checks that S3PO scene folders become time-ordered captured frames."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.import_s3po import import_re10k, import_scene


class ImportTest(unittest.TestCase):
    def test_dl3dv_keeps_the_first_300_names_in_order(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rgb = root / "dl3dv" / "2" / "rgb"
            rgb.mkdir(parents=True)
            for index in (2, 0, 1):
                Image.new("RGB", (8, 8)).save(rgb / f"{index:04d}.png")
            Image.new("RGB", (8, 8)).save(rgb / "0001-checkpoint.png")
            (rgb.parent / "gt").mkdir()
            (rgb.parent / "gt" / "000.txt").write_text("pose\n")
            destination = root / "out"
            written = import_scene(root, "dl3dv-2", destination)
            self.assertEqual([path.name for path in written], ["000000.png", "000001.png", "000002.png"])
            self.assertTrue((destination / "000000.png").is_file())
            self.assertEqual((destination.parent / "gt" / "000.txt").read_text(), "pose\n")


class LinkSplitTest(unittest.TestCase):
    def test_dense_excludes_test_frames(self):
        from src.data.link_split import link_split
        from src.data.prepare import prepare_scene

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            for index in range(10):
                Image.new("RGB", (16, 16)).save(source / f"{index:02d}.png")
            prepared = root / "prepared"
            prepare_scene("waymo-405841", source, prepared)
            linked = link_split(prepared, "dense", root / "dense")
            names = {path.name for path in linked}
            self.assertNotIn("000005.jpg", names)
            self.assertIn("000000.jpg", names)

    def test_re10k_skips_the_video_and_keeps_camera_metadata(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            images = root / "Re10k-1" / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(images / "00001.png")
            Image.new("RGB", (8, 8)).save(images / "00000.png")
            (images / "video.mp4").write_bytes(b"not a frame")
            (root / "Re10k-1" / "cameras.json").write_text("[]")
            written = import_re10k(root, root / "raw")
            self.assertEqual([path.name for path in written], ["000000.png", "000001.png"])
            self.assertTrue((root / "cameras.json").is_file())
            self.assertFalse((root / "raw" / "cameras.json").exists())


if __name__ == "__main__":
    unittest.main()
