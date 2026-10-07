"""Checks that scene preparation resizes frames and writes the split."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.prepare import prepare_scene


class PrepareTest(unittest.TestCase):
    def test_resize_and_split(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "source"
            destination = Path(raw) / "prepared"
            source.mkdir()
            for name, size in (("000.png", (1800, 900)), ("001.png", (800, 400))):
                Image.new("RGB", size, (10, 20, 30)).save(source / name)
            manifest = prepare_scene("waymo-405841", source, destination)
            self.assertEqual(manifest["frames"], ["000000.jpg", "000001.jpg"])
            with Image.open(destination / "images" / "000000.jpg") as image:
                self.assertEqual(image.size, (1600, 800))
            self.assertEqual(manifest["sparse"], [0])
            self.assertEqual(manifest["test"], [])
            self.assertEqual(manifest["dense"], [0, 1])
            self.assertEqual(manifest["width"], 1600)
            self.assertEqual(manifest["height"], 800)


if __name__ == "__main__":
    unittest.main()
