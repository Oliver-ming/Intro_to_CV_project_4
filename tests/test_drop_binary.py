"""Checks that COLMAP binary files are removed after a text export."""

import tempfile
import unittest
from pathlib import Path

from src.init.colmap import drop_binary_model


class DropBinaryTest(unittest.TestCase):
    def test_bins_are_removed_and_text_stays(self):
        with tempfile.TemporaryDirectory() as raw:
            model = Path(raw)
            (model / "images.bin").write_bytes(b"bin")
            (model / "images.txt").write_text("text\n")
            drop_binary_model(model)
            self.assertFalse((model / "images.bin").exists())
            self.assertEqual((model / "images.txt").read_text(), "text\n")


if __name__ == "__main__":
    unittest.main()
