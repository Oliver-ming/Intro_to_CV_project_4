"""Checks archive ingest and the Part 1 training command."""

import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from src.data.ingest import ingest_archive
from src.run.ingest_when_ready import archive_ready
from src.run.part1 import gaussian_training_command


class IngestTest(unittest.TestCase):
    def test_zip_becomes_a_prepared_scene(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rgb = root / "stage" / "405841" / "FRONT" / "rgb"
            rgb.mkdir(parents=True)
            Image.new("RGB", (20, 10), (1, 2, 3)).save(rgb / "000.png")
            archive = root / "405841.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for path in rgb.rglob("*.png"):
                    bundle.write(path, path.relative_to(root / "stage"))
            manifest = ingest_archive(
                "waymo-405841",
                archive,
                root / "extracted",
                root / "prepared",
            )
            self.assertEqual(manifest["frames"], ["000000.jpg"])
            self.assertTrue((root / "prepared" / "images" / "000000.jpg").is_file())

    def test_token_without_a_zip_is_not_ready(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            log = root / "download.log"
            log.write_text("WAYMO_OK\n")
            partial = root / "405841.zip"
            partial.write_bytes(b"not a zip")
            self.assertFalse(archive_ready(log, "WAYMO_OK", partial))
            with zipfile.ZipFile(partial, "w") as bundle:
                bundle.writestr("empty.txt", "ok")
            self.assertTrue(archive_ready(log, "WAYMO_OK", partial))


class CommandTest(unittest.TestCase):
    def test_command_reports_the_kerbl_checkpoints(self):
        command = gaussian_training_command(
            Path("source"),
            Path("model"),
            "python",
            Path("train.py"),
        )
        self.assertIn("30000", command)
        self.assertIn("7000", command)
        self.assertIn("--disable_viewer", command)


if __name__ == "__main__":
    unittest.main()
