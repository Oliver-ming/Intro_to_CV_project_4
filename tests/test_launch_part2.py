"""Checks that Part 2 prefers the separate i3dgs extension build."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.run.launch_part2 import launch_part2, part2_command, part2_environ


class LaunchPart2Test(unittest.TestCase):
    def test_pythonpath_puts_i3dgs_extensions_first(self):
        env = part2_environ({"PYTHONPATH": "/other"})
        first = env["PYTHONPATH"].split(os.pathsep)[0]
        self.assertTrue(first.endswith("third_party/i3dgs/py_ext"))
        self.assertIn("/other", env["PYTHONPATH"])

    def test_command_fixes_focal(self):
        command = part2_command(Path("images"), Path("model"), "python", 100.0)
        self.assertIn("--fix_focal", command)
        self.assertTrue(command[1].endswith("third_party/i3dgs/train.py"))

    def test_launch_uses_the_prepared_waymo_focal(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prepared = root / "prepared"
            images = prepared / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (8, 4)).save(images / "000000.jpg")
            (prepared / "split.json").write_text(json.dumps({
                "scene": "waymo-405841",
                "width": 1600,
                "height": 1067,
                "frames": ["000000.jpg"],
                "dense": [0],
                "sparse": [0],
            }))
            seen = {}

            def run(command, env):
                seen["command"] = command
                seen["env"] = env

            focal = launch_part2(prepared, root / "work", "sparse", "python", run)
            self.assertTrue((root / "work" / "sparse" / "images" / "000000.jpg").is_symlink())
        self.assertAlmostEqual(focal, 2066.697564417299 * 1600 / 1920)
        self.assertIn(str(focal), seen["command"])
        self.assertTrue(seen["env"]["PYTHONPATH"].split(os.pathsep)[0].endswith("py_ext"))

    def test_execute_writes_the_joint_clock(self):
        from src.run.launch_part2 import execute_part2

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prepared = root / "prepared"
            images = prepared / "images"
            images.mkdir(parents=True)
            Image.new("RGB", (8, 4)).save(images / "000000.jpg")
            (prepared / "split.json").write_text(json.dumps({
                "scene": "waymo-405841",
                "width": 1600,
                "height": 1067,
                "frames": ["000000.jpg"],
                "dense": [0],
                "sparse": [0],
            }))
            record = execute_part2(prepared, root / "work", "dense", "python", spawn=lambda command, env: None)
            self.assertTrue((root / "work" / "TRAIN_PART2_DENSE_OK").is_file())
            self.assertIn("joint_reconstruction_seconds", record)


if __name__ == "__main__":
    unittest.main()
