"""Checks that the VGGT initializer does not turn on bundle adjustment."""

import unittest
from pathlib import Path

from src.init.vggt import vggt_command


class VggtCommandTest(unittest.TestCase):
    def test_ba_is_not_requested(self):
        command = vggt_command(Path("scene"), "python", Path("demo_colmap.py"))
        self.assertNotIn("--use_ba", command)
        self.assertIn("--scene_dir", command)
        self.assertEqual(command[-1], "scene")


if __name__ == "__main__":
    unittest.main()
