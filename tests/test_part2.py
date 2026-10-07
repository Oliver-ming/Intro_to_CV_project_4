"""Checks the i3dgs focal scaling and command."""

import unittest
from pathlib import Path

from src.run.part2 import i3dgs_command, scaled_focal


class Part2Test(unittest.TestCase):
    def test_waymo_focal_shrinks_with_width(self):
        focal = scaled_focal(2066.697564417299, 1920)
        self.assertAlmostEqual(focal, 2066.697564417299 * 1600 / 1920)

    def test_command_fixes_focal_and_omits_poses(self):
        command = i3dgs_command(Path("images"), Path("model"), "python", Path("train.py"), 1722.0)
        self.assertIn("--fix_focal", command)
        self.assertIn("1722.0", command)
        self.assertEqual(command[command.index("--viewer_mode") + 1], "none")
        self.assertNotIn("extrinsic", " ".join(command).lower())


if __name__ == "__main__":
    unittest.main()
