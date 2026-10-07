"""Checks closed-loop helpers pick the newest checkpoint."""

import tempfile
import unittest
from pathlib import Path

from src.run.closed_loop_scene import latest_checkpoint


class ClosedLoopSceneTest(unittest.TestCase):
    def test_latest_checkpoint_is_the_highest_iteration(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "chkpnt1500.pth").write_bytes(b"a")
            (root / "chkpnt30000.pth").write_bytes(b"b")
            (root / "chkpnt3000.pth").write_bytes(b"c")
            self.assertEqual(latest_checkpoint(root).name, "chkpnt30000.pth")


if __name__ == "__main__":
    unittest.main()
