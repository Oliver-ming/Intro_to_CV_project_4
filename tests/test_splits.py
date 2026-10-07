"""Checks for the mandatory view split and input width."""

import unittest

from src.data.splits import output_size, scene_split, split_indices


class SplitTest(unittest.TestCase):
    def test_waymo_stride(self):
        parts = scene_split("waymo-405841", 25)
        self.assertEqual(parts["sparse"], [0, 10, 20])
        self.assertEqual(parts["test"], [5, 15])
        self.assertNotIn(5, parts["dense"])
        self.assertIn(0, parts["dense"])
        self.assertIn(1, parts["dense"])

    def test_thirty_frame_stride(self):
        parts = split_indices(60, 30)
        self.assertEqual(parts["sparse"], [0, 30])
        self.assertEqual(parts["test"], [15, 45])
        self.assertEqual(len(parts["dense"]) + len(parts["test"]), 60)

    def test_width_cap(self):
        self.assertEqual(output_size(1920, 1280), (1600, 1067))
        self.assertEqual(output_size(800, 600), (800, 600))


if __name__ == "__main__":
    unittest.main()
