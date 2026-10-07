"""Checks that hierarchy export keeps leaves and drops internal nodes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from src.reconstruct.hierarchy_gaussians import load_hierarchy_gaussians


class HierarchyGaussianTest(unittest.TestCase):
    def test_internal_node_is_not_exported(self):
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
            ("child_0", "i4"),
            ("parent", "i4"),
            ("id_in_hierarchy", "i4"),
        ]
        rows = np.array([
            (1, 2, 3, 0, 0, 0, 0.2, 0, 0, 0, 1, 0, 0, 0, -1, -1, 0),
            (9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 1),
        ], dtype=dtype)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "hierarchy.ply"
            PlyData([PlyElement.describe(rows, "vertex")], text=True).write(str(path))
            loaded = load_hierarchy_gaussians(path)
        self.assertEqual(loaded["xyz"].shape[0], 1)
        self.assertEqual(loaded["sh_degree"], 0)
        self.assertAlmostEqual(float(loaded["xyz"][0, 0]), 1.0)
        self.assertEqual(tuple(loaded["features_rest"].shape), (1, 0, 3))


if __name__ == "__main__":
    unittest.main()
