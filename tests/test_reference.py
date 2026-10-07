"""Checks reference-view selection and dropped-view counting."""

import unittest

import torch

from src.generate.reference import adjacent_pairs, dropped_view_count, nearest_training_index


class ReferenceTest(unittest.TestCase):
    def test_nearest_and_pairs(self):
        centers = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        self.assertEqual(nearest_training_index(torch.tensor([2.0, 0.0, 0.0]), centers), 2)
        self.assertEqual(adjacent_pairs([0, 20, 10]), [(0, 10), (10, 20)])
        self.assertEqual(adjacent_pairs([4]), [])
        self.assertEqual(dropped_view_count([0, 10, 20], [0, 20]), 1)


if __name__ == "__main__":
    unittest.main()
