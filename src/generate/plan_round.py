"""Choose one target pose between each pair of adjacent training views."""

from __future__ import annotations

import torch

from src.generate.reference import adjacent_pairs
from src.generate.target_view import midpoint_pose


def plan_round(indices: list[int], poses: dict[int, torch.Tensor]) -> list[tuple[int, int, torch.Tensor]]:
    """Return ``(left, right, midpoint pose)`` for every adjacent training pair.

    An empty list means fewer than two registered training views, so the round
    cannot start.
    """
    planned = []
    for left, right in adjacent_pairs(indices):
        planned.append((left, right, midpoint_pose(poses[left], poses[right])))
    return planned
