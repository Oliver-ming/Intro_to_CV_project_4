"""Which training views a target camera may use."""

from __future__ import annotations

import torch


def adjacent_pairs(indices: list[int]) -> list[tuple[int, int]]:
    """Pairs of training views in time order. Fewer than two views yields no pair."""
    ordered = sorted(indices)
    return list(zip(ordered, ordered[1:]))


def nearest_training_index(target_center: torch.Tensor, training_centers: torch.Tensor) -> int:
    """Index of the training camera closest to ``target_center``.

    ``training_centers`` is ``N,3``. The caller must not include test views.
    """
    if training_centers.ndim != 2 or training_centers.shape[1] != 3:
        raise ValueError("training_centers must be N,3")
    if training_centers.shape[0] == 0:
        raise ValueError("no training cameras")
    distances = torch.linalg.norm(training_centers - target_center, dim=1)
    return int(torch.argmin(distances))


def dropped_view_count(training_indices: list[int], estimated_indices: list[int]) -> int:
    """How many training views never received an estimated pose."""
    return len(set(training_indices) - set(estimated_indices))
