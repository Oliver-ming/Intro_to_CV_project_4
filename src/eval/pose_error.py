"""Pose error after one Sim(3) alignment on training views."""

from __future__ import annotations

import torch


def sim3_align(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return scale, rotation, and translation mapping ``source`` onto ``target``.

    Both inputs are ``N,3`` camera centers. The rotation is in ``SO(3)``.
    """
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("expected matching N,3 camera centers")
    if source.shape[0] < 3:
        raise ValueError("Sim(3) alignment needs at least three camera centers")
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular, v_t = torch.linalg.svd(covariance)
    correction = torch.eye(3, device=source.device, dtype=source.dtype)
    correction[2, 2] = torch.det(u @ v_t)
    rotation = u @ correction @ v_t
    variance = torch.sum(source_centered ** 2) / source.shape[0]
    scale = torch.trace(torch.diag(singular) @ correction) / variance
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def pose_error(estimated: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Sim(3)-aligned translation RMSE, in the same unit as ``ground_truth``."""
    scale, rotation, translation = sim3_align(estimated, ground_truth)
    aligned = scale * (estimated @ rotation.T) + translation
    squared = torch.sum((aligned - ground_truth) ** 2, dim=1)
    return torch.sqrt(torch.mean(squared)).item()
