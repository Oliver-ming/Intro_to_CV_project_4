"""Assemble one metrics record from rendered images and training poses."""

from __future__ import annotations

from collections.abc import Callable

import torch

from src.eval.pose_error import pose_error
from src.eval.rendering import psnr, ssim
from src.generate.reference import dropped_view_count


def score_run(
    prediction: torch.Tensor,
    target: torch.Tensor,
    estimated_centers: torch.Tensor,
    ground_truth_centers: torch.Tensor,
    training_indices: list[int],
    estimated_indices: list[int],
    scene: str,
    run: str,
    iteration: int,
    initialization_seconds: float,
    reconstruction_seconds: float,
    generation_seconds: float,
    joint_reconstruction_seconds: float,
    lpips_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict:
    """Score one checkpoint. ``lpips_fn`` must be VGG LPIPS."""
    return {
        "scene": scene,
        "run": run,
        "iteration": iteration,
        "psnr": float(psnr(prediction, target).mean()),
        "ssim": float(ssim(prediction, target).mean()),
        "lpips_vgg": float(lpips_fn(prediction, target).mean()),
        "pose_error": pose_error(estimated_centers, ground_truth_centers),
        "dropped_views": dropped_view_count(training_indices, estimated_indices),
        "initialization_seconds": initialization_seconds,
        "reconstruction_seconds": reconstruction_seconds,
        "generation_seconds": generation_seconds,
        "joint_reconstruction_seconds": joint_reconstruction_seconds,
    }
