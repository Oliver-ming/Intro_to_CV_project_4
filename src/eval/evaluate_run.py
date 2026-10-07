"""Score test views after photometric pose alignment and append one metrics line."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from src.eval.evaluation_pose import EVALUATION_POSE_STEPS, align_evaluation_pose
from src.eval.pose_delta import apply_pose_delta
from src.eval.save_metrics import save_metrics
from src.eval.score import score_run
from src.generate.reference import nearest_training_index


def initial_test_poses(
    test_centers: dict[int, torch.Tensor],
    training_centers: torch.Tensor,
    training_poses: list[torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Start each test camera at the nearest registered training pose."""
    if len(training_poses) != training_centers.shape[0]:
        raise ValueError("training poses and centers must match")
    poses = {}
    for index, center in test_centers.items():
        local = nearest_training_index(center, training_centers)
        poses[index] = training_poses[local]
    return poses


def aligned_renders(
    test_images: dict[int, torch.Tensor],
    initial_poses: dict[int, torch.Tensor],
    render_image: Callable[[torch.Tensor], torch.Tensor],
    steps: int = EVALUATION_POSE_STEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render every test view after its own 500-step pose fit. The model stays fixed."""
    renders = []
    targets = []
    for index in sorted(test_images):
        pose = initial_poses[index]
        target = test_images[index]

        def render(delta: torch.Tensor, pose: torch.Tensor = pose) -> torch.Tensor:
            return render_image(apply_pose_delta(pose, delta))

        delta = align_evaluation_pose(target, target, render, steps=steps)
        rendered = render(delta).detach()
        renders.append(rendered)
        targets.append(target.detach().to(device=rendered.device, dtype=rendered.dtype))
    return torch.cat(renders, dim=0), torch.cat(targets, dim=0)


def record_test_views(
    path: Path,
    test_images: dict[int, torch.Tensor],
    initial_poses: dict[int, torch.Tensor],
    render_image: Callable[[torch.Tensor], torch.Tensor],
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
    steps: int = EVALUATION_POSE_STEPS,
) -> dict:
    """Align test poses, score the frozen reconstruction, and append one JSONL record."""
    prediction, target = aligned_renders(test_images, initial_poses, render_image, steps=steps)
    record = score_run(
        prediction,
        target,
        estimated_centers,
        ground_truth_centers,
        training_indices,
        estimated_indices,
        scene,
        run,
        iteration,
        initialization_seconds,
        reconstruction_seconds,
        generation_seconds,
        joint_reconstruction_seconds,
        lpips_fn,
    )
    save_metrics(path, record)
    return record
