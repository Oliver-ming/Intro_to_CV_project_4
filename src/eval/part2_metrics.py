"""Pose error for a finished i3dgs run. Rendering metrics are separate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.eval.pose_error import pose_error
from src.run.i3dgs_export import registered_camera_to_world


def part2_pose_summary(
    metadata: dict,
    frame_names: list[str],
    training_indices: list[int],
    ground_truth_centers: torch.Tensor,
) -> dict:
    """Sim(3) error on registered training views, plus the dropped-view count."""
    poses = registered_camera_to_world(metadata)
    estimated = []
    truth = []
    for index in training_indices:
        name = frame_names[index]
        if name not in poses:
            continue
        estimated.append(torch.tensor(poses[name][:3, 3], dtype=torch.float32))
        truth.append(ground_truth_centers[index])
    if len(estimated) < 3:
        raise RuntimeError("Sim(3) pose error needs at least three registered training views")
    return {
        "registered_views": len(estimated),
        "dropped_views": len(training_indices) - len(estimated),
        "pose_error": pose_error(torch.stack(estimated), torch.stack(truth)),
    }


def write_part2_pose_summary(
    metadata_path: Path,
    split_path: Path,
    split_name: str,
    ground_truth_centers: torch.Tensor,
    destination: Path,
    joint_reconstruction_seconds: float,
) -> dict:
    """Append one pose summary for ``dense`` or ``sparse``."""
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads(split_path.read_text())
    summary = part2_pose_summary(
        metadata,
        manifest["frames"],
        manifest[split_name],
        ground_truth_centers,
    )
    record = {
        "scene": manifest["scene"],
        "run": f"i3dgs-{split_name}",
        "joint_reconstruction_seconds": joint_reconstruction_seconds,
        **summary,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record
