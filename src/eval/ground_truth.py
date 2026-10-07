"""Ground-truth camera centers for pose error. These never enter training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def waymo_camera_centers(ground_truth_dir: Path) -> torch.Tensor:
    """Camera centers from S3PO Waymo ``gt/*.txt`` files, in filename order.

    Each file is a 4x4 camera-to-world matrix. The center is its translation.
    """
    files = sorted(ground_truth_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no pose files in {ground_truth_dir}")
    centers = []
    for path in files:
        matrix = np.loadtxt(path, delimiter=" ").reshape(4, 4)
        centers.append(matrix[:3, 3])
    return torch.tensor(np.stack(centers), dtype=torch.float32)


def dl3dv_camera_centers(cameras_path: Path, begin: int = 0, end: int = 300) -> torch.Tensor:
    """Camera centers from S3PO ``cameras.json`` field ``cam_trans``."""
    poses = json.loads(cameras_path.read_text())
    if poses and isinstance(poses[0], dict) and "image_name" in poses[0]:
        poses = sorted(poses, key=lambda pose: pose["image_name"])
    poses = poses[begin:end]
    if not poses:
        raise ValueError(f"no poses in {cameras_path} for [{begin}, {end})")
    centers = [pose["cam_trans"] for pose in poses]
    return torch.tensor(centers, dtype=torch.float32)
