"""Load the training cameras that a generation round is allowed to see."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.generate.colmap_view import read_camera_to_world


def read_intrinsics(cameras_path: Path) -> np.ndarray:
    """3x3 intrinsics from a COLMAP PINHOLE or SIMPLE_PINHOLE cameras.txt."""
    for line in cameras_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        model = parts[1]
        if model == "SIMPLE_PINHOLE":
            focal, cx, cy = (float(value) for value in parts[4:7])
            fy = focal
        elif model == "PINHOLE":
            focal, fy, cx, cy = (float(value) for value in parts[4:8])
        else:
            raise ValueError(f"unsupported camera model {model}")
        intrinsics = np.eye(3, dtype=np.float64)
        intrinsics[0, 0] = focal
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        return intrinsics
    raise ValueError(f"no camera in {cameras_path}")


def load_training_views(scene_dir: Path) -> tuple[list[int], dict[int, torch.Tensor], dict[int, Image.Image]]:
    """Real training images only. Names are ``000000.jpg`` style frame indices.

    Generated views are skipped here. ``GenerationState`` adds them during the rounds.
    """
    poses = read_camera_to_world(scene_dir / "sparse" / "0" / "images.txt")
    indices = []
    pose_by_index = {}
    images = {}
    for name, pose in poses.items():
        if name.startswith("gen_"):
            continue
        index = int(Path(name).stem)
        indices.append(index)
        pose_by_index[index] = torch.tensor(pose, dtype=torch.float32)
        with Image.open(scene_dir / "images" / name) as image:
            images[index] = image.convert("RGB")
    return indices, pose_by_index, images
