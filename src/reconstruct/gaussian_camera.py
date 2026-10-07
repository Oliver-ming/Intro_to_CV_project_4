"""Camera matrices in the official 3DGS convention."""

from __future__ import annotations

import numpy as np
import torch


def gaussian_rotation_translation(camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``R, T`` pair stored by ``scene.cameras.Camera``.

    ``R`` is the camera-to-world rotation. ``T`` is the world-to-camera translation.
    """
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")
    world_to_camera = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    rotation = world_to_camera[:3, :3].T
    translation = world_to_camera[:3, 3]
    return rotation, translation


def world_view_transform(camera_to_world: np.ndarray) -> torch.Tensor:
    """The transposed world-to-view matrix that ``Camera`` puts on the rasterizer."""
    rotation, translation = gaussian_rotation_translation(camera_to_world)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.T
    matrix[:3, 3] = translation
    return torch.tensor(matrix, dtype=torch.float32).transpose(0, 1)


def camera_center(camera_to_world: np.ndarray) -> torch.Tensor:
    """Camera center in world coordinates, matching ``MiniCam.camera_center``."""
    return torch.inverse(world_view_transform(camera_to_world))[3, :3]


def make_minicam(camera_to_world: np.ndarray, width: int, height: int, focal_x: float, focal_y: float):
    """Build the camera object official 3DGS renders with. Poses stay frozen."""
    import math
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix

    fov_x = 2 * math.atan(width / (2 * focal_x))
    fov_y = 2 * math.atan(height / (2 * focal_y))
    view = world_view_transform(camera_to_world)
    projection = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y).transpose(0, 1)
    full = view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return MiniCam(width, height, fov_y, fov_x, 0.01, 100.0, view, full)
