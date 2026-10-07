"""Apply the 6-vector used by photometric evaluation poses."""

from __future__ import annotations

import torch


def apply_pose_delta(camera_to_world: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Rotate the camera by ``delta[:3]`` and add ``delta[3:]`` to its center.

    ``delta`` is an axis-angle plus a world translation. A zero vector leaves the
    pose unchanged, which is the start of the 500-step evaluation.
    """
    if camera_to_world.shape != (4, 4) or delta.shape != (6,):
        raise ValueError("expected a 4x4 pose and a 6-vector")
    updated = camera_to_world.clone()
    updated[:3, :3] = _rodrigues(delta[:3]) @ camera_to_world[:3, :3]
    updated[:3, 3] = camera_to_world[:3, 3] + delta[3:]
    return updated


def _rodrigues(omega: torch.Tensor) -> torch.Tensor:
    skew = torch.zeros(3, 3, device=omega.device, dtype=omega.dtype)
    skew[0, 1] = -omega[2]
    skew[0, 2] = omega[1]
    skew[1, 0] = omega[2]
    skew[1, 2] = -omega[0]
    skew[2, 0] = -omega[1]
    skew[2, 1] = omega[0]
    theta2 = torch.sum(omega * omega)
    theta = torch.sqrt(theta2 + 1e-12)
    sine_scale = torch.sin(theta) / theta
    cosine_scale = (1 - torch.cos(theta)) / (theta2 + 1e-12)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype)
    return eye + sine_scale * skew + cosine_scale * (skew @ skew)
