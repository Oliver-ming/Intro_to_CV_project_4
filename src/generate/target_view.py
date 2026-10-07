"""One target camera halfway between two estimated training poses."""

from __future__ import annotations

import torch


def midpoint_pose(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return the 4x4 pose halfway from ``first`` to ``second``.

    Translation is the arithmetic mean. Rotation is the spherical midpoint.
    Neither input is a ground-truth captured camera.
    """
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("poses must be 4x4")
    pose = torch.eye(4, device=first.device, dtype=first.dtype)
    pose[:3, :3] = _slerp_rotation(first[:3, :3], second[:3, :3], 0.5)
    pose[:3, 3] = 0.5 * (first[:3, 3] + second[:3, 3])
    return pose


def _slerp_rotation(first: torch.Tensor, second: torch.Tensor, fraction: float) -> torch.Tensor:
    start = _matrix_to_quaternion(first)
    end = _matrix_to_quaternion(second)
    if torch.dot(start, end) < 0:
        end = -end
    cosine = torch.clamp(torch.dot(start, end), -1.0, 1.0)
    angle = torch.acos(cosine)
    if float(angle) < 1e-6:
        blended = (1.0 - fraction) * start + fraction * end
    else:
        blended = (
            torch.sin((1.0 - fraction) * angle) * start + torch.sin(fraction * angle) * end
        ) / torch.sin(angle)
    blended = blended / torch.linalg.norm(blended)
    return _quaternion_to_matrix(blended)


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    trace = torch.trace(matrix)
    if trace > 0:
        scale = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = torch.diagonal(matrix)
        axis = int(torch.argmax(diagonal))
        if axis == 0:
            scale = torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = torch.stack((w, x, y, z))
    return quaternion / torch.linalg.norm(quaternion)


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        )
    )
