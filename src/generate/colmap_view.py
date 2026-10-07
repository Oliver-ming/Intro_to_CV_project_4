"""Add a generated view to a COLMAP text model without changing existing poses."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_camera_to_world(images_path: Path) -> dict[str, np.ndarray]:
    """Read COLMAP text images back into camera-to-world matrices, keyed by name."""
    poses = {}
    lines = images_path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith("#"):
            index += 1
            continue
        parts = line.split()
        qvec = np.array([float(value) for value in parts[1:5]], dtype=np.float64)
        translation = np.array([float(value) for value in parts[5:8]], dtype=np.float64)
        name = parts[9]
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = _qvec_to_rotation(qvec)
        world_to_camera[:3, 3] = translation
        poses[name] = np.linalg.inv(world_to_camera)
        index += 2
    return poses


def append_generated_view(
    images_path: Path,
    name: str,
    camera_to_world: np.ndarray,
    camera_id: int,
    image_id: int,
) -> None:
    """Append one image whose pose stays at ``camera_to_world``.

    COLMAP stores world-to-camera. The two-line image record has no 2D points.
    """
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")
    world_to_camera = np.linalg.inv(camera_to_world)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    qvec = _rotation_to_qvec(rotation)
    line = (
        f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
        f"{translation[0]} {translation[1]} {translation[2]} {camera_id} {name}\n\n"
    )
    with images_path.open("a") as handle:
        handle.write(line)


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        axis = int(np.argmax(np.diagonal(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    if qvec[0] < 0:
        qvec *= -1
    return qvec / np.linalg.norm(qvec)
