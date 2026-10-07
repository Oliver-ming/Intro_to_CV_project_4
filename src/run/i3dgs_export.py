"""Read i3dgs metadata into a standard 3DGS scene."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from src.init.export_colmap import write_text_scene
from src.reconstruct.export_gaussians import write_gaussian_checkpoint


POINT_CLOUD_CAP = 20_000


def _placeholder_points(xyz: torch.Tensor) -> torch.Tensor:
    """A short point cloud so 3DGS can open the scene before the checkpoint loads.

    The checkpoint keeps every exported Gaussian. The text cloud is only the
    loader's placeholder, capped so a large hierarchy does not become a huge file.
    """
    if xyz.shape[0] <= POINT_CLOUD_CAP:
        return xyz
    step = max(1, xyz.shape[0] // POINT_CLOUD_CAP)
    return xyz[::step][:POINT_CLOUD_CAP]


def registered_camera_to_world(metadata: dict) -> dict[str, np.ndarray]:
    """Poses for keyframes i3dgs kept. ``Rt`` in the file is world-to-camera.

    A keyframe marked ``is_test`` is not a training view. Missing names are skipped
    because they cannot be matched to an image file. The caller counts dropped views
    from the names that are absent.
    """
    poses = {}
    for frame in metadata["keyframes"]:
        info = frame["info"]
        if info.get("is_test"):
            continue
        name = info.get("name")
        if not name:
            continue
        world_to_camera = np.asarray(frame["Rt"], dtype=np.float64)
        poses[name] = np.linalg.inv(world_to_camera)
    return poses


def normalize_reconstruction(
    camera_to_world: np.ndarray,
    xyz: torch.Tensor,
    scaling: torch.Tensor,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, dict]:
    """Put cameras and Gaussians in a frame the 3DGS rasterizer can handle.

    i3dgs can leave translations in the tens of thousands. Those coordinates make
    the tile counter overflow. The shift is the mean camera center. The scale is
    1.1 times the farthest camera, matching the usual 3DGS scene radius. Log-scales
    lose ``log(radius)`` so the Gaussian size follows the new unit.
    """
    centers = camera_to_world[:, :3, 3]
    center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - center, axis=1).max() * 1.1)
    if radius < 1e-6:
        radius = 1.0
    normalized = np.array(camera_to_world, dtype=np.float64, copy=True)
    normalized[:, :3, 3] = (centers - center) / radius
    points = xyz.detach().cpu().to(dtype=torch.float32)
    center_tensor = torch.tensor(center, dtype=torch.float32)
    normalized_xyz = (points - center_tensor) / radius
    normalized_scaling = scaling.detach().cpu().to(dtype=torch.float32) - math.log(radius)
    frame = {"center": center.tolist(), "radius": radius}
    return normalized, normalized_xyz, normalized_scaling, frame


def keep_trainable_gaussians(
    camera_to_world: np.ndarray,
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    factor: float = 10.0,
) -> torch.Tensor:
    """Drop Gaussians that sit far outside the cameras or are far too large.

    i3dgs keeps a few triangulated outliers. Their coordinates make the 3DGS
    rasterizer overflow. ``factor`` is measured in scene radii around the mean
    camera center.
    """
    if factor <= 1:
        raise ValueError("factor must be greater than 1")
    centers = camera_to_world[:, :3, 3]
    center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - center, axis=1).max() * 1.1)
    if radius < 1e-6:
        radius = 1.0
    points = xyz.detach().cpu().to(dtype=torch.float32)
    center_tensor = torch.tensor(center, dtype=torch.float32)
    distance = torch.linalg.norm(points - center_tensor, dim=1)
    largest_scale = scaling.detach().cpu().to(dtype=torch.float32).max(dim=1).values
    return (distance < factor * radius) & (largest_scale < math.log(factor * radius))


def dropped_training_names(training_names: list[str], metadata: dict) -> list[str]:
    """Training images that i3dgs never registered. Order follows ``training_names``."""
    registered = set(registered_camera_to_world(metadata))
    return [name for name in training_names if name not in registered]


def export_part3_scene(
    scene_dir: Path,
    metadata: dict,
    training_names: list[str],
    images_dir: Path,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    sh_degree: int,
) -> dict:
    """Write registered images, frozen poses, and a 3DGS checkpoint.

    Fewer than two registered training views is a failure. The checkpoint is the
    i3dgs Gaussians. The point cloud exists so 3DGS can open the scene before
    ``--start_checkpoint`` replaces it.
    """
    dropped = dropped_training_names(training_names, metadata)
    poses = registered_camera_to_world(metadata)
    if len(poses) < 2:
        return {"failure": "fewer than two registered training views", "dropped_views": len(dropped)}
    names = sorted(poses)
    camera_to_world = np.stack([poses[name] for name in names])
    keep = keep_trainable_gaussians(camera_to_world, xyz, scaling)
    if int(keep.sum()) == 0:
        return {"failure": "no Gaussians left inside the camera scene", "dropped_views": len(dropped)}
    removed_gaussians = int((~keep).sum())
    xyz = xyz[keep]
    features_dc = features_dc[keep]
    features_rest = features_rest[keep]
    scaling = scaling[keep]
    rotation = rotation[keep]
    opacity = opacity[keep]
    camera_to_world, xyz, scaling, frame = normalize_reconstruction(camera_to_world, xyz, scaling)
    points = _placeholder_points(xyz.detach().cpu())
    colors = np.full((points.shape[0], 3), 200, dtype=np.int64)
    write_text_scene(scene_dir, names, camera_to_world, intrinsics, width, height, points.numpy(), colors)
    image_dir = scene_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        destination = image_dir / name
        if not destination.exists():
            destination.symlink_to((images_dir / name).resolve())
    checkpoint = scene_dir / "chkpnt0.pth"
    write_gaussian_checkpoint(
        checkpoint, xyz, features_dc, features_rest, scaling, rotation, opacity, sh_degree
    )
    return {
        "failure": None,
        "dropped_views": len(dropped),
        "removed_gaussians": removed_gaussians,
        "checkpoint": checkpoint,
        "frame": frame,
    }
