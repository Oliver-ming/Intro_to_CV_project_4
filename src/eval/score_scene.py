"""Score a finished Part 1 checkpoint on the held-out test frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.intrinsics import intrinsics_for_prepared
from src.data.undistort import read_pinhole, undistort_waymo
from src.eval.evaluate_run import record_test_views
from src.eval.rendering import lpips_vgg
from src.generate.colmap_view import read_camera_to_world
from src.init.colmap import camera_model_name
from src.reconstruct.render_view import load_gaussians, render_view


def nearest_training_pose(test_index: int, training_indices: list[int], poses: dict[int, torch.Tensor]) -> torch.Tensor:
    """Start a test camera at the training pose closest in frame index."""
    nearest = min(training_indices, key=lambda index: (abs(index - test_index), index))
    return poses[nearest]


def registered_training_poses(
    images_path: Path,
    frame_names: list[str],
    training_indices: list[int],
) -> tuple[list[int], dict[int, torch.Tensor]]:
    """Poses COLMAP kept. Missing names are dropped views."""
    recovered = read_camera_to_world(images_path)
    indices = []
    poses = {}
    for index in training_indices:
        name = frame_names[index]
        if name not in recovered:
            continue
        indices.append(index)
        poses[index] = torch.tensor(recovered[name], dtype=torch.float32)
    return indices, poses


def _prepared_intrinsics(scene: str, prepared: Path, manifest: dict) -> tuple[float, float, float, float]:
    """Pixel intrinsics, including DL3DV fractions stored beside the split."""
    normalized = None
    cameras = prepared / "cameras.json"
    if cameras.is_file():
        payload = json.loads(cameras.read_text())
        item = payload[0] if isinstance(payload, list) else payload
        normalized = (item["fx"], item["fy"], item["cx"], item["cy"])
    return intrinsics_for_prepared(scene, manifest["width"], manifest["height"], normalized)


def write_checkpoint_metrics(
    scene: str,
    prepared: Path,
    source: Path,
    model_dir: Path,
    metrics_path: Path,
    ground_truth_centers: torch.Tensor,
    iteration: int,
    initialization_seconds: float,
    reconstruction_seconds: float,
    steps: int = 500,
    render_image=None,
    lpips_fn=None,
    training_split: str = "dense",
    run_name: str = "colmap",
    joint_reconstruction_seconds: float = 0.0,
    generation_seconds: float = 0.0,
) -> dict:
    """Append one metrics line for ``iteration``. The reconstruction stays frozen."""
    manifest = json.loads((prepared / "split.json").read_text())
    training_indices, poses = registered_training_poses(
        source / "sparse" / "0" / "images.txt",
        manifest["frames"],
        manifest[training_split],
    )
    if len(training_indices) < 3:
        raise RuntimeError("Sim(3) pose error needs at least three registered training views")
    estimated_centers = torch.stack([poses[index][:3, 3] for index in training_indices])
    ground_truth = ground_truth_centers[training_indices]
    test_images = {}
    initial_poses = {}
    cameras_path = source / "sparse" / "0" / "cameras.txt"
    pinhole = None
    if (
        scene == "waymo-405841"
        and cameras_path.is_file()
        and camera_model_name(cameras_path.read_text()) == "PINHOLE"
        and source.name == "pinhole"
    ):
        pinhole = read_pinhole(cameras_path.read_text())
    render_width = pinhole[0] if pinhole else manifest["width"]
    render_height = pinhole[1] if pinhole else manifest["height"]
    for index in manifest["test"]:
        with Image.open(prepared / "images" / manifest["frames"][index]) as image:
            array = np.array(image.convert("RGB"))
        if pinhole is not None:
            array = undistort_waymo(array, manifest["width"], manifest["height"], cameras_path.read_text())
        tensor = torch.from_numpy(np.array(array)).float().permute(2, 0, 1) / 255
        test_images[index] = tensor.unsqueeze(0)
        initial_poses[index] = nearest_training_pose(index, training_indices, poses)
    if render_image is None:
        if pinhole is not None:
            _, _, focal_x, focal_y, _, _ = pinhole
        else:
            focal_x, focal_y, _, _ = _prepared_intrinsics(scene, prepared, manifest)
        model = load_gaussians(model_dir / f"chkpnt{iteration}.pth", sh_degree=3)

        def render_image(pose, model=model, focal_x=focal_x, focal_y=focal_y):
            rendered = render_view(
                model,
                pose.detach().cpu().numpy(),
                render_width,
                render_height,
                focal_x,
                focal_y,
            )
            return rendered["color"].detach().unsqueeze(0).clamp(0, 1)

    return record_test_views(
        metrics_path,
        test_images,
        initial_poses,
        render_image,
        estimated_centers,
        ground_truth,
        manifest[training_split],
        training_indices,
        scene,
        run_name,
        iteration,
        initialization_seconds,
        reconstruction_seconds,
        generation_seconds,
        joint_reconstruction_seconds,
        lpips_fn or lpips_vgg,
        steps=steps,
    )
