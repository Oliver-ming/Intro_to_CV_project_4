"""Turn one initializer's poses and depth into a COLMAP text scene."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.generate.colmap_view import append_generated_view


def depth_to_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    stride: int = 8,
) -> np.ndarray:
    """Unproject a depth map to world points. ``world_to_camera`` is OpenCV 4x4.

    Only finite, positive depths are returned. ``stride`` skips pixels so the
    initial cloud stays smaller than the image.
    """
    if depth.ndim != 2:
        raise ValueError("depth must be H,W")
    if stride <= 0:
        raise ValueError("stride must be positive")
    sampled = depth[::stride, ::stride]
    rows, cols = np.nonzero(np.isfinite(sampled) & (sampled > 0))
    if rows.size == 0:
        raise ValueError("depth map has no positive finite pixels")
    z = sampled[rows, cols]
    u = cols.astype(np.float64) * stride
    v = rows.astype(np.float64) * stride
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera = np.stack(((u - cx) / fx * z, (v - cy) / fy * z, z, np.ones_like(z)), axis=1)
    world = camera @ np.linalg.inv(world_to_camera).T
    return world[:, :3]


def scale_intrinsics(intrinsics: np.ndarray, image_size: tuple[int, int], depth_shape: tuple[int, int]) -> np.ndarray:
    """Scale a dataset camera from ``(width, height)`` onto a depth map ``(H, W)``."""
    width, height = image_size
    depth_height, depth_width = depth_shape
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    scaled = np.array(intrinsics, dtype=np.float64, copy=True)
    scaled[0, :] *= depth_width / width
    scaled[1, :] *= depth_height / height
    return scaled


def export_omega_scene(
    scene_dir: Path,
    image_names: list[str],
    depth_maps: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> Path:
    """Write VGGT-Ω geometry. Extrinsics stay camera-from-world. ``K`` is the dataset's.

    Depth is unprojected with ``K`` scaled to the depth-map resolution. The COLMAP
    camera keeps the image resolution so 3DGS trains on the shared pixels.
    """
    if depth_maps.shape[0] != len(image_names):
        raise ValueError("one depth map per image is required")
    clouds = []
    for index, name in enumerate(image_names):
        del name
        scaled = scale_intrinsics(intrinsics, (width, height), depth_maps[index].shape)
        clouds.append(depth_to_points(depth_maps[index], scaled, world_to_camera[index]))
    points = np.concatenate(clouds, axis=0)
    colors = np.full((points.shape[0], 3), 200, dtype=np.int64)
    camera_to_world = np.linalg.inv(world_to_camera)
    return write_text_scene(
        scene_dir,
        image_names,
        camera_to_world,
        intrinsics,
        width,
        height,
        points,
        colors,
    )


def write_text_scene(
    scene_dir: Path,
    image_names: list[str],
    camera_to_world: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    points: np.ndarray,
    colors: np.ndarray,
) -> Path:
    """Write ``sparse/0`` text files. Poses are camera-to-world, one per image."""
    if camera_to_world.shape != (len(image_names), 4, 4):
        raise ValueError("one 4x4 pose per image is required")
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length")
    model = scene_dir / "sparse" / "0"
    model.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    (model / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n"
    )
    images_path = model / "images.txt"
    images_path.write_text("# Image list with two lines of data per image:\n")
    for image_id, name in enumerate(image_names, start=1):
        append_generated_view(images_path, name, camera_to_world[image_id - 1], 1, image_id)
    point_lines = ["# 3D point list with one line of data per point:"]
    for index, (point, color) in enumerate(zip(points, colors), start=1):
        red, green, blue = (int(channel) for channel in color)
        point_lines.append(
            f"{index} {point[0]} {point[1]} {point[2]} {red} {green} {blue} 0"
        )
    (model / "points3D.txt").write_text("\n".join(point_lines) + "\n")
    return model
