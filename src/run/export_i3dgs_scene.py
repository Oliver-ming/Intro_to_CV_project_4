"""Turn a finished i3dgs model into a standard 3DGS scene."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.intrinsics import intrinsics_for_prepared
from src.reconstruct.hierarchy_gaussians import load_hierarchy_gaussians
from src.run.i3dgs_export import export_part3_scene


def export_i3dgs_scene(
    prepared: Path,
    model_dir: Path,
    images_dir: Path,
    scene_dir: Path,
    training_split: str,
) -> dict:
    """Write registered views, frozen poses, and ``chkpnt0.pth`` from one i3dgs split."""
    if training_split not in {"dense", "sparse"}:
        raise ValueError("training split must be dense or sparse")
    manifest = json.loads((prepared / "split.json").read_text())
    normalized = None
    cameras = prepared / "cameras.json"
    if cameras.is_file():
        payload = json.loads(cameras.read_text())
        item = payload[0] if isinstance(payload, list) else payload
        normalized = (item["fx"], item["fy"], item["cx"], item["cy"])
    focal_x, focal_y, principal_x, principal_y = intrinsics_for_prepared(
        manifest["scene"],
        manifest["width"],
        manifest["height"],
        normalized,
    )
    intrinsics = np.array(
        [[focal_x, 0.0, principal_x], [0.0, focal_y, principal_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    training_names = [manifest["frames"][index] for index in manifest[training_split]]
    gaussians = load_hierarchy_gaussians(model_dir / "hierarchy.ply")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    result = export_part3_scene(
        scene_dir,
        metadata,
        training_names,
        images_dir,
        intrinsics,
        manifest["width"],
        manifest["height"],
        gaussians["xyz"],
        gaussians["features_dc"],
        gaussians["features_rest"],
        gaussians["scaling"],
        gaussians["rotation"],
        gaussians["opacity"],
        gaussians["sh_degree"],
    )
    result["sh_degree"] = gaussians["sh_degree"]
    if result["failure"] is None:
        (scene_dir / "export.json").write_text(json.dumps({
            "scene": manifest["scene"],
            "split": training_split,
            "sh_degree": gaussians["sh_degree"],
            "gaussians": int(gaussians["xyz"].shape[0]),
            "dropped_views": result["dropped_views"],
            "removed_gaussians": result["removed_gaussians"],
            "checkpoint": str(result["checkpoint"]),
            "center": result["frame"]["center"],
            "radius": result["frame"]["radius"],
        }, indent=2) + "\n")
    return result
