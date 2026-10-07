"""Write one generated image and its frozen pose into a COLMAP scene."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.generate.colmap_view import append_generated_view


def write_generated_view(
    scene_dir: Path,
    image: Image.Image,
    name: str,
    camera_to_world: np.ndarray,
    camera_id: int,
    image_id: int,
) -> Path:
    """Save ``image`` under ``scene_dir/images`` and append its COLMAP pose.

    Existing cameras are left unchanged. The photometric loss later treats this
    image like a sparse view.
    """
    image_dir = scene_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    destination = image_dir / name
    image.save(destination)
    append_generated_view(
        scene_dir / "sparse" / "0" / "images.txt",
        name,
        camera_to_world,
        camera_id,
        image_id,
    )
    return destination
