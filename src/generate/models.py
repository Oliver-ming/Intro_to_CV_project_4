"""Call Depth Anything 3 and GeoQuery with the locked argument shapes."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# SD-Turbo at the shared 1600-wide training size does not fit in 24 GB beside
# the UNet weights. The written view is still the camera resolution.
MAX_GEOQUERY_SIDE = 512


def estimate_with_da3(model, image: Image.Image) -> torch.Tensor:
    """Depth of one reference view. The map is ``H, W`` before scaling."""
    prediction = model.inference([image])
    depth = np.asarray(prediction.depth)
    if depth.ndim == 3:
        depth = depth[0]
    return torch.as_tensor(depth, dtype=torch.float32)


def restore_with_geoquery(
    model,
    target_image: Image.Image,
    reference_image: Image.Image,
    reference_depth: np.ndarray,
    intrinsics: np.ndarray,
    reference_pose: np.ndarray,
    target_pose: np.ndarray,
) -> Image.Image:
    """One GeoQuery restoration. Poses are camera-to-world.

    Wide images are restored at ``MAX_GEOQUERY_SIDE`` and resampled back to the
    camera size. Intrinsics are scaled with the image so the geometry query
    matches the tensor GeoQuery sees.
    """
    original = target_image.size
    target_image, reference_image, intrinsics = _fit_geoquery(
        target_image, reference_image, intrinsics
    )
    result = model.sample(
        target_image=target_image,
        reference_image=reference_image,
        reference_depth=reference_depth,
        K=intrinsics,
        reference_pose=reference_pose,
        target_pose=target_pose,
    )
    if isinstance(result, tuple):
        result = result[0]
    if result.size != original:
        result = result.resize(original, Image.LANCZOS)
    return result


def _fit_geoquery(
    target: Image.Image,
    reference: Image.Image,
    intrinsics: np.ndarray,
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """Shrink both views so the long side is at most ``MAX_GEOQUERY_SIDE``."""
    width, height = target.size
    long_side = max(width, height)
    if long_side <= MAX_GEOQUERY_SIDE:
        return target, reference, intrinsics
    scale = MAX_GEOQUERY_SIDE / long_side
    new_width = max(8, int(width * scale) // 8 * 8)
    new_height = max(8, int(height * scale) // 8 * 8)
    scaled = np.array(intrinsics, dtype=np.float64, copy=True)
    scaled[0, :] *= new_width / width
    scaled[1, :] *= new_height / height
    return (
        target.resize((new_width, new_height), Image.LANCZOS),
        reference.resize((new_width, new_height), Image.LANCZOS),
        scaled,
    )
