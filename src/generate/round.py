"""One closed-loop generation round: depth, restore, write back."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.generate.depth_scale import align_reference_depth
from src.generate.plan_round import plan_round
from src.generate.reference import nearest_training_index
from src.generate.write_back import write_generated_view


def run_generation_round(
    round_index: int,
    indices: list[int],
    poses: dict[int, torch.Tensor],
    images: dict[int, Image.Image],
    intrinsics: np.ndarray,
    scene_dir: Path,
    camera_id: int,
    first_image_id: int,
    render_color: Callable[[torch.Tensor], Image.Image],
    render_depth: Callable[[int], torch.Tensor],
    estimate_depth: Callable[[Image.Image], torch.Tensor],
    restore: Callable[..., Image.Image],
) -> dict:
    """Insert one Generated View between each adjacent training pair.

    Depth Anything 3 runs only on the nearest training view. Its depth is scaled
    by the median ratio against that view's rendered depth, then GeoQuery
    restores the rendered target. The returned index list includes the new views
    so the next round can pair them.
    """
    started = time.perf_counter()
    planned = plan_round(indices, poses)
    if not planned:
        return {
            "generation_seconds": 0.0,
            "written": [],
            "indices": list(indices),
            "poses": poses,
            "failure": "fewer than two registered training views",
        }
    ordered = sorted(indices)
    centers = torch.stack([poses[index][:3, 3] for index in ordered])
    updated_indices = list(indices)
    updated_poses = dict(poses)
    written = []
    image_id = first_image_id
    for offset, (_left, _right, pose) in enumerate(planned):
        local = nearest_training_index(pose[:3, 3], centers)
        reference = ordered[local]
        rendered_depth = render_depth(reference)
        estimated = estimate_depth(images[reference])
        estimated = _match_depth(estimated, rendered_depth)
        aligned, _ratio = align_reference_depth(rendered_depth, estimated)
        restored = restore(
            render_color(pose),
            images[reference],
            aligned.detach().cpu().numpy(),
            intrinsics,
            poses[reference].detach().cpu().numpy(),
            pose.detach().cpu().numpy(),
        )
        name = f"gen_r{round_index}_{offset:04d}.jpg"
        write_generated_view(scene_dir, restored, name, pose.detach().cpu().numpy(), camera_id, image_id)
        slot = _sort_slot(_left, _right, updated_indices)
        updated_indices.append(slot)
        updated_poses[slot] = pose.detach()
        written.append({"name": name, "slot": slot, "image": restored})
        image_id += 1
    return {
        "generation_seconds": time.perf_counter() - started,
        "written": [item["name"] for item in written],
        "slots": [item["slot"] for item in written],
        "images": {item["slot"]: item["image"] for item in written},
        "indices": updated_indices,
        "poses": updated_poses,
        "next_image_id": image_id,
    }


def _match_depth(estimated: torch.Tensor, rendered: torch.Tensor) -> torch.Tensor:
    """Put Depth Anything 3's map on the rendered depth grid.

    The model resizes its input, so the two maps are not the same shape. The
    scale is still the median ratio on the overlapping positive pixels.
    """
    depth = estimated.detach().float()
    while depth.ndim > 2:
        depth = depth[0]
    target = rendered.shape[-2:]
    if depth.shape == target:
        return depth
    resized = torch.nn.functional.interpolate(
        depth.view(1, 1, *depth.shape),
        size=target,
        mode="bilinear",
        align_corners=False,
    )
    return resized.view(*target)


def _sort_slot(left: int, right: int, used: list[int]) -> int:
    """Integer between the pair so the next round orders this view between them."""
    slot = (left + right) // 2
    if slot in used or slot == left or slot == right:
        raise ValueError(f"no free index between {left} and {right}")
    return slot
